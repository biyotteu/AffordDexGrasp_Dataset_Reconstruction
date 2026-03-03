#!/usr/bin/env python3
"""
Phase C: Open-set Split 생성 (A/B)
- C1: Open-set A/B 정의
  * Split A: unseen cate set 1
  * Split B: unseen cate set 2
  * seen cate의 obj 80% train / 20% test + unseen obj 전부 test
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import yaml
import jsonlines
import numpy as np
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_open_set_splits(cfg):
    """
    Open-set A/B Split 생성

    논문 규칙:
    1. 전체 카테고리를 seen/unseen으로 나눔
    2. seen 카테고리의 obj: 80% train / 20% test
    3. unseen 카테고리의 obj: 전부 test
    4. Split A와 B는 서로 다른 unseen 카테고리를 사용
    """
    print("=" * 60)
    print("[C1] Open-set Split 생성")
    print("=" * 60)

    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
    splits_dir = Path(cfg['paths']['splits'])
    splits_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg['open_set']['seed']
    unseen_ratio = cfg['open_set']['unseen_ratio']
    seen_train_ratio = cfg['open_set']['seen_train_ratio']

    # 모든 메타 데이터 로드
    all_samples = []
    for split_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(split_file) as reader:
            for entry in reader:
                all_samples.append(entry)

    print(f"  총 샘플: {len(all_samples)}")

    # 카테고리별 오브젝트 수집
    cate_to_objs = defaultdict(set)
    obj_to_samples = defaultdict(list)

    for sample in all_samples:
        cate_id = str(sample.get('cate_id', 'unknown'))
        obj_id = str(sample.get('obj_id', 'unknown'))
        cate_to_objs[cate_id].add(obj_id)
        obj_to_samples[obj_id].append(sample['sample_id'])

    all_categories = sorted(cate_to_objs.keys())
    num_categories = len(all_categories)
    num_unseen = max(1, int(num_categories * unseen_ratio))

    print(f"  카테고리 수: {num_categories}")
    print(f"  Unseen 카테고리 수 (각 split): {num_unseen}")
    print(f"  Unseen 비율: {unseen_ratio*100:.0f}%")

    # 카테고리를 셔플하여 A/B split 생성
    rng = random.Random(seed)
    shuffled_cats = all_categories.copy()
    rng.shuffle(shuffled_cats)

    # Split A: 앞에서 num_unseen개를 unseen으로
    # Split B: 뒤에서 num_unseen개를 unseen으로 (겹치지 않게)
    unseen_a = set(shuffled_cats[:num_unseen])
    unseen_b = set(shuffled_cats[num_unseen:2*num_unseen])

    # 만약 카테고리가 충분하지 않으면 겹칠 수 있음
    if len(unseen_b) < num_unseen:
        # 부족한 만큼 나머지에서 추가
        remaining = [c for c in shuffled_cats if c not in unseen_a and c not in unseen_b]
        unseen_b.update(remaining[:num_unseen - len(unseen_b)])

    for split_label, unseen_cats in [("A", unseen_a), ("B", unseen_b)]:
        seen_cats = set(all_categories) - unseen_cats

        split_data = {
            "split_label": split_label,
            "seed": seed,
            "unseen_categories": sorted(unseen_cats),
            "seen_categories": sorted(seen_cats),
            "train_sample_ids": [],
            "test_sample_ids": [],
            "stats": {},
        }

        train_ids = []
        test_ids = []

        # Seen 카테고리: obj 80% train / 20% test
        for cate in sorted(seen_cats):
            objs = sorted(cate_to_objs[cate])
            rng_split = random.Random(seed + hash(cate))  # deterministic per category
            rng_split.shuffle(objs)

            n_train = max(1, int(len(objs) * seen_train_ratio))
            train_objs = set(objs[:n_train])
            test_objs = set(objs[n_train:])

            for obj in train_objs:
                train_ids.extend(obj_to_samples[obj])
            for obj in test_objs:
                test_ids.extend(obj_to_samples[obj])

        # Unseen 카테고리: 전부 test
        for cate in sorted(unseen_cats):
            for obj in cate_to_objs[cate]:
                test_ids.extend(obj_to_samples[obj])

        split_data["train_sample_ids"] = sorted(set(train_ids))
        split_data["test_sample_ids"] = sorted(set(test_ids))

        # 통계
        split_data["stats"] = {
            "num_seen_categories": len(seen_cats),
            "num_unseen_categories": len(unseen_cats),
            "num_train_samples": len(split_data["train_sample_ids"]),
            "num_test_samples": len(split_data["test_sample_ids"]),
            "num_train_objects": len([o for c in seen_cats for o in cate_to_objs[c]][:int(len(cate_to_objs)*seen_train_ratio)]),
            "num_test_objects": len(set([
                s.get('obj_id') for s in all_samples
                if s['sample_id'] in set(split_data["test_sample_ids"])
            ][:100])),  # approximate
        }

        # 저장
        output_path = splits_dir / f"open_set_{split_label}.json"
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        print(f"\n  [Split {split_label}]")
        print(f"    Unseen categories: {sorted(unseen_cats)[:5]}{'...' if len(unseen_cats)>5 else ''}")
        print(f"    Train samples: {len(split_data['train_sample_ids'])}")
        print(f"    Test samples: {len(split_data['test_sample_ids'])}")
        print(f"    → {output_path}")

    # 누수 검증
    verify_no_leakage(cfg)


def verify_no_leakage(cfg):
    """Open-set 누수 검증: 같은 obj_id가 train/test에 섞이지 않았는지"""
    print("\n  [검증] Open-set 누수 체크")

    splits_dir = Path(cfg['paths']['splits'])
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"

    # 메타 로드하여 sample_id → obj_id 매핑
    sid_to_obj = {}
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                sid_to_obj[entry['sample_id']] = entry.get('obj_id', 'unknown')

    for split_file in splits_dir.glob("open_set_*.json"):
        with open(split_file) as f:
            split_data = json.load(f)

        label = split_data['split_label']
        train_objs = set(sid_to_obj.get(sid, 'unk') for sid in split_data['train_sample_ids'])
        test_objs = set(sid_to_obj.get(sid, 'unk') for sid in split_data['test_sample_ids'])

        leaked = train_objs & test_objs
        # Note: seen category의 test split과 train split에서 같은 카테고리의 다른 obj는 OK
        # 하지만 같은 obj_id가 양쪽에 있으면 누수
        leaked_objs = set()
        train_obj_set = defaultdict(set)
        test_obj_set = defaultdict(set)

        for sid in split_data['train_sample_ids']:
            oid = sid_to_obj.get(sid, 'unk')
            train_obj_set[oid].add(sid)

        for sid in split_data['test_sample_ids']:
            oid = sid_to_obj.get(sid, 'unk')
            if oid in train_obj_set:
                leaked_objs.add(oid)
            test_obj_set[oid].add(sid)

        if leaked_objs:
            print(f"    ⚠️ Split {label}: {len(leaked_objs)}개 obj_id 누수! {list(leaked_objs)[:5]}")
        else:
            print(f"    ✅ Split {label}: 누수 없음")


def main():
    parser = argparse.ArgumentParser(description="Phase C: Open-set Split 생성")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_open_set_splits(cfg)

    print("\n" + "=" * 60)
    print("Phase C 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
