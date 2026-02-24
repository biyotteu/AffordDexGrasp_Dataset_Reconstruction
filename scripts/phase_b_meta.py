#!/usr/bin/env python3
"""
Phase B: 내부 표준 메타로 변환 & 무결성 리포트
- B1: 표준 포맷(meta) 생성
- B2: 통계/QC
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import yaml
import numpy as np
import jsonlines
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# B1: 표준 포맷 (meta) 생성
# ============================================================
def generate_standard_meta(cfg):
    """
    DexGYS train/test JSON + OakInk index → 표준 메타 JSONL

    표준 포맷:
    {
        "sample_id": "train_00001",
        "split": "train",
        "cate_id": 5,
        "obj_id": "obj_123",
        "action_id": 2,
        "guidance": "grab the mug by its handle",
        "translation": [x, y, z],          # 3D
        "rotation_aa": [rx, ry, rz],        # axis-angle 3D
        "joint_angles": [...],              # 22D ShadowHand qpos
        "dex_grasp_raw": [...],             # original 28D
        "mesh_path": "data/oakink/shape/...",
        "mesh_type": "real" | "virtual"
    }
    """
    print("=" * 60)
    print("[B1] 표준 메타 생성")
    print("=" * 60)

    proc_dir = Path(cfg['paths']['processed'])
    proc_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = proc_dir / "dexgys_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Mesh index 로드
    index_path = proc_dir / "obj_mesh_index.json"
    if index_path.exists():
        with open(index_path) as f:
            mesh_index = json.load(f)
    else:
        print("  ⚠️ mesh index 없음 - mesh_path 비워둠")
        mesh_index = {}

    dexgys_dir = Path(cfg['paths']['dexgys'])
    total_samples = 0
    missing_mesh = 0
    broken_grasp = 0

    for split_name in ['train', 'test']:
        # 파일 찾기
        pattern = f"*{split_name}*.json"
        json_files = list(dexgys_dir.glob(pattern))
        if not json_files:
            print(f"  ⚠️ {split_name} 파일 없음")
            continue

        with open(json_files[0]) as f:
            data = json.load(f)

        # 데이터 리스트 추출
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and 'data' in data:
            samples = data['data']
        elif isinstance(data, dict):
            # 중첩 구조일 수 있음 - 평탄화 시도
            samples = []
            for key, val in data.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            item['_parent_key'] = key
                            samples.append(item)
                elif isinstance(val, dict):
                    val['_parent_key'] = key
                    samples.append(val)
        else:
            samples = []

        print(f"  [{split_name}] 원본 샘플 수: {len(samples)}")

        output_path = meta_dir / f"{split_name}_meta.jsonl"
        with jsonlines.open(output_path, mode='w') as writer:
            for idx, sample in enumerate(tqdm(samples, desc=f"  {split_name} 변환")):
                if not isinstance(sample, dict):
                    continue

                sample_id = f"{split_name}_{idx:06d}"

                # dex_grasp 추출 및 분해
                grasp_raw = None
                for key in ['dex_grasp', 'grasp', 'hand_pose', 'grasp_pose']:
                    if key in sample:
                        grasp_raw = np.array(sample[key], dtype=np.float32)
                        break

                if grasp_raw is None or grasp_raw.shape[-1] != 28:
                    broken_grasp += 1
                    continue

                # 28D 분해
                t = grasp_raw[..., :3].tolist()
                r = grasp_raw[..., 3:6].tolist()
                q = grasp_raw[..., 6:28].tolist()

                # obj_id 추출
                obj_id = None
                for key in ['obj_id', 'object_id', 'oid']:
                    if key in sample:
                        obj_id = str(sample[key])
                        break

                # cate_id, action_id
                cate_id = sample.get('cate_id', sample.get('category', sample.get('cat_id', -1)))
                action_id = sample.get('action_id', sample.get('intent', sample.get('action', -1)))

                # guidance / language
                guidance = ""
                for key in ['guidance', 'language', 'text', 'instruction', 'command']:
                    if key in sample:
                        guidance = sample[key] if isinstance(sample[key], str) else str(sample[key])
                        break

                # mesh 경로 매핑
                mesh_info = mesh_index.get(obj_id, {})
                mesh_path = mesh_info.get('mesh_path', '')
                mesh_type = mesh_info.get('type', 'unknown')

                if not mesh_path:
                    missing_mesh += 1

                meta_entry = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "cate_id": cate_id,
                    "obj_id": obj_id,
                    "action_id": action_id,
                    "guidance": guidance,
                    "translation": t if isinstance(t[0], float) else t,
                    "rotation_aa": r if isinstance(r[0], float) else r,
                    "joint_angles": q if isinstance(q[0], float) else q,
                    "dex_grasp_raw": grasp_raw.tolist(),
                    "mesh_path": mesh_path,
                    "mesh_type": mesh_type,
                }

                writer.write(meta_entry)
                total_samples += 1

        print(f"  [{split_name}] 변환 완료: {output_path}")

    print(f"\n  총 변환 샘플: {total_samples}")
    print(f"  누락 메쉬: {missing_mesh} ({missing_mesh/max(total_samples,1)*100:.1f}%)")
    print(f"  깨진 그래스프: {broken_grasp}")

    return {
        "total_samples": total_samples,
        "missing_mesh": missing_mesh,
        "broken_grasp": broken_grasp,
    }


# ============================================================
# B2: 통계 / QC
# ============================================================
def run_qc(cfg):
    """메타 데이터 통계 및 품질 검사"""
    print("\n" + "=" * 60)
    print("[B2] 통계/QC 리포트")
    print("=" * 60)

    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
    stats = {
        "total_samples": 0,
        "splits": {},
        "cate_distribution": {},
        "action_distribution": {},
        "guidance_stats": {},
        "obj_id_sample_counts": {},
        "bad_samples": [],
    }

    for split_name in ['train', 'test']:
        meta_file = meta_dir / f"{split_name}_meta.jsonl"
        if not meta_file.exists():
            continue

        cate_counter = Counter()
        action_counter = Counter()
        obj_counter = Counter()
        guidance_lengths = []
        guidance_set = set()
        bad = []

        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                stats["total_samples"] += 1

                cate_counter[str(entry.get('cate_id', 'unknown'))] += 1
                action_counter[str(entry.get('action_id', 'unknown'))] += 1
                obj_counter[str(entry.get('obj_id', 'unknown'))] += 1

                g = entry.get('guidance', '')
                guidance_lengths.append(len(g))
                guidance_set.add(g)

                # Bad sample check
                issues = []
                if not entry.get('mesh_path'):
                    issues.append("no_mesh")
                if entry.get('cate_id', -1) == -1:
                    issues.append("no_cate")
                if not g:
                    issues.append("no_guidance")

                # 그래스프 값 범위 체크
                t = entry.get('translation', [0, 0, 0])
                if isinstance(t, list) and len(t) == 3:
                    if any(abs(v) > 10 for v in t):
                        issues.append("extreme_translation")

                if issues:
                    bad.append({
                        "sample_id": entry['sample_id'],
                        "issues": issues,
                    })

        stats["splits"][split_name] = {
            "count": sum(cate_counter.values()),
            "num_categories": len(cate_counter),
            "num_objects": len(obj_counter),
            "num_actions": len(action_counter),
            "unique_guidance": len(guidance_set),
            "avg_guidance_length": np.mean(guidance_lengths) if guidance_lengths else 0,
        }

        stats["cate_distribution"][split_name] = dict(cate_counter.most_common())
        stats["action_distribution"][split_name] = dict(action_counter.most_common())

        # obj_id별 샘플 수 편차
        obj_counts = list(obj_counter.values())
        stats["obj_id_sample_counts"][split_name] = {
            "mean": float(np.mean(obj_counts)) if obj_counts else 0,
            "std": float(np.std(obj_counts)) if obj_counts else 0,
            "min": int(min(obj_counts)) if obj_counts else 0,
            "max": int(max(obj_counts)) if obj_counts else 0,
        }

        stats["bad_samples"].extend(bad)

        print(f"\n  [{split_name}]")
        print(f"    샘플 수: {stats['splits'][split_name]['count']}")
        print(f"    카테고리 수: {stats['splits'][split_name]['num_categories']}")
        print(f"    오브젝트 수: {stats['splits'][split_name]['num_objects']}")
        print(f"    유니크 guidance: {stats['splits'][split_name]['unique_guidance']}")
        print(f"    평균 guidance 길이: {stats['splits'][split_name]['avg_guidance_length']:.1f} chars")
        print(f"    obj당 샘플 수: mean={stats['obj_id_sample_counts'][split_name]['mean']:.1f}, "
              f"std={stats['obj_id_sample_counts'][split_name]['std']:.1f}")

    # Bad samples 저장
    bad_path = Path(cfg['paths']['processed']) / "bad_samples.jsonl"
    if stats["bad_samples"]:
        with jsonlines.open(bad_path, mode='w') as writer:
            for item in stats["bad_samples"]:
                writer.write(item)
        print(f"\n  ⚠️ Bad samples: {len(stats['bad_samples'])}개 → {bad_path}")
    else:
        print(f"\n  ✅ Bad samples 없음")

    # 통계 저장
    stats_path = Path(cfg['paths']['processed']) / "stats.json"
    # numpy/set 변환
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  통계 저장: {stats_path}")

    return stats


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase B: 표준 메타 생성 + QC")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "meta", "qc"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "meta"]:
        generate_standard_meta(cfg)

    if args.step in ["all", "qc"]:
        run_qc(cfg)

    print("\n" + "=" * 60)
    print("Phase B 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
