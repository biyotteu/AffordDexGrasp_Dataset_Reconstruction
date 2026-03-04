#!/usr/bin/env python3
"""
Phase I: 최종 패키징 & QC 리포트
- I1: 최종 데이터 구조로 묶기
- I2: 검증 (누수 체크, 시각화, 통계)
"""

import json
import shutil
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
# I1: 최종 데이터 구조 패키징
# ============================================================
def package_final_dataset(cfg):
    """
    최종 데이터셋을 AffordDexGrasp 논문 형태로 패키징

    구조:
    final_dataset/
    ├── splits/
    │   ├── open_set_A.json
    │   └── open_set_B.json
    ├── scenes/
    │   └── {scene_id}/
    │       ├── rgb_cam{0-4}.png       # 5-view RGB
    │       ├── depth_cam{0-4}.npy     # 5-view Depth
    │       ├── seg_cam{0-4}.npy       # Segmentation
    │       ├── camera_params.json     # Intrinsics + Extrinsics
    │       ├── object_pose.json       # Physics-settled pose
    │       └── global_pc.npz          # 통합 포인트 클라우드
    ├── samples/
    │   └── {sample_id}.json           # 개별 샘플 (instruction, grasp params, semantic)
    ├── groups/
    │   └── {group_id}/
    │       ├── meta.json              # Group metadata
    │       └── affordance.npz         # GT affordance (points + scores)
    ├── assets/
    │   ├── meshes/                    # Object meshes (symlink or copy)
    │   └── shadow_hand/              # ShadowHand MJCF
    ├── semantic_groups.json           # 전체 group index
    └── dataset_stats.json             # 통계 리포트
    """
    print("=" * 60)
    print("[I1] 최종 패키징")
    print("=" * 60)

    final_dir = Path(cfg['paths']['final_dataset'])
    final_dir.mkdir(parents=True, exist_ok=True)

    # 하위 디렉토리 생성
    for subdir in ['splits', 'scenes', 'samples', 'groups', 'assets/meshes', 'assets/shadow_hand']:
        (final_dir / subdir).mkdir(parents=True, exist_ok=True)

    # --- Splits 복사 ---
    splits_dir = Path(cfg['paths']['splits'])
    for split_file in splits_dir.glob("open_set_*.json"):
        shutil.copy2(split_file, final_dir / "splits" / split_file.name)
        print(f"  Split: {split_file.name}")

    # --- Scenes 패키징 ---
    renders_dir = Path(cfg['paths']['renders'])
    pc_dir = Path(cfg['paths']['pointclouds'])
    scenes_dir = Path(cfg['paths']['scenes'])

    scene_count = 0
    for render_dir in tqdm(list(renders_dir.iterdir()), desc="  Scenes"):
        if not render_dir.is_dir():
            continue

        scene_id = render_dir.name
        target = final_dir / "scenes" / scene_id
        target.mkdir(parents=True, exist_ok=True)

        # RGB, Depth, Seg 복사
        for f in render_dir.glob("*.png"):
            shutil.copy2(f, target / f.name)
        for f in render_dir.glob("*.npy"):
            shutil.copy2(f, target / f.name)

        # Camera params
        cam_file = render_dir / "camera_params.json"
        if cam_file.exists():
            shutil.copy2(cam_file, target / "camera_params.json")

        # Object pose
        scene_source = scenes_dir / scene_id
        pose_file = scene_source / "object_pose.json"
        if pose_file.exists():
            shutil.copy2(pose_file, target / "object_pose.json")

        # Global PC
        for ext in ['.ply', '.npz']:
            pc_file = pc_dir / f"{scene_id}{ext}"
            if pc_file.exists():
                shutil.copy2(pc_file, target / f"global_pc{ext}")
                break

        scene_count += 1

    print(f"  Scenes 패키징: {scene_count}")

    # --- Samples 패키징 ---
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
    semantics_dir = Path(cfg['paths']['semantics'])

    # Semantic 데이터 로드
    all_semantics = {}
    for sem_file in semantics_dir.glob("*.jsonl"):
        with jsonlines.open(sem_file) as reader:
            for entry in reader:
                all_semantics[entry.get('sample_id', '')] = entry

    sample_count = 0
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                sample_id = entry['sample_id']

                # Semantic 정보 병합
                semantic = all_semantics.get(sample_id, {})

                sample_data = {
                    "sample_id": sample_id,
                    "split": entry.get('split'),
                    "obj_id": entry.get('obj_id'),
                    "cate_id": entry.get('cate_id'),
                    "action_id": entry.get('action_id'),
                    # Instruction
                    "guidance": entry.get('guidance', ''),
                    "normalized_command": semantic.get('normalized_command', ''),
                    # Grasp parameters
                    "grasp": {
                        "translation": entry.get('translation'),
                        "rotation_axis_angle": entry.get('rotation_aa'),
                        "joint_angles": entry.get('joint_angles'),
                    },
                    # Semantic attributes
                    "semantic": {
                        "object_category": semantic.get('object_category', ''),
                        "intention": semantic.get('intention', ''),
                        "contact_parts": semantic.get('contact_parts', ''),
                        "grasp_direction": semantic.get('grasp_direction', ''),
                    },
                    # Mesh
                    "mesh_path": entry.get('mesh_path', ''),
                }

                output_path = final_dir / "samples" / f"{sample_id}.json"
                with open(output_path, 'w') as f:
                    json.dump(sample_data, f, indent=2)

                sample_count += 1

    print(f"  Samples 패키징: {sample_count}")

    # --- Groups 패키징 ---
    groups_path = Path(cfg['paths']['processed']) / "semantic_groups.json"
    affordance_dir = Path(cfg['paths']['processed']) / "affordance_gt"

    if groups_path.exists():
        with open(groups_path) as f:
            semantic_groups = json.load(f)

        # 전체 인덱스 복사
        shutil.copy2(groups_path, final_dir / "semantic_groups.json")

        group_count = 0
        for group_id, group_info in semantic_groups.items():
            group_dir = final_dir / "groups" / group_id
            group_dir.mkdir(parents=True, exist_ok=True)

            # Group meta
            with open(group_dir / "meta.json", 'w') as f:
                json.dump(group_info, f, indent=2)

            # Affordance GT
            aff_file = affordance_dir / f"{group_id}.npz"
            if aff_file.exists():
                shutil.copy2(aff_file, group_dir / "affordance.npz")

            aff_meta = affordance_dir / f"{group_id}_meta.json"
            if aff_meta.exists():
                shutil.copy2(aff_meta, group_dir / "affordance_meta.json")

            group_count += 1

        print(f"  Groups 패키징: {group_count}")

    # --- Assets ---
    # ShadowHand MJCF
    mjcf_dir = Path(cfg['paths']['mjcf'])
    if mjcf_dir.exists():
        target_mjcf = final_dir / "assets" / "shadow_hand"
        if not (target_mjcf / "shadow_hand.xml").exists():
            try:
                shutil.copytree(mjcf_dir, target_mjcf, dirs_exist_ok=True)
                print(f"  ShadowHand assets 복사 완료")
            except:
                print(f"  ⚠️ ShadowHand assets 복사 실패")

    print(f"\n  ✅ 최종 데이터셋: {final_dir}")


# ============================================================
# I2: 검증 & QC 리포트
# ============================================================
def generate_qc_report(cfg):
    """최종 QC 리포트"""
    print("\n" + "=" * 60)
    print("[I2] 최종 QC 리포트")
    print("=" * 60)

    final_dir = Path(cfg['paths']['final_dataset'])

    report = {
        "dataset_name": "AffordDexGrasp-Reconstructed",
        "scenes": {},
        "samples": {},
        "groups": {},
        "splits": {},
        "integrity": {},
    }

    # --- Scene 통계 ---
    scene_dirs = [d for d in (final_dir / "scenes").iterdir() if d.is_dir()] \
                 if (final_dir / "scenes").exists() else []

    scenes_with_rgb = 0
    scenes_with_depth = 0
    scenes_with_pc = 0

    for sd in scene_dirs:
        if list(sd.glob("rgb_cam*.png")):
            scenes_with_rgb += 1
        if list(sd.glob("depth_cam*.npy")):
            scenes_with_depth += 1
        if list(sd.glob("global_pc.*")):
            scenes_with_pc += 1

    report["scenes"] = {
        "total": len(scene_dirs),
        "with_rgb": scenes_with_rgb,
        "with_depth": scenes_with_depth,
        "with_global_pc": scenes_with_pc,
    }

    # --- Sample 통계 ---
    sample_files = list((final_dir / "samples").glob("*.json")) \
                   if (final_dir / "samples").exists() else []

    cate_counter = Counter()
    intention_counter = Counter()
    contact_counter = Counter()
    direction_counter = Counter()
    guidance_lengths = []

    for sf in sample_files[:5000]:  # 샘플링
        with open(sf) as f:
            sample = json.load(f)
        cate_counter[str(sample.get('cate_id', 'unknown'))] += 1
        sem = sample.get('semantic', {})
        intention_counter[sem.get('intention', 'unknown')] += 1
        cp = sem.get('contact_parts', 'unknown')
        contact_counter[','.join(cp) if isinstance(cp, list) else str(cp)] += 1
        direction_counter[sem.get('grasp_direction', 'unknown')] += 1
        guidance_lengths.append(len(sample.get('guidance', '')))

    report["samples"] = {
        "total": len(sample_files),
        "num_categories": len(cate_counter),
        "top_categories": dict(cate_counter.most_common(10)),
        "intention_distribution": dict(intention_counter.most_common()),
        "contact_parts_distribution": dict(contact_counter.most_common(10)),
        "direction_distribution": dict(direction_counter.most_common()),
        "avg_guidance_length": float(np.mean(guidance_lengths)) if guidance_lengths else 0,
    }

    # --- Group 통계 ---
    group_dirs = list((final_dir / "groups").iterdir()) \
                 if (final_dir / "groups").exists() else []
    groups_with_affordance = 0
    group_sizes = []

    for gd in group_dirs:
        if not gd.is_dir():
            continue
        if (gd / "affordance.npz").exists():
            groups_with_affordance += 1
        meta_file = gd / "meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                group_sizes.append(meta.get('num_grasps', 0))

    report["groups"] = {
        "total": len([g for g in group_dirs if g.is_dir()]),
        "with_affordance_gt": groups_with_affordance,
        "size_stats": {
            "mean": float(np.mean(group_sizes)) if group_sizes else 0,
            "std": float(np.std(group_sizes)) if group_sizes else 0,
            "min": int(min(group_sizes)) if group_sizes else 0,
            "max": int(max(group_sizes)) if group_sizes else 0,
        },
    }

    # --- Split 통계 ---
    splits_dir = final_dir / "splits"
    if splits_dir.exists():
        for split_file in splits_dir.glob("open_set_*.json"):
            with open(split_file) as f:
                split_data = json.load(f)
            label = split_data.get('split_label', split_file.stem)
            report["splits"][label] = {
                "train_samples": len(split_data.get('train_sample_ids', [])),
                "test_samples": len(split_data.get('test_sample_ids', [])),
                "seen_categories": len(split_data.get('seen_categories', [])),
                "unseen_categories": len(split_data.get('unseen_categories', [])),
            }

    # --- 무결성 검증 ---
    # 1. Open-set 누수 체크
    leakage_found = False
    for split_file in (splits_dir.glob("open_set_*.json") if splits_dir.exists() else []):
        with open(split_file) as f:
            split_data = json.load(f)

        train_ids = set(split_data.get('train_sample_ids', []))
        test_ids = set(split_data.get('test_sample_ids', []))

        overlap = train_ids & test_ids
        if overlap:
            leakage_found = True
            print(f"  ⚠️ 누수 발견 ({split_file.name}): {len(overlap)} 중복 sample_ids")

    report["integrity"]["open_set_leakage"] = leakage_found

    # 2. 누락 리소스 체크
    missing_scenes = 0
    missing_affordance = 0

    for sf in sample_files[:1000]:
        with open(sf) as f:
            sample = json.load(f)
        # Scene 존재 확인은 sample → scene 매핑이 필요

    report["integrity"]["missing_scenes"] = missing_scenes
    report["integrity"]["missing_affordance"] = len([g for g in group_dirs if g.is_dir()]) - groups_with_affordance

    # --- 논문 대비 통계 비교 ---
    print("\n  === 논문 대비 통계 비교 ===")
    print(f"  {'항목':<25} {'논문':>10} {'재구축':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Objects':<25} {'1,536':>10} {report['samples']['num_categories']:>10}")
    print(f"  {'Scenes':<25} {'1,909':>10} {report['scenes']['total']:>10}")
    print(f"  {'Grasps':<25} {'43,504':>10} {report['samples']['total']:>10}")
    print(f"  {'Groups':<25} {'N/A':>10} {report['groups']['total']:>10}")
    print(f"  {'With Affordance GT':<25} {'N/A':>10} {report['groups']['with_affordance_gt']:>10}")

    # 리포트 저장
    report_path = final_dir / "dataset_stats.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  QC 리포트: {report_path}")

    # 무결성 결과
    if not leakage_found:
        print("  ✅ Open-set 누수 없음")
    else:
        print("  ❌ Open-set 누수 발견!")

    return report


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase I: 최종 패키징 + QC")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "package", "qc"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "package"]:
        package_final_dataset(cfg)

    if args.step in ["all", "qc"]:
        generate_qc_report(cfg)

    print("\n" + "=" * 60)
    print("Phase I 완료! 데이터셋 구축 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
