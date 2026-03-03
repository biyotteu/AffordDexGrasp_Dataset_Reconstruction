#!/usr/bin/env python3
"""좌표계 디버깅: Phase H의 실제 affordance_gt 출력 + mesh PC vs grasp params 비교"""
import json, yaml, sys
import numpy as np
import jsonlines
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_config(p="configs/pipeline_config.yaml"):
    with open(p) as f: return yaml.safe_load(f)

cfg = load_config()

# 1. semantic groups 로드
groups_path = Path(cfg['paths']['processed']) / "semantic_groups.json"
with open(groups_path) as f:
    groups = json.load(f)

# 2. meta 로드
meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
all_meta = {}
for mf in meta_dir.glob("*_meta.jsonl"):
    with jsonlines.open(mf) as r:
        for e in r:
            all_meta[e['sample_id']] = e

# 3. mesh index 로드
index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
mesh_index = {}
if index_path.exists():
    with open(index_path) as f:
        mesh_index = json.load(f)

# 4. affordance_gt 디렉토리
affordance_dir = Path(cfg['paths']['processed']) / "affordance_gt"

for gid, ginfo in list(groups.items())[:3]:
    scene_id = ginfo['scene_id']
    sample_ids = ginfo['sample_ids']

    print(f"\n{'='*60}")
    print(f"Group: {gid}, Scene: {scene_id}")
    print(f"Samples: {len(sample_ids)}")

    # --- A: Phase H 결과 (affordance_gt NPZ) ---
    npz_path = affordance_dir / f"{gid}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        pts = data['object_points']
        scores = data['affordance_scores']
        print(f"\n  [Affordance GT 결과]")
        print(f"    Points: {pts.shape}")
        print(f"    Obj center: {pts.mean(axis=0)}")
        print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"    Score mean: {scores.mean():.4f}")
        high_pct = (scores > 0.5).mean() * 100
        peak_pct = (scores > 0.8).mean() * 100
        print(f"    High (>0.5): {(scores > 0.5).sum()} pts ({high_pct:.1f}%)")
        print(f"    Peak (>0.8): {(scores > 0.8).sum()} pts ({peak_pct:.1f}%)")
        if 'raw_distances' in data:
            raw = data['raw_distances']
            print(f"    Raw dist: min={raw.min():.6f}, max={raw.max():.6f}, mean={raw.mean():.6f}")
    else:
        print(f"  ⚠️ Affordance GT 없음 ({npz_path})")

    # --- B: Mesh 기반 Object PC (Phase H가 사용하는 것) ---
    obj_points_mesh = None
    for sid in sample_ids:
        if sid in all_meta:
            obj_id = all_meta[sid].get('obj_id')
            if obj_id and obj_id in mesh_index:
                try:
                    mesh = trimesh.load(mesh_index[obj_id]['mesh_path'], force='mesh')
                    obj_points_mesh, _ = trimesh.sample.sample_surface(mesh, 4096)
                    print(f"\n  [Object Mesh PC] (Phase H가 실제 사용)")
                    print(f"    obj_id: {obj_id}")
                    print(f"    Center: {obj_points_mesh.mean(axis=0)}")
                    print(f"    X: [{obj_points_mesh[:,0].min():.4f}, {obj_points_mesh[:,0].max():.4f}]")
                    print(f"    Y: [{obj_points_mesh[:,1].min():.4f}, {obj_points_mesh[:,1].max():.4f}]")
                    print(f"    Z: [{obj_points_mesh[:,2].min():.4f}, {obj_points_mesh[:,2].max():.4f}]")
                except Exception as e:
                    print(f"  ⚠️ Mesh 로드 실패: {e}")
            break

    # --- C: Grasp 파라미터 ---
    translations = []
    for sid in sample_ids[:5]:
        if sid not in all_meta:
            continue
        m = all_meta[sid]
        t = np.array(m['translation'])
        translations.append(t)

        if len(translations) <= 2:
            r = np.array(m['rotation_aa'])
            q = np.array(m['joint_angles'])
            print(f"\n  [Grasp {sid}]")
            print(f"    translation: {t}")
            print(f"    rotation_aa: {r}")
            print(f"    joint_angles (first 6): {q[:6]}")

    if translations:
        translations = np.array(translations)
        print(f"\n  [Grasp translations 요약 ({len(translations)}개)]")
        print(f"    Mean: {translations.mean(axis=0)}")

    # --- D: 좌표계 비교 (mesh PC vs grasp) ---
    if obj_points_mesh is not None and len(translations) > 0:
        obj_center = obj_points_mesh.mean(axis=0)
        grasp_center = translations.mean(axis=0)
        dist = np.linalg.norm(obj_center - grasp_center)
        print(f"\n  [좌표계 비교 - Mesh vs Grasp]")
        print(f"    Mesh obj center: {obj_center}")
        print(f"    Grasp center:    {grasp_center}")
        print(f"    거리: {dist:.4f}m")
        if dist > 0.5:
            print(f"    ⚠️ 거리가 너무 큼! 좌표계 불일치")
        elif dist > 0.15:
            print(f"    ⚠️ 거리가 다소 큼")
        else:
            print(f"    ✅ 좌표계 일치 (hand가 object 근처)")
