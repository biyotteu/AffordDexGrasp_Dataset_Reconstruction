#!/usr/bin/env python3
"""
Phase F: 5-view RGB-D 렌더링 + Global Point Cloud 생성
- F1: 5대 카메라 세팅
- F2: RGB/Depth/Seg 렌더
- F3: Partial PC → Global PC merge
"""

import json
import os
import argparse
import subprocess
from pathlib import Path

import yaml
import numpy as np
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# blenderproc 버전 확인 + 자동 다운그레이드
# ============================================================
BPROC_TARGET_VERSION = "2.7.0"  # → Blender 3.5.1 → Python 3.10 (conda와 일치)
RENDER_WORKER_SCRIPT = Path("scripts") / "phase_f_render_worker.py"


def _ensure_blenderproc():
    """blenderproc 2.7.0이 설치되어 있는지 확인하고, 아니면 자동 설치/다운그레이드"""
    try:
        result = subprocess.run(
            ["python", "-c",
             "import blenderproc; print(blenderproc.__version__)"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            installed_ver = result.stdout.strip()
            if installed_ver == BPROC_TARGET_VERSION:
                print(f"  blenderproc {installed_ver} OK")
                return True
            else:
                print(f"  blenderproc {installed_ver} → {BPROC_TARGET_VERSION} 다운그레이드 중...")
        else:
            print(f"  blenderproc 미설치, {BPROC_TARGET_VERSION} 설치 중...")
    except FileNotFoundError:
        print(f"  blenderproc 미설치, {BPROC_TARGET_VERSION} 설치 중...")

    r = subprocess.run(
        ["pip", "install", f"blenderproc=={BPROC_TARGET_VERSION}"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"  ❌ blenderproc 설치 실패: {r.stderr[-500:]}")
        return False

    print(f"  blenderproc {BPROC_TARGET_VERSION} 설치 완료")
    return True


# ============================================================
# F1 + F2: BlenderProc 기반 5-view RGB-D 렌더링
# ============================================================
def run_rendering(cfg, max_scenes=None):
    """모든 scene 렌더링 (blenderproc run - 옛날 방식 그대로)"""
    print("=" * 60)
    print("[F1+F2] 5-view RGB-D 렌더링")
    print("=" * 60)

    worker_path = RENDER_WORKER_SCRIPT
    if not worker_path.exists():
        print(f"  ⚠️ Render worker 스크립트 없음: {worker_path}")
        return

    # blenderproc 2.7.0 확인/설치
    if not _ensure_blenderproc():
        print("  ❌ blenderproc 준비 실패")
        return

    scenes_dir = Path(cfg['paths']['scenes'])
    renders_dir = Path(cfg['paths']['renders'])
    renders_dir.mkdir(parents=True, exist_ok=True)
    textures_dir = str(Path(cfg['paths']['textures']).resolve())

    scene_dirs = [d for d in scenes_dir.iterdir() if d.is_dir() and (d / "job.json").exists()]
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    fail_count = 0
    for scene_dir in tqdm(scene_dirs, desc="  렌더링"):
        scene_id = scene_dir.name
        output = renders_dir / scene_id

        if output.exists() and (output / "rgb_cam0.png").exists():
            continue  # 이미 렌더링됨

        # 옛날 방식 그대로: blenderproc run (Blender 3.5.1 자동 다운로드)
        worker_abs = str(worker_path.resolve())
        cmd = [
            "blenderproc", "run",
            worker_abs,
            "--scene_dir", str(scene_dir),
            "--renders_dir", str(renders_dir),
            "--textures_dir", textures_dir,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )

            # 렌더링 결과 파일 존재 여부로 실제 성공 판단
            rgb0 = output / "rgb_cam0.png"
            render_actually_succeeded = rgb0.exists()

            if result.returncode != 0 or not render_actually_succeeded:
                stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                tb_start = -1
                for li, line in enumerate(stderr_lines):
                    if 'Traceback' in line or 'Error' in line:
                        tb_start = li
                if tb_start >= 0:
                    err_msg = '\n'.join(stderr_lines[tb_start:])
                else:
                    err_msg = '\n'.join(stderr_lines[-10:])

                reason = "returncode=" + str(result.returncode)
                if not render_actually_succeeded:
                    reason += ", rgb_cam0.png 없음"
                tqdm.write(f"    ⚠️ {scene_id} 렌더링 실패 ({reason}):")
                tqdm.write(f"       {err_msg[:500]}")

                log_dir = Path("logs/render_failures")
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / f"{scene_id}.log", 'w') as lf:
                    lf.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}")
                fail_count += 1
        except subprocess.TimeoutExpired:
            tqdm.write(f"    ⚠️ {scene_id} 타임아웃 (600s)")
            fail_count += 1
        except Exception as e:
            tqdm.write(f"    ⚠️ {scene_id} 에러: {e}")
            fail_count += 1

    print(f"\n  렌더링 실패: {fail_count}개 / 전체: {len(scene_dirs)}개")


# ============================================================
# F3: Partial PC → Global PC Merge
# ============================================================
def merge_point_clouds(cfg, max_scenes=None):
    """
    Depth + intrinsics/extrinsics → back-projection → world transform → merge
    Voxel downsample + outlier 제거 + table 제거(segmentation)
    """
    print("\n" + "=" * 60)
    print("[F3] Global Point Cloud 생성")
    print("=" * 60)

    renders_dir = Path(cfg['paths']['renders'])
    pc_dir = Path(cfg['paths']['pointclouds'])
    pc_dir.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d
    except ImportError:
        print("  ⚠️ open3d 미설치. pip install open3d")
        print("  numpy 기반 대안으로 진행...")
        o3d = None

    scene_dirs = [d for d in renders_dir.iterdir() if d.is_dir()]
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    num_points_target = cfg['affordance']['num_object_points']  # 4096
    import gc

    for idx, scene_dir in enumerate(tqdm(scene_dirs, desc="  PC 생성")):
        scene_id = scene_dir.name
        cam_file = scene_dir / "camera_params.json"

        if not cam_file.exists():
            continue

        # 이미 생성된 PC 스킵
        if (pc_dir / f"{scene_id}.ply").exists() or (pc_dir / f"{scene_id}.npz").exists():
            continue

        # 100개마다 GC 실행 (Open3D 메모리 누수 방지)
        if idx % 100 == 0:
            gc.collect()

        try:
            with open(cam_file) as f:
                cam_params = json.load(f)

            K = np.array(cam_params['intrinsics'])
            W = cam_params['image_width']
            H = cam_params['image_height']

            all_points = []
            all_colors = []

            for cam_idx in range(cam_params['num_cameras']):
                depth_file = scene_dir / f"depth_cam{cam_idx}.npy"
                rgb_file = scene_dir / f"rgb_cam{cam_idx}.png"
                seg_file = scene_dir / f"seg_cam{cam_idx}.npy"

                if not depth_file.exists():
                    continue

                depth = np.load(depth_file)

                # RGB 로드
                try:
                    from PIL import Image
                    rgb = np.array(Image.open(rgb_file)) / 255.0
                except:
                    rgb = np.ones((H, W, 3)) * 0.5

                # Segmentation 로드 (table 제거용)
                seg = None
                if seg_file.exists():
                    seg = np.load(seg_file)

                # Extrinsics (cam2world)
                cam2world = np.array(cam_params['extrinsics'][cam_idx])

                # Back-projection
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                u, v = np.meshgrid(np.arange(W), np.arange(H))
                z = depth

                # Valid depth mask
                valid = (z > 0) & (z < 10.0)  # reasonable range

                # Table 제거 (seg == 0이 table)
                if seg is not None:
                    valid = valid & (seg != 0)

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)
                colors = rgb[valid]

                # Camera → World 변환
                R = cam2world[:3, :3]
                t = cam2world[:3, 3]
                points_world = (R @ points_cam.T).T + t

                all_points.append(points_world)
                all_colors.append(colors)

            if not all_points:
                continue

            # Merge
            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)

            if o3d is not None:
                # Open3D 기반 처리
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_points)
                pcd.colors = o3d.utility.Vector3dVector(all_colors[:, :3])

                # Voxel downsample (config에서 설정)
                pc_cfg = cfg.get('pointcloud', {})
                voxel_size = pc_cfg.get('voxel_size', 0.002)
                pcd = pcd.voxel_down_sample(voxel_size)

                # Outlier 제거 (config에서 설정)
                nb_neighbors = pc_cfg.get('outlier_nb_neighbors', 20)
                std_ratio = pc_cfg.get('outlier_std_ratio', 2.0)
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

                # 최종 포인트 수 조정
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)

                if len(points) > num_points_target:
                    indices = np.random.choice(len(points), num_points_target, replace=False)
                    points = points[indices]
                    colors = colors[indices]
                elif len(points) < num_points_target and len(points) > 0:
                    repeat = num_points_target // len(points) + 1
                    points = np.tile(points, (repeat, 1))[:num_points_target]
                    colors = np.tile(colors, (repeat, 1))[:num_points_target]

                # 저장
                final_pcd = o3d.geometry.PointCloud()
                final_pcd.points = o3d.utility.Vector3dVector(points)
                final_pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(pc_dir / f"{scene_id}.ply"), final_pcd)

            else:
                # Numpy fallback
                if len(all_points) > num_points_target:
                    indices = np.random.choice(len(all_points), num_points_target, replace=False)
                    all_points = all_points[indices]
                    all_colors = all_colors[indices]

                # NPZ 저장
                np.savez_compressed(
                    pc_dir / f"{scene_id}.npz",
                    xyz=all_points.astype(np.float32),
                    rgb=all_colors[:, :3].astype(np.float32),
                )

        except Exception as e:
            print(f"    ⚠️ {scene_id} PC 생성 실패: {e}")
            continue

    # 통계
    pc_files = list(pc_dir.glob("*.ply")) + list(pc_dir.glob("*.npz"))
    print(f"\n  Global PC 생성: {len(pc_files)}개")
    print(f"  목표 포인트 수: {num_points_target}")


def verify_pointclouds(cfg):
    """포인트 클라우드 품질 확인 (좌표계, 스케일 등)"""
    print("\n  [검증] 포인트 클라우드 QC")

    pc_dir = Path(cfg['paths']['pointclouds'])
    pc_files = list(pc_dir.glob("*.ply")) + list(pc_dir.glob("*.npz"))

    if not pc_files:
        print("    ⚠️ PC 파일 없음")
        return

    try:
        import open3d as o3d
        use_o3d = True
    except:
        use_o3d = False

    stats = {
        "total": len(pc_files),
        "point_counts": [],
        "bbox_ranges": [],
    }

    for pc_file in pc_files[:min(10, len(pc_files))]:  # 샘플 10개만 체크
        if pc_file.suffix == '.ply' and use_o3d:
            pcd = o3d.io.read_point_cloud(str(pc_file))
            points = np.asarray(pcd.points)
        elif pc_file.suffix == '.npz':
            data = np.load(pc_file)
            points = data['xyz']
        else:
            continue

        stats["point_counts"].append(len(points))

        if len(points) > 0:
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            bbox_range = bbox_max - bbox_min
            stats["bbox_ranges"].append(bbox_range.tolist())

    if stats["point_counts"]:
        print(f"    포인트 수: mean={np.mean(stats['point_counts']):.0f}, "
              f"min={min(stats['point_counts'])}, max={max(stats['point_counts'])}")

    if stats["bbox_ranges"]:
        ranges = np.array(stats["bbox_ranges"])
        print(f"    BBox range (x): {ranges[:,0].mean():.3f}m")
        print(f"    BBox range (y): {ranges[:,1].mean():.3f}m")
        print(f"    BBox range (z): {ranges[:,2].mean():.3f}m")

        # 스케일 이상 감지
        if ranges.max() > 5.0:
            print("    ⚠️ 스케일 이상 감지! 좌표계 확인 필요")
        elif ranges.max() < 0.001:
            print("    ⚠️ 스케일 너무 작음! 단위 확인 필요")
        else:
            print("    ✅ 스케일 정상")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase F: 5-view 렌더링 + Global PC")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "render", "merge", "verify"],
                       default="all")
    parser.add_argument("--max_scenes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "render"]:
        run_rendering(cfg, max_scenes=args.max_scenes)

    if args.step in ["all", "merge"]:
        merge_point_clouds(cfg, max_scenes=args.max_scenes)

    if args.step in ["all", "verify"]:
        verify_pointclouds(cfg)

    print("\n" + "=" * 60)
    print("Phase F 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
