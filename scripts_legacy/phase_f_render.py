#!/usr/bin/env python3
"""
Phase F: 5-view RGB-D 렌더링 + Global Point Cloud 생성
- F1: 5대 카메라 세팅
- F2: RGB/Depth/Seg 렌더
- F3: Partial PC → Global PC merge
"""

import json
import argparse
import math
from pathlib import Path

import yaml
import numpy as np
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# F1 + F2: BlenderProc 기반 5-view RGB-D 렌더링
# ============================================================
RENDER_WORKER_SCRIPT = '''import blenderproc as bproc
import sys
import os
import json
import argparse
import numpy as np


def assign_material(blender_obj, color_rgba, roughness=0.5, metallic=0.0):
    """Assign a solid-color Principled BSDF material to an object if it has none."""
    import bpy
    mesh_obj = blender_obj.blender_obj
    if mesh_obj.data.materials:
        return
    mat = bpy.data.materials.new(name="auto_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color_rgba
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    mesh_obj.data.materials.append(mat)


def render_scene(scene_dir, renders_dir):
    """5-view RGB-D rendering - object only, no table/pedestal"""
    bproc.init()

    # --- Load scene ---
    job_file = os.path.join(scene_dir, "job.json")
    pose_file = os.path.join(scene_dir, "object_pose.json")

    with open(job_file) as f:
        job = json.load(f)

    scene_id = job["scene_id"]
    output_dir = os.path.join(renders_dir, scene_id)
    os.makedirs(output_dir, exist_ok=True)

    # Try textured mesh first, then uv_mesh, fallback to original
    obj_id = job["obj_id"]
    textured_mesh = os.path.join("textures", obj_id, "textured_mesh.obj")
    uv_mesh = os.path.join("textures", obj_id, "uv_mesh.obj")
    original_mesh = job["mesh_path"]

    if os.path.exists(textured_mesh):
        mesh_path = textured_mesh
        has_texture = True
    elif os.path.exists(uv_mesh):
        mesh_path = uv_mesh
        has_texture = True
    else:
        mesh_path = original_mesh
        has_texture = False

    import bpy
    from mathutils import Vector

    # --- Object only (no table, no pedestal) ---
    try:
        objs = bproc.loader.load_obj(mesh_path)
    except:
        objs = [bproc.loader.load_ply(mesh_path)]

    target_obj = objs[0] if isinstance(objs, list) else objs
    target_obj.set_name("object_" + str(obj_id))
    target_obj.set_cp("category_id", 1)

    # Check if the loaded mesh actually has REAL materials
    mesh_blender_check = target_obj.blender_obj
    actually_has_material = False
    if mesh_blender_check.data.materials:
        for mat in mesh_blender_check.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        w, h = node.image.size[0], node.image.size[1]
                        if w >= 64 and h >= 64:
                            actually_has_material = True
                            break
                if actually_has_material:
                    break

    if not actually_has_material:
        color_seed = hash(obj_id) % 1000
        np.random.seed(color_seed)
        r = np.random.uniform(0.3, 0.85)
        g = np.random.uniform(0.3, 0.85)
        b = np.random.uniform(0.3, 0.85)
        assign_material(target_obj, (r, g, b, 1.0))
        print("  Assigned random color: (" + str(round(r,2)) + ", " + str(round(g,2)) + ", " + str(round(b,2)) + ")")

    # --- Position object at origin ---
    target_obj.set_location([0, 0, 0])
    target_obj.set_rotation_euler([0, 0, 0])
    bpy.context.view_layer.update()

    mesh_blender = target_obj.blender_obj
    bbox_local = [Vector(c) for c in mesh_blender.bound_box]
    local_coords = np.array([[v.x, v.y, v.z] for v in bbox_local])
    obj_size = local_coords.max(axis=0) - local_coords.min(axis=0)
    obj_diagonal = float(np.linalg.norm(obj_size))

    # Apply Phase D pose rotation if available
    if os.path.exists(pose_file):
        with open(pose_file) as f:
            pose_data = json.load(f)
        rot_euler = pose_data.get("rotation_euler", [0, 0, 0])
        target_obj.set_rotation_euler(rot_euler)
    else:
        target_obj.set_rotation_euler([0, 0, 0])

    # Center object at origin (after rotation)
    bpy.context.view_layer.update()
    bbox_world = [mesh_blender.matrix_world @ Vector(c) for c in mesh_blender.bound_box]
    world_coords = np.array([[v.x, v.y, v.z] for v in bbox_world])
    obj_center_offset = (world_coords.max(axis=0) + world_coords.min(axis=0)) / 2.0
    target_obj.set_location(-obj_center_offset)

    bpy.context.view_layer.update()
    obj_center = np.array([0.0, 0.0, 0.0])  # object is now centered at origin

    print("Object bbox size: " + str([round(x, 4) for x in obj_size.tolist()]))
    print("Object diagonal: " + str(round(obj_diagonal, 4)) + "m")

    # --- Camera auto-fit ---
    lens_mm = 50
    half_fov = np.arctan(18.0 / lens_mm)
    target_fill = 0.45  # object fills ~45% of frame (larger since no table)
    auto_distance = max(0.12, min(2.0, obj_diagonal / (2.0 * np.tan(half_fov) * target_fill)))

    print("Auto camera distance: " + str(round(auto_distance, 4)) + "m")

    # --- Lighting (3-point studio setup around origin) ---
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([0, -0.5, 1.0])
    light.set_energy(150)

    light2 = bproc.types.Light()
    light2.set_type("POINT")
    light2.set_location([0.7, 0.7, 0.6])
    light2.set_energy(60)

    light3 = bproc.types.Light()
    light3.set_type("POINT")
    light3.set_location([-0.5, -0.3, 0.5])
    light3.set_energy(40)

    # Light background (neutral white-gray)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.92, 0.92, 0.94, 1.0)
        bg_node.inputs[1].default_value = 0.8  # ambient strength

    # --- Camera setup ---
    cam_cfg = job["camera"]
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=lens_mm)

    camera_poses = []
    elevation = np.radians(cam_cfg["lateral_elevation_deg"])

    # 4 lateral cameras
    for i in range(cam_cfg["lateral_count"]):
        azimuth = np.radians(i * cam_cfg["lateral_azimuth_spacing_deg"])

        cam_x = auto_distance * np.cos(azimuth) * np.cos(elevation)
        cam_y = auto_distance * np.sin(azimuth) * np.cos(elevation)
        cam_z = auto_distance * np.sin(elevation)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = obj_center - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Build camera matrix manually to guarantee Z-up orientation
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # forward is nearly parallel to up, use Y as fallback
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # BlenderProc cam2world: columns = [right, up, -forward, pos]
        # Blender camera looks along -Z, Y is up in camera space
        R = np.column_stack([right, cam_up, -forward])
        cam2world = np.eye(4)
        cam2world[:3, :3] = R
        cam2world[:3, 3] = cam_pos

        bproc.camera.add_camera_pose(cam2world)
        camera_poses.append(cam2world.tolist())

    # 1 top-down camera
    cam_pos_td = np.array([0.0, 0.0, auto_distance * 0.85])
    forward_td = np.array([0.0, 0.0, -1.0])
    right_td = np.array([1.0, 0.0, 0.0])
    up_td = np.array([0.0, 1.0, 0.0])
    R_td = np.column_stack([right_td, up_td, -forward_td])
    cam2world_td = np.eye(4)
    cam2world_td[:3, :3] = R_td
    cam2world_td[:3, 3] = cam_pos_td
    bproc.camera.add_camera_pose(cam2world_td)
    camera_poses.append(cam2world_td.tolist())

    # --- Render ---
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    try:
        bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    except Exception:
        pass

    data = bproc.renderer.render()

    # --- Save outputs ---
    K = bproc.camera.get_intrinsics_as_K_matrix()

    cam_params = {
        "intrinsics": K.tolist(),
        "image_width": 640,
        "image_height": 480,
        "num_cameras": len(camera_poses),
        "extrinsics": camera_poses,
        "lens_mm": lens_mm,
    }

    with open(os.path.join(output_dir, "camera_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)

    from PIL import Image
    for i in range(len(camera_poses)):
        rgb = data["colors"][i]
        Image.fromarray(rgb).save(os.path.join(output_dir, "rgb_cam" + str(i) + ".png"))

        depth = data["depth"][i]
        np.save(os.path.join(output_dir, "depth_cam" + str(i) + ".npy"), depth)

        # Segmentation 저장 (배경 필터링용)
        for seg_key in ["category_id_segmaps", "class_segmaps", "segmentation"]:
            if seg_key in data:
                seg = data[seg_key][i]
                np.save(os.path.join(output_dir, "seg_cam" + str(i) + ".npy"), seg)
                break

    print("Render complete: " + output_dir)
    print("  Object: " + str(obj_id) + " | Textured: " + str(has_texture))
    print("  Camera distance: " + str(round(auto_distance, 3)) + "m | Lens: " + str(lens_mm) + "mm")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--renders_dir", default="renders")
    args = parser.parse_args()

    render_scene(args.scene_dir, args.renders_dir)
'''


def create_render_worker(cfg):
    """렌더링 worker 스크립트 생성"""
    worker_path = Path("scripts") / "phase_f_render_worker.py"
    with open(worker_path, 'w', encoding='utf-8') as f:
        f.write(RENDER_WORKER_SCRIPT)
    print(f"  Render worker: {worker_path}")
    return worker_path


def run_rendering(cfg, max_scenes=None):
    """모든 scene 렌더링"""
    print("=" * 60)
    print("[F1+F2] 5-view RGB-D 렌더링")
    print("=" * 60)

    import subprocess

    worker_path = create_render_worker(cfg)
    scenes_dir = Path(cfg['paths']['scenes'])
    renders_dir = Path(cfg['paths']['renders'])
    renders_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = [d for d in scenes_dir.iterdir() if d.is_dir() and (d / "job.json").exists()]
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    for scene_dir in tqdm(scene_dirs, desc="  렌더링"):
        scene_id = scene_dir.name
        output = renders_dir / scene_id

        if output.exists() and (output / "rgb_cam0.png").exists():
            continue  # 이미 렌더링됨

        cmd = [
            "blenderproc", "run",
            str(worker_path),
            "--scene_dir", str(scene_dir),
            "--renders_dir", str(renders_dir),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                # stderr에서 실제 에러만 추출 (Traceback 또는 Error 포함 라인)
                stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
                # Traceback 찾기
                tb_start = -1
                for li, line in enumerate(stderr_lines):
                    if 'Traceback' in line:
                        tb_start = li
                if tb_start >= 0:
                    err_msg = '\n'.join(stderr_lines[tb_start:])
                else:
                    err_msg = '\n'.join(stderr_lines[-10:])
                print(f"    ⚠️ {scene_id} 렌더링 실패:")
                print(f"       {err_msg[:500]}")

                # 실패 로그 저장
                log_dir = Path("logs/render_failures")
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(log_dir / f"{scene_id}.log", 'w') as lf:
                    lf.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}")
        except Exception as e:
            print(f"    ⚠️ {scene_id} 에러: {e}")


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

                # Voxel downsample
                voxel_size = 0.002  # 2mm
                pcd = pcd.voxel_down_sample(voxel_size)

                # Outlier 제거
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

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
