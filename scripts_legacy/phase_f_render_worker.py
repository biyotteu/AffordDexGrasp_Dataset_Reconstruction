#!/usr/bin/env python3
"""
BlenderProc 5-view RGB-D Rendering Worker
Run with: blenderproc run phase_f_render_worker.py --scene_dir <dir>
"""
import sys
import os
import json
import argparse
import numpy as np

import blenderproc as bproc


def render_scene(scene_dir, renders_dir):
    """5대 카메라로 RGB-D + Segmentation 렌더링"""
    bproc.init()

    # --- Scene 로드 ---
    job_file = os.path.join(scene_dir, "job.json")
    pose_file = os.path.join(scene_dir, "object_pose.json")

    with open(job_file) as f:
        job = json.load(f)

    scene_id = job["scene_id"]
    output_dir = os.path.join(renders_dir, scene_id)
    os.makedirs(output_dir, exist_ok=True)

    # 텍스처 적용 메쉬 로드 시도
    obj_id = job["obj_id"]
    textured_mesh = os.path.join("textures", obj_id, "textured_mesh.obj")
    original_mesh = job["mesh_path"]

    mesh_path = textured_mesh if os.path.exists(textured_mesh) else original_mesh

    # --- Table ---
    table_cfg = job["table"]
    table = bproc.object.create_primitive(
        "CUBE",
        scale=[table_cfg["size"][0]/2, table_cfg["size"][1]/2, table_cfg["size"][2]/2],
    )
    table.set_location([0, 0, table_cfg["height"] - table_cfg["size"][2]/2])
    table.set_name("table")
    table.set_cp("category_id", 0)

    # --- Object ---
    try:
        objs = bproc.loader.load_obj(mesh_path)
    except:
        objs = [bproc.loader.load_ply(mesh_path)]

    target_obj = objs[0] if isinstance(objs, list) else objs
    target_obj.set_name(f"object_{obj_id}")
    target_obj.set_cp("category_id", 1)

    # Object pose 적용
    if os.path.exists(pose_file):
        with open(pose_file) as f:
            pose_data = json.load(f)
        target_obj.set_location(pose_data["location"])
        target_obj.set_rotation_euler(pose_data["rotation_euler"])
        obj_center = np.array(pose_data["location"])
    else:
        obj_center = np.array([0, 0, table_cfg["height"] + 0.05])
        target_obj.set_location(obj_center.tolist())

    # --- Lighting ---
    light = bproc.types.Light()
    light.set_type("AREA")
    light.set_location([0, 0, 3])
    light.set_energy(300)

    # 환경광
    bproc.world.set_world_background_color([0.5, 0.5, 0.5])

    # --- 카메라 설정 ---
    cam_cfg = job["camera"]
    bproc.camera.set_intrinsics_from_blender_params(
        lens=35,
        image_resolution_x=640,
        image_resolution_y=480,
    )

    camera_poses = []

    # 4개 사선 카메라
    elevation = np.radians(cam_cfg["lateral_elevation_deg"])
    distance = cam_cfg["distance_from_center"]

    for i in range(cam_cfg["lateral_count"]):
        azimuth = np.radians(i * cam_cfg["lateral_azimuth_spacing_deg"])

        cam_x = obj_center[0] + distance * np.cos(azimuth) * np.cos(elevation)
        cam_y = obj_center[1] + distance * np.sin(azimuth) * np.cos(elevation)
        cam_z = obj_center[2] + distance * np.sin(elevation)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        rot = bproc.camera.rotation_from_forward_vec(obj_center - cam_pos)
        cam2world = bproc.math.build_transformation_mat(cam_pos, rot)
        bproc.camera.add_camera_pose(cam2world)
        camera_poses.append(cam2world.tolist())

    # 1개 top-down 카메라
    topdown_h = cam_cfg["topdown_height"]
    cam_pos = np.array([obj_center[0], obj_center[1], obj_center[2] + topdown_h])
    rot = bproc.camera.rotation_from_forward_vec(obj_center - cam_pos)
    cam2world = bproc.math.build_transformation_mat(cam_pos, rot)
    bproc.camera.add_camera_pose(cam2world)
    camera_poses.append(cam2world.tolist())

    # --- 렌더링 ---
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])

    data = bproc.renderer.render()

    # --- 저장 ---
    K = bproc.camera.get_intrinsics_as_K_matrix()

    cam_params = {
        "intrinsics": K.tolist(),
        "image_width": 640,
        "image_height": 480,
        "num_cameras": len(camera_poses),
        "extrinsics": camera_poses,
    }

    with open(os.path.join(output_dir, "camera_params.json"), 'w') as f:
        json.dump(cam_params, f, indent=2)

    # RGB, Depth, Segmentation 저장
    for i in range(len(camera_poses)):
        # RGB
        from PIL import Image
        rgb = data["colors"][i]
        Image.fromarray(rgb).save(os.path.join(output_dir, f"rgb_cam{i}.png"))

        # Depth (float32 as .npy)
        depth = data["depth"][i]
        np.save(os.path.join(output_dir, f"depth_cam{i}.npy"), depth)

        # Segmentation
        seg = data["category_id_segmaps"][i]
        np.save(os.path.join(output_dir, f"seg_cam{i}.npy"), seg)

    print(f"렌더링 완료: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--renders_dir", default="renders")
    args = parser.parse_args()

    render_scene(args.scene_dir, args.renders_dir)
