#!/usr/bin/env python3
"""
Phase F Render Worker: 5-view RGB-D 렌더링 (object only, auto-fit camera)
실행: blenderproc run phase_f_render_worker.py --scene_dir <dir> --renders_dir <dir>
"""
import blenderproc as bproc
import sys
import os
import json
import argparse
import numpy as np


def assign_material(blender_obj, color_rgba, roughness=0.5, metallic=0.0):
    """재질이 없는 오브젝트에 Principled BSDF 재질 할당"""
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

    # 텍스처 적용된 메쉬 우선, UV 메쉬, 원본 순서로 시도
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

    # --- Object 로드 (테이블 없이 오브젝트만) ---
    try:
        objs = bproc.loader.load_obj(mesh_path)
    except:
        objs = [bproc.loader.load_ply(mesh_path)]

    target_obj = objs[0] if isinstance(objs, list) else objs
    target_obj.set_name("object_" + str(obj_id))
    target_obj.set_cp("category_id", 1)

    # 실제 재질이 있는지 확인
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

    # --- 원점에 오브젝트 배치 ---
    target_obj.set_location([0, 0, 0])
    target_obj.set_rotation_euler([0, 0, 0])
    bpy.context.view_layer.update()

    mesh_blender = target_obj.blender_obj
    bbox_local = [Vector(c) for c in mesh_blender.bound_box]
    local_coords = np.array([[v.x, v.y, v.z] for v in bbox_local])
    obj_size = local_coords.max(axis=0) - local_coords.min(axis=0)
    obj_diagonal = float(np.linalg.norm(obj_size))

    # Phase D pose 회전 적용
    if os.path.exists(pose_file):
        with open(pose_file) as f:
            pose_data = json.load(f)
        rot_euler = pose_data.get("rotation_euler", [0, 0, 0])
        target_obj.set_rotation_euler(rot_euler)
    else:
        target_obj.set_rotation_euler([0, 0, 0])

    # 회전 후 원점 중심 정렬
    bpy.context.view_layer.update()
    bbox_world = [mesh_blender.matrix_world @ Vector(c) for c in mesh_blender.bound_box]
    world_coords = np.array([[v.x, v.y, v.z] for v in bbox_world])
    obj_center_offset = (world_coords.max(axis=0) + world_coords.min(axis=0)) / 2.0
    target_obj.set_location(-obj_center_offset)

    bpy.context.view_layer.update()
    obj_center = np.array([0.0, 0.0, 0.0])

    print("Object bbox size: " + str([round(x, 4) for x in obj_size.tolist()]))
    print("Object diagonal: " + str(round(obj_diagonal, 4)) + "m")

    # --- 카메라 자동 거리 계산 ---
    lens_mm = 50
    half_fov = np.arctan(18.0 / lens_mm)
    target_fill = 0.45
    auto_distance = max(0.12, min(2.0, obj_diagonal / (2.0 * np.tan(half_fov) * target_fill)))

    print("Auto camera distance: " + str(round(auto_distance, 4)) + "m")

    # --- 3-point 스튜디오 조명 ---
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

    # 배경색 (중립 흰회색)
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.92, 0.92, 0.94, 1.0)
        bg_node.inputs[1].default_value = 0.8

    # --- 카메라 설정 ---
    cam_cfg = job["camera"]
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=lens_mm)

    camera_poses = []
    elevation = np.radians(cam_cfg["lateral_elevation_deg"])

    # 4개 사선 카메라
    for i in range(cam_cfg["lateral_count"]):
        azimuth = np.radians(i * cam_cfg["lateral_azimuth_spacing_deg"])

        cam_x = auto_distance * np.cos(azimuth) * np.cos(elevation)
        cam_y = auto_distance * np.sin(azimuth) * np.cos(elevation)
        cam_z = auto_distance * np.sin(elevation)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        forward = obj_center - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Z-up 방향 보장을 위한 카메라 매트릭스 수동 생성
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        cam_up = np.cross(right, forward)
        cam_up = cam_up / np.linalg.norm(cam_up)

        # BlenderProc cam2world: columns = [right, up, -forward, pos]
        R = np.column_stack([right, cam_up, -forward])
        cam2world = np.eye(4)
        cam2world[:3, :3] = R
        cam2world[:3, 3] = cam_pos

        bproc.camera.add_camera_pose(cam2world)
        camera_poses.append(cam2world.tolist())

    # 1개 top-down 카메라
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

    # --- 렌더링 ---
    bproc.renderer.set_max_amount_of_samples(128)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    try:
        bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    except Exception:
        pass

    data = bproc.renderer.render()

    # --- 결과 저장 ---
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
