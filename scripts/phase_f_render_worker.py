# Phase F Render Worker: 5-view RGB-D 렌더링 (object only, auto-fit camera)
# 실행: blenderproc run worker.py --scene_dir <dir> --renders_dir <dir>
# blenderproc 2.7.0 → Blender 3.5.1 → Python 3.10 (conda와 일치)
import blenderproc as bproc
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


def find_best_texture_image(tex_dir):
    """텍스처 이미지 탐색 (material_*.jpeg 우선)

    우선순위:
      1) material_0.jpeg / material_0.jpg  (원본 텍스처)
      2) stage1/res-0/albedo.png           (paint3d 최종)
      3) material_0.png                    (크기 >= 64x64만)
      4) albedo.png (루트)                 (paint3d 중간 결과물)
    """
    import glob

    # 1) material_*.jpeg / material_*.jpg
    for pattern in ["material_*.jpeg", "material_*.jpg"]:
        matches = sorted(glob.glob(os.path.join(tex_dir, pattern)))
        for m in matches:
            # 너무 작은 파일 제외 (1KB 미만)
            if os.path.getsize(m) > 1024:
                return m, "material_jpeg"

    # 2) stage1/res-0/albedo.png
    stage1 = os.path.join(tex_dir, "stage1", "res-0", "albedo.png")
    if os.path.exists(stage1):
        return stage1, "stage1_albedo"

    # 3) material_0.png (크기 체크 — o25101의 2x2 같은 더미 제외)
    mat_png = os.path.join(tex_dir, "material_0.png")
    if os.path.exists(mat_png) and os.path.getsize(mat_png) > 1024:
        return mat_png, "material_png"

    # 4) 루트 albedo.png (10KB 미만은 빈/더미 이미지로 간주)
    root_albedo = os.path.join(tex_dir, "albedo.png")
    if os.path.exists(root_albedo) and os.path.getsize(root_albedo) > 10240:
        return root_albedo, "root_albedo"

    return None, "none"


def apply_texture_image(blender_obj, image_path):
    """메쉬에 텍스처 이미지를 적용 (기존 material 보존, 없으면 생성)"""
    import bpy

    mesh_data = blender_obj.blender_obj.data
    has_uv = bool(mesh_data.uv_layers and len(mesh_data.uv_layers) > 0)
    if not has_uv:
        print("  UV 없음 → 텍스처 적용 불가")
        return False

    abs_path = os.path.abspath(image_path)

    if len(mesh_data.materials) == 0:
        # material이 아예 없음 → 새로 생성
        mat = bpy.data.materials.new(name="texture_material")
        mat.use_nodes = True
        mesh_data.materials.append(mat)

    # 첫 번째 material의 BSDF에 텍스처 연결
    mat = mesh_data.materials[0]
    if not mat or not mat.use_nodes:
        return False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return False

    # 기존 Base Color 연결 해제
    for link in list(links):
        if link.to_socket == bsdf.inputs["Base Color"]:
            links.remove(link)

    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(abs_path)
    tex_node.image.colorspace_settings.name = 'sRGB'
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    print("  Applied texture: " + os.path.basename(image_path))
    return True


def render_scene(scene_dir, renders_dir, textures_dir=None):
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

    obj_id = job["obj_id"]
    if textures_dir:
        tex_dir = os.path.join(textures_dir, obj_id)
    else:
        tex_dir = os.path.join("textures", obj_id)

    # 텍스처 이미지 먼저 탐색 (material_*.jpeg 우선)
    best_texture_path, tex_source = find_best_texture_image(tex_dir)
    print("  Best texture image: " + tex_source
          + (" (" + os.path.basename(best_texture_path) + ")" if best_texture_path else ""))

    # 메쉬 탐색 — material_*.jpeg가 있으면 uv_mesh.obj 사용 (material.mtl이 jpeg 참조)
    # material_*.jpeg 없으면 stage1 → textured_mesh 순서
    stage1_mesh = os.path.join(tex_dir, "stage1", "res-0", "mesh.obj")
    textured_mesh = os.path.join(tex_dir, "textured_mesh.obj")
    uv_mesh = os.path.join(tex_dir, "uv_mesh.obj")
    original_mesh = job["mesh_path"]

    if tex_source == "material_jpeg" and os.path.exists(uv_mesh):
        # material_*.jpeg 존재 → uv_mesh.obj (material.mtl → material_0.jpeg)
        mesh_path = uv_mesh
        has_texture = True
        print("  Mesh: uv_mesh.obj (for material_jpeg)")
    elif os.path.exists(stage1_mesh):
        mesh_path = stage1_mesh
        has_texture = True
        print("  Mesh: stage1/res-0/mesh.obj (paint3d final)")
    elif os.path.exists(textured_mesh):
        mesh_path = textured_mesh
        has_texture = True
        print("  Mesh: textured_mesh.obj")
    elif os.path.exists(uv_mesh):
        mesh_path = uv_mesh
        has_texture = True
        print("  Mesh: uv_mesh.obj")
    else:
        mesh_path = original_mesh
        has_texture = False
        print("  Mesh: original (no texture)")

    import bpy
    from mathutils import Vector

    # --- Object 로드 ---
    try:
        objs = bproc.loader.load_obj(mesh_path)
    except Exception:
        try:
            import bpy as _bpy
            _bpy.ops.import_mesh.ply(filepath=mesh_path)
            imported = _bpy.context.selected_objects[0]
            from blenderproc.python.types.MeshObjectUtility import MeshObject
            objs = [MeshObject(imported)]
        except Exception as e:
            print("  ERROR: mesh load failed: " + str(e))
            return None

    target_obj = objs[0] if isinstance(objs, list) else objs
    target_obj.set_name("object_" + str(obj_id))
    target_obj.set_cp("category_id", 1)

    # --- 텍스처 체크 ---
    # .mtl 자동 로드로 TEX_IMAGE가 생겼는지 확인
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
                            print("  .mtl auto-loaded texture: " + node.image.name
                                  + " (" + str(w) + "x" + str(h) + ")")
                            break
                if actually_has_material:
                    break

    # .mtl 자동 로드 실패 시 → best_texture_path로 수동 적용
    if not actually_has_material and best_texture_path:
        actually_has_material = apply_texture_image(target_obj, best_texture_path)

    # 여전히 텍스처 없으면 랜덤 단색
    if not actually_has_material:
        color_seed = hash(obj_id) % 1000
        np.random.seed(color_seed)
        r = np.random.uniform(0.3, 0.85)
        g = np.random.uniform(0.3, 0.85)
        b = np.random.uniform(0.3, 0.85)
        assign_material(target_obj, (r, g, b, 1.0))
        print("  Assigned random color: (" + str(round(r,2)) + ", "
              + str(round(g,2)) + ", " + str(round(b,2)) + ")")

    # --- Position object at origin ---
    target_obj.set_location([0, 0, 0])
    target_obj.set_rotation_euler([0, 0, 0])
    mesh_blender = target_obj.blender_obj

    # Apply Phase D pose rotation if available
    if os.path.exists(pose_file):
        with open(pose_file) as f:
            pose_data = json.load(f)
        rot_euler = pose_data.get("rotation_euler", [0, 0, 0])
        target_obj.set_rotation_euler(rot_euler)

    # rotation 적용 후 world-space bbox 계산
    bpy.context.view_layer.update()
    bbox_world = [mesh_blender.matrix_world @ Vector(c) for c in mesh_blender.bound_box]
    world_coords = np.array([[v.x, v.y, v.z] for v in bbox_world])

    # obj_size: rotation 반영된 실제 world bbox 크기
    obj_size = world_coords.max(axis=0) - world_coords.min(axis=0)
    obj_diagonal = float(np.linalg.norm(obj_size))

    # 중심을 원점으로 이동
    obj_center_offset = (world_coords.max(axis=0) + world_coords.min(axis=0)) / 2.0
    target_obj.set_location(-obj_center_offset)

    bpy.context.view_layer.update()
    obj_center = np.array([0.0, 0.0, 0.0])

    print("Object bbox size (world): " + str([round(x, 4) for x in obj_size.tolist()]))
    print("Object diagonal: " + str(round(obj_diagonal, 4)) + "m")

    # --- Camera auto-fit ---
    # 물체 크기에 맞게 카메라 거리 자동 조절
    # target_fill=0.30: 물체가 화면의 ~30%를 채움 (여백 충분히 확보)
    # min 0.35m: 아무리 작아도 35cm 이상 떨어짐 (과도한 확대 방지)
    lens_mm = 50
    half_fov = np.arctan(18.0 / lens_mm)
    target_fill = 0.30
    auto_distance = max(0.35, min(2.0, obj_diagonal / (2.0 * np.tan(half_fov) * target_fill)))

    print("Auto camera distance: " + str(round(auto_distance, 4)) + "m")

    # --- Lighting (3-point studio) ---
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

    # Background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.92, 0.92, 0.94, 1.0)
        bg_node.inputs[1].default_value = 0.8

    # --- Camera setup ---
    # rotation_from_forward_vec + build_transformation_mat 사용
    # up_axis='Z': Blender는 Z-up 좌표계이므로 카메라의 "위"를 Z축으로 설정해야
    #              렌더링 이미지에서 물체가 올바른 방향으로 보임
    cam_cfg = job["camera"]
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=lens_mm)

    camera_poses = []
    base_elevation_deg = cam_cfg["lateral_elevation_deg"]

    # 납작한 물체 보정: Z높이가 XY 대비 작으면 elevation을 높여서
    # 거의 옆면만 보이는 것을 방지 (접시, 트레이 등)
    z_height = obj_size[2] if len(obj_size) > 2 else obj_size.min()
    xy_extent = max(obj_size[0], obj_size[1]) if len(obj_size) > 1 else obj_size.max()
    flatness_ratio = z_height / max(xy_extent, 1e-6)

    if flatness_ratio < 0.3:
        # 매우 납작함 (접시, 키보드 등) → elevation 45~55°
        elevation_deg = max(base_elevation_deg, 45.0 + (0.3 - flatness_ratio) * 30)
        elevation_deg = min(elevation_deg, 55.0)
        print("  Flat object (ratio=" + str(round(flatness_ratio, 2))
              + ") → elevation " + str(round(elevation_deg, 1)) + "°")
    else:
        elevation_deg = base_elevation_deg

    elevation = np.radians(elevation_deg)

    # 4 lateral cameras
    for i in range(cam_cfg["lateral_count"]):
        azimuth = np.radians(i * cam_cfg["lateral_azimuth_spacing_deg"])

        cam_x = auto_distance * np.cos(azimuth) * np.cos(elevation)
        cam_y = auto_distance * np.sin(azimuth) * np.cos(elevation)
        cam_z = auto_distance * np.sin(elevation)

        cam_pos = np.array([cam_x, cam_y, cam_z])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(obj_center - cam_pos, up_axis='Z')
        cam2world = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world)
        camera_poses.append(cam2world.tolist())

    # 1 top-down camera
    cam_pos_td = np.array([0.0, 0.0, auto_distance * 0.85])
    rotation_matrix_td = bproc.camera.rotation_from_forward_vec(obj_center - cam_pos_td, up_axis='Z')
    cam2world_td = bproc.math.build_transformation_mat(cam_pos_td, rotation_matrix_td)
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

        for seg_key in ["category_id_segmaps", "class_segmaps", "segmentation"]:
            if seg_key in data:
                seg = data[seg_key][i]
                np.save(os.path.join(output_dir, "seg_cam" + str(i) + ".npy"), seg)
                break

    print("Render complete: " + output_dir)
    print("  Object: " + str(obj_id) + " | Texture: " + tex_source)
    print("  Camera distance: " + str(round(auto_distance, 3)) + "m | Lens: " + str(lens_mm) + "mm")
    return output_dir


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--renders_dir", default="renders")
    parser.add_argument("--textures_dir", default=None,
                        help="텍스처 루트 디렉토리 (절대경로 권장)")
    args = parser.parse_args(argv)

    render_scene(args.scene_dir, args.renders_dir, args.textures_dir)
