import blenderproc as bproc  # 반드시 첫 줄이어야 함 (BlenderProc 요구사항)
# Phase D BlenderProc Worker: Tabletop scene 생성 + Physics settle
# 실행: blenderproc run phase_d_blenderproc_worker.py --job <job.json> --output_dir <dir>
import sys
import os
import json
import argparse
import numpy as np


def create_tabletop_scene(job, output_dir):
    """단일 오브젝트 tabletop scene 생성"""
    bproc.init()

    # --- Table 생성 ---
    table_cfg = job["table"]
    table = bproc.object.create_primitive(
        "CUBE",
        scale=[table_cfg["size"][0]/2, table_cfg["size"][1]/2, table_cfg["size"][2]/2],
    )
    table.set_location([0, 0, table_cfg["height"] - table_cfg["size"][2]/2])
    table.set_name("table")
    table.set_cp("category_id", 0)

    # 물리 속성: 테이블은 고정
    table.enable_rigidbody(
        active=False,
        collision_shape="BOX",
    )

    # --- Object 로드 ---
    mesh_path = job["mesh_path"]
    try:
        objs = bproc.loader.load_obj(mesh_path)
    except:
        # PLY 파일인 경우
        objs = bproc.loader.load_ply(mesh_path)

    target_obj = objs[0] if isinstance(objs, list) else objs
    target_obj.set_name("object_" + str(job['obj_id']))
    target_obj.set_cp("category_id", 1)

    # 오브젝트를 테이블 위에 배치 (약간 위에서 떨어뜨림)
    target_obj.set_location([0, 0, table_cfg["height"] + 0.15])
    target_obj.set_rotation_euler(np.random.uniform(0, 2*np.pi, 3).tolist())

    # 물리 속성: 오브젝트는 활성
    target_obj.enable_rigidbody(
        active=True,
        collision_shape="CONVEX_HULL",
        mass=0.5,
    )

    # --- Physics Simulation ---
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=4.0,
        check_object_interval=0.5,
    )

    # 최종 pose 저장
    final_pose = {
        "location": target_obj.get_location().tolist(),
        "rotation_euler": target_obj.get_rotation_euler().tolist(),
        "rotation_matrix": target_obj.get_rotation_mat().tolist(),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "object_pose.json"), 'w') as f:
        json.dump(final_pose, f, indent=2)

    print("Object final pose saved: " + output_dir + "/object_pose.json")

    # --- 5대 카메라 설정 ---
    cam_cfg = job["camera"]
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(lens=35)

    obj_center = np.array(final_pose["location"])

    # 4개 사선 카메라
    lateral_elevation = np.radians(cam_cfg["lateral_elevation_deg"])
    distance = cam_cfg["distance_from_center"]

    for i in range(cam_cfg["lateral_count"]):
        azimuth = np.radians(i * cam_cfg["lateral_azimuth_spacing_deg"])

        cam_x = obj_center[0] + distance * np.cos(azimuth) * np.cos(lateral_elevation)
        cam_y = obj_center[1] + distance * np.sin(azimuth) * np.cos(lateral_elevation)
        cam_z = obj_center[2] + distance * np.sin(lateral_elevation)

        cam_pos = [cam_x, cam_y, cam_z]
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            obj_center - np.array(cam_pos)
        )
        cam2world = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world)

    # 1개 top-down 카메라
    topdown_height = cam_cfg["topdown_height"]
    cam_pos = [obj_center[0], obj_center[1], obj_center[2] + topdown_height]
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        obj_center - np.array(cam_pos)
    )
    cam2world = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world)

    # 카메라 파라미터 저장
    cam_data = {
        "num_cameras": cam_cfg["lateral_count"] + cam_cfg["topdown_count"],
        "intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist(),
        "extrinsics": [],
    }
    for i in range(cam_data["num_cameras"]):
        cam2world_i = bproc.camera.get_camera_pose(i)
        cam_data["extrinsics"].append(cam2world_i.tolist())

    with open(os.path.join(output_dir, "camera_params.json"), 'w') as f:
        json.dump(cam_data, f, indent=2)

    return output_dir


if __name__ == "__main__":
    # Blender 직접 실행 시 argv에 Blender 인자가 포함됨
    # "--" 뒤의 인자만 파싱해야 함
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]  # blenderproc run 방식 fallback

    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Path to job JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args(argv)

    with open(args.job) as f:
        job = json.load(f)

    create_tabletop_scene(job, args.output_dir)
