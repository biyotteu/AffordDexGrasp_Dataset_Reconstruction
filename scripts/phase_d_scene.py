#!/usr/bin/env python3
"""
Phase D: Scene 구성(BlenderProc) + Physics + Collision Filtering
- D1: Scene job 생성
- D2: 물리 기반 배치 (BlenderProc physics settle)
- D3: Invalid grasp 필터링 (충돌 검사)

Requirements:
  pip install blenderproc
  # BlenderProc will download Blender automatically on first run
"""

import json
import os
import argparse
import uuid
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import jsonlines
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# D1: Scene Job 생성
# ============================================================
def generate_scene_jobs(cfg):
    """
    각 오브젝트에 대해 scene job을 생성
    - tabletop scene (기본)
    - 1~3 objects per scene
    """
    print("=" * 60)
    print("[D1] Scene Job 생성")
    print("=" * 60)

    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
    scenes_dir = Path(cfg['paths']['scenes'])
    scenes_dir.mkdir(parents=True, exist_ok=True)

    # 메타 로드
    obj_to_samples = defaultdict(list)
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                obj_id = entry.get('obj_id')
                if obj_id:
                    obj_to_samples[obj_id].append(entry)

    # Mesh index 로드
    index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
    mesh_index = {}
    if index_path.exists():
        with open(index_path) as f:
            mesh_index = json.load(f)

    jobs = []
    for obj_id, samples in tqdm(obj_to_samples.items(), desc="  Scene job 생성"):
        if obj_id not in mesh_index:
            continue

        mesh_info = mesh_index[obj_id]
        scene_id = f"scene_{obj_id}_{uuid.uuid4().hex[:8]}"

        job = {
            "scene_id": scene_id,
            "obj_id": obj_id,
            "mesh_path": mesh_info['mesh_path'],
            "mesh_type": mesh_info.get('type', 'unknown'),
            "num_samples": len(samples),
            "sample_ids": [s['sample_id'] for s in samples],
            "scene_type": "tabletop",  # or "shelf"
            "table": cfg['scene']['table'],
            "camera": cfg['scene']['camera_layout'],
        }
        jobs.append(job)

    # 저장
    jobs_path = scenes_dir / "jobs.jsonl"
    with jsonlines.open(jobs_path, mode='w') as writer:
        for job in jobs:
            writer.write(job)

    print(f"  Scene jobs: {len(jobs)}개 → {jobs_path}")
    return jobs


# ============================================================
# D2: BlenderProc 물리 기반 배치
# ============================================================
WORKER_SCRIPT = Path("scripts") / "phase_d_blenderproc_worker.py"

# Blender 실행 경로 자동 탐색
DEFAULT_BLENDER_PATHS = [
    Path.home() / "blender" / "blender-4.2.1-linux-x64" / "blender",
    Path.home() / "blender" / "blender-3.6.0-linux-x64" / "blender",
    Path("/usr/local/bin/blender"),
    Path("/usr/bin/blender"),
    Path("/snap/bin/blender"),
]


def _find_blender(cfg):
    """Blender 실행 파일 경로 탐색"""
    # config에 지정된 경로 우선
    blender_path = cfg.get('scene', {}).get('blender_path')
    if blender_path and Path(blender_path).exists():
        return str(blender_path)

    # 기본 경로 탐색
    for p in DEFAULT_BLENDER_PATHS:
        if p.exists():
            return str(p)

    # which로 탐색
    import subprocess
    try:
        result = subprocess.run(["which", "blender"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return None


def _get_blender_python(blender_path):
    """Blender 내장 Python 경로 반환"""
    blender_dir = Path(blender_path).parent
    # Blender 4.x: <blender_dir>/<version>/python/bin/python3.xx
    for version_dir in sorted(blender_dir.glob("[0-9]*.[0-9]*"), reverse=True):
        python_dir = version_dir / "python" / "bin"
        for pybin in sorted(python_dir.glob("python3*"), reverse=True):
            if pybin.is_file() and not pybin.name.endswith("-config"):
                return str(pybin)
    return None


def _ensure_blenderproc_in_blender(blender_python):
    """Blender 내장 Python에 blenderproc이 설치되어 있는지 확인하고, 없으면 설치"""
    import subprocess

    # blenderproc 설치 확인
    result = subprocess.run(
        [blender_python, "-c", "import blenderproc; print(blenderproc.__version__)"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        version = result.stdout.strip()
        print(f"  ✅ Blender Python에 blenderproc {version} 설치됨")
        return True

    # 설치 시도
    print("  📦 Blender Python에 blenderproc 설치 중...")
    result = subprocess.run(
        [blender_python, "-m", "pip", "install", "blenderproc"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0:
        print("  ✅ blenderproc 설치 완료")
        return True
    else:
        # pip가 없으면 ensurepip 먼저
        subprocess.run(
            [blender_python, "-m", "ensurepip", "--upgrade"],
            capture_output=True, text=True, timeout=60
        )
        result = subprocess.run(
            [blender_python, "-m", "pip", "install", "blenderproc"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("  ✅ blenderproc 설치 완료 (ensurepip 후)")
            return True

    print(f"  ❌ blenderproc 설치 실패: {result.stderr[-300:]}")
    return False


def run_physics_settle(cfg, max_scenes=None):
    """모든 scene에 대해 BlenderProc physics settle 실행"""
    print("\n" + "=" * 60)
    print("[D2] Physics 기반 배치 실행")
    print("=" * 60)

    import subprocess

    scenes_dir = Path(cfg['paths']['scenes'])
    jobs_path = scenes_dir / "jobs.jsonl"

    if not jobs_path.exists():
        print("  ⚠️ jobs.jsonl 없음 - D1 먼저 실행")
        return

    worker_path = WORKER_SCRIPT
    if not worker_path.exists():
        print(f"  ⚠️ Worker 스크립트 없음: {worker_path}")
        return

    # --- Blender 직접 실행 방식 (conda numpy 충돌 방지) ---
    blender_path = _find_blender(cfg)
    if not blender_path:
        print("  ❌ Blender를 찾을 수 없습니다.")
        print("     config에 scene.blender_path를 지정하거나, ~/blender/ 에 설치하세요.")
        return
    print(f"  Blender: {blender_path}")

    blender_python = _get_blender_python(blender_path)
    if blender_python:
        print(f"  Blender Python: {blender_python}")
        if not _ensure_blenderproc_in_blender(blender_python):
            print("  ❌ Blender에 blenderproc 설치 실패 - 중단")
            return
    else:
        print("  ⚠️ Blender 내장 Python을 찾을 수 없음 - blenderproc run 방식으로 fallback")

    # 환경 변수 정리: conda 경로 완전 제거 (Blender 자체 Python만 사용)
    clean_env = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX',
                'CONDA_DEFAULT_ENV', 'CONDA_PYTHON_EXE']:
        clean_env.pop(key, None)
    # LD_LIBRARY_PATH에서 conda 관련 경로 제거
    if 'LD_LIBRARY_PATH' in clean_env:
        ld_paths = clean_env['LD_LIBRARY_PATH'].split(':')
        ld_paths = [p for p in ld_paths if 'anaconda' not in p and 'conda' not in p]
        clean_env['LD_LIBRARY_PATH'] = ':'.join(ld_paths) if ld_paths else ''

    jobs = []
    with jsonlines.open(jobs_path) as reader:
        for job in reader:
            jobs.append(job)

    if max_scenes:
        jobs = jobs[:max_scenes]

    fail_count = 0
    consecutive_fails = 0
    MAX_CONSECUTIVE_FAILS = 20

    for idx, job in enumerate(tqdm(jobs, desc="  Physics settle")):
        scene_dir = scenes_dir / job['scene_id']
        result_pose = scene_dir / "object_pose.json"

        # 이미 완료된 scene은 스킵
        if result_pose.exists():
            consecutive_fails = 0
            continue

        scene_dir.mkdir(parents=True, exist_ok=True)

        # Job 파일 저장
        job_file = scene_dir / "job.json"
        with open(job_file, 'w') as f:
            json.dump(job, f, indent=2)

        # mesh 파일 존재 확인
        mesh_path = Path(job.get('mesh_path', ''))
        if not mesh_path.exists():
            print(f"    ⚠️ {job['scene_id']} 스킵: mesh 없음 → {mesh_path}")
            fail_count += 1
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print(f"\n  ❌ 연속 {MAX_CONSECUTIVE_FAILS}회 실패 - mesh 파일 없음")
                break
            continue

        # Blender 직접 실행 (--python-use-system-env 없이 → conda 오염 방지)
        worker_abs = str(worker_path.resolve())
        cmd = [
            blender_path, "--background", "--python", worker_abs,
            "--",  # Blender args와 script args 구분
            "--job", str(job_file),
            "--output_dir", str(scene_dir),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                env=clean_env,
            )
            if result.returncode != 0:
                stderr_lines = result.stderr.strip().split('\n')
                error_tail = '\n'.join(stderr_lines[-5:]) if len(stderr_lines) > 5 else result.stderr
                print(f"    ⚠️ {job['scene_id']} 실패:\n{error_tail}")

                log_file = scene_dir / "error.log"
                with open(log_file, 'w') as f:
                    f.write(f"=== STDERR ===\n{result.stderr}\n\n=== STDOUT ===\n{result.stdout}")

                fail_count += 1
                consecutive_fails += 1
            else:
                consecutive_fails = 0

        except subprocess.TimeoutExpired:
            print(f"    ⚠️ {job['scene_id']} 타임아웃 (300초)")
            fail_count += 1
            consecutive_fails += 1
        except FileNotFoundError:
            print(f"    ❌ Blender 실행 파일 없음: {blender_path}")
            break

        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            print(f"\n  ❌ 연속 {MAX_CONSECUTIVE_FAILS}회 실패 - 중단")
            print(f"     에러 로그: {scene_dir}/error.log")
            break

        # 50개마다 /tmp 캐시 정리
        if (idx + 1) % 50 == 0:
            import glob, shutil
            for tmp_dir in glob.glob("/tmp/blender_proc_*"):
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except:
                    pass

    print(f"\n  실패: {fail_count}개 / 전체: {len(jobs)}개")


# ============================================================
# D3: Invalid Grasp 필터링 (충돌 검사)
# ============================================================
def filter_invalid_grasps(cfg):
    """
    Hand mesh/포인트와 table/shelf 충돌 체크
    Invalid grasp 제거
    """
    print("\n" + "=" * 60)
    print("[D3] Invalid Grasp 필터링 (충돌 검사)")
    print("=" * 60)

    import trimesh

    scenes_dir = Path(cfg['paths']['scenes'])
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"

    # 테이블 geometry
    table_cfg = cfg['scene']['table']
    table_half = np.array(table_cfg['size']) / 2
    table_center = np.array([0, 0, table_cfg['height']])

    # 메타 로드
    all_meta = {}
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                all_meta[entry['sample_id']] = entry

    total_valid = 0
    total_invalid = 0

    for scene_dir in tqdm(list(scenes_dir.iterdir()), desc="  충돌 검사"):
        if not scene_dir.is_dir():
            continue

        job_file = scene_dir / "job.json"
        pose_file = scene_dir / "object_pose.json"

        if not job_file.exists():
            continue

        with open(job_file) as f:
            job = json.load(f)

        # Object pose (있으면 사용, 없으면 원점)
        obj_pose = np.eye(4)
        if pose_file.exists():
            with open(pose_file) as f:
                pose_data = json.load(f)
                obj_loc = np.array(pose_data.get('location', [0, 0, 0]))
                obj_pose[:3, 3] = obj_loc

        valid_grasps = []
        invalid_grasps = []

        for sample_id in job.get('sample_ids', []):
            if sample_id not in all_meta:
                continue

            meta = all_meta[sample_id]
            t = np.array(meta['translation'])

            # 간단한 충돌 검사: 손 위치(translation)가 테이블 내부인지
            # 더 정확한 검사는 FK로 hand mesh를 생성한 후 수행
            hand_pos = t + obj_pose[:3, 3]  # scene 좌표로 변환

            # 테이블 충돌: 손 위치가 테이블 높이 아래이면 invalid
            table_top_z = table_cfg['height']
            collision = False

            if hand_pos[2] < table_top_z - 0.02:  # 2cm margin
                # 손이 테이블 평면 내에 있는지
                if (abs(hand_pos[0] - table_center[0]) < table_half[0] and
                    abs(hand_pos[1] - table_center[1]) < table_half[1]):
                    collision = True

            if collision:
                invalid_grasps.append(sample_id)
                total_invalid += 1
            else:
                valid_grasps.append({
                    "sample_id": sample_id,
                    "hand_position_scene": hand_pos.tolist(),
                    "collision": False,
                })
                total_valid += 1

        # 유효 grasps 저장
        valid_path = scene_dir / "valid_grasps.jsonl"
        with jsonlines.open(valid_path, mode='w') as writer:
            for vg in valid_grasps:
                writer.write(vg)

    print(f"\n  유효 grasps: {total_valid}")
    print(f"  무효 grasps (충돌): {total_invalid}")
    print(f"  유효율: {total_valid / max(total_valid + total_invalid, 1) * 100:.1f}%")


def filter_grasps_detailed(cfg):
    """
    상세 충돌 검사: FK로 hand mesh를 생성하여 정밀 검사
    (Phase H의 hand surface point 생성과 연계)
    """
    print("\n  [D3-상세] FK 기반 정밀 충돌 검사")

    try:
        import pytorch_kinematics as pk
        import torch
    except ImportError:
        print("  ⚠️ pytorch_kinematics 미설치. 간단 검사만 수행됨.")
        print("  pip install pytorch_kinematics")
        return

    # ShadowHand MJCF 로드
    mjcf_dir = Path(cfg['paths']['mjcf'])
    xml_files = list(mjcf_dir.rglob("*.xml"))
    if not xml_files:
        print("  ⚠️ ShadowHand MJCF 없음")
        return

    # FK chain 로드
    chain = pk.build_chain_from_mjcf(open(xml_files[0]).read())
    print(f"  ShadowHand FK chain 로드: {xml_files[0].name}")
    print(f"  DoF: {chain.n_joints}")

    # 테이블 AABB
    table_cfg = cfg['scene']['table']
    table_min = np.array([
        -table_cfg['size'][0]/2,
        -table_cfg['size'][1]/2,
        table_cfg['height'] - table_cfg['size'][2]
    ])
    table_max = np.array([
        table_cfg['size'][0]/2,
        table_cfg['size'][1]/2,
        table_cfg['height']
    ])

    print(f"  Table AABB: min={table_min}, max={table_max}")
    print("  FK 기반 정밀 필터링은 Phase H와 함께 실행됩니다.")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase D: Scene 구성 + 충돌 필터링")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "jobs", "physics", "filter", "filter_detailed"],
                       default="all")
    parser.add_argument("--max_scenes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "jobs"]:
        generate_scene_jobs(cfg)

    if args.step in ["all", "physics"]:
        run_physics_settle(cfg, max_scenes=args.max_scenes)

    if args.step in ["all", "filter"]:
        filter_invalid_grasps(cfg)

    if args.step == "filter_detailed":
        filter_grasps_detailed(cfg)

    print("\n" + "=" * 60)
    print("Phase D 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
