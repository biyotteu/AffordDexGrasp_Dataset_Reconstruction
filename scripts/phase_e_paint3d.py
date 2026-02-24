#!/usr/bin/env python3
"""
Phase E: Paint3D 텍스처 생성/적용 (MLLM용 Realistic RGB 준비)
- E0: Paint3D conda 환경 감지/생성
- E1: 메쉬 전처리 (UV unwrap)
- E2: Paint3D 실행 (별도 conda 환경)

Paint3D는 Python 3.8 + PyTorch 1.12.1 + CUDA 11.3 기반이므로
메인 파이프라인(Python 3.10)과 별도의 conda 환경에서 실행합니다.

사전 준비:
  # 1) Paint3D conda 환경 자동 생성 (E0에서 처리)
  #    또는 수동:
  #    conda env create -f thirdparty/Paint3D/environment.yaml
  #    conda activate paint3d
  #    pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
  #
  # 2) Paint3D 클론:
  #    git clone https://github.com/OpenTexture/Paint3D.git thirdparty/Paint3D
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

import yaml
import numpy as np


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Worker 스크립트 경로 (별도 파일로 분리됨)
PAINT3D_WORKER_SCRIPT = Path("scripts") / "phase_e_paint3d_worker.py"
UV_UNWRAP_WORKER_SCRIPT = Path("scripts") / "phase_e_uv_unwrap_worker.py"


# ============================================================
# E0: Paint3D conda 환경 감지/생성
# ============================================================
def _get_conda_executable():
    """conda 실행 파일 경로 반환"""
    for name in ["conda", "mamba", "micromamba"]:
        try:
            result = subprocess.run(
                [name, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return name
        except FileNotFoundError:
            continue
    return None


def _conda_env_exists(conda_exe, env_name):
    """conda 환경이 존재하는지 확인"""
    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for env_path in data.get("envs", []):
                if Path(env_path).name == env_name:
                    return True
    except Exception:
        pass
    return False


def _get_conda_python(conda_exe, env_name):
    """conda 환경의 python 실행 파일 경로 반환"""
    try:
        result = subprocess.run(
            [conda_exe, "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def setup_paint3d_env(cfg):
    """
    Paint3D conda 환경 감지 및 생성

    반환: (python_path, paint3d_dir) 또는 (None, paint3d_dir)
    """
    print("=" * 60)
    print("[E0] Paint3D 환경 설정")
    print("=" * 60)

    paint3d_cfg = cfg.get('paint3d', {})
    env_name = paint3d_cfg.get('conda_env', 'paint3d')
    clone_dir = Path(paint3d_cfg.get('clone_dir', 'thirdparty/Paint3D'))
    conda_python_override = paint3d_cfg.get('conda_python')

    # --- 1) conda_python이 직접 지정되어 있으면 그것을 사용 ---
    if conda_python_override:
        python_path = Path(conda_python_override)
        if python_path.exists():
            print(f"  ✅ config에서 직접 지정된 python: {python_path}")
            _clone_paint3d_if_needed(clone_dir, paint3d_cfg)
            return str(python_path), str(clone_dir)
        else:
            print(f"  ⚠️ 지정된 python 경로 없음: {python_path}")

    # --- 2) conda 실행 파일 확인 ---
    conda_exe = _get_conda_executable()
    if not conda_exe:
        print("  ⚠️ conda/mamba를 찾을 수 없습니다")
        print("    → Paint3D를 사용하려면 conda를 설치하거나")
        print("      config에 paint3d.conda_python 경로를 직접 지정하세요")
        _clone_paint3d_if_needed(clone_dir, paint3d_cfg)
        return None, str(clone_dir)

    print(f"  conda 실행 파일: {conda_exe}")

    # --- 3) Paint3D 클론 ---
    _clone_paint3d_if_needed(clone_dir, paint3d_cfg)

    # --- 4) conda 환경 확인/생성 ---
    if _conda_env_exists(conda_exe, env_name):
        print(f"  ✅ conda 환경 '{env_name}' 존재")
    else:
        print(f"  conda 환경 '{env_name}' 없음 → 생성 중...")
        env_yaml = clone_dir / "environment.yaml"
        if env_yaml.exists():
            print(f"    environment.yaml 사용: {env_yaml}")
            result = subprocess.run(
                [conda_exe, "env", "create", "-f", str(env_yaml)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"    ⚠️ conda env create 실패:")
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"      {line}")
                print(f"\n    수동 생성 방법:")
                print(f"      conda env create -f {env_yaml}")
                print(f"      conda activate {env_name}")
                print(f"      pip install kaolin==0.13.0 -f {paint3d_cfg.get('kaolin_wheel_url', '')}")
                return None, str(clone_dir)
            print(f"    ✅ conda 환경 '{env_name}' 생성 완료")
        else:
            # environment.yaml 없으면 수동 생성
            print(f"    environment.yaml 없음 → 수동 conda 환경 생성")
            result = subprocess.run(
                [conda_exe, "create", "-n", env_name,
                 "python=3.8.5", "pytorch=1.12.1", "torchvision=0.13.1",
                 "cudatoolkit=11.3", "-c", "pytorch", "-c", "defaults", "-y"],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"    ⚠️ conda create 실패")
                return None, str(clone_dir)
            print(f"    ✅ 기본 환경 생성 완료 (추가 의존성 설치 필요)")

        # kaolin 설치
        kaolin_ver = paint3d_cfg.get('kaolin_version', '0.13.0')
        kaolin_url = paint3d_cfg.get('kaolin_wheel_url', '')
        print(f"    kaolin {kaolin_ver} 설치 중...")
        subprocess.run(
            [conda_exe, "run", "-n", env_name,
             "pip", "install", f"kaolin=={kaolin_ver}", "-f", kaolin_url],
            capture_output=True, text=True, timeout=300,
        )

    # --- 5) python 경로 확인 ---
    python_path = _get_conda_python(conda_exe, env_name)
    if python_path:
        print(f"  ✅ Paint3D python: {python_path}")
        # 주요 의존성 확인
        _check_paint3d_deps_in_env(conda_exe, env_name)
    else:
        print(f"  ⚠️ conda 환경 '{env_name}'에서 python을 찾을 수 없습니다")

    return python_path, str(clone_dir)


def _clone_paint3d_if_needed(clone_dir, paint3d_cfg):
    """Paint3D repo가 없으면 클론"""
    if not clone_dir.exists():
        repo_url = paint3d_cfg.get('repo', 'https://github.com/OpenTexture/Paint3D')
        print(f"  Paint3D 클론 중: {repo_url}")
        try:
            subprocess.run(
                ["git", "clone", repo_url, str(clone_dir)],
                check=True, capture_output=True, text=True, timeout=120,
            )
            print(f"  ✅ 클론 완료: {clone_dir}")
        except Exception as e:
            print(f"  ⚠️ 클론 실패: {e}")
    else:
        print(f"  ✅ Paint3D 디렉토리: {clone_dir}")


def _check_paint3d_deps_in_env(conda_exe, env_name):
    """Paint3D conda 환경 내 핵심 의존성 확인"""
    check_script = (
        "import sys; "
        "print(f'python={sys.version.split()[0]}'); "
        "import torch; print(f'torch={torch.__version__}'); "
        "print(f'cuda={torch.version.cuda}'); "
        "print(f'gpu_available={torch.cuda.is_available()}'); "
        "try:\n"
        "    import kaolin; print(f'kaolin={kaolin.__version__}')\n"
        "except: print('kaolin=NOT_FOUND'); "
        "try:\n"
        "    import diffusers; print(f'diffusers={diffusers.__version__}')\n"
        "except: print('diffusers=NOT_FOUND')"
    )
    try:
        result = subprocess.run(
            [conda_exe, "run", "-n", env_name, "python", "-c", check_script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, val = line.split('=', 1)
                    status = "✅" if val != "NOT_FOUND" else "❌"
                    print(f"    {status} {key}: {val}")
        else:
            print(f"    ⚠️ 의존성 체크 실패")
    except Exception:
        pass


# ============================================================
# E1: 메쉬 전처리 (UV Unwrap)
# ============================================================
def _uv_unwrap_subprocess(mesh_path, output_path):
    """Run xatlas UV unwrap in a subprocess to isolate segfaults.

    UV unwrap은 메인 환경(python 3.10)에서 실행 가능 (trimesh + xatlas).
    """
    worker = UV_UNWRAP_WORKER_SCRIPT
    if not worker.exists():
        print(f"  ⚠️ UV unwrap worker 스크립트 없음: {worker}")
        return "failed"

    result = subprocess.run(
        [sys.executable, str(worker),
         "--mesh_path", str(mesh_path),
         "--output_path", str(output_path)],
        capture_output=True, text=True, timeout=60
    )
    stdout = result.stdout.strip()
    if result.returncode == 0 and "HAS_UV" in stdout:
        return "has_uv"
    elif result.returncode == 0 and "UNWRAPPED" in stdout:
        return "unwrapped"
    elif result.returncode == 2:
        return "no_xatlas"
    else:
        return "failed"


def preprocess_meshes_uv(cfg):
    """
    UV가 없는 메쉬에 UV unwrap 수행
    각 메쉬를 subprocess에서 처리 (xatlas segfault 격리)
    메인 환경(python 3.10)에서 실행됨 — Paint3D 환경 불필요
    """
    print("\n" + "=" * 60)
    print("[E1] 메쉬 UV 전처리")
    print("=" * 60)

    textures_dir = Path(cfg['paths']['textures'])
    textures_dir.mkdir(parents=True, exist_ok=True)

    if not UV_UNWRAP_WORKER_SCRIPT.exists():
        print(f"  ⚠️ UV unwrap worker 없음: {UV_UNWRAP_WORKER_SCRIPT}")
        return

    index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
    if not index_path.exists():
        print("  ⚠️ mesh index 없음")
        return

    with open(index_path) as f:
        mesh_index = json.load(f)

    uv_stats = {"has_uv": 0, "no_uv_unwrapped": 0, "failed": 0, "skipped": 0}
    xatlas_available = True

    for obj_id, info in tqdm(mesh_index.items(), desc="  UV 체크"):
        mesh_path = info['mesh_path']
        if not Path(mesh_path).exists():
            uv_stats["skipped"] += 1
            continue

        obj_tex_dir = textures_dir / obj_id
        obj_tex_dir.mkdir(parents=True, exist_ok=True)

        output_path = obj_tex_dir / "uv_mesh.obj"
        if output_path.exists():
            uv_stats["has_uv"] += 1
            continue

        try:
            result = _uv_unwrap_subprocess(
                str(Path(mesh_path).resolve()),
                str(output_path.resolve())
            )
            if result == "has_uv":
                uv_stats["has_uv"] += 1
            elif result == "unwrapped":
                uv_stats["no_uv_unwrapped"] += 1
            elif result == "no_xatlas":
                if xatlas_available:
                    print("    xatlas not installed. pip install xatlas")
                    xatlas_available = False
                uv_stats["failed"] += 1
            else:
                uv_stats["failed"] += 1
        except subprocess.TimeoutExpired:
            uv_stats["failed"] += 1
        except Exception:
            uv_stats["failed"] += 1

        # uv_mesh.obj가 없으면 trimesh로 변환 시도
        if not output_path.exists():
            try:
                import trimesh
                m = trimesh.load(str(Path(mesh_path).resolve()), force='mesh')
                m.export(str(output_path))
                print(f"    {obj_id}: converted {Path(mesh_path).suffix} -> .obj (no UV)")
            except Exception as conv_e:
                try:
                    shutil.copy2(mesh_path, str(output_path))
                    print(f"    {obj_id}: copied as .obj fallback")
                except Exception:
                    print(f"    {obj_id}: conversion failed: {conv_e}")

    print(f"\n  UV 있음: {uv_stats['has_uv']}")
    print(f"  UV unwrap 성공: {uv_stats['no_uv_unwrapped']}")
    print(f"  실패/fallback: {uv_stats['failed']}")
    print(f"  스킵 (메쉬 없음): {uv_stats['skipped']}")

    return uv_stats


# ============================================================
# E2: Paint3D 실행 (별도 conda 환경)
# ============================================================
def _build_paint3d_cmd(python_path, conda_exe, env_name, worker_path, args_list):
    """
    Paint3D worker 실행 명령 구성

    python_path가 있으면 직접 실행, 없으면 conda run 사용
    """
    if python_path:
        return [python_path, str(worker_path)] + args_list
    elif conda_exe and env_name:
        return [conda_exe, "run", "--no-capture-output", "-n", env_name,
                "python", str(worker_path)] + args_list
    else:
        return None


def _run_fallback_texture(mesh_path, output_dir):
    """Paint3D 없이 fallback 텍스처 생성 (메인 환경에서 실행)"""
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(hash(str(mesh_path)) % 2**31)
    color = rng.randint(100, 230, 3).tolist()

    texture_size = 1024
    try:
        from PIL import Image
        albedo = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        albedo_img = Image.fromarray(albedo)
        albedo_img.save(os.path.join(output_dir, "albedo.png"))
    except Exception:
        pass

    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    if not os.path.exists(textured_path):
        try:
            shutil.copy2(str(mesh_path), textured_path)
        except Exception:
            pass


def run_paint3d(cfg, max_objects=None, paint3d_python=None, paint3d_dir=None):
    """모든 오브젝트에 대해 Paint3D 텍스처 생성"""
    print("\n" + "=" * 60)
    print("[E2] Paint3D 텍스처 생성")
    print("=" * 60)

    paint3d_cfg = cfg.get('paint3d', {})
    env_name = paint3d_cfg.get('conda_env', 'paint3d')
    conda_exe = _get_conda_executable()
    clone_dir = paint3d_dir or paint3d_cfg.get('clone_dir', 'thirdparty/Paint3D')
    sd_config_stage1 = paint3d_cfg.get('sd_config_stage1',
                                        'controlnet/config/depth_based_inpaint_template.yaml')
    sd_config_stage2 = paint3d_cfg.get('sd_config_stage2',
                                        'controlnet/config/UV_based_inpaint_template.yaml')
    render_config = paint3d_cfg.get('render_config', 'paint3d/config/train_config_paint3d.py')

    # Paint3D 사용 가능 여부 판단
    use_paint3d = False
    if paint3d_python and Path(paint3d_python).exists():
        use_paint3d = True
        print(f"  Paint3D python: {paint3d_python}")
    elif conda_exe and _conda_env_exists(conda_exe, env_name):
        use_paint3d = True
        print(f"  Paint3D conda 환경: {env_name} (via {conda_exe})")
    else:
        print("  ⚠️ Paint3D 환경 없음 → fallback(단색 텍스처) 모드")
        print(f"    해결방법:")
        print(f"    1. conda env create -f {clone_dir}/environment.yaml")
        print(f"    2. 또는 config에 paint3d.conda_python 경로를 지정")

    # Worker 확인
    worker_path = PAINT3D_WORKER_SCRIPT.resolve()
    if not worker_path.exists():
        print(f"  ⚠️ Paint3D worker 없음: {worker_path}")
        use_paint3d = False

    # Paint3D 디렉토리 확인
    if not Path(clone_dir).exists():
        print(f"  ⚠️ Paint3D 디렉토리 없음: {clone_dir}")
        use_paint3d = False

    textures_dir = Path(cfg['paths']['textures'])
    min_texture_size = paint3d_cfg.get('min_texture_size', 512)
    timeout = cfg.get('timeouts', {}).get('paint3d', 600)

    # Mesh index 로드
    index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
    if not index_path.exists():
        print("  ⚠️ mesh index 없음")
        return

    with open(index_path) as f:
        mesh_index = json.load(f)

    items = list(mesh_index.items())
    if max_objects:
        items = items[:max_objects]

    stats = {"success": 0, "fallback": 0, "skipped": 0, "failed": 0}

    for obj_id, info in tqdm(items, desc="  Paint3D 실행"):
        obj_tex_dir = textures_dir / obj_id
        uv_mesh = obj_tex_dir / "uv_mesh.obj"

        if not uv_mesh.exists():
            uv_mesh = Path(info['mesh_path'])
        if not uv_mesh.exists():
            stats["skipped"] += 1
            continue

        # 이미 완료된 것 스킵
        if _has_valid_texture(obj_tex_dir, min_texture_size):
            stats["skipped"] += 1
            continue

        # 카테고리 기반 프롬프트 생성
        obj_name = info.get('name', 'object')
        prompt = f"a realistic {obj_name}, photorealistic texture, detailed surface"

        if use_paint3d:
            worker_args = [
                "--mesh_path", str(uv_mesh.resolve()),
                "--output_dir", str(obj_tex_dir.resolve()),
                "--prompt", prompt,
                "--paint3d_dir", str(Path(clone_dir).resolve()),
                "--sd_config_stage1", sd_config_stage1,
                "--sd_config_stage2", sd_config_stage2,
                "--render_config", render_config,
            ]

            cmd = _build_paint3d_cmd(
                paint3d_python, conda_exe, env_name, worker_path, worker_args
            )
            if not cmd:
                _run_fallback_texture(str(uv_mesh), str(obj_tex_dir))
                stats["fallback"] += 1
                continue

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                )
                if result.returncode == 0:
                    stats["success"] += 1
                else:
                    # Worker 내부에서 이미 fallback 처리함
                    if (obj_tex_dir / "albedo.png").exists():
                        stats["fallback"] += 1
                    else:
                        stats["failed"] += 1
                    if result.stderr:
                        err_lines = result.stderr.strip().split('\n')[-3:]
                        tqdm.write(f"    ⚠️ {obj_id}: {' | '.join(err_lines)}")
            except subprocess.TimeoutExpired:
                tqdm.write(f"    ⚠️ {obj_id} 타임아웃 ({timeout}s)")
                _run_fallback_texture(str(uv_mesh), str(obj_tex_dir))
                stats["fallback"] += 1
            except Exception as e:
                tqdm.write(f"    ⚠️ {obj_id} 에러: {e}")
                _run_fallback_texture(str(uv_mesh), str(obj_tex_dir))
                stats["fallback"] += 1
        else:
            _run_fallback_texture(str(uv_mesh), str(obj_tex_dir))
            stats["fallback"] += 1

    print(f"\n  [결과]")
    print(f"    Paint3D 성공: {stats['success']}")
    print(f"    Fallback: {stats['fallback']}")
    print(f"    실패: {stats['failed']}")
    print(f"    스킵: {stats['skipped']}")

    check_texture_quality(cfg)


def _has_valid_texture(obj_tex_dir, min_texture_size):
    """해당 오브젝트에 유효한 텍스처가 있는지 확인"""
    # Paint3D 원본 결과 (material_*.png)
    mat_files = list(obj_tex_dir.glob("material_*.png"))
    if mat_files:
        try:
            from PIL import Image
            for mf in mat_files:
                img = Image.open(str(mf))
                if img.size[0] >= min_texture_size and img.size[1] >= min_texture_size:
                    return True
        except Exception:
            pass

    # Fallback 결과 (albedo.png + textured_mesh.obj)
    if (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "albedo.png").exists():
        return True

    return False


def check_texture_quality(cfg):
    """텍스처 품질 확인"""
    textures_dir = Path(cfg['paths']['textures'])

    total = 0
    has_albedo = 0
    has_textured_mesh = 0
    has_paint3d_material = 0

    for obj_dir in textures_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        total += 1
        if (obj_dir / "albedo.png").exists():
            has_albedo += 1
        if (obj_dir / "textured_mesh.obj").exists():
            has_textured_mesh += 1
        if list(obj_dir.glob("material_*.png")):
            has_paint3d_material += 1

    print(f"\n  [텍스처 품질 체크]")
    print(f"    총 오브젝트: {total}")
    print(f"    Paint3D 텍스처: {has_paint3d_material} ({has_paint3d_material/max(total,1)*100:.1f}%)")
    print(f"    albedo (Paint3D+fallback): {has_albedo} ({has_albedo/max(total,1)*100:.1f}%)")
    print(f"    textured mesh: {has_textured_mesh}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase E: Paint3D 텍스처 생성")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "setup", "uv", "paint3d", "check"],
                       default="all")
    parser.add_argument("--max_objects", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    paint3d_python = None
    paint3d_dir = None

    if args.step in ["all", "setup"]:
        paint3d_python, paint3d_dir = setup_paint3d_env(cfg)

    if args.step in ["all", "uv"]:
        preprocess_meshes_uv(cfg)

    if args.step in ["all", "paint3d"]:
        if paint3d_python is None and args.step != "all":
            # setup을 따로 안 돌렸으면 여기서 감지
            paint3d_python, paint3d_dir = setup_paint3d_env(cfg)
        run_paint3d(cfg, max_objects=args.max_objects,
                    paint3d_python=paint3d_python, paint3d_dir=paint3d_dir)

    if args.step == "check":
        check_texture_quality(cfg)

    print("\n" + "=" * 60)
    print("Phase E 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
