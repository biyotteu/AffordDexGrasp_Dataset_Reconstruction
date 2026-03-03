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
        # 주요 의존성 확인 및 누락 시 자동 설치
        missing = _check_paint3d_deps_in_env(conda_exe, env_name)
        if missing:
            _install_missing_deps(conda_exe, env_name, missing, clone_dir)
            # 재확인
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
    """Paint3D conda 환경 내 핵심 의존성 확인. 누락된 패키지 목록 반환."""
    # 개별적으로 체크 (하나가 실패해도 나머지 체크 가능)
    deps_to_check = {
        "torch": "import torch; print(torch.__version__)",
        "cuda": "import torch; print(torch.version.cuda)",
        "gpu": "import torch; print(torch.cuda.is_available())",
        "kaolin": "import kaolin; print(kaolin.__version__)",
        "diffusers": "import diffusers; print(diffusers.__version__)",
        "transformers": "import transformers; print(transformers.__version__)",
        "accelerate": "import accelerate; print(accelerate.__version__)",
    }

    missing = []
    for name, script in deps_to_check.items():
        try:
            result = subprocess.run(
                [conda_exe, "run", "-n", env_name, "python", "-c", script],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                val = result.stdout.strip().split('\n')[-1]
                print(f"    ✅ {name}: {val}")
            else:
                print(f"    ❌ {name}: NOT_FOUND")
                missing.append(name)
        except Exception:
            print(f"    ❌ {name}: CHECK_FAILED")
            missing.append(name)

    return missing


# Paint3D가 필요로 하는 pip 패키지 (environment.yaml 기반)
PAINT3D_PIP_DEPS = [
    "huggingface_hub<0.24.0",   # diffusers 0.25.0과 호환 (cached_download 필요)
    "diffusers==0.25.0",
    "accelerate==0.29.2",
    "transformers==4.27.1",
    "omegaconf==2.1.1",
    "pytorch-lightning==1.4.2",
    "einops==0.3.0",
    "kornia==0.6",
    "open_clip_torch==2.0.2",
    "trimesh==3.20.2",
    "xatlas==0.0.7",
    "loguru==0.7.2",
    "albumentations==1.3.0",
    "imageio==2.9.0",
    "webdataset==0.2.5",
]


def _install_missing_deps(conda_exe, env_name, missing, clone_dir):
    """누락된 Paint3D 의존성 자동 설치"""
    print(f"\n  📦 누락된 의존성 설치 중: {', '.join(missing)}")

    # environment.yaml의 pip 의존성 전체 설치가 가장 확실
    env_yaml = Path(clone_dir) / "environment.yaml"
    yaml_installed = False
    if env_yaml.exists():
        print(f"    environment.yaml에서 pip 의존성 설치...")
        try:
            with open(env_yaml) as f:
                env_data = yaml.safe_load(f)
            pip_deps = []
            for dep in env_data.get('dependencies', []):
                if isinstance(dep, dict) and 'pip' in dep:
                    pip_deps = dep['pip']
                    break

            if pip_deps:
                result = subprocess.run(
                    [conda_exe, "run", "-n", env_name,
                     "pip", "install"] + pip_deps,
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    print(f"    ✅ pip 의존성 {len(pip_deps)}개 설치 완료")
                    yaml_installed = True
                else:
                    print(f"    ⚠️ 일부 설치 실패, 개별 설치 시도...")
        except Exception as e:
            print(f"    ⚠️ environment.yaml 파싱 실패: {e}")

    if not yaml_installed:
        # 폴백: 핵심 패키지만 개별 설치
        for pkg in PAINT3D_PIP_DEPS:
            pkg_name = pkg.split('==')[0].split('<')[0].split('>')[0]
            if pkg_name in missing or pkg_name in ['diffusers', 'accelerate', 'transformers', 'huggingface_hub']:
                print(f"    설치: {pkg}")
                result = subprocess.run(
                    [conda_exe, "run", "-n", env_name, "pip", "install", pkg],
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    print(f"    ✅ {pkg}")
                else:
                    err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown"
                    print(f"    ❌ {pkg}: {err}")

    # 항상 huggingface_hub 버전 핀 적용 (yaml 설치가 최신 버전을 넣을 수 있으므로)
    print(f"    huggingface_hub<0.24.0 핀 적용 중...")
    result = subprocess.run(
        [conda_exe, "run", "-n", env_name,
         "pip", "install", "huggingface_hub<0.24.0"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        print(f"    ✅ huggingface_hub 다운그레이드 완료")
    else:
        err = result.stderr.strip().split('\n')[-1] if result.stderr else "unknown"
        print(f"    ⚠️ huggingface_hub 다운그레이드 실패: {err}")


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
    """Paint3D 없이 fallback 텍스처 생성 (메인 환경에서 실행)

    albedo.png + paint3d.mtl + textured_mesh.obj 를 생성하여
    BlenderProc load_obj()가 자동으로 텍스처를 읽을 수 있게 함.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)

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

    # .mtl 생성 + .obj에 mtllib 패치 (BlenderProc이 텍스처를 자동 로드하도록)
    albedo_file = out_path / "albedo.png"
    if albedo_file.exists():
        mtl_content = ("# Fallback material\n"
                       "newmtl paint3d_material\n"
                       "Ka 1.000 1.000 1.000\n"
                       "Kd 1.000 1.000 1.000\n"
                       "Ks 0.000 0.000 0.000\n"
                       "Ns 10.0\nd 1.0\nillum 1\n"
                       "map_Kd albedo.png\n")
        (out_path / "paint3d.mtl").write_text(mtl_content)

        # textured_mesh.obj에 mtllib 패치
        obj_file = Path(textured_path)
        if obj_file.exists():
            try:
                lines = obj_file.read_text().splitlines()
                new_lines = []
                has_mtllib = False
                has_usemtl = False
                for line in lines:
                    s = line.strip()
                    if s.startswith("mtllib "):
                        new_lines.append("mtllib paint3d.mtl")
                        has_mtllib = True
                    elif s.startswith("usemtl "):
                        new_lines.append("usemtl paint3d_material")
                        has_usemtl = True
                    else:
                        new_lines.append(line)
                if not has_mtllib:
                    new_lines.insert(0, "mtllib paint3d.mtl")
                if not has_usemtl:
                    for i, line in enumerate(new_lines):
                        if line.strip().startswith("f "):
                            new_lines.insert(i, "usemtl paint3d_material")
                            break
                obj_file.write_text("\n".join(new_lines) + "\n")
            except Exception:
                pass


def run_paint3d(cfg, max_objects=None, paint3d_python=None, paint3d_dir=None, force=False):
    """모든 오브젝트에 대해 Paint3D 텍스처 생성. force=True이면 기존 결과 무시."""
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

        # 이미 완료된 것 스킵 (force 모드에서는 fallback만 있는 경우 재실행)
        if not force and _has_valid_texture(obj_tex_dir, min_texture_size):
            stats["skipped"] += 1
            continue
        if force and _has_valid_texture(obj_tex_dir, min_texture_size):
            # force 모드: .paint3d_done 마커가 있으면 진짜 Paint3D 성공 → 스킵
            if (obj_tex_dir / PAINT3D_DONE_MARKER).exists():
                stats["skipped"] += 1
                continue
            # 마커 없음 = fallback 단색이거나 원본 material → 재실행
            for f in ["albedo.png", "textured_mesh.obj"]:
                fp = obj_tex_dir / f
                if fp.exists():
                    fp.unlink()

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
                    # .paint3d_done 마커 or albedo.png → Paint3D 성공
                    if (obj_tex_dir / PAINT3D_DONE_MARKER).exists():
                        stats["success"] += 1
                    elif (obj_tex_dir / "albedo.png").exists():
                        stats["success"] += 1
                    else:
                        stats["fallback"] += 1
                else:
                    # Worker 내부에서 이미 fallback 처리함
                    if (obj_tex_dir / "albedo.png").exists():
                        stats["fallback"] += 1
                    else:
                        stats["failed"] += 1

                    # 첫 3개 실패에 대해 상세 에러 출력 (stdout + stderr 모두 확인)
                    if stats["fallback"] + stats["failed"] <= 3:
                        all_output = ""
                        if result.stderr:
                            all_output += result.stderr
                        if result.stdout:
                            all_output += result.stdout
                        if all_output:
                            err_lines = all_output.strip().split('\n')[-10:]
                            tqdm.write(f"    ⚠️ {obj_id} 실패 (returncode={result.returncode}):")
                            for line in err_lines:
                                tqdm.write(f"      {line}")
                        else:
                            tqdm.write(f"    ⚠️ {obj_id} 실패 (returncode={result.returncode}, 출력 없음)")

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


PAINT3D_DONE_MARKER = ".paint3d_done"  # Paint3D 완료 마커 파일


def _has_valid_texture(obj_tex_dir, min_texture_size):
    """해당 오브젝트에 Paint3D가 생성한 유효한 텍스처가 있는지 확인"""
    # Paint3D 완료 마커가 있으면 확실히 완료된 것
    if (obj_tex_dir / PAINT3D_DONE_MARKER).exists():
        return True

    # Fallback 결과 (albedo.png + textured_mesh.obj) — fallback도 "처리 완료"로 간주
    if (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "albedo.png").exists():
        return True

    # 주의: material_*.png는 원본 메쉬 자체 텍스처일 수 있으므로
    # Paint3D 결과 판정에 사용하지 않음 (마커 파일로만 판단)
    return False


def check_texture_quality(cfg):
    """텍스처 품질 확인"""
    textures_dir = Path(cfg['paths']['textures'])

    total = 0
    paint3d_done = 0       # .paint3d_done 마커 있음 (진짜 Paint3D 성공)
    has_albedo = 0
    has_textured_mesh = 0
    has_original_material = 0  # 원본 메쉬 material (Paint3D가 아님)
    fallback_only = 0      # albedo.png만 있고 마커 없음 (단색 fallback)

    for obj_dir in textures_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        total += 1

        marker = obj_dir / PAINT3D_DONE_MARKER
        albedo = obj_dir / "albedo.png"
        textured = obj_dir / "textured_mesh.obj"
        mat_files = list(obj_dir.glob("material_*.png"))

        if marker.exists():
            paint3d_done += 1
        if albedo.exists():
            has_albedo += 1
        if textured.exists():
            has_textured_mesh += 1
        if mat_files and not marker.exists():
            has_original_material += 1  # 원본 메쉬에 포함된 material
        if albedo.exists() and not marker.exists():
            fallback_only += 1

    print(f"\n  [텍스처 품질 체크]")
    print(f"    총 오브젝트: {total}")
    print(f"    Paint3D 성공 (마커): {paint3d_done} ({paint3d_done/max(total,1)*100:.1f}%)")
    print(f"    Fallback 단색: {fallback_only} ({fallback_only/max(total,1)*100:.1f}%)")
    print(f"    원본 material 보유: {has_original_material}")
    print(f"    albedo 보유: {has_albedo}")
    print(f"    textured mesh: {has_textured_mesh}")
    print(f"    미처리: {total - paint3d_done - fallback_only}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase E: Paint3D 텍스처 생성")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "setup", "uv", "paint3d", "check"],
                       default="all")
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                       help="기존 fallback 텍스처를 무시하고 Paint3D로 재실행")
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
                    paint3d_python=paint3d_python, paint3d_dir=paint3d_dir,
                    force=args.force)

    if args.step == "check":
        check_texture_quality(cfg)

    print("\n" + "=" * 60)
    print("Phase E 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
