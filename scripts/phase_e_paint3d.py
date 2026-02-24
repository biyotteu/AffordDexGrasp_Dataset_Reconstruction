#!/usr/bin/env python3
"""
Phase E: Paint3D 텍스처 생성/적용 (MLLM용 Realistic RGB 준비)
- E1: 메쉬 전처리 (UV unwrap)
- E2: Paint3D 실행 (PyTorch Nightly cu128 포팅)

Requirements:
  # 1) PyTorch Nightly for Blackwell (RTX PRO 6000):
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

  # 2) Paint3D 클론 및 설치:
  git clone https://github.com/OpenTexture/Paint3D.git
  cd Paint3D
  pip install -r requirements.txt

  # 3) Paint3D 모델 체크포인트 다운로드 (README 참조)
"""

import os
import sys
import json
import argparse
import subprocess
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
# E0: Paint3D 환경 설정 (Blackwell 호환)
# ============================================================
def setup_paint3d_env(cfg):
    """
    Paint3D를 Blackwell (RTX PRO 6000) 에서 실행하기 위한 환경 설정

    주의사항:
    - Paint3D는 원래 PyTorch 1.12 + CUDA 11.6용
    - Blackwell은 sm_120/sm_122 → CUDA 12.8+ 필요
    - PyTorch Nightly (cu128) 사용
    """
    print("=" * 60)
    print("[E0] Paint3D 환경 설정 (Blackwell 포팅)")
    print("=" * 60)

    paint3d_dir = Path("thirdparty/Paint3D")

    if not paint3d_dir.exists():
        print("  Paint3D 클론 중...")
        subprocess.run([
            "git", "clone", cfg['paint3d']['repo'],
            str(paint3d_dir)
        ], check=True)

    # PyTorch 버전 확인
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            print(f"  Compute Capability: {cap[0]}.{cap[1]}")
            if cap[0] >= 12:
                print("  ✅ Blackwell 아키텍처 감지")
            else:
                print(f"  ℹ️ SM {cap[0]}.{cap[1]} 아키텍처")
    except:
        print("  ⚠️ PyTorch 미설치")
        print("  설치 명령:")
        print("  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")

    # Paint3D 의존성 패치 안내
    print("\n  [Blackwell 포팅 주의사항]")
    print("  1. torch.cuda.amp.autocast → torch.amp.autocast('cuda') 로 변경")
    print("  2. xformers가 Blackwell 미지원시 → attention fallback 사용")
    print("  3. TORCH_CUDA_ARCH_LIST='12.0' 환경변수 설정")
    print("  4. diffusers 최신 버전 사용 (Blackwell 호환)")

    return paint3d_dir


# ============================================================
# E1: 메쉬 전처리 (UV Unwrap)
# ============================================================
def _uv_unwrap_subprocess(mesh_path, output_path):
    """Run xatlas UV unwrap in a subprocess to isolate segfaults.

    UV unwrap 로직은 phase_e_uv_unwrap_worker.py에 구현되어 있음.
    """
    worker = UV_UNWRAP_WORKER_SCRIPT
    if not worker.exists():
        # Worker 파일이 없으면 에러
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
    """
    print("\n" + "=" * 60)
    print("[E1] 메쉬 UV 전처리")
    print("=" * 60)

    textures_dir = Path(cfg['paths']['textures'])
    textures_dir.mkdir(parents=True, exist_ok=True)

    # Worker 스크립트 존재 확인
    if not UV_UNWRAP_WORKER_SCRIPT.exists():
        print(f"  ⚠️ UV unwrap worker 없음: {UV_UNWRAP_WORKER_SCRIPT}")
        return

    # Mesh index 로드
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
        # Skip if already done
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
        except Exception as e:
            uv_stats["failed"] += 1

        # Ensure uv_mesh.obj always exists: convert ply/other to obj via trimesh
        if not output_path.exists():
            try:
                import trimesh
                m = trimesh.load(str(Path(mesh_path).resolve()), force='mesh')
                m.export(str(output_path))
                print(f"    {obj_id}: converted {Path(mesh_path).suffix} -> .obj (no UV)")
            except Exception as conv_e:
                # Last resort: just copy the file with .obj extension
                import shutil
                try:
                    shutil.copy2(mesh_path, str(output_path))
                    print(f"    {obj_id}: copied as .obj fallback")
                except:
                    print(f"    {obj_id}: conversion failed: {conv_e}")

    print(f"\n  UV 있음: {uv_stats['has_uv']}")
    print(f"  UV unwrap 성공: {uv_stats['no_uv_unwrapped']}")
    print(f"  실패/fallback: {uv_stats['failed']}")
    print(f"  스킵 (메쉬 없음): {uv_stats['skipped']}")

    return uv_stats


# ============================================================
# E2: Paint3D 실행
# ============================================================
def _check_paint3d_deps():
    """Paint3D 의존성 사전 체크. 문제 있으면 에러 메시지와 함께 False 반환."""
    issues = []

    # 1) kaolin 체크
    try:
        import kaolin
        ver = getattr(kaolin, '__version__', 'unknown')
        if ver == '0.1':
            issues.append(
                "kaolin이 PyPI placeholder(0.1)입니다. 진짜 kaolin을 설치하세요:\n"
                "    pip uninstall kaolin -y\n"
                "    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html"
            )
        else:
            print(f"  ✅ kaolin {ver}")
    except ImportError:
        issues.append(
            "kaolin이 설치되어 있지 않습니다:\n"
            "    pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html"
        )

    # 2) diffusers 체크
    try:
        import diffusers
        print(f"  ✅ diffusers {diffusers.__version__}")
    except ImportError:
        issues.append("diffusers가 설치되어 있지 않습니다:\n    pip install diffusers")

    # 3) Paint3D 디렉토리 체크
    paint3d_dir = Path("thirdparty/Paint3D")
    if not paint3d_dir.exists():
        issues.append(
            "Paint3D가 클론되어 있지 않습니다:\n"
            "    git clone https://github.com/OpenTexture/Paint3D.git thirdparty/Paint3D"
        )
    else:
        print(f"  ✅ Paint3D dir: {paint3d_dir}")

    # 4) Paint3D import 테스트
    if not issues:
        try:
            import importlib
            sys.path.insert(0, str(paint3d_dir))
            importlib.import_module("paint3d.paint3d")
            print("  ✅ Paint3D import OK")
        except Exception as e:
            issues.append(f"Paint3D import 실패: {e}")

    # 5) GPU 체크
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"  ✅ GPU: {gpu} (sm_{cap[0]}{cap[1]})")
        else:
            issues.append("CUDA GPU를 사용할 수 없습니다.")
    except Exception:
        pass

    return issues


def _run_fallback_texture(mesh_path, output_dir):
    """Paint3D 없이 fallback 텍스처 생성 (단색 albedo + textured_mesh.obj 복사)"""
    import shutil
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.RandomState(hash(mesh_path) % 2**31)
    color = rng.randint(100, 230, 3).tolist()

    texture_size = 1024
    try:
        from PIL import Image
        albedo = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        albedo_img = Image.fromarray(albedo)
        albedo_img.save(os.path.join(output_dir, "albedo.png"))
    except Exception:
        pass

    # textured_mesh.obj 복사 (원본 메쉬 + mtl 참조 유지)
    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    if not os.path.exists(textured_path):
        try:
            shutil.copy2(mesh_path, textured_path)
        except Exception:
            pass


def run_paint3d(cfg, max_objects=None):
    """모든 오브젝트에 대해 Paint3D 텍스처 생성"""
    print("\n" + "=" * 60)
    print("[E2] Paint3D 텍스처 생성")
    print("=" * 60)

    # ===== 의존성 사전 체크 =====
    print("\n  [의존성 체크]")
    dep_issues = _check_paint3d_deps()
    use_paint3d = True
    if dep_issues:
        use_paint3d = False
        print("\n  ⚠️ Paint3D 의존성 미충족 → fallback(단색 텍스처) 모드로 실행합니다.")
        for i, issue in enumerate(dep_issues, 1):
            print(f"    {i}. {issue}")
        print("\n  ℹ️  Blackwell GPU (sm_120)의 경우:")
        print("      pip install torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu128")
        print("      pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html")

    # Worker 스크립트 확인
    worker_path = PAINT3D_WORKER_SCRIPT
    if not worker_path.exists():
        print(f"  ⚠️ Paint3D worker 스크립트 없음: {worker_path}")
        use_paint3d = False

    textures_dir = Path(cfg['paths']['textures'])

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

    # OakInk 카테고리 매핑 로드 (텍스처 프롬프트용)
    cat_map = {}
    cat_file = Path(cfg['paths']['oakink']) / "shape" / "metaV2" / "yodaobject_cat.json"
    if cat_file.exists():
        with open(cat_file) as f:
            cat_map = json.load(f)

    min_texture_size = cfg.get('paint3d', {}).get('min_texture_size', 512)

    for obj_id, info in tqdm(items, desc="  Paint3D 실행"):
        obj_tex_dir = textures_dir / obj_id
        uv_mesh = obj_tex_dir / "uv_mesh.obj"

        if not uv_mesh.exists():
            uv_mesh = Path(info['mesh_path'])

        if not uv_mesh.exists():
            continue

        # Paint3D 진짜 결과가 있으면 스킵
        mat_files = sorted(
            list(obj_tex_dir.glob("material_*.png"))
            + list(obj_tex_dir.glob("material_*.jpg"))
            + list(obj_tex_dir.glob("material_*.jpeg"))
        )
        if mat_files:
            has_good_texture = False
            try:
                from PIL import Image as _PilImg
                for mf in mat_files:
                    _mat = _PilImg.open(str(mf))
                    _w, _h = _mat.size
                    if _w >= min_texture_size and _h >= min_texture_size:
                        has_good_texture = True
                        break
            except Exception:
                pass
            if has_good_texture:
                continue  # 정상 텍스처 → 스킵
            else:
                for mf in mat_files:
                    print(f"    {obj_id}: {mf.name} is dummy, removing...")
                    mf.unlink()
        elif (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "textured_mesh.mtl").exists():
            pass  # mtl이 있지만 material png가 없는 경우 → 재시도 필요

        # textured_mesh.obj + albedo.png 이미 있으면 스킵
        if (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "albedo.png").exists():
            continue

        # 카테고리 기반 프롬프트 생성
        obj_name = info.get('name', 'object')
        prompt = f"a realistic {obj_name}, photorealistic texture, detailed surface"

        if use_paint3d:
            cmd = [
                sys.executable, str(worker_path),
                "--mesh_path", str(uv_mesh),
                "--output_dir", str(obj_tex_dir),
                "--prompt", prompt,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    err_msg = result.stderr.strip().split('\n')[-5:]
                    print(f"    ⚠️ {obj_id} 실패: {' | '.join(err_msg)}")
                # 텍스처 결과 검증
                new_mats = sorted(obj_tex_dir.glob("material_*.png"))
                if new_mats:
                    from PIL import Image as _PilChk
                    for mf in new_mats:
                        _img = _PilChk.open(str(mf))
                        _w, _h = _img.size
                        if _w < min_texture_size or _h < min_texture_size:
                            log_dir = Path("logs/paint3d_failures")
                            log_dir.mkdir(parents=True, exist_ok=True)
                            log_file = log_dir / f"{obj_id}.log"
                            with open(log_file, 'w') as lf:
                                lf.write(f"=== {obj_id} ===\nmesh: {uv_mesh}\n")
                                lf.write(f"material: {mf.name} = {_w}x{_h}\n\n")
                                lf.write(f"=== STDOUT ===\n{result.stdout or '(empty)'}\n")
                                lf.write(f"\n=== STDERR ===\n{result.stderr or '(empty)'}\n")
                            print(f"    ⚠️ {obj_id}: {mf.name}={_w}x{_h} (dummy). Log: {log_file}")
                            break
            except subprocess.TimeoutExpired:
                print(f"    ⚠️ {obj_id} 타임아웃")
            except Exception as e:
                print(f"    ⚠️ {obj_id} 에러: {e}")
        else:
            _run_fallback_texture(str(uv_mesh), str(obj_tex_dir))

    # 체크포인트: 텍스처 품질 확인
    check_texture_quality(cfg)


def check_texture_quality(cfg):
    """텍스처 품질 확인"""
    textures_dir = Path(cfg['paths']['textures'])

    total = 0
    has_albedo = 0
    has_textured_mesh = 0

    for obj_dir in textures_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        total += 1
        if (obj_dir / "albedo.png").exists():
            has_albedo += 1
        if (obj_dir / "textured_mesh.obj").exists():
            has_textured_mesh += 1

    print(f"\n  [텍스처 품질 체크]")
    print(f"    총 오브젝트: {total}")
    print(f"    albedo 생성: {has_albedo} ({has_albedo/max(total,1)*100:.1f}%)")
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

    if args.step in ["all", "setup"]:
        setup_paint3d_env(cfg)

    if args.step in ["all", "uv"]:
        preprocess_meshes_uv(cfg)

    if args.step in ["all", "paint3d"]:
        run_paint3d(cfg, max_objects=args.max_objects)

    if args.step in ["all", "check"]:
        check_texture_quality(cfg)

    print("\n" + "=" * 60)
    print("Phase E 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
