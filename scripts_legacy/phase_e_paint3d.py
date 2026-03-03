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
    """Run xatlas UV unwrap in a subprocess to isolate segfaults."""
    script = f'''
import sys
import trimesh
import numpy as np
try:
    import xatlas
except ImportError:
    sys.exit(2)

mesh_path = "{mesh_path}"
output_path = "{output_path}"

def clean_mesh(mesh):
    """Clean mesh to prevent xatlas segfaults."""
    # 1. Remove degenerate faces (zero-area triangles)
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    # 2. Remove duplicate faces
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    # 3. Merge duplicate vertices (within tolerance)
    mesh.merge_vertices()
    # 4. Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    # 5. Fix normals
    if not mesh.is_watertight:
        mesh.fix_normals()
    return mesh

try:
    mesh = trimesh.load(mesh_path, force='mesh')

    # Check: must have valid faces and vertices
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        mesh.export(output_path)
        print("FAILED: too few vertices/faces")
        sys.exit(1)

    # Check if already has UV
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
        if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            mesh.export(output_path)
            print("HAS_UV")
            sys.exit(0)

    # Clean mesh before xatlas
    mesh = clean_mesh(mesh)

    # Validate after cleaning
    if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
        print("FAILED: mesh degenerate after cleaning")
        sys.exit(1)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)

    # Ensure contiguous arrays
    verts = np.ascontiguousarray(verts)
    faces = np.ascontiguousarray(faces)

    # Validate face indices
    if faces.max() >= verts.shape[0]:
        print("FAILED: face index out of bounds")
        mesh.export(output_path)
        sys.exit(1)

    vmapping, indices, uvs = xatlas.parametrize(verts, faces)
    mesh_with_uv = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        visual=trimesh.visual.TextureVisuals(uv=uvs),
    )
    mesh_with_uv.export(output_path)
    print("UNWRAPPED")
    sys.exit(0)
except Exception as e:
    # fallback: export without UV
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.export(output_path)
    except:
        pass
    print(f"FAILED: {{e}}")
    sys.exit(1)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
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
PAINT3D_RUNNER_SCRIPT = '''# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Paint3D Texture Generation Runner
Adapted for PyTorch Nightly + cu128 (Blackwell compatibility)
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Blackwell compatibility patch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

def patch_for_blackwell():
    """Patch deprecated torch.cuda.amp.autocast for Blackwell GPUs"""
    import torch.cuda.amp as amp
    if not hasattr(amp, '_original_autocast'):
        amp._original_autocast = amp.autocast
        amp.autocast = lambda *args, **kwargs: torch.amp.autocast('cuda', *args, **kwargs)

def generate_texture(mesh_path, output_dir, prompt="a realistic textured household object",
                     paint3d_dir="thirdparty/Paint3D"):
    """Generate texture using Paint3D pipeline"""
    patch_for_blackwell()
    sys.path.insert(0, paint3d_dir)

    try:
        from paint3d.models.textured_mesh import TexturedMeshModel
        from paint3d.paint3d import Paint3DPipeline

        pipeline = Paint3DPipeline(
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result = pipeline.run(
            mesh_path=mesh_path,
            prompt=prompt,
            output_dir=output_dir,
        )
        print(f"Texture done: {output_dir}")
        return True

    except ImportError as e:
        print(f"Paint3D import failed: {e}")
        return generate_texture_fallback(mesh_path, output_dir, prompt)

    except Exception as e:
        print(f"Paint3D failed: {e}")
        return generate_texture_fallback(mesh_path, output_dir, prompt)


def generate_texture_fallback(mesh_path, output_dir, prompt):
    """Fallback: generate solid-color texture so MLLM can still recognize shape"""
    import shutil
    os.makedirs(output_dir, exist_ok=True)

    print("  Fallback: solid color texture")
    rng = np.random.RandomState(hash(mesh_path) % 2**31)
    color = rng.randint(100, 230, 3).tolist()

    # Generate albedo texture (solid color PNG) - no trimesh needed
    texture_size = 1024
    try:
        from PIL import Image
        albedo = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        albedo_img = Image.fromarray(albedo)
        albedo_path = os.path.join(output_dir, "albedo.png")
        albedo_img.save(albedo_path)
    except ImportError:
        # Even without PIL, create a minimal PPM file (no dependencies)
        albedo_path = os.path.join(output_dir, "albedo.ppm")
        header = "P6 " + str(texture_size) + " " + str(texture_size) + " 255 "
        with open(albedo_path, 'wb') as f:
            f.write(header.encode('ascii'))
            pixel = bytes(color)
            f.write(pixel * (texture_size * texture_size))
        # Rename to .png for consistency
        png_path = os.path.join(output_dir, "albedo.png")
        shutil.move(albedo_path, png_path)
        albedo_path = png_path

    # Copy original mesh as "textured" mesh (keeps original geometry)
    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    try:
        shutil.copy2(mesh_path, textured_path)
    except Exception:
        pass

    print(f"  Fallback texture saved: {albedo_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a realistic textured household object")
    parser.add_argument("--paint3d_dir", default="thirdparty/Paint3D")
    args = parser.parse_args()

    generate_texture(args.mesh_path, args.output_dir, args.prompt, args.paint3d_dir)
'''


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

    # Write worker script with explicit UTF-8 encoding
    worker_path = Path("scripts") / "phase_e_paint3d_worker.py"
    with open(worker_path, 'w', encoding='utf-8') as f:
        f.write(PAINT3D_RUNNER_SCRIPT)
    print(f"  Worker script: {worker_path}")

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

    for obj_id, info in tqdm(items, desc="  Paint3D 실행"):
        obj_tex_dir = textures_dir / obj_id
        uv_mesh = obj_tex_dir / "uv_mesh.obj"

        if not uv_mesh.exists():
            uv_mesh = Path(info['mesh_path'])

        if not uv_mesh.exists():
            continue

        # Paint3D 진짜 결과가 있으면 스킵
        # material_*.{png,jpg,jpeg} 중 하나라도 512x512 이상이면 정상 텍스처로 판단
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
                    if _w >= 512 and _h >= 512:
                        has_good_texture = True
                        break
            except Exception:
                pass
            if has_good_texture:
                continue  # 정상 텍스처 → 스킵
            else:
                # 모든 material이 더미 → 삭제하고 재시도
                for mf in mat_files:
                    print(f"    {obj_id}: {mf.name} is dummy, removing...")
                    mf.unlink()
        # textured_mesh.obj + .mtl도 있으면 (material이 정상 참조) 스킵
        elif (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "textured_mesh.mtl").exists():
            # mtl이 있지만 material png가 없는 경우 → 재시도 필요
            pass

        # textured_mesh.obj + albedo.png 이미 있으면 스킵
        if (obj_tex_dir / "textured_mesh.obj").exists() and (obj_tex_dir / "albedo.png").exists():
            continue

        # 카테고리 기반 프롬프트 생성
        obj_name = info.get('name', 'object')
        prompt = f"a realistic {obj_name}, photorealistic texture, detailed surface"

        if use_paint3d:
            # Paint3D subprocess 실행
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
                        if _w < 512 or _h < 512:
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
            # Paint3D 없이 fallback 직접 실행
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
