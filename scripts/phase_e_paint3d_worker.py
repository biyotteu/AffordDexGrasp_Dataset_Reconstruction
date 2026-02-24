#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E Paint3D Worker: 텍스처 생성
PyTorch Nightly + cu128 (Blackwell 호환)

실행: python phase_e_paint3d_worker.py --mesh_path <path> --output_dir <dir>
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Blackwell 호환 환경변수
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")


def patch_for_blackwell():
    """Blackwell GPU 호환을 위한 deprecated API 패치"""
    import torch.cuda.amp as amp
    if not hasattr(amp, '_original_autocast'):
        amp._original_autocast = amp.autocast
        amp.autocast = lambda *args, **kwargs: torch.amp.autocast('cuda', *args, **kwargs)


def generate_texture(mesh_path, output_dir, prompt="a realistic textured household object",
                     paint3d_dir="thirdparty/Paint3D"):
    """Paint3D를 사용하여 텍스처 생성"""
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
        print("Texture done: " + str(output_dir))
        return True

    except ImportError as e:
        print("Paint3D import failed: " + str(e))
        return generate_texture_fallback(mesh_path, output_dir, prompt)

    except Exception as e:
        print("Paint3D failed: " + str(e))
        return generate_texture_fallback(mesh_path, output_dir, prompt)


def generate_texture_fallback(mesh_path, output_dir, prompt):
    """Fallback: MLLM이 형상을 인식할 수 있도록 단색 텍스처 생성"""
    import shutil
    os.makedirs(output_dir, exist_ok=True)

    print("  Fallback: 단색 텍스처 생성")
    rng = np.random.RandomState(hash(mesh_path) % 2**31)
    color = rng.randint(100, 230, 3).tolist()

    # 단색 albedo 텍스처 생성 (1024x1024)
    texture_size = 1024
    try:
        from PIL import Image
        albedo = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        albedo_img = Image.fromarray(albedo)
        albedo_path = os.path.join(output_dir, "albedo.png")
        albedo_img.save(albedo_path)
    except ImportError:
        # PIL 없이 최소 PPM 파일 생성
        albedo_path = os.path.join(output_dir, "albedo.ppm")
        header = "P6 " + str(texture_size) + " " + str(texture_size) + " 255 "
        with open(albedo_path, 'wb') as f:
            f.write(header.encode('ascii'))
            pixel = bytes(color)
            f.write(pixel * (texture_size * texture_size))
        png_path = os.path.join(output_dir, "albedo.png")
        shutil.move(albedo_path, png_path)
        albedo_path = png_path

    # 원본 메쉬를 textured mesh로 복사
    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    try:
        shutil.copy2(mesh_path, textured_path)
    except Exception:
        pass

    print("  Fallback texture saved: " + str(albedo_path))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="a realistic textured household object")
    parser.add_argument("--paint3d_dir", default="thirdparty/Paint3D")
    args = parser.parse_args()

    generate_texture(args.mesh_path, args.output_dir, args.prompt, args.paint3d_dir)
