#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E Paint3D Worker: Paint3D CLI 파이프라인 호출 (별도 conda 환경에서 실행)

이 스크립트는 Paint3D conda 환경(python 3.8 + pytorch 1.12.1)에서 실행됩니다.
메인 파이프라인(python 3.10)에서 subprocess로 호출합니다.

Paint3D 2단계 파이프라인:
  Stage 1 (coarse):  depth-conditioned inpainting → coarse texture
  Stage 2 (refine):  UV-position-conditioned refinement → refined texture

실행:
  conda run -n paint3d python phase_e_paint3d_worker.py \
      --mesh_path <path> --output_dir <dir> --prompt <text> \
      --paint3d_dir thirdparty/Paint3D

종료 코드:
  0: 성공 (텍스처 생성 완료)
  1: Paint3D 실행 실패 → fallback 단색 텍스처 생성
  2: Paint3D 디렉토리 없음
"""
import os
import sys
import argparse
import subprocess
import shutil
import numpy as np
from pathlib import Path


# Paint3D 실제 config 경로 (clone_dir 기준 상대경로)
DEFAULT_SD_CONFIG_STAGE1 = "controlnet/config/depth_based_inpaint_template.yaml"
DEFAULT_SD_CONFIG_STAGE2 = "controlnet/config/UV_based_inpaint_template.yaml"
DEFAULT_RENDER_CONFIG = "paint3d/config/train_config_paint3d.py"


def run_paint3d_cli(mesh_path, output_dir, prompt, paint3d_dir,
                    sd_config_stage1=DEFAULT_SD_CONFIG_STAGE1,
                    sd_config_stage2=DEFAULT_SD_CONFIG_STAGE2,
                    render_config=DEFAULT_RENDER_CONFIG):
    """
    Paint3D 2단계 파이프라인 실행

    Stage 1: Coarse texture (depth-conditioned ControlNet inpainting)
      - sd_config: controlnet/config/depth_based_inpaint_template.yaml
      - models: runwayml/stable-diffusion-v1-5, lllyasviel/control_v11f1p_sd15_depth
    Stage 2: Refined texture (UV-position-conditioned refinement)
      - sd_config: controlnet/config/UV_based_inpaint_template.yaml
      - models: GeorgeQi/Paint3d_UVPos_Control, lllyasviel/control_v11p_sd15_inpaint
    """
    paint3d_path = Path(paint3d_dir).resolve()
    if not paint3d_path.exists():
        print(f"ERROR: Paint3D directory not found: {paint3d_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    stage1_outdir = os.path.join(output_dir, "stage1")
    stage2_outdir = output_dir

    # Config 파일 경로 확인
    stage1_sd_config = paint3d_path / sd_config_stage1
    stage2_sd_config = paint3d_path / sd_config_stage2
    render_cfg = paint3d_path / render_config if render_config else None

    if not stage1_sd_config.exists():
        print(f"ERROR: Stage 1 sd_config not found: {stage1_sd_config}")
        print(f"  Paint3D를 올바르게 클론했는지 확인하세요:")
        print(f"  git clone https://github.com/OpenTexture/Paint3D.git {paint3d_dir}")
        return False

    # --- Stage 1: Coarse texture ---
    stage1_script = paint3d_path / "pipeline_paint3d_stage1.py"
    if not stage1_script.exists():
        print(f"ERROR: Stage 1 script not found: {stage1_script}")
        return False

    stage1_cmd = [
        sys.executable,
        str(stage1_script),
        "--sd_config", str(stage1_sd_config),
        "--render_config", str(render_cfg) if render_cfg and render_cfg.exists() else "",
        "--mesh_path", str(Path(mesh_path).resolve()),
        "--prompt", prompt,
        "--outdir", str(Path(stage1_outdir).resolve()),
    ]
    # render_config 없으면 인자 제거
    if not render_cfg or not render_cfg.exists():
        stage1_cmd = [a for a in stage1_cmd if a]  # 빈 문자열 제거

    print(f"[Stage 1] Coarse texture generation...")
    print(f"  sd_config: {stage1_sd_config.name}")
    result1 = subprocess.run(
        stage1_cmd,
        capture_output=True, text=True,
        cwd=str(paint3d_path),
        timeout=600,
    )

    if result1.returncode != 0:
        print(f"[Stage 1] FAILED (returncode={result1.returncode})")
        if result1.stderr:
            err_lines = result1.stderr.strip().split('\n')[-10:]
            for line in err_lines:
                print(f"  stderr: {line}")
        return False

    # Stage 1 결과 확인
    stage1_path = Path(stage1_outdir)
    stage1_textures = sorted(
        list(stage1_path.glob("*.png")) + list(stage1_path.glob("*.jpg")),
        key=lambda f: f.stat().st_size, reverse=True
    )
    if not stage1_textures:
        # 하위 디렉토리 탐색 (Paint3D가 subdirectory에 저장할 수도 있음)
        stage1_textures = sorted(
            list(stage1_path.rglob("*.png")) + list(stage1_path.rglob("*.jpg")),
            key=lambda f: f.stat().st_size, reverse=True
        )

    if not stage1_textures:
        print("[Stage 1] No texture output found")
        return False

    print(f"[Stage 1] OK: {len(stage1_textures)} texture files")
    print(f"  Best: {stage1_textures[0].name} ({stage1_textures[0].stat().st_size // 1024}KB)")

    # --- Stage 2: Refined texture ---
    stage2_script = paint3d_path / "pipeline_paint3d_stage2.py"
    if not stage2_script.exists():
        print("[Stage 2] Script not found, using Stage 1 output as final")
        _copy_results_to_output(stage1_outdir, output_dir, mesh_path)
        return True

    if not stage2_sd_config.exists():
        print(f"[Stage 2] sd_config not found: {stage2_sd_config}")
        print("  Using Stage 1 output as final")
        _copy_results_to_output(stage1_outdir, output_dir, mesh_path)
        return True

    texture_path = str(stage1_textures[0])

    stage2_cmd = [
        sys.executable,
        str(stage2_script),
        "--sd_config", str(stage2_sd_config),
        "--render_config", str(render_cfg) if render_cfg and render_cfg.exists() else "",
        "--mesh_path", str(Path(mesh_path).resolve()),
        "--texture_path", texture_path,
        "--prompt", prompt,
        "--outdir", str(Path(stage2_outdir).resolve()),
    ]
    if not render_cfg or not render_cfg.exists():
        stage2_cmd = [a for a in stage2_cmd if a]

    print(f"[Stage 2] Texture refinement...")
    print(f"  sd_config: {stage2_sd_config.name}")
    result2 = subprocess.run(
        stage2_cmd,
        capture_output=True, text=True,
        cwd=str(paint3d_path),
        timeout=600,
    )

    if result2.returncode != 0:
        print(f"[Stage 2] FAILED (returncode={result2.returncode})")
        if result2.stderr:
            err_lines = result2.stderr.strip().split('\n')[-10:]
            for line in err_lines:
                print(f"  stderr: {line}")
        print("[Stage 2] Falling back to Stage 1 output")
        _copy_results_to_output(stage1_outdir, output_dir, mesh_path)
        return True

    print("[Stage 2] OK")
    _copy_results_to_output(stage2_outdir, output_dir, mesh_path)
    return True


def _copy_results_to_output(src_dir, output_dir, mesh_path):
    """Paint3D 출력 결과를 표준 파일명으로 정리"""
    src_path = Path(src_dir)
    out_path = Path(output_dir)

    # texture 파일 → albedo.png (가장 큰 파일)
    texture_files = sorted(
        list(src_path.rglob("*.png")) + list(src_path.rglob("*.jpg")),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    # stage1/ 하위 파일은 제외 (이미 처리된 것)
    texture_files = [f for f in texture_files if f.name != "albedo.png"]

    if texture_files and not (out_path / "albedo.png").exists():
        best = texture_files[0]
        shutil.copy2(str(best), str(out_path / "albedo.png"))
        print(f"  albedo.png <- {best.name} ({best.stat().st_size // 1024}KB)")

    # textured_mesh.obj
    textured = out_path / "textured_mesh.obj"
    if not textured.exists():
        # Paint3D가 생성한 .obj 탐색
        paint3d_objs = list(src_path.rglob("*.obj"))
        if paint3d_objs:
            shutil.copy2(str(paint3d_objs[0]), str(textured))
        else:
            try:
                shutil.copy2(str(mesh_path), str(textured))
            except Exception:
                pass

    # .mtl 파일
    for mtl in src_path.rglob("*.mtl"):
        dst_mtl = out_path / mtl.name
        if not dst_mtl.exists():
            shutil.copy2(str(mtl), str(dst_mtl))


def generate_texture_fallback(mesh_path, output_dir):
    """Fallback: MLLM이 형상을 인식할 수 있도록 단색 텍스처 생성"""
    os.makedirs(output_dir, exist_ok=True)

    print("  Fallback: 단색 텍스처 생성")
    rng = np.random.RandomState(hash(str(mesh_path)) % 2**31)
    color = rng.randint(100, 230, 3).tolist()

    texture_size = 1024
    try:
        from PIL import Image
        albedo = np.full((texture_size, texture_size, 3), color, dtype=np.uint8)
        albedo_img = Image.fromarray(albedo)
        albedo_path = os.path.join(output_dir, "albedo.png")
        albedo_img.save(albedo_path)
        print(f"  Fallback texture: {albedo_path} (color={color})")
    except ImportError:
        albedo_path = os.path.join(output_dir, "albedo.ppm")
        header = f"P6 {texture_size} {texture_size} 255 "
        with open(albedo_path, 'wb') as f:
            f.write(header.encode('ascii'))
            pixel = bytes(color)
            f.write(pixel * (texture_size * texture_size))
        png_path = os.path.join(output_dir, "albedo.png")
        shutil.move(albedo_path, png_path)

    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    if not os.path.exists(textured_path):
        try:
            shutil.copy2(str(mesh_path), textured_path)
        except Exception:
            pass

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paint3D Worker (conda 환경에서 실행)")
    parser.add_argument("--mesh_path", required=True, help="입력 메쉬 경로")
    parser.add_argument("--output_dir", required=True, help="텍스처 출력 디렉토리")
    parser.add_argument("--prompt", default="a realistic textured household object")
    parser.add_argument("--paint3d_dir", default="thirdparty/Paint3D", help="Paint3D 클론 경로")
    parser.add_argument("--sd_config_stage1", default=DEFAULT_SD_CONFIG_STAGE1)
    parser.add_argument("--sd_config_stage2", default=DEFAULT_SD_CONFIG_STAGE2)
    parser.add_argument("--render_config", default=DEFAULT_RENDER_CONFIG)
    parser.add_argument("--fallback_only", action="store_true", help="Paint3D 없이 fallback만 실행")
    args = parser.parse_args()

    if args.fallback_only:
        generate_texture_fallback(args.mesh_path, args.output_dir)
        sys.exit(0)

    success = run_paint3d_cli(
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        prompt=args.prompt,
        paint3d_dir=args.paint3d_dir,
        sd_config_stage1=args.sd_config_stage1,
        sd_config_stage2=args.sd_config_stage2,
        render_config=args.render_config,
    )

    if not success:
        print("\nPaint3D failed -> generating fallback texture")
        generate_texture_fallback(args.mesh_path, args.output_dir)
        sys.exit(1)

    sys.exit(0)
