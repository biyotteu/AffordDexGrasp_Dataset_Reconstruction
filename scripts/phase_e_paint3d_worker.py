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
        print(f"ERROR: Paint3D directory not found: {paint3d_dir}", file=sys.stderr)
        return False

    os.makedirs(output_dir, exist_ok=True)
    stage1_outdir = os.path.join(output_dir, "stage1")
    stage2_outdir = output_dir

    # --- 의존성 사전 체크 ---
    dep_check_failed = False
    for mod_name in ["torch", "kaolin", "diffusers"]:
        try:
            __import__(mod_name)
        except ImportError:
            print(f"ERROR: '{mod_name}' not installed in current Python ({sys.executable})", file=sys.stderr)
            dep_check_failed = True
    if dep_check_failed:
        print(f"ERROR: Paint3D 의존성 부족. conda 환경 확인 필요:", file=sys.stderr)
        print(f"  conda activate paint3d && pip install torch kaolin diffusers", file=sys.stderr)
        return False

    # Config 파일 경로 확인
    stage1_sd_config = paint3d_path / sd_config_stage1
    stage2_sd_config = paint3d_path / sd_config_stage2
    render_cfg = paint3d_path / render_config if render_config and render_config.strip() else None

    if not stage1_sd_config.exists():
        print(f"ERROR: Stage 1 sd_config not found: {stage1_sd_config}", file=sys.stderr)
        print(f"  Paint3D를 올바르게 클론했는지 확인하세요:", file=sys.stderr)
        print(f"  git clone https://github.com/OpenTexture/Paint3D.git {paint3d_dir}", file=sys.stderr)
        return False

    # --- Stage 1: Coarse texture ---
    stage1_script = paint3d_path / "pipeline_paint3d_stage1.py"
    if not stage1_script.exists():
        print(f"ERROR: Stage 1 script not found: {stage1_script}", file=sys.stderr)
        return False

    stage1_cmd = [
        sys.executable,
        str(stage1_script),
        "--sd_config", str(stage1_sd_config),
        "--mesh_path", str(Path(mesh_path).resolve()),
        "--prompt", prompt,
        "--outdir", str(Path(stage1_outdir).resolve()),
    ]
    # render_config가 있으면 추가
    if render_cfg and render_cfg.exists():
        stage1_cmd.extend(["--render_config", str(render_cfg)])

    print(f"[Stage 1] Coarse texture generation...")
    print(f"  sd_config: {stage1_sd_config.name}")
    result1 = subprocess.run(
        stage1_cmd,
        capture_output=True, text=True,
        cwd=str(paint3d_path),
        timeout=600,
    )

    if result1.returncode != 0:
        print(f"[Stage 1] FAILED (returncode={result1.returncode})", file=sys.stderr)
        if result1.stderr:
            err_lines = result1.stderr.strip().split('\n')[-10:]
            for line in err_lines:
                print(f"  stderr: {line}", file=sys.stderr)
        if result1.stdout:
            out_lines = result1.stdout.strip().split('\n')[-5:]
            for line in out_lines:
                print(f"  stdout: {line}", file=sys.stderr)
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
        Path(output_dir, ".paint3d_done").write_text("stage1_only")
        return True

    if not stage2_sd_config.exists():
        print(f"[Stage 2] sd_config not found: {stage2_sd_config}")
        print("  Using Stage 1 output as final")
        _copy_results_to_output(stage1_outdir, output_dir, mesh_path)
        Path(output_dir, ".paint3d_done").write_text("stage1_only")
        return True

    texture_path = str(stage1_textures[0])

    stage2_cmd = [
        sys.executable,
        str(stage2_script),
        "--sd_config", str(stage2_sd_config),
        "--mesh_path", str(Path(mesh_path).resolve()),
        "--texture_path", texture_path,
        "--prompt", prompt,
        "--outdir", str(Path(stage2_outdir).resolve()),
    ]
    if render_cfg and render_cfg.exists():
        stage2_cmd.extend(["--render_config", str(render_cfg)])

    print(f"[Stage 2] Texture refinement...")
    print(f"  sd_config: {stage2_sd_config.name}")
    result2 = subprocess.run(
        stage2_cmd,
        capture_output=True, text=True,
        cwd=str(paint3d_path),
        timeout=600,
    )

    if result2.returncode != 0:
        print(f"[Stage 2] FAILED (returncode={result2.returncode})", file=sys.stderr)
        if result2.stderr:
            err_lines = result2.stderr.strip().split('\n')[-10:]
            for line in err_lines:
                print(f"  stderr: {line}", file=sys.stderr)
        if result2.stdout:
            out_lines = result2.stdout.strip().split('\n')[-5:]
            for line in out_lines:
                print(f"  stdout: {line}", file=sys.stderr)
        print("[Stage 2] Falling back to Stage 1 output", file=sys.stderr)
        _copy_results_to_output(stage1_outdir, output_dir, mesh_path)
        Path(output_dir, ".paint3d_done").write_text("stage1_only")
        return True

    print("[Stage 2] OK")
    _copy_results_to_output(stage2_outdir, output_dir, mesh_path)
    # Paint3D 성공 마커 생성
    Path(output_dir, ".paint3d_done").write_text("ok")
    return True


def _copy_results_to_output(src_dir, output_dir, mesh_path):
    """Paint3D 출력 결과를 표준 파일명으로 정리

    최종 구조:
      output_dir/
        albedo.png          ← Paint3D 텍스처 (가장 큰 PNG)
        textured_mesh.obj   ← 메쉬 (mtllib paint3d.mtl 참조)
        paint3d.mtl         ← albedo.png를 참조하는 material
    """
    src_path = Path(src_dir)
    out_path = Path(output_dir)

    # --- 1) texture 파일 → albedo.png (가장 큰 파일) ---
    texture_files = sorted(
        list(src_path.rglob("*.png")) + list(src_path.rglob("*.jpg")),
        key=lambda f: f.stat().st_size,
        reverse=True,
    )
    # 이미 복사된 albedo.png는 제외
    texture_files = [f for f in texture_files if f.name != "albedo.png"]

    if texture_files and not (out_path / "albedo.png").exists():
        best = texture_files[0]
        shutil.copy2(str(best), str(out_path / "albedo.png"))
        print(f"  albedo.png <- {best.name} ({best.stat().st_size // 1024}KB)")

    # --- 2) textured_mesh.obj ---
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

    # --- 3) paint3d.mtl 생성 (albedo.png를 Base Color로 참조) ---
    if (out_path / "albedo.png").exists():
        _write_mtl_for_albedo(out_path)
        _patch_obj_mtllib(textured, "paint3d.mtl")


def _write_mtl_for_albedo(out_path):
    """albedo.png를 참조하는 .mtl 파일 생성"""
    mtl_path = out_path / "paint3d.mtl"
    mtl_content = """# Paint3D generated material
newmtl paint3d_material
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ns 10.0
d 1.0
illum 1
map_Kd albedo.png
"""
    mtl_path.write_text(mtl_content)
    print(f"  paint3d.mtl 생성 (map_Kd -> albedo.png)")


def _patch_obj_mtllib(obj_path, mtl_name):
    """OBJ 파일의 mtllib 참조를 강제 교체하고 usemtl 확인

    BlenderProc의 load_obj()가 .mtl을 자동으로 읽으려면
    OBJ 파일에 'mtllib xxx.mtl'과 'usemtl yyy' 지시어가 있어야 함.
    """
    obj_path = Path(obj_path)
    if not obj_path.exists():
        return

    try:
        lines = obj_path.read_text().splitlines()
    except Exception:
        return

    new_lines = []
    has_mtllib = False
    has_usemtl = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            # 기존 mtllib을 교체
            new_lines.append(f"mtllib {mtl_name}")
            has_mtllib = True
        elif stripped.startswith("usemtl "):
            # 기존 usemtl을 paint3d material로 교체
            new_lines.append("usemtl paint3d_material")
            has_usemtl = True
        else:
            new_lines.append(line)

    # mtllib이 없었으면 파일 맨 앞에 삽입
    if not has_mtllib:
        new_lines.insert(0, f"mtllib {mtl_name}")

    # usemtl이 없었으면 첫 번째 'f ' (face) 앞에 삽입
    if not has_usemtl:
        for i, line in enumerate(new_lines):
            if line.strip().startswith("f "):
                new_lines.insert(i, "usemtl paint3d_material")
                break

    obj_path.write_text("\n".join(new_lines) + "\n")
    print(f"  textured_mesh.obj 패치: mtllib={mtl_name}, usemtl=paint3d_material")


def generate_texture_fallback(mesh_path, output_dir):
    """Fallback: MLLM이 형상을 인식할 수 있도록 단색 텍스처 생성

    단색 albedo.png + paint3d.mtl + textured_mesh.obj 를 생성하여
    BlenderProc load_obj()가 자동으로 텍스처를 읽을 수 있게 함.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir)

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

    # textured_mesh.obj 복사
    textured_path = os.path.join(output_dir, "textured_mesh.obj")
    if not os.path.exists(textured_path):
        try:
            shutil.copy2(str(mesh_path), textured_path)
        except Exception:
            pass

    # .mtl 생성 + .obj에 mtllib 패치 (load_obj가 자동으로 텍스처 읽도록)
    if (out_path / "albedo.png").exists():
        _write_mtl_for_albedo(out_path)
        _patch_obj_mtllib(Path(textured_path), "paint3d.mtl")

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
