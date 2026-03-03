#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 Paint3D 텍스처 디렉토리에 paint3d.mtl 생성 + textured_mesh.obj 패치 일괄 적용

이미 albedo.png와 textured_mesh.obj가 생성되어 있지만,
paint3d.mtl이 없거나 OBJ에 mtllib 연결이 안 된 경우 사용.

Usage:
    python scripts/fix_mtl_linkage.py --textures_dir data/textures
    python scripts/fix_mtl_linkage.py --textures_dir data/textures --dry_run
"""
import argparse
from pathlib import Path


def write_mtl(out_dir: Path) -> Path:
    mtl_path = out_dir / "paint3d.mtl"
    mtl_path.write_text(
        "# Paint3D generated material\n"
        "newmtl paint3d_material\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        "Ks 0.000 0.000 0.000\n"
        "Ns 10.0\n"
        "d 1.0\n"
        "illum 1\n"
        "map_Kd albedo.png\n"
    )
    return mtl_path


def patch_obj(obj_path: Path, mtl_name: str = "paint3d.mtl"):
    lines = obj_path.read_text().splitlines()
    new_lines = []
    has_mtllib = False
    has_usemtl = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("mtllib "):
            new_lines.append(f"mtllib {mtl_name}")
            has_mtllib = True
        elif stripped.startswith("usemtl "):
            new_lines.append("usemtl paint3d_material")
            has_usemtl = True
        else:
            new_lines.append(line)

    if not has_mtllib:
        new_lines.insert(0, f"mtllib {mtl_name}")

    if not has_usemtl:
        for i, line in enumerate(new_lines):
            if line.strip().startswith("f "):
                new_lines.insert(i, "usemtl paint3d_material")
                break

    obj_path.write_text("\n".join(new_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="기존 텍스처에 paint3d.mtl + OBJ 패치 일괄 적용")
    parser.add_argument("--textures_dir", type=str, required=True,
                        help="텍스처 루트 디렉토리 (data/textures)")
    parser.add_argument("--dry_run", action="store_true",
                        help="실제 수정 없이 대상만 출력")
    args = parser.parse_args()

    tex_root = Path(args.textures_dir)
    if not tex_root.exists():
        print(f"ERROR: {tex_root} 없음")
        return

    fixed = 0
    skipped = 0
    no_albedo = 0

    for obj_dir in sorted(tex_root.iterdir()):
        if not obj_dir.is_dir():
            continue

        albedo = obj_dir / "albedo.png"
        obj_file = obj_dir / "textured_mesh.obj"

        if not albedo.exists():
            no_albedo += 1
            continue

        if not obj_file.exists():
            print(f"  SKIP {obj_dir.name}: albedo.png 있지만 textured_mesh.obj 없음")
            skipped += 1
            continue

        # 이미 paint3d.mtl이 있고 OBJ에 mtllib가 있으면 skip
        mtl_exists = (obj_dir / "paint3d.mtl").exists()
        if mtl_exists:
            obj_text = obj_file.read_text()
            if "mtllib paint3d.mtl" in obj_text:
                skipped += 1
                continue

        if args.dry_run:
            print(f"  [DRY] {obj_dir.name}: mtl={'있음' if mtl_exists else '생성'}, obj=패치")
        else:
            write_mtl(obj_dir)
            patch_obj(obj_file)
            # .paint3d_done 마커도 없으면 생성
            marker = obj_dir / ".paint3d_done"
            if not marker.exists():
                marker.write_text("success")
            print(f"  OK {obj_dir.name}")

        fixed += 1

    print(f"\n완료: {fixed}개 수정, {skipped}개 이미 정상, {no_albedo}개 albedo 없음")


if __name__ == "__main__":
    main()
