#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase E UV Unwrap Worker: xatlas 기반 UV unwrap (segfault 격리용 별도 프로세스)

실행: python phase_e_uv_unwrap_worker.py --mesh_path <path> --output_path <path>

종료 코드:
  0: 성공 (HAS_UV 또는 UNWRAPPED)
  1: 실패 (FAILED)
  2: xatlas 미설치
"""
import sys
import argparse
import numpy as np


def clean_mesh(mesh):
    """xatlas segfault 방지를 위한 메쉬 정제"""
    # 1. 퇴화 면(zero-area triangles) 제거
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    # 2. 중복 면 제거
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    # 3. 중복 정점 병합
    mesh.merge_vertices()
    # 4. 미참조 정점 제거
    mesh.remove_unreferenced_vertices()
    # 5. 노멀 수정
    if not mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def uv_unwrap(mesh_path, output_path):
    """메쉬에 UV unwrap 수행"""
    import trimesh

    try:
        import xatlas
    except ImportError:
        print("FAILED: xatlas not installed")
        sys.exit(2)

    try:
        mesh = trimesh.load(mesh_path, force='mesh')

        # 유효성 검사: 최소 정점/면 수
        if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
            mesh.export(output_path)
            print("FAILED: too few vertices/faces")
            sys.exit(1)

        # 이미 UV가 있는지 확인
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
            if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                mesh.export(output_path)
                print("HAS_UV")
                sys.exit(0)

        # xatlas 전 메쉬 정제
        mesh = clean_mesh(mesh)

        # 정제 후 유효성 재검사
        if mesh.vertices.shape[0] < 3 or mesh.faces.shape[0] < 1:
            print("FAILED: mesh degenerate after cleaning")
            sys.exit(1)

        verts = np.ascontiguousarray(np.asarray(mesh.vertices, dtype=np.float32))
        faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.uint32))

        # 면 인덱스 범위 검사
        if faces.max() >= verts.shape[0]:
            print("FAILED: face index out of bounds")
            mesh.export(output_path)
            sys.exit(1)

        # xatlas UV parametrize
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
        # 실패 시 UV 없이 원본 메쉬 저장
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            mesh.export(output_path)
        except:
            pass
        print(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UV Unwrap Worker")
    parser.add_argument("--mesh_path", required=True, help="입력 메쉬 경로")
    parser.add_argument("--output_path", required=True, help="출력 메쉬 경로")
    args = parser.parse_args()

    uv_unwrap(args.mesh_path, args.output_path)
