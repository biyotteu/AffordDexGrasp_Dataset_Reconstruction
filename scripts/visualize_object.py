#!/usr/bin/env python3
"""
단일 객체 3D 메쉬 시각화 → 이미지 저장

사용법:
  # 특정 객체 디렉토리 지정
  python scripts/visualize_object.py --obj_dir textures/bottle_001

  # obj_id로 지정 (textures/ 아래에서 찾음)
  python scripts/visualize_object.py --obj_id bottle_001

  # 여러 각도 렌더링
  python scripts/visualize_object.py --obj_dir textures/bottle_001 --views 6

  # 출력 경로 지정
  python scripts/visualize_object.py --obj_dir textures/bottle_001 -o output.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def load_mesh_trimesh(mesh_path, texture_path=None):
    """trimesh로 메쉬 로드 (텍스처 포함)"""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force='mesh')

    # 텍스처 적용 시도
    if texture_path and Path(texture_path).exists():
        try:
            tex_img = Image.open(texture_path)
            material = trimesh.visual.texture.SimpleMaterial(image=tex_img)
            if mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=mesh.visual.uv, material=material, image=tex_img
                )
        except Exception as e:
            print(f"  텍스처 적용 실패: {e}")

    return mesh


def render_mesh_matplotlib(mesh, elevations, azimuths, title="", figsize=(16, 5)):
    """matplotlib로 메쉬를 여러 각도에서 렌더링"""
    n_views = len(elevations)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, n_views, figure=fig)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # 중심 정규화
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    vertices = (vertices - center) / scale

    # 면 색상 결정
    face_colors = None
    try:
        if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
            fc = np.array(mesh.visual.face_colors)
            if fc.shape[0] == len(faces) and fc.shape[1] >= 3:
                face_colors = fc[:, :3] / 255.0
    except Exception:
        pass

    if face_colors is None:
        # vertex color 시도
        try:
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vc = np.array(mesh.visual.vertex_colors)[:, :3] / 255.0
                face_colors = vc[faces].mean(axis=1)
        except Exception:
            pass

    if face_colors is None:
        face_colors = np.full((len(faces), 3), 0.7)

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    view_labels = []
    for i in range(n_views):
        if n_views <= 4:
            labels = ["정면", "우측", "상단", "후면"]
            view_labels = labels[:n_views]
        else:
            view_labels.append(f"View {i+1}")

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        ax = fig.add_subplot(gs[0, i], projection='3d')

        # 면 단위 밝기 (간단한 램버트 셰이딩)
        light_dir = np.array([0.5, 0.8, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)

        normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]]
        )
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals = normals / norms

        brightness = np.clip(np.dot(normals, light_dir), 0.15, 1.0)
        shaded = face_colors * brightness[:, np.newaxis]
        shaded = np.clip(shaded, 0, 1)

        # 폴리곤 컬렉션
        tri_verts = vertices[faces]
        poly = Poly3DCollection(
            tri_verts, facecolors=shaded,
            edgecolors='none', linewidths=0.1, alpha=1.0
        )
        ax.add_collection3d(poly)

        margin = 0.6
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_zlim(-margin, margin)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        if i < len(view_labels):
            ax.set_title(view_labels[i], fontsize=11, pad=-5)

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)

    fig.tight_layout()
    return fig


def visualize_object_dir(obj_dir, output_path=None, n_views=4):
    """객체 디렉토리 내 메쉬+텍스처를 시각화"""
    obj_dir = Path(obj_dir)
    if not obj_dir.exists():
        print(f"디렉토리 없음: {obj_dir}")
        return None

    # 메쉬 파일 선택 (우선순위: textured_mesh.obj > uv_mesh.obj)
    mesh_candidates = ["textured_mesh.obj", "uv_mesh.obj"]
    mesh_path = None
    for name in mesh_candidates:
        p = obj_dir / name
        if p.exists() and p.stat().st_size > 0:
            mesh_path = p
            break

    if mesh_path is None:
        print(f"  메쉬 파일 없음: {obj_dir}")
        return None

    # 텍스처 파일 선택 (우선순위: albedo.png > material_0.png)
    texture_candidates = ["albedo.png", "material_0.png"]
    texture_path = None
    for name in texture_candidates:
        p = obj_dir / name
        if p.exists():
            texture_path = p
            break

    print(f"  메쉬: {mesh_path.name}")
    print(f"  텍스처: {texture_path.name if texture_path else '없음'}")

    # 파일 목록 출력
    files = sorted(obj_dir.iterdir())
    print(f"  파일 목록: {', '.join(f.name for f in files if f.is_file())}")

    # 메쉬 로드
    mesh = load_mesh_trimesh(mesh_path, texture_path)
    print(f"  정점: {len(mesh.vertices):,}  면: {len(mesh.faces):,}")

    # 뷰 각도 설정
    if n_views == 1:
        elevations = [25]
        azimuths = [135]
    elif n_views == 2:
        elevations = [25, 25]
        azimuths = [135, 225]
    elif n_views == 3:
        elevations = [25, 25, 80]
        azimuths = [135, 225, 0]
    elif n_views == 4:
        elevations = [25, 25, 80, 25]
        azimuths = [135, 225, 0, 315]
    else:
        elevations = [25] * n_views
        azimuths = np.linspace(0, 360, n_views, endpoint=False).tolist()

    obj_id = obj_dir.name
    fig = render_mesh_matplotlib(
        mesh, elevations, azimuths,
        title=obj_id,
        figsize=(5 * n_views, 5)
    )

    # 텍스처 이미지도 함께 보여주기
    if texture_path:
        tex_fig, tex_axes = plt.subplots(1, 2, figsize=(10, 5))
        tex_img = Image.open(texture_path)
        tex_axes[0].imshow(tex_img)
        tex_axes[0].set_title(f"텍스처: {texture_path.name}")
        tex_axes[0].axis('off')

        # material_0.png도 있으면 같이 보여줌
        mat_path = obj_dir / "material_0.png"
        if mat_path.exists() and mat_path != texture_path:
            mat_img = Image.open(mat_path)
            tex_axes[1].imshow(mat_img)
            tex_axes[1].set_title(f"Material: {mat_path.name}")
        else:
            tex_axes[1].set_visible(False)
        tex_axes[1].axis('off')
        tex_fig.tight_layout()

    # 저장
    if output_path is None:
        output_path = obj_dir / "preview.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close('all')
    print(f"  저장: {output_path}")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="객체 3D 메쉬 시각화")
    parser.add_argument("--obj_dir", type=str, help="객체 텍스처 디렉토리 경로")
    parser.add_argument("--obj_id", type=str, help="객체 ID (textures/ 아래에서 찾음)")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--views", type=int, default=4, help="렌더링 뷰 수 (1-8)")
    parser.add_argument("-o", "--output", type=str, help="출력 이미지 경로")
    args = parser.parse_args()

    # 객체 디렉토리 결정
    if args.obj_dir:
        obj_dir = Path(args.obj_dir)
    elif args.obj_id:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        textures_dir = Path(cfg['paths']['textures'])
        obj_dir = textures_dir / args.obj_id
    else:
        # 인자 없으면 textures/ 아래 첫 번째 디렉토리 사용
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        textures_dir = Path(cfg['paths']['textures'])
        dirs = sorted([d for d in textures_dir.iterdir() if d.is_dir()])
        if not dirs:
            print("textures/ 디렉토리가 비어있습니다")
            sys.exit(1)
        obj_dir = dirs[0]
        print(f"자동 선택: {obj_dir.name}")

    print(f"\n객체 시각화: {obj_dir}")
    result = visualize_object_dir(obj_dir, output_path=args.output, n_views=args.views)
    if result:
        print(f"\n완료: {result}")
    else:
        print("\n시각화 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
