#!/usr/bin/env python3
"""
Affordance GT 시각화 스크립트
- 3D point cloud + affordance heatmap (여러 각도)
- 렌더링 이미지와 나란히 배치
- 그룹별 메타 정보 표시

Usage:
  python scripts/visualize_affordance.py --config configs/pipeline_config.yaml
  python scripts/visualize_affordance.py --config configs/pipeline_config.yaml --group group_00000
  python scripts/visualize_affordance.py --config configs/pipeline_config.yaml --num 10
"""

import json
import argparse
from pathlib import Path

import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_pointcloud_3views(ax_list, points, scores, title=""):
    """3개 각도에서 point cloud를 시각화"""
    norm = Normalize(vmin=0, vmax=max(scores.max(), 0.01))
    colors = cm.hot(norm(scores))

    # (front, side, top) 3가지 뷰
    views = [
        (25, -60, "Front-Left"),
        (25, 30, "Front-Right"),
        (80, 0, "Top"),
    ]

    for ax, (elev, azim, view_name) in zip(ax_list, views):
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors, s=1.5, alpha=0.8, edgecolors='none'
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{view_name}", fontsize=9)
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Y', fontsize=7)
        ax.set_zlabel('Z', fontsize=7)
        ax.tick_params(labelsize=6)

        # 축 비율 동일하게
        max_range = np.ptp(points, axis=0).max() / 2
        mid = points.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def visualize_group(group_npz_path, meta_path, renders_dir, output_path):
    """단일 그룹 시각화"""
    data = np.load(group_npz_path)
    points = data['object_points']
    scores = data['affordance_scores']

    # 메타 로드
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    scene_id = meta.get('scene_id', 'unknown')
    group_id = meta.get('group_id', group_npz_path.stem)
    num_grasps = meta.get('num_grasps_in_group', 0)
    sigma = meta.get('sigma', 0)

    # 렌더 이미지 로드
    render_dir = renders_dir / scene_id
    rgb_path = render_dir / "rgb_cam0.png"

    has_render = rgb_path.exists()

    # --- Figure 구성 ---
    if has_render:
        fig = plt.figure(figsize=(18, 8))
        # 왼쪽: 렌더 이미지, 오른쪽: 3D views + 히스토그램
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # 렌더 이미지 (왼쪽 큰 칸)
        ax_img = fig.add_subplot(gs[:, 0])
        from PIL import Image
        img = np.array(Image.open(rgb_path))
        ax_img.imshow(img)
        ax_img.set_title(f"Render: {scene_id}", fontsize=10)
        ax_img.axis('off')

        # 3D views (오른쪽 위 3칸)
        ax_3d = [
            fig.add_subplot(gs[0, 1], projection='3d'),
            fig.add_subplot(gs[0, 2], projection='3d'),
            fig.add_subplot(gs[0, 3], projection='3d'),
        ]

        # 히스토그램 (오른쪽 아래 왼쪽)
        ax_hist = fig.add_subplot(gs[1, 1])

        # 통계 텍스트 (오른쪽 아래 가운데)
        ax_text = fig.add_subplot(gs[1, 2:])
    else:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        ax_3d = [
            fig.add_subplot(gs[0, 0], projection='3d'),
            fig.add_subplot(gs[0, 1], projection='3d'),
            fig.add_subplot(gs[0, 2], projection='3d'),
        ]
        ax_hist = fig.add_subplot(gs[1, 0])
        ax_text = fig.add_subplot(gs[1, 1:])

    # 3D point cloud (3가지 각도)
    plot_pointcloud_3views(ax_3d, points, scores)

    # Colorbar
    sm = cm.ScalarMappable(cmap='hot', norm=Normalize(vmin=0, vmax=max(scores.max(), 0.01)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_3d, fraction=0.02, pad=0.1)
    cbar.set_label('Affordance Score', fontsize=8)

    # 히스토그램
    ax_hist.hist(scores, bins=50, color='coral', edgecolor='black', alpha=0.8)
    ax_hist.set_xlabel('Affordance Score', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.set_title('Score Distribution', fontsize=10)
    ax_hist.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='0.5 threshold')
    ax_hist.legend(fontsize=7)

    # 통계 텍스트
    ax_text.axis('off')
    stats_text = (
        f"Group: {group_id}\n"
        f"Scene: {scene_id}\n"
        f"Grasps in group: {num_grasps}\n"
        f"Points: {len(points)}\n"
        f"σ (avg NN dist): {sigma:.6f}\n"
        f"\n"
        f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n"
        f"Score mean: {scores.mean():.4f}\n"
        f"Score std: {scores.std():.4f}\n"
        f"\n"
        f"High (>0.5): {(scores > 0.5).sum()} pts ({(scores > 0.5).mean()*100:.1f}%)\n"
        f"Peak (>0.8): {(scores > 0.8).sum()} pts ({(scores > 0.8).mean()*100:.1f}%)\n"
        f"Near-zero (<0.01): {(scores < 0.01).sum()} pts ({(scores < 0.01).mean()*100:.1f}%)"
    )

    quality = "GOOD" if scores.max() > 0.5 and (scores > 0.5).mean() > 0.01 else "CHECK"
    if scores.max() < 0.01:
        quality = "BAD (all zeros)"

    stats_text += f"\n\nQuality: {quality}"

    ax_text.text(0.05, 0.95, stats_text, transform=ax_text.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f"Affordance GT: {group_id} ({scene_id})", fontsize=13, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'group_id': group_id,
        'scene_id': scene_id,
        'score_max': float(scores.max()),
        'score_mean': float(scores.mean()),
        'high_pct': float((scores > 0.5).mean() * 100),
        'quality': quality,
    }


def main():
    parser = argparse.ArgumentParser(description="Affordance GT 시각화")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--group", type=str, default=None, help="특정 그룹 ID")
    parser.add_argument("--num", type=int, default=5, help="시각화할 그룹 수")
    parser.add_argument("--output_dir", type=str, default=None, help="출력 디렉토리")
    args = parser.parse_args()

    cfg = load_config(args.config)

    affordance_dir = Path(cfg['paths']['processed']) / "affordance_gt"
    renders_dir = Path(cfg['paths']['renders'])

    if args.output_dir:
        viz_dir = Path(args.output_dir)
    else:
        viz_dir = affordance_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 그룹 목록
    if args.group:
        npz_files = [affordance_dir / f"{args.group}.npz"]
    else:
        npz_files = sorted(affordance_dir.glob("*.npz"))

    if not npz_files:
        print("⚠️ Affordance GT 파일 없음!")
        print(f"  경로: {affordance_dir}")
        return

    print(f"시각화 대상: {len(npz_files)} 그룹 (최대 {args.num}개)")
    print(f"출력 폴더: {viz_dir}")
    print()

    results = []
    for i, npz_path in enumerate(npz_files[:args.num]):
        if not npz_path.exists():
            print(f"  ⚠️ {npz_path.name} 없음")
            continue

        group_id = npz_path.stem
        meta_path = affordance_dir / f"{group_id}_meta.json"
        output_path = viz_dir / f"{group_id}_viz.png"

        print(f"  [{i+1}/{min(len(npz_files), args.num)}] {group_id}...", end=" ", flush=True)

        try:
            result = visualize_group(npz_path, meta_path, renders_dir, output_path)
            results.append(result)
            print(f"max={result['score_max']:.3f}, mean={result['score_mean']:.3f}, "
                  f"high={result['high_pct']:.1f}% [{result['quality']}]")
        except Exception as e:
            print(f"실패: {e}")

    # 전체 요약
    if results:
        print(f"\n{'='*60}")
        print(f"시각화 완료: {len(results)}개")
        print(f"출력 폴더: {viz_dir}")

        goods = [r for r in results if r['quality'] == 'GOOD']
        bads = [r for r in results if 'BAD' in r['quality']]
        checks = [r for r in results if r['quality'] == 'CHECK']

        print(f"  GOOD: {len(goods)}, CHECK: {len(checks)}, BAD: {len(bads)}")

        if bads:
            print(f"\n  ⚠️ BAD 그룹 (score 전부 0):")
            for r in bads:
                print(f"    {r['group_id']} ({r['scene_id']})")

        # 전체 요약 이미지 생성 (max score 분포)
        if len(results) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            max_scores = [r['score_max'] for r in results]
            mean_scores = [r['score_mean'] for r in results]

            axes[0].bar(range(len(max_scores)), max_scores, color='coral')
            axes[0].set_xlabel('Group index')
            axes[0].set_ylabel('Max affordance score')
            axes[0].set_title('Max Score per Group')
            axes[0].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5)

            axes[1].bar(range(len(mean_scores)), mean_scores, color='skyblue')
            axes[1].set_xlabel('Group index')
            axes[1].set_ylabel('Mean affordance score')
            axes[1].set_title('Mean Score per Group')

            plt.tight_layout()
            summary_path = viz_dir / "summary.png"
            plt.savefig(summary_path, dpi=150)
            plt.close()
            print(f"\n  요약 차트: {summary_path}")


if __name__ == "__main__":
    main()
