#!/bin/bash
# ============================================================
# AffordDexGrasp Dataset Reconstruction - Master Pipeline
# ============================================================
# Usage:
#   ./run_pipeline.sh [phase] [options]
#
# Examples:
#   ./run_pipeline.sh all          # 전체 파이프라인
#   ./run_pipeline.sh a            # Phase A만
#   ./run_pipeline.sh a,b,c        # Phase A, B, C
#   ./run_pipeline.sh g --max 100  # Phase G, 최대 100 샘플
# ============================================================

set -e

CONFIG="configs/pipeline_config.yaml"
PHASE="${1:-all}"
MAX_SAMPLES="${2:-}"

echo "============================================================"
echo " AffordDexGrasp Dataset Reconstruction Pipeline"
echo " Phase: $PHASE"
echo " Config: $CONFIG"
echo "============================================================"
echo ""

# 환경 확인
echo "[ENV] Python: $(python3 --version)"
echo "[ENV] PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "[ENV] CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "[ENV] GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")' 2>/dev/null || echo 'N/A')"
echo ""

run_phase() {
    local phase=$1
    echo ""
    echo "============================================================"
    echo " Running Phase $phase"
    echo "============================================================"

    case $phase in
        a|A)
            echo "[Phase A] 원천 데이터 확보"
            python3 scripts/phase_a_download.py --config $CONFIG --step all
            ;;
        b|B)
            echo "[Phase B] 표준 메타 생성 + QC"
            python3 scripts/phase_b_meta.py --config $CONFIG --step all
            ;;
        c|C)
            echo "[Phase C] Open-set Split 생성"
            python3 scripts/phase_c_splits.py --config $CONFIG
            ;;
        d|D)
            echo "[Phase D] Scene 구성 + Physics + 충돌 필터링"
            if [ -n "$MAX_SAMPLES" ]; then
                python3 scripts/phase_d_scene.py --config $CONFIG --step all --max_scenes $MAX_SAMPLES
            else
                python3 scripts/phase_d_scene.py --config $CONFIG --step all
            fi
            ;;
        e|E)
            echo "[Phase E] Paint3D 텍스처 생성"
            if [ -n "$MAX_SAMPLES" ]; then
                python3 scripts/phase_e_paint3d.py --config $CONFIG --step all --max_objects $MAX_SAMPLES
            else
                python3 scripts/phase_e_paint3d.py --config $CONFIG --step all
            fi
            ;;
        f|F)
            echo "[Phase F] 5-view RGB-D 렌더링 + Global PC"
            if [ -n "$MAX_SAMPLES" ]; then
                python3 scripts/phase_f_render.py --config $CONFIG --step all --max_scenes $MAX_SAMPLES
            else
                python3 scripts/phase_f_render.py --config $CONFIG --step all
            fi
            ;;
        g|G)
            echo "[Phase G] MLLM Semantic 추출 + Group 생성"
            if [ -n "$MAX_SAMPLES" ]; then
                python3 scripts/phase_g_mllm.py --config $CONFIG --step all --max_samples $MAX_SAMPLES
            else
                python3 scripts/phase_g_mllm.py --config $CONFIG --step all
            fi
            ;;
        g_rule)
            echo "[Phase G-Rule] 규칙 기반 Semantic 추출 (MLLM 없이)"
            python3 scripts/phase_g_mllm.py --config $CONFIG --step extract_rule
            python3 scripts/phase_g_mllm.py --config $CONFIG --step group
            ;;
        h|H)
            echo "[Phase H] GT Affordance 생성"
            if [ -n "$MAX_SAMPLES" ]; then
                python3 scripts/phase_h_affordance.py --config $CONFIG --step all --max_groups $MAX_SAMPLES
            else
                python3 scripts/phase_h_affordance.py --config $CONFIG --step all
            fi
            ;;
        i|I)
            echo "[Phase I] 최종 패키징 + QC"
            python3 scripts/phase_i_package.py --config $CONFIG --step all
            ;;
        *)
            echo "Unknown phase: $phase"
            exit 1
            ;;
    esac
}

# 실행
if [ "$PHASE" = "all" ]; then
    for p in a b c d e f g h i; do
        run_phase $p
    done
elif [[ "$PHASE" == *","* ]]; then
    # 콤마로 구분된 phase 목록
    IFS=',' read -ra PHASES <<< "$PHASE"
    for p in "${PHASES[@]}"; do
        run_phase $(echo $p | xargs)
    done
else
    run_phase $PHASE
fi

echo ""
echo "============================================================"
echo " Pipeline Complete!"
echo "============================================================"
