#!/bin/bash
# ============================================================
# BlenderProc 전용 conda 환경 생성 (Python 3.11)
# ============================================================
# 문제: conda afforddex 환경은 Python 3.10인데, Blender 4.2.1은
#       Python 3.11을 사용. blenderproc run이 PYTHONPATH에 conda
#       site-packages를 설정하면 Python 3.10용 numpy가 Blender의
#       Python 3.11에서 로드되어 ABI 불일치 (multiarray) 에러 발생.
#
# 해결: Python 3.11 전용 conda 환경에서 blenderproc run 실행
# ============================================================

set -e

ENV_NAME="bproc"
PYTHON_VERSION="3.11"
BLENDERPROC_VERSION="2.8.0"

echo "========================================="
echo "BlenderProc 전용 conda 환경 생성"
echo "  환경 이름: ${ENV_NAME}"
echo "  Python: ${PYTHON_VERSION}"
echo "  BlenderProc: ${BLENDERPROC_VERSION}"
echo "========================================="

# 기존 환경 확인
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "기존 '${ENV_NAME}' 환경이 존재합니다."
    read -p "삭제 후 재생성하시겠습니까? (y/N) " answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "기존 환경을 유지합니다. blenderproc 설치 상태만 확인합니다."
        conda run -n ${ENV_NAME} python -c "import blenderproc; print('blenderproc', blenderproc.__version__)" 2>/dev/null \
            || conda run -n ${ENV_NAME} pip install blenderproc==${BLENDERPROC_VERSION}
        echo "완료!"
        exit 0
    fi
fi

# 환경 생성
echo ""
echo "[1/2] conda 환경 생성 (Python ${PYTHON_VERSION})..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# blenderproc 설치
echo ""
echo "[2/2] blenderproc ${BLENDERPROC_VERSION} 설치..."
conda run -n ${ENV_NAME} pip install blenderproc==${BLENDERPROC_VERSION}

# 검증
echo ""
echo "========================================="
echo "검증 중..."
echo "========================================="
conda run -n ${ENV_NAME} python -c "
import sys
print(f'Python: {sys.version}')

import numpy
print(f'numpy: {numpy.__version__}')

import blenderproc
print(f'blenderproc: {blenderproc.__version__}')

print()
print('모든 의존성 정상!')
"

echo ""
echo "========================================="
echo "설치 완료!"
echo ""
echo "사용법:"
echo "  python3 scripts/phase_f_render.py --config configs/pipeline_config.yaml --step render"
echo ""
echo "또는 직접 테스트:"
echo "  conda run -n ${ENV_NAME} blenderproc run --custom-blender-path /home/biyotteu/blender/blender-4.2.1-linux-x64 scripts/phase_f_render_worker.py --scene_dir scenes/scene_xxx --renders_dir renders"
echo "========================================="
