# AffordDexGraspData

**AffordDexGrasp 데이터셋 재구축 파이프라인** — ICCV 2025 논문 ([arXiv 2503.07360](https://arxiv.org/abs/2503.07360)) 기반

DexGYSNet 데이터셋으로부터 AffordDexGrasp 학습용 데이터를 자동 생성하는 end-to-end 파이프라인입니다. 원본 논문의 데이터 구축 과정(Scene 구성 → 렌더링 → Semantic 추출 → GT Affordance 생성)을 재현

## 파이프라인 개요

```
Phase A: 원천 데이터 다운로드 (DexGYS, OakInk, ShadowHand)
   ↓
Phase B: 표준 메타 변환 (28D grasp → t/r/q 분해)
   ↓
Phase C: Open-set Split 생성 (seen/unseen 카테고리)
   ↓
Phase D: BlenderProc Scene 구성 (tabletop + physics settle)
   ↓
Phase E: Paint3D 텍스처 적용
   ↓
Phase F: 5-view RGB-D 렌더링 + 포인트 클라우드 생성
   ↓
Phase G: MLLM Semantic 추출 (Qwen3-VL-32B)
   ↓
Phase H: GT Affordance 생성 (ShadowHand FK + 논문 공식)
   ↓
Phase I: 최종 패키징 + QC
```

## 사용 환경

- **GPU**: NVIDIA GPU 96GB
- **Python**: 3.10+
- **PyTorch**: Nightly (cu128)
- **저장공간**: ~500GB 

## 빠른 시작

```bash
# 1. 메인 환경 설정 (Python 3.10)
conda create -n afforddex python=3.10
conda activate afforddex

# PyTorch 설치 (Blackwell GPU)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 의존성 설치
pip install -r requirements.txt

# 2. Paint3D 환경 설정 (Python 3.8, Phase E 전용)
git clone https://github.com/OpenTexture/Paint3D.git thirdparty/Paint3D
conda env create -f thirdparty/Paint3D/environment.yaml
conda run -n paint3d pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html

# 3. 전체 파이프라인 실행
conda activate afforddex
./run_pipeline.sh all

# 또는 단계별 실행
./run_pipeline.sh a          # 데이터 다운로드
./run_pipeline.sh b          # 메타 변환
./run_pipeline.sh h          # GT Affordance 생성
```

## 주요 Phase 상세

### Phase A: 원천 데이터 확보

```bash
python scripts/phase_a_download.py --step all
```

세 가지 데이터를 다운로드합니다:

- **DexGYS** (HuggingFace `wyl2077/DexGYS`): 49,433 grasp 샘플 (28D = translation 3 + axis-angle 3 + joint angles 22)
- **OakInk** (HuggingFace `oakink/OakInk-v1`): 1,801 object 메쉬 (real 101 + virtual 1,700)
- **ShadowHand MJCF**: mujoco_menagerie 모델 + **DexGraspNet 원본 모델** (`shadow_hand_wrist_free.xml`)

DexGraspNet 모델만 별도 다운로드:
```bash
python scripts/phase_a_download.py --step dexgraspnet
```

### Phase H: GT Affordance 생성

```bash
python scripts/phase_h_affordance.py
```

논문 핵심 공식을 구현합니다:

1. **Hand surface points**: ShadowHand FK (Forward Kinematics)로 grasp parameter → 3D hand surface points
2. **Distance map**: `d_i^g = min_{h ∈ H_g} ||p_i - h||` (각 grasp별 최소 거리)
3. **Group union**: `d_i = min_g d_i^g` (semantic group 내 모든 grasp의 union)
4. **Gaussian smoothing**: σ = average nearest-neighbor distance
5. **Score 변환**: `a_i = exp(-d_i / σ)` (0~1 affordance score)

Hand 모델은 DexGraspNet의 `shadow_hand_wrist_free.xml`을 사용합니다 (root=palm, 22 DOF). DexGYS grasp parameter와 동일한 모델이므로 좌표계 보정 없이 정확한 FK가 가능합니다.

## 프로젝트 구조

```
AffordDexGraspData/
├── configs/
│   └── pipeline_config.yaml      # 전체 파이프라인 설정
├── scripts/
│   ├── phase_a_download.py       # 데이터 다운로드
│   ├── phase_b_meta.py           # 메타 변환
│   ├── phase_c_splits.py         # Open-set split
│   ├── phase_d_scene.py          # Scene 구성
│   ├── phase_d_blenderproc_worker.py
│   ├── phase_e_paint3d.py        # Paint3D 텍스처
│   ├── phase_e_paint3d_worker.py
│   ├── phase_f_render.py         # RGB-D 렌더링
│   ├── phase_f_render_worker.py
│   ├── phase_g_mllm.py           # MLLM semantic 추출
│   ├── phase_h_affordance.py     # GT Affordance 생성
│   ├── phase_i_package.py        # 최종 패키징
│   ├── debug_coords.py           # 좌표계 디버깅
│   └── visualize_affordance.py   # Affordance 시각화
├── docs/
│   ├── SETUP_AND_RUN.md          # 상세 설정/실행 가이드
│   ├── PAPER_IMPLEMENTATION.md   # 논문 vs 구현 매핑 (수식, 설정값, 차이점)
│   ├── DESIGN_DECISIONS.md       # 논문 미공개 부분 자체 설계 결정 사항
│   └── PIPELINE_VERIFICATION.md  # 파이프라인 검증 보고서
├── run_pipeline.sh               # 실행 스크립트 (Linux)
├── run_pipeline.bat              # 실행 스크립트 (Windows)
└── requirements.txt
```

## 출력 데이터 구조

```
final_dataset/
├── splits/open_set_{A,B}.json
├── scenes/{scene_id}/
│   ├── rgb_cam{0-4}.png
│   ├── depth_cam{0-4}.npy
│   ├── camera_params.json
│   └── global_pc.npz
├── groups/{group_id}/
│   ├── meta.json        # semantic attributes
│   └── affordance.npz   # object_points + affordance_scores
├── samples/{sample_id}.json
├── semantic_groups.json
└── dataset_stats.json
```

## 설정

모든 파이프라인 설정은 `configs/pipeline_config.yaml`에서 관리합니다. 주요 설정:

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `affordance.num_object_points` | 4096 | Object point cloud 크기 |
| `affordance.num_hand_surface_points` | 2048 | Grasp당 hand surface points |
| `mllm.model_name` | Qwen3-VL-32B | MLLM 모델 |
| `scene.num_cameras` | 5 | 렌더링 카메라 수 (4 lateral + 1 top) |
| `rendering.image_width` | 640 | 렌더링 해상도 |

## 문서

| 문서 | 내용 |
|------|------|
| [SETUP_AND_RUN.md](docs/SETUP_AND_RUN.md) | 환경 설정, 의존성 설치, 단계별 실행 가이드 |
| [PAPER_IMPLEMENTATION.md](docs/PAPER_IMPLEMENTATION.md) | 논문 수식·설정값과 구현 간 매핑, 차이점 분석 |
| [DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) | 논문 미공개 부분 자체 설계 결정 사항 (프롬프트, fallback, 매핑 테이블 등) |

## 참고

- 논문: *AffordDexGrasp: Affordance-Driven Dexterous Grasping* (ICCV 2025)
- 원본 데이터셋: [DexGYSNet](https://huggingface.co/datasets/wyl2077/DexGYS)
- Hand 모델: [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) (`shadow_hand_wrist_free.xml`)
- Object 메쉬: [OakInk-v1](https://huggingface.co/datasets/oakink/OakInk-v1)
