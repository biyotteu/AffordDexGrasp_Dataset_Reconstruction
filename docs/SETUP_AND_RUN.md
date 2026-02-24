# AffordDexGrasp 데이터셋 재구축 - 설정 및 실행 가이드

## 0. 사용환경경

- **GPU**: NVIDIA GPU (VRAM 96GB)
- **Python**: 3.10+
- **저장공간**: ~500GB (전체 데이터셋)


## 1. 환경 설정

```bash
# conda 환경 생성
conda create -n afforddex python=3.10
conda activate afforddex

# PyTorch 설치 (Blackwell GPU의 경우 cu128 필요)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 기본 의존성
pip install -r requirements.txt

# BlenderProc (Phase D, F)
pip install blenderproc

# xatlas (UV unwrap용, Phase E)
pip install xatlas

# pytorch_kinematics (Phase H)
pip install pytorch_kinematics mujoco

# MLLM (Phase G) - vLLM 권장
pip install vllm>=0.6.0

# Blackwell 환경변수 (sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
```

## 2. 실행 순서

### 전체 파이프라인 (권장)
```bash
./run_pipeline.sh all
```

### 단계별 실행
```bash
# Phase A: 데이터 다운로드
./run_pipeline.sh a

# Phase A (DexGraspNet 모델만): Phase H에 필수
python scripts/phase_a_download.py --step dexgraspnet

# Phase B: 메타 데이터 생성
./run_pipeline.sh b

# Phase C: Open-set 분할
./run_pipeline.sh c

# Phase D: Scene 구성 (BlenderProc 필요)
./run_pipeline.sh d

# Phase E: Paint3D 텍스처
./run_pipeline.sh e

# Phase F: RGB-D 렌더링 + 포인트 클라우드
./run_pipeline.sh f

# Phase G: MLLM Semantic 추출
./run_pipeline.sh g
# 또는 MLLM 없이 규칙 기반:
./run_pipeline.sh g_rule

# Phase H: GT Affordance 생성
./run_pipeline.sh h

# Phase I: 최종 패키징
./run_pipeline.sh i
```

### 테스트 실행 (소량)
```bash
# 10개 scene/object로 테스트
./run_pipeline.sh d 10
./run_pipeline.sh e 10
./run_pipeline.sh g 100
./run_pipeline.sh h 50
```

## 3. Phase별 상세 설명

### Phase A: 원천 데이터 확보

HuggingFace 및 GitHub에서 세 가지 데이터를 다운로드합니다.

- **DexGYS** (`wyl2077/DexGYS`): train 39,462 + test 9,971 = 49,433 샘플. 각 샘플은 28D grasp parameter (t(3) + axis-angle(3) + qpos(22)) + guidance 텍스트로 구성
- **OakInk** (`oakink/OakInk-v1`): OakInkObjectsV2 (real 101개) + OakInkVirtualObjectsV2 (virtual 1,700개) = 1,801 object 메쉬. `align_ds/` 디렉토리의 정렬/다운샘플 메쉬 사용
- **ShadowHand MJCF**: mujoco_menagerie에서 ShadowHand 모델 다운로드
- **DexGraspNet Hand Model**: `shadow_hand_wrist_free.xml` + 18개 OBJ 메쉬를 PKU-EPIC/DexGraspNet에서 다운로드. 이 모델은 DexGYS grasp parameter와 동일한 reference frame을 사용하므로 Phase H에서 정확한 FK 계산에 필수

추가로 `obj_mesh_index.json`을 생성하여 DexGYS의 모든 obj_id를 OakInk 메쉬 경로에 매핑합니다 (1,649개 매칭).

### Phase B: 표준 메타 생성

DexGYS의 28D grasp를 분해하여 표준 메타 포맷으로 변환합니다. 각 샘플에 고유 sample_id를 부여하고, obj_id → mesh 경로 매핑, 누락/깨진 샘플 통계를 생성합니다.

### Phase C: Open-set Split

카테고리 기반 seen/unseen 분할을 생성합니다 (Split A, B 두 가지). 논문에서 unseen 카테고리 리스트를 공개하지 않았으므로 동등 구현합니다. Seen의 80% train / 20% test + unseen 전부 test.

### Phase D: Scene 구성

BlenderProc으로 tabletop scene을 생성합니다. Physics settle로 오브젝트를 자연스럽게 배치하고, hand-table 충돌 기반으로 invalid grasp를 필터링합니다.

### Phase E: Paint3D 텍스처

UV unwrap (xatlas) 후 Paint3D로 realistic 텍스처를 생성합니다. Paint3D 실행 실패 시 기본 색상 텍스처로 자동 대체됩니다.

참고: Paint3D는 kaolin 0.18.0 + PyTorch 2.8.0 + cu128 필요. `pipeline_config.yaml`의 `paint3d` 섹션 참조.

### Phase F: 5-view RGB-D 렌더링

4개 사선(lateral) + 1개 탑다운(top-down) 카메라로 RGB + Depth + Segmentation을 렌더링합니다. Depth back-projection으로 partial PC를 생성하고, 5개 뷰를 merge하여 global point cloud (4096 pts)를 생성합니다.

### Phase G: MLLM Semantic 추출

Qwen3-VL-32B (또는 설정된 MLLM)로 렌더링된 RGB와 guidance 텍스트를 입력하여 semantic attribute를 추출합니다: `{object_category, intention, contact_parts, grasp_direction}`. 동일 attribute를 가진 샘플들을 semantic group으로 묶습니다.

### Phase H: GT Affordance 생성

DexGraspNet의 `shadow_hand_wrist_free.xml` 모델로 Forward Kinematics를 수행합니다.

핵심 구현:

1. **Hand Model**: DexGraspNet 원본 모델 사용 (root=`robot0:palm`, 22 DOF). mujoco_menagerie 모델은 fallback으로만 사용. DexGraspNet 모델이 없으면 `python scripts/phase_a_download.py --step dexgraspnet` 실행
2. **FK**: pytorch_kinematics로 grasp parameter (t, r, q) → link transform 계산. DexGraspNet XML은 pk 호환성을 위해 자동 전처리 (불필요 태그/속성 제거)
3. **Object PC**: 렌더링된 PC가 아닌 원본 메쉬에서 직접 샘플링 (좌표계 일치)
4. **Scoring**: 논문 원래 공식 `a_i = exp(-d_i / σ)` 그대로 사용

기대 출력: 접촉 영역에서 max score 0.5~0.8, 전체 평균 0.02~0.05, high affordance (>0.5) 0.1~1%

### Phase I: 최종 패키징

표준 디렉토리 구조로 패키징하고 QC 리포트를 생성

## 4. 디버깅 도구

### 좌표계 확인
```bash
python scripts/debug_coords.py
```

Phase H가 사용하는 mesh PC, affordance_gt 결과, grasp parameter를 비교합니다. Mesh obj center와 Grasp center의 거리가 0.15m 이내이면 정상입니다.

### Affordance 시각화
```bash
python scripts/visualize_affordance.py
```

GT affordance의 3D 시각화와 score 분포 히스토그램을 생성

## 5. 트러블슈팅

### Blackwell GPU 문제
```bash
# PyTorch가 GPU를 못 찾는 경우
python -c "import torch; print(torch.cuda.is_available())"

# sm_120 에러 → nightly 빌드 확인
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# CUDA 버전 확인
nvidia-smi
nvcc --version
```

### DexGraspNet 모델 로드 실패

`pytorch_kinematics 로드 실패` 에러가 나면 Phase H가 menagerie fallback을 사용합니다. DexGraspNet 모델이 있는지 확인:
```bash
ls data/mjcf_dexgraspnet/shadow_hand_wrist_free.xml
ls data/mjcf_dexgraspnet/meshes/*.obj
```

없으면 다시 다운로드:
```bash
python scripts/phase_a_download.py --step dexgraspnet
```

### Paint3D 실패

Fallback 텍스처가 자동 생성됩니다. MLLM은 형상 인식이 주목적이므로 단색이어도 작동합니다.

### MLLM OOM
```bash
# 양자화 사용 (config에서 수정)
# configs/pipeline_config.yaml → mllm.quantization: "awq"

# 또는 더 작은 모델
# mllm.model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
```

## 6. 출력 구조
```
final_dataset/
├── splits/open_set_{A,B}.json
├── scenes/{scene_id}/
│   ├── rgb_cam{0-4}.png
│   ├── depth_cam{0-4}.npy
│   ├── camera_params.json
│   └── global_pc.npz
├── samples/{sample_id}.json
├── groups/{group_id}/
│   ├── meta.json
│   └── affordance.npz
├── assets/shadow_hand/
├── semantic_groups.json
└── dataset_stats.json
```
