# AffordDexGraspData Pipeline - Phase별 파라미터 가이드

모든 Phase는 `--config configs/pipeline_config.yaml` 옵션으로 설정 파일을 지정할 수 있습니다 (기본값).

---

## Phase A: Download (원천 데이터 확보)

DexGYS labels, OakInk meshes, ShadowHand MJCF 다운로드 및 로더 정합.

```bash
python scripts/phase_a_download.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | 실행 단계 선택 |

**`--step` 옵션:**
- `all` — 전체 실행
- `dexgys` — DexGYS 데이터셋 다운로드 (HuggingFace)
- `oakink` — OakInk 메쉬/메타 다운로드
- `shadowhand` — ShadowHand MJCF 다운로드
- `dexgraspnet` — DexGraspNet 핸드 모델 다운로드
- `index` — obj_id → mesh path 인덱스 생성
- `test` — 샘플 로딩 + 포인트 클라우드 테스트

**관련 Config 키:** `paths.dexgys`, `paths.oakink`, `paths.mjcf`, `dexgys.hf_repo`, `shadow_hand.mjcf_url`

---

## Phase B: Meta (표준 메타 변환 + QC)

DexGYS JSON을 내부 표준 JSONL 포맷으로 변환하고 무결성 리포트 생성.

```bash
python scripts/phase_b_meta.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | `meta` / `qc` / `all` |

**출력:** `processed/meta_all.jsonl` — 각 샘플의 grasp 파라미터(translation, rotation_aa, joint_angles) + 메타데이터

**관련 Config 키:** `paths.processed`, `paths.dexgys`

---

## Phase C: Splits (Open-set Split 생성)

Unseen category generalization을 위한 A/B split 생성.

```bash
python scripts/phase_c_splits.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |

**분할 전략:**
- Seen categories: 80% train / 20% test (오브젝트 단위)
- Unseen categories: 100% test
- Split A와 B는 서로 다른 unseen category 세트 사용

**관련 Config 키:** `open_set.unseen_ratio`, `open_set.seen_train_ratio`, `open_set.seed`

---

## Phase D: Scene (BlenderProc 물리 시뮬레이션 + 충돌 필터링)

BlenderProc으로 테이블 위 물체 배치(Physics Settle) 후 충돌 검사로 유효 grasp 필터링.

```bash
python scripts/phase_d_scene.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | 실행 단계 선택 |
| `--max_scenes` | int | `None` | **샘플 제한** — 처리할 최대 장면 수 |

**`--step` 옵션:**
- `all` — 전체 실행
- `jobs` — scene job 파일 생성
- `physics` — BlenderProc 물리 시뮬레이션
- `filter` — 테이블 관통 grasp 제거
- `filter_detailed` — FK 기반 정밀 충돌 필터링

**사용 예시:**
```bash
# 5개 장면만 테스트
python scripts/phase_d_scene.py --step physics --max_scenes 5
```

**요구사항:** `blenderproc==2.7.0` (→ Blender 3.5.1 → Python 3.10)

**관련 Config 키:** `scene.table`, `scene.camera_layout`, `paths.scenes`

---

## Phase E: Paint3D (텍스처 생성)

Paint3D로 메쉬에 텍스처를 생성/적용 (UV unwrap → Paint3D 2-stage → 품질 체크).

```bash
python scripts/phase_e_paint3d.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | 실행 단계 선택 |
| `--max_objects` | int | `None` | **샘플 제한** — 처리할 최대 오브젝트 수 |
| `--force` | flag | `False` | 기존 fallback 텍스처를 무시하고 Paint3D로 재실행 |

**`--step` 옵션:**
- `all` — 전체 실행
- `setup` — Paint3D conda 환경 감지/생성
- `uv` — xatlas UV unwrap
- `paint3d` — Paint3D 텍스처 생성
- `check` — 텍스처 품질 검증

**사용 예시:**
```bash
# 3개 오브젝트만 텍스처 생성
python scripts/phase_e_paint3d.py --step paint3d --max_objects 3

# 실패한 오브젝트 재시도
python scripts/phase_e_paint3d.py --step paint3d --force
```

**요구사항:** 별도 conda 환경 (Python 3.8 + PyTorch 1.12.1 + CUDA 11.3)

**텍스처 출력 구조:**
```
textures/{obj_id}/
├── material_0.jpeg     ← 원본 텍스처 (우선 사용)
├── uv_mesh.obj         ← material.mtl → material_0.jpeg
├── stage1/res-0/
│   ├── albedo.png      ← Paint3D 최종 결과물
│   └── mesh.obj        ← mesh.mtl → albedo.png
├── textured_mesh.obj   ← paint3d.mtl → albedo.png
└── albedo.png          ← Paint3D 중간 결과물
```

**관련 Config 키:** `paint3d.conda_env`, `paint3d.clone_dir`, `paths.textures`

---

## Phase F: Render (5-View RGB-D 렌더링 + Global Point Cloud)

BlenderProc으로 5시점 RGB-D 렌더링 후 depth를 back-projection하여 글로벌 포인트 클라우드 생성.

```bash
python scripts/phase_f_render.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | 실행 단계 선택 |
| `--max_scenes` | int | `None` | **샘플 제한** — 처리할 최대 장면 수 |

**`--step` 옵션:**
- `all` — 전체 실행
- `render` — 5-view RGB-D 렌더링
- `merge` — depth → point cloud 변환 + 병합
- `verify` — PC 품질 검증

**사용 예시:**
```bash
# 3개 장면만 렌더링
python scripts/phase_f_render.py --step render --max_scenes 3
```

**카메라 설정 (pipeline_config.yaml):**
```yaml
scene:
  camera_layout:
    lateral_count: 4              # 측면 카메라 수
    lateral_elevation_deg: 30     # 측면 카메라 앙각
    lateral_azimuth_spacing_deg: 90  # 방위각 간격
    topdown_count: 1              # 탑다운 카메라
    distance_from_center: 0.8     # 카메라-물체 거리 (m)
    topdown_height: 1.0           # 탑다운 높이 (m)
```

**렌더링 출력:**
```
renders/{scene_id}/
├── rgb_cam0~4.png         # 640x480 RGB
├── depth_cam0~4.npy       # depth map
├── seg_cam0~4.npy         # segmentation
└── camera_params.json     # intrinsics + extrinsics
```

**자동 카메라 보정:**
- `target_fill=0.30` — 물체가 화면의 ~30% 차지
- 최소 거리 0.35m — 과도한 확대 방지
- 납작한 물체(flatness_ratio < 0.3) → elevation 자동 증가 (45~55°)

**관련 Config 키:** `scene.camera_layout`, `affordance.num_object_points`, `pointcloud.*`, `paths.renders`

---

## Phase G: MLLM (Semantic 추출 + Grouping)

MLLM(Qwen3-VL)으로 각 grasp의 의미 정보를 추출하고 semantic group으로 묶음.

```bash
python scripts/phase_g_mllm.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | 실행 단계 선택 |
| `--max_samples` | int | `None` | **샘플 제한** — 처리할 최대 샘플 수 |
| `--use_vllm` | flag | `True` | vLLM 추론 사용 (기본 활성) |
| `--no_vllm` | flag | `False` | vLLM 대신 transformers 사용 |

**`--step` 옵션:**
- `all` — 전체 실행
- `extract` — MLLM semantic 추출
- `extract_rule` — 규칙 기반 fallback (MLLM 없이)
- `group` — semantic group 생성

**사용 예시:**
```bash
# 10개 샘플만 semantic 추출
python scripts/phase_g_mllm.py --step extract --max_samples 10

# GPU 없을 때 규칙 기반으로
python scripts/phase_g_mllm.py --step extract_rule
```

**추출 속성:**
```json
{
  "object_category": "mug",
  "intention": "pick up",
  "contact_parts": "handle",
  "grasp_direction": "from above",
  "normalized_command": "...",
  "reasoning": "..."
}
```

**관련 Config 키:** `mllm.model_name`, `mllm.quantization`, `mllm.max_tokens`, `mllm.num_rgb_views`

---

## Phase H: Affordance (GT Affordance 생성)

FK(Forward Kinematics)로 hand surface points를 계산하고, object point cloud와의 거리 기반 affordance score 생성.

```bash
python scripts/phase_h_affordance.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | `generate` / `visualize` / `all` |
| `--max_groups` | int | `None` | **샘플 제한** — 처리할 최대 그룹 수 |
| `--group_id` | str | `None` | 특정 그룹만 시각화 (visualize 단계) |

**사용 예시:**
```bash
# 5개 그룹만 affordance 생성
python scripts/phase_h_affordance.py --step generate --max_groups 5

# 특정 그룹 시각화
python scripts/phase_h_affordance.py --step visualize --group_id grp_001
```

**Affordance 계산 파이프라인:**
1. **H1**: FK → hand surface points (2048개)
2. **H2**: 각 object point와 가장 가까운 hand point 거리 계산
3. **H3**: Gaussian smoothing (σ = avg nearest neighbor distance)
4. **H4**: 거리 → score 변환: `a_i = exp(-d_i / σ)`

**출력:**
```
affordance_gt/{group_id}.npz
├── object_points: (4096, 3) float32
├── affordance_scores: (4096,) float32    # 0.0 ~ 1.0
├── raw_distances: (4096,) float32
└── smoothed_distances: (4096,) float32
```

**관련 Config 키:** `affordance.num_object_points`, `affordance.num_hand_surface_points`, `affordance.knn_neighbors`, `affordance.score_method`

---

## Phase I: Package (최종 패키징 + QC)

모든 출력을 final_dataset/ 구조로 패키징하고 QC 리포트 생성.

```bash
python scripts/phase_i_package.py [OPTIONS]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `--config` | str | `configs/pipeline_config.yaml` | 설정 파일 경로 |
| `--step` | choice | `all` | `package` / `qc` / `all` |

**최종 데이터셋 구조:**
```
final_dataset/
├── splits/           # open_set_A.json, open_set_B.json
├── scenes/{id}/      # RGB-D, depth, camera, point cloud
├── samples/{id}.json # 개별 샘플 (grasp + semantic)
├── groups/{id}/      # affordance .npz + meta
├── assets/           # meshes, shadow_hand
└── dataset_stats.json
```

**관련 Config 키:** `paths.final_dataset`, 모든 paths.*

---

## 빠른 참조: 샘플 제한 옵션

| Phase | 파라미터 | 예시 |
|-------|---------|------|
| D | `--max_scenes N` | `python scripts/phase_d_scene.py --max_scenes 5` |
| E | `--max_objects N` | `python scripts/phase_e_paint3d.py --max_objects 5` |
| F | `--max_scenes N` | `python scripts/phase_f_render.py --max_scenes 5` |
| G | `--max_samples N` | `python scripts/phase_g_mllm.py --max_samples 5` |
| H | `--max_groups N` | `python scripts/phase_h_affordance.py --max_groups 5` |
| I | — | 전체 패키징 (제한 없음) |

---

## 개별 단계 실행 예시

```bash
# Phase D: physics만 5개 장면
python scripts/phase_d_scene.py --step physics --max_scenes 5

# Phase E: UV unwrap만
python scripts/phase_e_paint3d.py --step uv

# Phase F: 렌더링 3개 + PC 병합
python scripts/phase_f_render.py --step render --max_scenes 3
python scripts/phase_f_render.py --step merge --max_scenes 3

# Phase G: 규칙 기반 semantic (GPU 불필요)
python scripts/phase_g_mllm.py --step extract_rule

# Phase H: 특정 그룹 시각화
python scripts/phase_h_affordance.py --step visualize --group_id grp_001
```
