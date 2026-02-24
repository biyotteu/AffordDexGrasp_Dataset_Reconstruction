# 논문-구현 매핑 문서

**논문**: *AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance*
**저자**: Yi-Lin Wei, Mu Lin, Yuhao Lin, Jian-Jian Jiang, Xiao-Ming Wu, Ling-An Zeng, Wei-Shi Zheng
**학회**: ICCV 2025 | [arXiv 2503.07360](https://arxiv.org/abs/2503.07360)

---

## 1. 논문 구조 → 파이프라인 Phase 매핑

| 논문 섹션 | 내용 | 구현 Phase | 코드 |
|-----------|------|-----------|------|
| Sec 4. Dataset | 원천 데이터 (DexGYSNet + OakInk) | Phase A | `phase_a_download.py` |
| Sec 4. Dataset | Grasp parameter 분해 (28D → t/r/q) | Phase B | `phase_b_meta.py` |
| Sec 4. Dataset | Open-set split (A/B) | Phase C | `phase_c_splits.py` |
| Sec 4. Dataset | Tabletop scene 구성 (BlenderProc) | Phase D | `phase_d_scene.py` |
| Sec 4. Dataset | Realistic 텍스처 (Paint3D) | Phase E | `phase_e_paint3d.py` |
| Sec 4. Dataset | 5-view RGB-D 렌더링 + PC | Phase F | `phase_f_render.py` |
| Sec 3.2 | MLLM intention pre-understanding | Phase G | `phase_g_mllm.py` |
| Sec 3.1 | Generalizable-Instructive Affordance GT | Phase H | `phase_h_affordance.py` |
| - | 최종 패키징 + QC | Phase I | `phase_i_package.py` |

---

## 2. Section 3.1: Generalizable-Instructive Affordance Representation

### 2.1 논문 정의

논문은 기존 fine-grained contact map(손 표면과 오브젝트 간 접촉 거리)이 unseen 카테고리에 일반화하기 어렵다는 문제를 지적합니다. 대안으로 *generalizable-instructive affordance*를 제안합니다. 이 affordance는 두 가지 특성을 가집니다:

- **Generalizable**: 카테고리에 독립적인 semantic 속성(intention, contact parts, direction)과 오브젝트의 local structure를 활용하여 unseen 카테고리에 일반화
- **Instructive**: Dexterous grasp 생성을 효과적으로 가이드할 수 있는 fine-grained 정보 제공

동일 semantic group 내 모든 grasp의 잠재적 파지 가능 영역(graspable region)을 하나의 affordance map으로 통합합니다.

### 2.2 수학적 정의

논문의 GT affordance 생성 공식:

1. **Hand surface points**: FK(Forward Kinematics)로 grasp parameter → 손 표면 점군

   각 grasp g에 대해 hand surface points `H_g = {h_1, ..., h_Nh}`를 생성

2. **Contact distance** (각 grasp별 최소 거리):

   `d_i^g = min_{h ∈ H_g} ||p_i - h||`

   여기서 `p_i`는 오브젝트 포인트 클라우드의 i번째 점

3. **Group union** (semantic group 내 모든 grasp의 union):

   `d_i = min_g d_i^g`

4. **Gaussian smoothing**:

   `σ = average nearest-neighbor distance of object point cloud`

   KNN 이웃의 가중 평균으로 distance smoothing

5. **Score 변환**:

   `a_i = exp(-d_i / σ)`

   0~1 범위의 affordance score로 변환. σ가 작으므로(~2mm) 접촉점 근처에서만 높은 값

### 2.3 구현

**파일**: `scripts/phase_h_affordance.py`

| 논문 개념 | 구현 함수/클래스 | 설명 |
|-----------|-----------------|------|
| Hand surface points (H_g) | `ShadowHandFK.compute_hand_surface_points()` | FK → link transform → mesh surface sampling |
| FK model | `ShadowHandFK._load_pk_model()` | pytorch_kinematics로 MJCF chain 빌드 |
| d_i^g (per-grasp distance) | `compute_distance_map()` | `cKDTree.query()`로 각 점의 최소 거리 |
| d_i (group union) | `compute_group_union_distance()` | 그룹 내 모든 grasp의 `np.minimum` |
| σ (avg NN distance) | `compute_avg_nn_distance()` | KNN k=5의 평균 거리 |
| Gaussian smoothing | `gaussian_smooth_affordance()` | KNN=20 이웃의 Gaussian 가중 평균 |
| a_i = exp(-d/σ) | `distance_to_score()` | `np.exp(-distances / sigma)` |

### 2.4 구현 시 발견한 핵심 이슈

**Hand 모델 불일치**: 논문은 DexGraspNet의 `shadow_hand_wrist_free.xml`을 사용하지만, 초기 구현에서 mujoco_menagerie의 ShadowHand를 사용하여 좌표계 불일치 발생. DexGraspNet 모델은 root=palm (22 DOF)이고, menagerie 모델은 root=forearm (24 DOF, wrist 2 + finger 22). DexGYS의 grasp parameter `t`는 palm 위치를 직접 지정하므로, DexGraspNet 모델을 사용해야 offset 보정 없이 정확한 FK가 가능.

**Object PC 좌표계**: 렌더링된 포인트 클라우드는 world coordinate (Z~1.75m)에 있지만, grasp parameter는 mesh local coordinate (원점 근처)에 있음. Phase H에서는 원본 메쉬에서 직접 샘플링하여 좌표계 일치.

**MJCF 호환성**: DexGraspNet XML에는 pytorch_kinematics가 지원하지 않는 MuJoCo 속성들(`apirate`, `nuser_*`, `condim`, `class` 등)이 포함. ElementTree로 전처리하여 FK에 필요한 최소 요소만 남김 (`_sanitize_mjcf_for_pk()`).

---

## 3. Section 3.2: AffordDexGrasp Framework

### 3.1 Intention Pre-Understanding (MLLM)

#### 논문 내용

논문은 MLLM(GPT-4o)을 사용하여 grasp에 필요한 핵심 semantic cue를 추출합니다:

- **object_category**: 오브젝트 카테고리 (예: mug, bottle)
- **intention**: 사용자 의도 (예: use it, hand over)
- **contact_parts**: 접촉할 오브젝트 부위 (예: handle, body)
- **grasp_direction**: 파지 방향, 6방향으로 이산화 (front, back, left, right, up, down)

이 4가지 속성이 동일한 샘플들을 하나의 *semantic group*으로 묶어 공유 affordance map을 생성합니다. MLLM은 이 정보를 간결한 문장으로 정리합니다 (예: "use the mug from the left by contacting the handle").

Grasp direction은 먼저 이미지 좌표에서 추출한 후, 카메라 pose를 이용해 world 좌표로 변환하고, 가장 가까운 6방향 축으로 이산화합니다.

#### 구현

**파일**: `scripts/phase_g_mllm.py`

| 논문 개념 | 구현 | 차이점 |
|-----------|------|--------|
| MLLM | GPT-4o → **Qwen3-VL-32B** | 로컬 실행 가능, 비용 절감 |
| 입력 | rendered RGB + guidance 텍스트 | 논문과 동일 구조 |
| 출력 | {category, intention, contact_parts, direction} | 논문과 동일 4가지 속성 |
| Chain-of-Thought | 프롬프트 내 단계적 추론 유도 | 논문 프롬프트 미공개 → 자체 설계 |
| Semantic grouping | 동일 속성 → 동일 group_key | 논문과 동일 |
| Fallback | 규칙 기반 추출 (`--step extract_rule`) | 논문에 없음 (MLLM 미사용 시 대안) |

**설정** (`configs/pipeline_config.yaml`):
```yaml
mllm:
  model_name: "Qwen/Qwen3-VL-32B-Instruct"
  quantization: null
  num_rgb_views: 1
  semantic_attrs:
    - "object_category"
    - "intention"
    - "contact_parts"
    - "grasp_direction"
```

## 4. Section 4: Dataset Construction

### 4.1 원천 데이터

#### 논문 내용

- **DexGYSNet**: 50,000쌍의 dexterous grasp + language guidance, 1,800 household objects
- **OakInk**: 오브젝트 메쉬 데이터
- **ShadowHand**: 22 DOF, DexGraspNet의 `shadow_hand_wrist_free.xml`
- Grasp 표현: 28D = t(3) + axis-angle(3) + qpos(22)

#### 구현

**파일**: `scripts/phase_a_download.py`

| 논문 데이터 | 다운로드 소스 | 구현 함수 |
|------------|-------------|----------|
| DexGYSNet | HuggingFace `wyl2077/DexGYS` | `download_dexgys()` |
| OakInk 메쉬 | HuggingFace `oakink/OakInk-v1` | `download_oakink()` |
| ShadowHand (menagerie) | PKU 미러 / GitHub | `download_shadowhand()` |
| ShadowHand (DexGraspNet) | GitHub `PKU-EPIC/DexGraspNet` | `download_dexgraspnet_hand()` |
| obj_id → mesh 매핑 | OakInk metaV2 | `build_mesh_index()` |

**데이터 통계**:
```
DexGYS: train 39,462 + test 9,971 = 49,433 samples
OakInk: real 101 + virtual 1,700 = 1,801 objects
DexGYS obj_id 매칭: 1,649/1,649 (100%)
```

### 4.2 Open-set Split

#### 논문 내용

- 33 카테고리, 1,536 objects, 1,909 scenes, 43,504 grasps
- Open Set A: 10 unseen 카테고리 (9,688 samples, 808 objects) + 159 unseen objects from seen categories
- Open Set B: 10 unseen 카테고리 (29,744 samples, 568 objects) + 202 unseen objects from seen categories
- 규칙: seen의 80% train / 20% test + unseen 전부 test

#### 구현

**파일**: `scripts/phase_c_splits.py`

논문에서 unseen 카테고리 리스트를 공개하지 않았으므로 동등 구현합니다. `configs/pipeline_config.yaml`에서 설정:

```yaml
open_set:
  num_splits: 2        # A and B
  unseen_ratio: 0.2    # 20% categories as unseen
  seen_train_ratio: 0.8 # 80% of seen for train
  seed: 42
```

### 4.3 Scene 구성

#### 논문 내용

BlenderProc으로 tabletop scene을 구성합니다. 오브젝트를 테이블 위에 배치하고, physics settle로 자연스러운 포즈를 만듭니다. Hand-table 충돌을 감지하여 invalid grasp를 필터링합니다. 오브젝트는 shelf 위에 올려 테이블과의 충돌을 방지합니다.

#### 구현

**파일**: `scripts/phase_d_scene.py`, `scripts/phase_d_blenderproc_worker.py`

| 논문 설정 | 구현 설정 (`pipeline_config.yaml`) |
|-----------|-----------------------------------|
| Tabletop scene | `scene.table.size: [1.0, 1.0, 0.02]` |
| Table height | `scene.table.height: 0.75` |
| Physics settle | `scene.physics.settle_steps: 500` |
| 카메라 | `scene.num_cameras: 5` (4 lateral + 1 top-down) |
| 충돌 검사 | BlenderProc collision detection |

### 4.4 텍스처 (Paint3D)

#### 논문 내용

렌더링된 realistic RGB를 MLLM에 입력하기 위해 Paint3D로 오브젝트 메쉬에 텍스처를 적용합니다.

#### 구현

**파일**: `scripts/phase_e_paint3d.py`, `scripts/phase_e_paint3d_worker.py`

Paint3D는 kaolin 0.18.0 + PyTorch 2.8.0 + cu128 필요. 실패 시 기본 색상 텍스처로 자동 대체. Phase G의 MLLM은 형상 인식이 주목적이므로 단색이어도 작동합니다.

### 4.5 렌더링

#### 논문 내용

5개 카메라 뷰(4 lateral + 1 top-down)로 RGB-D를 렌더링합니다. Depth를 back-projection하여 partial PC를 생성하고, 5개 뷰를 merge하여 global point cloud (4096 points)를 만듭니다.

#### 구현

**파일**: `scripts/phase_f_render.py`, `scripts/phase_f_render_worker.py`

| 논문 설정 | 구현 설정 |
|-----------|----------|
| 5 cameras | 4 lateral (30° elevation, 90° spacing) + 1 top-down |
| RGB + Depth | 640×480, Depth EXR format |
| Segmentation mask | 선택사항 (`rendering.use_segmentation: true`) |
| PC merge | Open3D voxel downsample (2mm) → 4096 points |

---

## 5. ShadowHand FK 상세

### 5.1 논문의 Hand 모델

논문은 DexGraspNet의 ShadowHand 모델을 사용합니다:

- **MJCF**: `shadow_hand_wrist_free.xml`
- **Root body**: `robot0:palm` (forearm/wrist 없음)
- **DOF**: 22 finger joints
- **Grasp 표현**: t(3, palm 위치) + r(3, axis-angle 회전) + q(22, joint angles)
- **좌표계**: object local frame 기준

Joint 구성 (22 DOF):
```
FF: FFJ3, FFJ2, FFJ1, FFJ0 (4 joints, index finger)
MF: MFJ3, MFJ2, MFJ1, MFJ0 (4 joints, middle finger)
RF: RFJ3, RFJ2, RFJ1, RFJ0 (4 joints, ring finger)
LF: LFJ4, LFJ3, LFJ2, LFJ1, LFJ0 (5 joints, little finger + metacarpal)
TH: THJ4, THJ3, THJ2, THJ1, THJ0 (5 joints, thumb)
```

### 5.2 구현

**파일**: `scripts/phase_h_affordance.py` → `ShadowHandFK` 클래스

#### Model loading flow:

```
ShadowHandFK.load()
  ├── 1) DexGraspNet 모델 우선 시도
  │   └── _load_pk_model(xml_path, "dexgraspnet")
  │       ├── _sanitize_mjcf_for_pk()  ← XML 전처리
  │       └── pk.build_chain_from_mjcf()
  │
  └── 2) Fallback: mujoco_menagerie
      └── _load_pk_model(xml_path, "menagerie")
```

#### XML 전처리 (`_sanitize_mjcf_for_pk`):

pytorch_kinematics는 MuJoCo의 일부 스키마만 지원하므로, ElementTree로 다음을 제거:

| 제거 태그 | 이유 |
|-----------|------|
| `<option>` | `apirate` 등 pk 미지원 속성 |
| `<size>` | `njmax`, `nconmax` 등 |
| `<default>` | class 기반 기본값 (pk 미지원) |
| `<contact>` | 충돌 쌍 정의 (FK에 불필요) |
| `<tendon>` | 건 제약 (FK에 불필요) |
| `<actuator>` | 액추에이터 (FK에 불필요) |
| `<sensor>` | 센서 (FK에 불필요) |
| `<site>` | 사이트 마커 (FK에 불필요) |
| `<light>` | 조명 (FK에 불필요) |

추가: `<compiler meshdir>` 경로를 로컬 구조에 맞게 수정 (`./mjcf/meshes/` → `./meshes/`)

#### Surface point sampling flow:

```
compute_hand_surface_points(t, r, q)
  ├── R_global = Rotation.from_rotvec(r).as_matrix()
  ├── _compute_pk(t, R_global, q)
  │   ├── chain.forward_kinematics(q_tensor)  ← 22 joints FK
  │   ├── for each link:
  │   │   ├── _match_link_mesh(link_name)
  │   │   ├── trimesh.sample.sample_surface(mesh, n_pts)
  │   │   └── pts_world = R_global @ (link_rot @ pts + link_pos) + t
  │   └── 2048 points로 down/up-sample
  └── (fallback) _compute_fallback()  ← 간단한 근사
```

#### Link mesh 매칭 (`_match_link_mesh`):

DexGraspNet과 menagerie의 메쉬 이름이 다르므로 fuzzy matching:

| FK link name | DexGraspNet mesh | menagerie mesh |
|-------------|-----------------|----------------|
| `ffproximal` | `F3` | `f_proximal` |
| `ffmiddle` | `F2` | `f_proximal` (재사용) |
| `ffdistal` | `F1` | `f_distal_pst` |
| `thproximal` | `TH3_z` | `th_proximal` |
| `thmiddle` | `TH2_z` | `th_middle` |
| `thdistal` | `TH1_z` | `th_distal_pst` |
| `palm` (body) | - | - (box geom 사용) |
| `lfmetacarpal` | `lfmetacarpal` | `lf_metacarpal` |

Mesh scale: MJCF에서 `scale="0.001 0.001 0.001"` 파싱하여 자동 적용 (mm → m 변환).

---

## 6. 논문 vs 구현 차이점 요약

| 항목 | 논문 | 구현 | 사유 |
|------|------|------|------|
| MLLM | GPT-4o | Qwen3-VL-32B | 로컬 실행, 비용 절감, 성능 유사 |
| MLLM 프롬프트 | 미공개 | 자체 설계 (CoT) | 논문 프롬프트 비공개 |
| Open-set 카테고리 | 공식 리스트 | 동등 구현 | 논문 리스트 비공개 |
| Leap Hand | 지원 | 미포함 | ShadowHand만 필수 |
| Object PC 소스 | 렌더링 PC (world coords) | 메쉬 직접 샘플링 | 좌표계 일치 위해 (아래 참고) |

### Object PC 소스에 대한 참고

논문에서는 렌더링된 global PC를 사용하지만, 이 PC는 world coordinate에 있고 grasp parameter는 object local coordinate에 있습니다. 본 구현에서는 이 좌표계 차이를 해결하기 위해 원본 메쉬에서 직접 샘플링합니다. 이는 논문의 의도(오브젝트 표면과 손 표면 간 거리 계산)와 동일한 결과를 냅니다. 향후 Phase F의 렌더링 좌표계를 object local로 변환하는 방식으로 논문과 완전히 동일하게 만들 수 있습니다.

---

## 7. 설정 파일 매핑

`configs/pipeline_config.yaml`의 주요 설정과 논문 대응:

```yaml
# 논문 Sec 4: Dataset statistics
dexgys:
  grasp_dim: 28          # 논문: t(3) + r(3) + q(22) = 28D

# 논문 Sec 5: ShadowHand
shadow_hand:
  dof: 22                # 논문: 22 DOF finger joints
  dexgraspnet_xml: "shadow_hand_wrist_free.xml"  # DexGraspNet 원본

# 논문 Sec 3.1: Affordance GT
affordance:
  num_object_points: 4096      # 논문: PointNet++ 입력 크기
  num_hand_surface_points: 2048 # FK → surface sampling per grasp
  knn_neighbors: 20            # Gaussian smoothing KNN
  score_method: "gaussian"     # 논문: exp(-d/σ)

# 논문 Sec 3.2: MLLM
mllm:
  model_name: "Qwen/Qwen3-VL-32B-Instruct"  # 논문: GPT-4o
  semantic_attrs:              # 논문 Sec 3.2의 4가지 key cues
    - "object_category"
    - "intention"
    - "contact_parts"
    - "grasp_direction"

# 논문 Sec 4: Open-set split
open_set:
  num_splits: 2          # 논문: A and B
  unseen_ratio: 0.2      # 논문: ~10 unseen categories each

# 논문 Sec 4: Scene & Rendering
scene:
  num_cameras: 5         # 논문: 4 lateral + 1 top-down
rendering:
  image_width: 640
  image_height: 480
pointcloud:
  num_final_points: 4096 # 논문: 4096 points
```

---

## 8. GT Affordance 품질 기준

논문 공식 `a_i = exp(-d_i / σ)`의 특성상, σ ≈ 0.002m (~2mm, 4096 points의 avg NN distance)이므로 affordance map은 매우 국소적(sharp)입니다. 접촉점에서 5mm만 떨어져도 score가 0.08로 떨어집니다.

검증된 정상 범위:

| 지표 | 값 | 의미 |
|------|-----|------|
| Union dist min | ~0.0006m | 거의 접촉 (0.6mm) |
| Score max | 0.3~0.8 | Smoothing 후 최대값 |
| Score mean | 0.01~0.05 | 대부분 비접촉 영역 |
| High (>0.5) | 0.05%~2% | 직접 접촉 영역 |
| Near-zero (<0.01) | 60%~90% | 비접촉 영역 |

이는 논문의 설계 의도에 부합합니다. AFM은 이 sharp한 GT affordance를 학습 타겟으로 사용하며, PointNet++가 오브젝트의 local structure와 language semantics를 기반으로 이를 예측합니다.

---

## 9. 학습 하이퍼파라미터 (참고)

논문에서 공개한 학습 설정 (본 프로젝트에서 직접 사용하지 않지만 참고용):

| 항목 | AFM | GFM |
|------|-----|-----|
| Batch size | 16 | 64 |
| Learning rate | 2.0×10⁻⁴ (decay) | 2.0×10⁻⁴ (decay) |
| Backbone | PointNet++ (SA+FP) | PointNet++ (SA) |
| Language encoder | CLIP (pretrained) | CLIP (pretrained) |
| Direction encoder | MLP | MLP |
| - | - | Transformer decoder |
| Loss (λ_pose) | - | 10 |
| Loss (λ_chamfer) | - | 1 |
| Loss (λ_tip) | - | 2 |
