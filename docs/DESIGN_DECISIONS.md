# 자체 설계 결정 사항 (논문 미공개 부분)

논문 *AffordDexGrasp* (ICCV 2025, arXiv 2503.07360)에서 공개되지 않은 구현 세부사항을 자체적으로 설계한 항목들을 정리합니다.

---

## 1. Phase G — MLLM 프롬프트 설계

**파일**: `scripts/phase_g_mllm.py`

논문은 GPT-4o를 사용한 "Chain-of-Thought + Visual Prompt" 방식이라고 언급하지만, 실제 프롬프트 내용은 미공개입니다.

### 1-1. System Prompt

```
You are an expert in robotic grasping and manipulation.
You analyze images of objects on a table and extract semantic information
about how a robotic dexterous hand (ShadowHand) should grasp objects
based on human language instructions.

You must extract structured semantic attributes that describe the grasping intention.
Always respond in the specified JSON format.
```

**설계 의도**: MLLM이 일반 대화 모드가 아닌 로보틱 그래스핑 전문가 역할을 유지하도록 role-setting. JSON 출력 형식 강제.

### 1-2. 단건 추출 프롬프트 (EXTRACTION_PROMPT_TEMPLATE)

CoT 4단계 구조로 설계:

| 단계 | 내용 | 추출 속성 |
|------|------|-----------|
| Step 1 | Object Recognition | `object_category` |
| Step 2 | Intention Understanding | `intention` |
| Step 3 | Contact Part Identification | `contact_parts` |
| Step 4 | Grasp Direction | `grasp_direction` |

출력 필드: `reasoning`, `object_category`, `intention`, `contact_parts`, `grasp_direction`, `normalized_command`

**설계 의도**: 논문의 4가지 semantic attribute와 1:1 매핑. reasoning 필드로 CoT 유도. normalized_command로 guidance 텍스트 표준화.

### 1-3. 배치 추출 프롬프트 (BATCH_EXTRACTION_PROMPT)

같은 scene(같은 물체)의 여러 guidance를 한 번의 MLLM 호출로 처리. JSON array 형태로 응답 요청.

**설계 의도**: 같은 물체에 대한 object_category를 일관되게 추출 + GPU inference 횟수 감소.

### 1-4. 모델 선택

| 항목 | 논문 | 구현 | 이유 |
|------|------|------|------|
| MLLM 모델 | GPT-4o | Qwen3-VL-32B-Instruct | 로컬 실행 가능, API 비용 없음, 성능 유사 |

---

## 2. Phase G — 규칙 기반 Fallback (extract_rule)

**파일**: `scripts/phase_g_mllm.py` → `extract_semantics_rule_based()`

논문에 없는 기능. MLLM을 사용할 수 없는 환경(VRAM 부족, 모델 미설치)에서 DexGYS의 `guidance` 텍스트를 키워드 매칭으로 분석하는 대체 수단.

### 키워드 매핑 테이블

**intention:**

| 추출값 | 매칭 키워드 |
|--------|-------------|
| `use` | use, utilize, operate, employ |
| `pick_up` | pick up, grab, take, lift, hold |
| `hand_over` | hand over, pass, give, deliver |
| `pour` | pour, dispense, empty |
| `open` | open, unscrew, twist open |
| `close` | close, shut, seal |
| `cut` | cut, slice, chop |
| `press` | press, push, squeeze |

**contact_parts:**

| 추출값 | 매칭 키워드 |
|--------|-------------|
| `handle` | handle, grip, knob |
| `body` | body, main, barrel, shaft |
| `lid` | lid, cap, cover, top |
| `rim` | rim, edge, brim, lip |
| `base` | base, bottom, foot |
| `neck` | neck, spout, nozzle |
| `blade` | blade, cutting edge |
| `trigger` | trigger, button, switch |

**grasp_direction:**

| 추출값 | 매칭 키워드 |
|--------|-------------|
| `from_right` | from the right, right side, from right |
| `from_left` | from the left, left side, from left |
| `from_above` | from above, from top, top down, from the top |
| `from_front` | from the front, from front, facing |
| `from_behind` | from behind, from the back, from back |
| `from_side` | from the side, lateral, sideways |

**실행**:
```bash
python scripts/phase_g_mllm.py --step extract_rule
```

---

## 3. Phase G — 소규모 그룹 병합 ("merged")

**파일**: `scripts/phase_g_mllm.py` → `generate_semantic_groups()`

논문에 없는 기능.

### 문제

Semantic group key = `(scene_id, intention, contact_parts, grasp_direction)`. MLLM이 세밀하게 분류하면 어떤 그룹에는 grasp가 1개뿐인 경우가 생김. Grasp 1개짜리 그룹은 Phase H에서 union affordance를 계산할 때 대표성이 없고 noise가 큼.

### 해결 방식

```python
MIN_GROUP_SIZE = 2

# 그룹 크기 < 2이면:
# (scene_id, intention, contact_parts, direction)
# →
# (scene_id, intention, "merged", "merged")
```

같은 `(scene_id, intention)`을 공유하는 소규모 그룹들을 `contact_parts="merged"`, `grasp_direction="merged"`로 합쳐서 하나의 더 큰 그룹으로 통합.

### semantic_groups.json에서의 표현

```json
{
  "group_00012": {
    "scene_id": "scene_abc",
    "intention": "pour",
    "contact_parts": "merged",
    "grasp_direction": "merged",
    "num_grasps": 5
  }
}
```

`contact_parts`와 `grasp_direction`이 `"merged"`인 그룹은 단일 semantic label이 아닌 여러 label의 합쳐진 그룹임을 의미.

---

## 4. Phase C — Open-set Split 카테고리 선정

**파일**: `scripts/phase_c_splits.py`

논문은 Split A/B 각각 10개의 unseen 카테고리를 사용한다고 명시하지만, 카테고리 목록을 공개하지 않음.

### 동등 구현 방식

```python
# configs/pipeline_config.yaml
open_set:
  num_splits: 2
  unseen_ratio: 0.2   # 전체 카테고리의 20%를 unseen으로
  seen_train_ratio: 0.8
  seed: 42
```

전체 카테고리를 `seed=42` 기준으로 셔플 후 앞 20%를 Split A unseen, 그 다음 20%를 Split B unseen으로 사용. 논문과 동일 비율·구조이나 구체적인 카테고리 조합은 다를 수 있음.

**규칙 (논문과 동일)**:
- Seen 카테고리 오브젝트: 80% train / 20% test
- Unseen 카테고리 오브젝트: 전부 test only
- Split A와 B는 서로 다른 unseen 카테고리 집합 사용

---

## 5. Phase H — Object Point Cloud 소스

**파일**: `scripts/phase_h_affordance.py`

### 논문 방식 vs 구현 방식

| | 논문 | 구현 |
|--|------|------|
| Object PC 소스 | Phase F에서 렌더링한 global PC (world coordinates) | Object mesh에서 직접 샘플링 |
| 좌표계 | World frame (Z ≈ 1.75m 부근) | Object local frame (원점 근처) |

### 이유

Grasp parameter `t`는 object local frame 기준 palm 위치. 렌더링된 PC는 world frame에 있어 좌표계 불일치 발생. 메쉬에서 직접 샘플링하면 grasp parameter와 같은 local frame을 사용하므로 별도의 좌표 변환 없이 정확한 거리 계산 가능.

**결과**: Phase H의 affordance score 계산 결과는 논문과 동일. 다만 최종 데이터셋의 `object_points`가 world frame이 아닌 local frame 기준으로 저장됨.

---

## 6. Phase H — MJCF 전처리 (_sanitize_mjcf_for_pk)

**파일**: `scripts/phase_h_affordance.py` → `ShadowHandFK._sanitize_mjcf_for_pk()`

논문에 언급 없음. `pytorch_kinematics`가 MuJoCo MJCF의 일부 태그/속성을 지원하지 않아 FK 로드 실패가 발생하여 전처리 추가.

### 제거 대상

| 제거 태그/속성 | 이유 |
|----------------|------|
| `<option>` | `apirate` 등 pk 미지원 속성 |
| `<size>` | `njmax`, `nconmax` 등 |
| `<default>` | class 기반 기본값 (pk 미지원) |
| `<contact>` | 충돌 쌍 정의 (FK에 불필요) |
| `<tendon>` | 건 제약 (FK에 불필요) |
| `<actuator>` | 액추에이터 (FK에 불필요) |
| `<sensor>` | 센서 (FK에 불필요) |
| `<site>` | 사이트 마커 (FK에 불필요) |
| `<light>` | 조명 (FK에 불필요) |

추가로 `<compiler meshdir>` 경로를 로컬 디렉토리 구조에 맞게 수정 (`./mjcf/meshes/` → `./meshes/`).

---

## 7. Phase H — Link-Mesh 이름 매핑 (_match_link_mesh)

**파일**: `scripts/phase_h_affordance.py` → `ShadowHandFK._match_link_mesh()`

논문에 언급 없음. DexGraspNet 모델과 mujoco_menagerie 모델의 link 이름 및 mesh 파일명이 달라 fuzzy matching 테이블을 자체 구축.

### 매핑 테이블

| FK link name | DexGraspNet mesh | menagerie mesh |
|-------------|-----------------|----------------|
| `ffproximal` | `F3` | `f_proximal` |
| `ffmiddle` | `F2` | `f_proximal` (재사용) |
| `ffdistal` | `F1` | `f_distal_pst` |
| `thproximal` | `TH3_z` | `th_proximal` |
| `thmiddle` | `TH2_z` | `th_middle` |
| `thdistal` | `TH1_z` | `th_distal_pst` |
| `palm` | box geom 근사 | box geom 근사 |
| `lfmetacarpal` | `lfmetacarpal` | `lf_metacarpal` |

Mesh scale: MJCF `scale="0.001 0.001 0.001"` 파싱하여 자동 적용 (mm → m).

---

## 8. Phase E — Paint3D Fallback (단색 텍스처)

**파일**: `scripts/phase_e_paint3d_worker.py` → `generate_texture_fallback()`

논문에 언급 없음. Paint3D 실행 환경(kaolin 0.18.0 + PyTorch nightly + CUDA 12.8)이 불충족되거나 Paint3D 자체 오류 시 파이프라인이 중단되지 않도록 Fallback 추가.

### 동작

1. Paint3D import 실패 또는 실행 오류 발생 시 자동 전환
2. `hash(mesh_path) % 2^31` 기반 RNG로 물체마다 일관된 단색 결정 (RGB 각 100~230 범위)
3. PIL로 512×512 단색 텍스처 이미지 생성
4. 메쉬에 UV 좌표 할당 후 저장

**Phase G 영향**: Phase G의 MLLM은 물체의 형상을 인식하는 것이 주목적이므로, 단색 텍스처여도 정상 동작함.

---

## 9. Phase E — Paint3D 텍스처 생성 프롬프트

**파일**: `scripts/phase_e_paint3d.py`, `scripts/phase_e_paint3d_worker.py`

논문은 Paint3D로 텍스처를 생성한다고 언급하지만, 텍스처 생성에 사용하는 텍스트 프롬프트는 미공개.

### 프롬프트 구조

**기본값** (worker 직접 실행 시):
```
"a realistic textured household object"
```

**실제 사용값** (phase_e_paint3d.py에서 물체별로 생성):
```python
prompt = f"a realistic {obj_name}, photorealistic texture, detailed surface"
# 예: "a realistic mug, photorealistic texture, detailed surface"
#     "a realistic scissors, photorealistic texture, detailed surface"
```

`obj_name`은 DexGYS 메타데이터의 `name` 필드에서 가져옴. 물체 이름을 알 수 없을 경우 `"object"`로 대체.

### 설계 의도

Paint3D는 text-to-texture diffusion 모델. 프롬프트가 구체적일수록 해당 물체에 어울리는 realistic 텍스처가 생성됨. Phase G에서 MLLM이 물체의 형상과 카테고리를 인식하는 데 도움이 되도록 가능한 한 물체명을 포함하는 방향으로 설계.

---

## 요약

| 항목 | Phase | 논문 공개 여부 | 자체 설계 내용 |
|------|-------|--------------|---------------|
| MLLM 프롬프트 (System/User/Batch) | G | ❌ 미공개 | CoT 4단계 구조, JSON 출력 형식 |
| MLLM 모델 | G | GPT-4o | Qwen3-VL-32B (로컬 실행) |
| 규칙 기반 Fallback | G | ❌ 없음 | 키워드 매핑 테이블 |
| 소규모 그룹 병합 ("merged") | G | ❌ 없음 | MIN_GROUP_SIZE=2 기준 통합 |
| Open-set 카테고리 목록 | C | ❌ 미공개 | seed=42 랜덤 20% 동등 구현 |
| Object PC 소스 | H | 렌더링 PC | 메쉬 직접 샘플링 |
| MJCF 전처리 | H | ❌ 없음 | pytorch_kinematics 호환용 XML sanitize |
| Link-Mesh 이름 매핑 | H | ❌ 없음 | DexGraspNet/menagerie 간 fuzzy matching |
| Paint3D Fallback | E | ❌ 없음 | 단색 텍스처 자동 생성 |
| Paint3D 텍스처 프롬프트 | E | ❌ 미공개 | `"a realistic {obj_name}, photorealistic texture, detailed surface"` |
