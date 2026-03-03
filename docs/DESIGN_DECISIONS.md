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

```
Look at the following image(s) of an object placed on a table.
A human has given the following instruction for how to grasp this object:

**Instruction:** "{guidance}"

Please analyze the instruction and the object in the image step by step:

**Step 1 - Object Recognition:** What is the object? Identify its category.
**Step 2 - Intention Understanding:** What does the user want to do with the object? (e.g., "use it", "hand it over", "pick it up", "pour from it", "open it")
**Step 3 - Contact Part Identification:** Which specific part of the object should the hand contact? (e.g., "handle", "body", "lid", "rim", "base", "neck", "trigger", "blade")
**Step 4 - Grasp Direction:** From which direction should the hand approach? (e.g., "from the right", "from the left", "from above", "from the front", "from behind", "from the side")

Now provide your analysis in the following JSON format:
{
    "reasoning": "<your step-by-step reasoning>",
    "object_category": "<object category name>",
    "intention": "<what the user wants to do>",
    "contact_parts": "<specific part(s) to contact>",
    "grasp_direction": "<approach direction>",
    "normalized_command": "<a standardized version of the instruction>"
}

Important guidelines:
- Be specific about contact parts (not just "the object" but which part)
- Use consistent terminology for directions (left/right/above/front/behind/side)
- If the instruction is ambiguous, make a reasonable inference based on the object type
- The normalized_command should be a clean, standardized version of the original instruction
```

CoT 4단계 구조:

| 단계 | 내용 | 추출 속성 |
|------|------|-----------|
| Step 1 | Object Recognition | `object_category` |
| Step 2 | Intention Understanding | `intention` |
| Step 3 | Contact Part Identification | `contact_parts` |
| Step 4 | Grasp Direction | `grasp_direction` |

**설계 의도**: 논문의 4가지 semantic attribute와 1:1 매핑. reasoning 필드로 CoT 유도. normalized_command로 guidance 텍스트 표준화. 각 Step에 예시 후보를 열거하여 MLLM 출력 일관성 확보.

### 1-3. 배치 추출 프롬프트 (BATCH_EXTRACTION_PROMPT)

```
Look at the image of an object. Multiple human instructions are given for grasping it.
For EACH instruction, reason step by step:
1. Object Recognition: What is this object?
2. Intention Understanding: What does the user want to do? (use, pick up, hand over, pour, open, etc.)
3. Contact Part Identification: Which specific part should the hand contact? (handle, body, lid, rim, base, neck, blade, trigger, etc.)
4. Grasp Direction: From which direction to approach? (from the left / right / above / front / behind / side)

{instruction_list}

Respond with a JSON array of exactly {num_instructions} entries, one per instruction, in order:
[
    {
        "reasoning": "<your step-by-step reasoning>",
        "object_category": "<object category>",
        "intention": "<what the user wants to do>",
        "contact_parts": "<specific part(s) to contact>",
        "grasp_direction": "<approach direction>",
        "normalized_command": "<standardized version of instruction>"
    },
    ...
]
Be specific about contact parts (not just "the object"). Use consistent direction terms (left/right/above/front/behind/side).
If an instruction is ambiguous, make a reasonable inference based on the object type.
```

같은 scene(같은 물체)의 여러 guidance를 한 번의 MLLM 호출로 처리. `{instruction_list}`에 번호가 매겨진 guidance 텍스트가 들어가고, `{num_instructions}`에 개수가 들어감.

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

## 8. Phase E — Paint3D 별도 conda 환경 실행

**파일**: `scripts/phase_e_paint3d.py`, `scripts/phase_e_paint3d_worker.py`

논문에 언급 없음. Paint3D는 Python 3.8 + PyTorch 1.12.1 + CUDA 11.3 기반이고, 메인 파이프라인은 Python 3.10 기반이므로 동일 환경에서 실행 불가. 별도 conda 환경을 자동 감지/생성하여 subprocess로 호출하는 방식으로 해결.

### 실행 구조

```
phase_e_paint3d.py (메인 환경, Python 3.10)
    │
    ├── E0: conda 환경 감지 → 없으면 environment.yaml로 자동 생성
    ├── E1: UV unwrap (메인 환경에서 실행, xatlas)
    └── E2: subprocess 호출
            │
            └── conda run -n paint3d python phase_e_paint3d_worker.py
                    │
                    ├── Stage 1: pipeline_paint3d_stage1.py (depth-conditioned)
                    │   sd_config: controlnet/config/depth_based_inpaint_template.yaml
                    │
                    └── Stage 2: pipeline_paint3d_stage2.py (UV-position)
                        sd_config: controlnet/config/UV_based_inpaint_template.yaml
```

### conda 환경 관리

`pipeline_config.yaml` 설정:
```yaml
paint3d:
  conda_env: "paint3d"      # conda 환경 이름
  conda_python: null         # 자동 감지. 직접 지정도 가능
```

환경 감지 순서:
1. `conda_python` 직접 경로 지정 → 그대로 사용
2. `conda env list`에서 환경 탐색 → `conda run` 사용
3. 둘 다 없으면 → `environment.yaml`로 자동 생성 시도

### Paint3D Pretrained 모델

HuggingFace에서 첫 실행 시 자동 다운로드:
- `runwayml/stable-diffusion-v1-5`
- `lllyasviel/control_v11f1p_sd15_depth` (Stage 1)
- `lllyasviel/control_v11p_sd15_inpaint` (Stage 2)
- `GeorgeQi/Paint3d_UVPos_Control` (Stage 2)
- `GeorgeQi/realisticVisionV13_v13` (Stage 2 img2img)

### Fallback (단색 텍스처)

Paint3D 실패 시 자동 전환:
1. `hash(mesh_path) % 2^31` 기반 RNG로 물체마다 일관된 단색 결정 (RGB 각 100~230 범위)
2. PIL로 1024×1024 단색 텍스처 이미지 생성
3. 원본 메쉬를 textured_mesh.obj로 복사

**Phase G 영향**: MLLM은 형상 인식이 주목적이므로 단색 텍스처여도 정상 동작함.

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

## 10. Phase D/F — Blender 직접 실행 (blenderproc run 우회)

**파일**: `scripts/phase_d_scene.py`, `scripts/phase_f_render.py`

논문에 언급 없음. BlenderProc CLI(`blenderproc run worker.py`)를 사용하면 conda 환경의 numpy가 Blender 내장 Python 3.11 환경에 주입되어 버전 충돌(`numpy.dtype size changed`)이 발생하는 문제를 해결.

### 문제

`blenderproc run`은 내부적으로 Blender를 실행할 때 현재 Python 환경의 `site-packages`를 Blender의 `PYTHONPATH`에 추가합니다. conda 환경에 numpy 2.x가 설치되어 있으면 Blender 내장 numpy 1.x와 충돌하여 런타임 에러 발생.

### 해결 방식

```
기존: blenderproc run phase_d_blenderproc_worker.py --args
변경: blender --background --python phase_d_blenderproc_worker.py -- --args
```

1. `_find_blender(cfg)`: blenderproc이 설치한 Blender 바이너리 경로를 탐색 (`blenderproc path` → `resources/blender/blender`)
2. `_ensure_blenderproc_in_blender(blender_python)`: Blender 내장 Python에 blenderproc 패키지 설치 확인
3. `clean_env`: `PYTHONPATH`, `PYTHONHOME`, `CONDA_PREFIX`, `CONDA_DEFAULT_ENV`, `CONDA_PYTHON_EXE` 제거 + `LD_LIBRARY_PATH`에서 conda 경로 제거
4. `--` argv separator: Blender 자체 인자와 스크립트 인자를 분리. Worker에서 `sys.argv[sys.argv.index("--") + 1:]`으로 파싱

### Worker import 제약

BlenderProc은 worker 스크립트의 **첫 줄**에 `import blenderproc as bproc`이 있어야 합니다. Docstring이나 다른 import가 먼저 오면 `RuntimeError`가 발생합니다:

```python
import blenderproc as bproc  # 반드시 첫 줄
# 이 위에 docstring이나 다른 코드가 올 수 없음
import sys
import json
...
```

---

## 11. Phase E/F — OBJ-MTL 텍스처 연결 체인

**파일**: `scripts/phase_e_paint3d_worker.py`, `scripts/phase_e_paint3d.py`, `scripts/phase_f_render_worker.py`

논문에 언급 없음. Paint3D가 생성하는 `albedo.png`를 BlenderProc이 자동으로 로드하도록 OBJ-MTL 연결 체인을 구축.

### Paint3D 출력 구조

```
textures/{obj_id}/
├── albedo.png            # UV-mapped 텍스처 아틀라스
├── textured_mesh.obj     # UV 좌표 포함 메쉬
├── paint3d.mtl           # 자동 생성된 material 파일
├── .paint3d_done         # Paint3D 성공 마커
├── material_0.png        # (있을 수 있음) Paint3D 내부 중간 결과
├── material.mtl          # (있을 수 있음) Paint3D 내부 생성
└── uv_mesh.obj           # UV unwrap된 원본 메쉬
```

### MTL 생성 + OBJ 패치

Paint3D가 albedo.png를 생성한 뒤, worker가 자동으로:

1. `paint3d.mtl` 생성:
```
newmtl paint3d_material
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
Ns 10.0
d 1.0
illum 1
map_Kd albedo.png
```

2. `textured_mesh.obj`에 `mtllib paint3d.mtl` 및 `usemtl paint3d_material` 삽입/패치

이 체인이 있어야 BlenderProc의 `load_obj()`가 .obj → .mtl → albedo.png 순서로 자동 텍스처 로딩을 수행합니다.

### Fallback 텍스처도 동일 체인 적용

단색 fallback 텍스처(Paint3D 실패 시)에도 동일한 .mtl 생성 + .obj 패치가 적용되어 BlenderProc에서 일관된 텍스처 로딩이 가능합니다.

---

## 12. Phase F — 3단계 텍스처 Fallback 체인

**파일**: `scripts/phase_f_render_worker.py`

논문에 언급 없음. BlenderProc 렌더링에서 텍스처 로딩 실패에 대비한 3단계 fallback:

| 우선순위 | 방식 | 설명 |
|---------|------|------|
| 1 | MTL 자동 로드 | `load_obj()` → .mtl의 `map_Kd` 자동 인식. Paint3D chain 정상 시 이 방식 사용 |
| 2 | `assign_texture_from_image()` | Blender Principled BSDF 노드에 직접 Image Texture 연결. .mtl 누락/파싱 실패 시 |
| 3 | 랜덤 단색 | `hash(obj_id)` 기반 일관된 RGB 색상 할당 |

`assign_texture_from_image()` 함수는 Blender의 Material/Node 시스템을 직접 조작합니다:
```
Material → Principled BSDF → Base Color ← Image Texture (albedo.png)
```

### 텍스처 소스 우선순위

Phase F는 텍스처 메쉬를 다음 순서로 탐색합니다:
1. `textures/{obj_id}/textured_mesh.obj` (Paint3D 출력)
2. `textures/{obj_id}/uv_mesh.obj` (UV unwrap만 된 메쉬)
3. 원본 `data/objects/{category}/{obj_id}/align_ds/mesh.obj`

### render_info.json

각 렌더링 결과 디렉토리에 메타데이터를 저장합니다:
```json
{
  "obj_id": "bottle_001",
  "mesh_source": "paint3d_textured",
  "texture_source": "mtl_auto",
  "auto_distance": 1.85,
  "num_cameras": 5
}
```

---

## 13. Phase E — .paint3d_done 마커 시스템

**파일**: `scripts/phase_e_paint3d.py`, `scripts/phase_e_paint3d_worker.py`

논문에 언급 없음. Paint3D가 성공적으로 텍스처를 생성했는지 구별하기 위한 마커 파일 시스템.

### 문제

Paint3D 출력 디렉토리에 `material_0.png`, `material.mtl` 등이 있어도 이것이 Paint3D 자체 생성물인지 원본 메쉬에 포함된 파일인지 구분 불가. Fallback 단색 텍스처도 동일한 파일명(`albedo.png`)을 사용하므로 성공/실패 판별 불가.

### 해결 방식

`.paint3d_done` 파일을 성공 마커로 사용:

| 마커 내용 | 의미 |
|----------|------|
| `"success"` | Stage 1 + Stage 2 모두 성공 |
| `"stage1_only"` | Stage 2 없이 Stage 1만 완료 |
| `"fallback:{color}"` | Paint3D 실패, 단색 대체 (색상 정보 포함) |

force 재실행 시 `.paint3d_done` 존재 여부로 이전 Paint3D 결과를 판별하고, `material_*.png` 존재만으로는 성공 판정하지 않습니다.

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
| Paint3D conda 환경 분리 | E | ❌ 없음 | 별도 conda 환경 자동 감지/생성 + subprocess 호출 |
| Paint3D Fallback | E | ❌ 없음 | 단색 텍스처 자동 생성 |
| Paint3D 텍스처 프롬프트 | E | ❌ 미공개 | `"a realistic {obj_name}, photorealistic texture, detailed surface"` |
| Blender 직접 실행 | D/F | ❌ 없음 | `blenderproc run` 우회로 numpy 충돌 방지 |
| OBJ-MTL 텍스처 체인 | E/F | ❌ 없음 | paint3d.mtl 자동 생성 + OBJ 패치 + 3단계 fallback |
| .paint3d_done 마커 | E | ❌ 없음 | Paint3D 성공 여부 판별 마커 파일 시스템 |
