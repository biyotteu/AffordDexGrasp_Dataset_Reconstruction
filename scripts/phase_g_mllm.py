#!/usr/bin/env python3
"""
Phase G: MLLM 기반 Semantic 추출 → Semantic Group 구성
- G1: MLLM semantic cue 추출 (Qwen2.5-VL-72B)
- G2: Semantic group 생성

논문 핵심:
  - 텍스처 적용된 RGB를 MLLM에 입력
  - Chain-of-Thought + Visual Prompt 사용
  - 추출: {object_category, intention, contact_parts, grasp_direction}
  - Semantic group: 같은 (object, intention, contact_parts, direction)을 공유하는 grasps를 묶음

Requirements:
  pip install transformers accelerate vllm qwen-vl-utils Pillow
  # OR for faster inference:
  pip install vllm
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import jsonlines
from tqdm import tqdm


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# MLLM Prompts (논문 미공개 → 새로 설계)
# ============================================================

# Chain-of-Thought 프롬프트 (논문의 CoT + Visual Prompt 방식 재현)
SYSTEM_PROMPT = """You are an expert in robotic grasping and manipulation.
You analyze images of objects on a table and extract semantic information
about how a robotic dexterous hand (ShadowHand) should grasp objects
based on human language instructions.

You must extract structured semantic attributes that describe the grasping intention.
Always respond in the specified JSON format."""

EXTRACTION_PROMPT_TEMPLATE = """Look at the following image(s) of an object placed on a table.
A human has given the following instruction for how to grasp this object:

**Instruction:** "{guidance}"

Please analyze the instruction and the object in the image step by step:

**Step 1 - Object Recognition:** What is the object? Identify its category.
**Step 2 - Intention Understanding:** What does the user want to do with the object? (e.g., "use it", "hand it over", "pick it up", "pour from it", "open it")
**Step 3 - Contact Part Identification:** Which specific part of the object should the hand contact? (e.g., "handle", "body", "lid", "rim", "base", "neck", "trigger", "blade")
**Step 4 - Grasp Direction:** From which direction should the hand approach? (e.g., "from the right", "from the left", "from above", "from the front", "from behind", "from the side")

Now provide your analysis in the following JSON format:
```json
{{
    "reasoning": "<your step-by-step reasoning>",
    "object_category": "<object category name>",
    "intention": "<what the user wants to do>",
    "contact_parts": "<specific part(s) to contact>",
    "grasp_direction": "<approach direction>",
    "normalized_command": "<a standardized version of the instruction>"
}}
```

Important guidelines:
- Be specific about contact parts (not just "the object" but which part)
- Use consistent terminology for directions (left/right/above/front/behind/side)
- If the instruction is ambiguous, make a reasonable inference based on the object type
- The normalized_command should be a clean, standardized version of the original instruction"""

# 배치 프롬프트: 같은 scene의 여러 guidance를 한번에 처리 (논문 방식 CoT 포함)
BATCH_EXTRACTION_PROMPT = """Look at the image of an object. Multiple human instructions are given for grasping it.
For EACH instruction, reason step by step:
1. Object Recognition: What is this object?
2. Intention Understanding: What does the user want to do? (use, pick up, hand over, pour, open, etc.)
3. Contact Part Identification: Which specific part should the hand contact? (handle, body, lid, rim, base, neck, blade, trigger, etc.)
4. Grasp Direction: From which direction to approach? (from the left / right / above / front / behind / side)

{instruction_list}

Respond with a JSON array of exactly {num_instructions} entries, one per instruction, in order:
```json
[
    {{
        "reasoning": "<your step-by-step reasoning>",
        "object_category": "<object category>",
        "intention": "<what the user wants to do>",
        "contact_parts": "<specific part(s) to contact>",
        "grasp_direction": "<approach direction>",
        "normalized_command": "<standardized version of instruction>"
    }},
    ...
]
```
Be specific about contact parts (not just "the object"). Use consistent direction terms (left/right/above/front/behind/side).
If an instruction is ambiguous, make a reasonable inference based on the object type."""


# ============================================================
# G1: MLLM Semantic Cue 추출
# ============================================================

class MLLMExtractor:
    """Qwen2.5-VL-72B 기반 Semantic 추출기"""

    def __init__(self, cfg, use_vllm=True):
        self.cfg = cfg
        self.model_name = cfg['mllm']['model_name']
        self.use_vllm = use_vllm
        self.model = None
        self.processor = None

    def load_model(self):
        """모델 로드"""
        print(f"  모델 로드: {self.model_name}")

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_vllm(self):
        """vLLM 기반 로드 (Linux only - Windows 미지원)"""
        import platform
        if platform.system() == "Windows":
            print("  ⚠️ vLLM은 Windows 미지원, transformers로 대체")
            self.use_vllm = False
            self._load_transformers()
            return

        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoProcessor

            # Processor 로드 (apply_chat_template 용)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            print("  Processor loaded")

            self.model = LLM(
                model=self.model_name,
                quantization=self.cfg['mllm'].get('quantization'),
                tensor_parallel_size=1,
                max_model_len=4096,
                trust_remote_code=True,
            )
            self.sampling_params = SamplingParams(
                temperature=self.cfg['mllm']['temperature'],
                max_tokens=self.cfg['mllm']['max_tokens'],
            )
            print("  vLLM load complete")
        except (ImportError, Exception) as e:
            print(f"  ⚠️ vLLM load failed ({e}), falling back to transformers")
            self.use_vllm = False
            self._load_transformers()

    def _load_transformers(self):
        """Transformers 기반 로드 - Qwen3-VL / Qwen2.5-VL 모두 지원"""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        quant = self.cfg['mllm'].get('quantization')

        print(f"  Model: {self.model_name}")
        print(f"  Quantization: {quant or 'none (BF16)'}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Determine dtype
        if quant in ('awq', 'gptq'):
            dtype = "auto"  # let model config decide
        else:
            dtype = torch.bfloat16

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  ⚠️ Load failed: {e}")
            print(f"\n  === TROUBLESHOOTING ===")
            print(f"  Option A: Use rule-based extraction: --step extract_rule")
            print(f"  Option B: Use smaller model (Qwen3-VL-8B-Instruct)")
            print(f"  Option C: Check VRAM with nvidia-smi")
            raise

    def extract_semantics(self, guidance, image_paths):
        """
        단일 샘플의 semantic 추출

        Args:
            guidance: 사용자 지시문
            image_paths: RGB 이미지 경로 리스트 (1~3장)

        Returns:
            dict: {object_category, intention, contact_parts, grasp_direction, normalized_command}
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(guidance=guidance)

        if self.use_vllm:
            return self._extract_vllm(prompt, image_paths)
        else:
            return self._extract_transformers(prompt, image_paths)

    def _extract_vllm(self, prompt, image_paths):
        """vLLM 단일 추출 (fallback용)"""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Check model initialization.")
        results = self._extract_vllm_batch([(prompt, image_paths)])
        return results[0]

    def _extract_vllm_batch(self, prompt_image_pairs):
        """vLLM 배치 추출 - 여러 (prompt, image_paths)를 한번에 처리"""
        if self.processor is None:
            raise RuntimeError("Processor not loaded. Check model initialization.")
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        requests = []
        valid_indices = []

        for idx, (prompt, image_paths) in enumerate(prompt_image_pairs):
            try:
                images = []
                for p in image_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        images.append(img)
                    except:
                        continue

                if not images:
                    continue

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)

                requests.append({
                    "prompt": text,
                    "multi_modal_data": {"image": image_inputs},
                })
                valid_indices.append(idx)
            except Exception as e:
                print(f"    배치 준비 실패 [{idx}]: {e}")
                continue

        # 결과 배열 초기화 (default로)
        results = [self._default_semantic() for _ in prompt_image_pairs]

        if not requests:
            return results

        try:
            outputs = self.model.generate(requests, sampling_params=self.sampling_params, use_tqdm=False)
            for out_idx, vi in enumerate(valid_indices):
                try:
                    response = outputs[out_idx].outputs[0].text
                    results[vi] = self._parse_response(response)
                except Exception as e:
                    print(f"    응답 파싱 실패 [{vi}]: {e}")
        except Exception as e:
            print(f"    vLLM 배치 추출 실패: {e}")

        return results

    def extract_scene_batch(self, guidances, image_paths, max_per_call=30):
        """같은 scene의 모든 guidance를 한번에 처리 (1 MLLM 호출 per scene)"""
        if not self.use_vllm or self.processor is None:
            # fallback: 개별 처리
            results = []
            for g in guidances:
                prompt = EXTRACTION_PROMPT_TEMPLATE.format(guidance=g)
                try:
                    r = self.extract_semantics(g, image_paths)
                except:
                    r = self._default_semantic()
                results.append(r)
            return results

        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        all_results = [self._default_semantic() for _ in guidances]

        # 이미지 로드 (한번만)
        images = []
        for p in image_paths:
            try:
                images.append(PILImage.open(p).convert("RGB"))
            except:
                continue
        if not images:
            return all_results

        # max_per_call 단위로 나눠서 호출
        for chunk_start in range(0, len(guidances), max_per_call):
            chunk = guidances[chunk_start:chunk_start + max_per_call]

            # 프롬프트 구성
            instruction_list = "\n".join(
                f"**Instruction {i}:** \"{g}\"" for i, g in enumerate(chunk)
            )
            prompt = BATCH_EXTRACTION_PROMPT.format(
                instruction_list=instruction_list,
                num_instructions=len(chunk),
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in images],
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(messages)

                # max_tokens 늘리기 (배치 출력이 길어짐)
                from vllm import SamplingParams
                batch_params = SamplingParams(
                    temperature=self.sampling_params.temperature,
                    max_tokens=min(4096, self.sampling_params.max_tokens * 2),
                )

                outputs = self.model.generate(
                    {"prompt": text, "multi_modal_data": {"image": image_inputs}},
                    sampling_params=batch_params,
                    use_tqdm=False,
                )

                response = outputs[0].outputs[0].text
                parsed = self._parse_batch_response(response, len(chunk))

                for i, p in enumerate(parsed):
                    all_results[chunk_start + i] = p

            except Exception as e:
                print(f"    scene 배치 추출 실패: {e}")

        return all_results

    def _parse_batch_response(self, response, expected_count):
        """배치 JSON 배열 응답 파싱 (짧은 키 + 긴 키 모두 지원)"""
        import re
        results = []

        # JSON 배열 추출
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                entries = json.loads(json_match.group())
                if isinstance(entries, list):
                    for entry in entries:
                        results.append({
                            'object_category': entry.get('cat', entry.get('object_category', 'unknown')),
                            'intention': entry.get('intent', entry.get('intention', 'unknown')),
                            'contact_parts': entry.get('parts', entry.get('contact_parts', 'unknown')),
                            'grasp_direction': entry.get('dir', entry.get('grasp_direction', 'unknown')),
                            'normalized_command': entry.get('cmd', entry.get('normalized_command', '')),
                        })
            except json.JSONDecodeError:
                pass

        # 부족한 만큼 default로 채우기
        while len(results) < expected_count:
            results.append(self._default_semantic())

        return results[:expected_count]

    def _extract_transformers(self, prompt, image_paths):
        """Transformers 추출"""
        import torch
        from PIL import Image

        images = []
        image_tokens = ""
        for i, p in enumerate(image_paths):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                image_tokens += f"<image>\n"
            except:
                continue

        if not images:
            return self._default_semantic()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg['mllm']['max_tokens'],
                    temperature=self.cfg['mllm']['temperature'],
                    do_sample=True,
                )

            response = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return self._parse_response(response)

        except Exception as e:
            print(f"    Transformers 추출 실패: {e}")
            return self._default_semantic()

    def _parse_response(self, response):
        """MLLM 응답에서 JSON 추출"""
        try:
            # JSON 블록 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                # 직접 JSON 파싱 시도
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    return self._default_semantic()

            # 필수 키 검증
            result = {
                "object_category": data.get("object_category", "unknown"),
                "intention": data.get("intention", "unknown"),
                "contact_parts": data.get("contact_parts", "unknown"),
                "grasp_direction": data.get("grasp_direction", "unknown"),
                "normalized_command": data.get("normalized_command", ""),
                "reasoning": data.get("reasoning", ""),
                "raw_response": response[:500],
            }

            return result

        except (json.JSONDecodeError, Exception) as e:
            print(f"    JSON 파싱 실패: {e}")
            return self._default_semantic()

    def _default_semantic(self):
        return {
            "object_category": "unknown",
            "intention": "unknown",
            "contact_parts": "unknown",
            "grasp_direction": "unknown",
            "normalized_command": "",
            "reasoning": "",
            "raw_response": "",
        }


def extract_all_semantics(cfg, max_samples=None, use_vllm=True):
    """모든 샘플에 대해 MLLM semantic 추출"""
    print("=" * 60)
    print("[G1] MLLM Semantic Cue 추출")
    print("=" * 60)

    semantics_dir = Path(cfg['paths']['semantics'])
    semantics_dir.mkdir(parents=True, exist_ok=True)

    renders_dir = Path(cfg['paths']['renders'])
    scenes_dir = Path(cfg['paths']['scenes'])
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"

    # MLLM 초기화
    extractor = MLLMExtractor(cfg, use_vllm=use_vllm)
    extractor.load_model()

    # obj_id → scene_id 인덱스 미리 구축 (O(n) 한 번만)
    print("  scene 인덱스 구축 중...")
    obj_to_scene = {}
    for scene_dir in scenes_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        job_file = scene_dir / "job.json"
        if job_file.exists():
            try:
                with open(job_file) as f:
                    job = json.load(f)
                obj_to_scene[job.get('obj_id', '')] = job['scene_id']
            except:
                continue
    print(f"  scene 인덱스: {len(obj_to_scene)}개")

    # 메타 로드 (scene별로 처리)
    scene_to_samples = defaultdict(list)
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                obj_id = entry.get('obj_id', '')
                scene_id = obj_to_scene.get(obj_id)
                if scene_id:
                    scene_to_samples[scene_id].append(entry)

    import time

    total_scenes = len(scene_to_samples)
    total_samples = sum(len(v) for v in scene_to_samples.values())

    num_views = cfg['mllm']['num_rgb_views']  # 대표 뷰 수
    processed = 0
    skipped_scenes = 0
    skipped_samples = 0
    new_scenes = 0
    new_samples = 0
    unknown_count = 0
    failed_scenes = 0

    # 먼저 이미 처리된 것 카운트
    for scene_id, samples in scene_to_samples.items():
        output_path = semantics_dir / f"{scene_id}.jsonl"
        if output_path.exists():
            skipped_scenes += 1
            skipped_samples += len(samples)

    remaining_scenes = total_scenes - skipped_scenes
    remaining_samples = total_samples - skipped_samples

    print(f"  전체: {total_scenes} scenes, {total_samples} 샘플")
    print(f"  이미 완료: {skipped_scenes} scenes, {skipped_samples} 샘플 ({skipped_samples/max(total_samples,1)*100:.1f}%)")
    print(f"  남은 처리: {remaining_scenes} scenes, {remaining_samples} 샘플")
    print()

    if remaining_scenes == 0:
        print("  모든 scene이 이미 처리되었습니다!")
        processed = skipped_samples
    else:
        start_time = time.time()
        scene_list = list(scene_to_samples.items())
        log_interval = max(1, remaining_scenes // 50)  # ~50번 로그 출력

        for scene_idx, (scene_id, samples) in enumerate(scene_list):
            if max_samples and (skipped_samples + new_samples) >= max_samples:
                break

            # 이미 처리된 scene 스킵
            output_path = semantics_dir / f"{scene_id}.jsonl"
            if output_path.exists():
                processed += len(samples)
                continue

            # 렌더링된 RGB 이미지 선택 (대표 1~3장)
            render_dir = renders_dir / scene_id
            rgb_files = sorted(render_dir.glob("rgb_cam*.png"))[:num_views]

            if not rgb_files:
                failed_scenes += 1
                continue

            image_paths = [str(f) for f in rgb_files]

            # 같은 scene = 같은 이미지 → 1 MLLM 호출로 전체 처리
            valid_samples = [s for s in samples if s.get('guidance', '')]
            guidances = [s['guidance'] for s in valid_samples]

            # scene당 1회 호출 (내부에서 30개씩 chunk)
            batch_results = extractor.extract_scene_batch(guidances, image_paths, max_per_call=10)

            scene_semantics = []
            for i, sample in enumerate(valid_samples):
                semantic = batch_results[i]
                semantic['sample_id'] = sample['sample_id']
                semantic['scene_id'] = scene_id
                semantic['guidance_original'] = sample['guidance']

                if semantic['contact_parts'] == 'unknown':
                    unknown_count += 1

                scene_semantics.append(semantic)
                processed += 1
                new_samples += 1

            # Scene별 저장
            if scene_semantics:
                with jsonlines.open(output_path, mode='w') as writer:
                    for s in scene_semantics:
                        writer.write(s)
                new_scenes += 1

            # 진행률 출력 (일정 간격 또는 처음 10개)
            if new_scenes <= 10 or new_scenes % log_interval == 0:
                done_total = skipped_samples + new_samples
                elapsed = time.time() - start_time
                avg_per_scene = elapsed / new_scenes
                eta_sec = avg_per_scene * (remaining_scenes - new_scenes)
                if eta_sec > 3600:
                    eta_str = f"{eta_sec/3600:.1f}h"
                elif eta_sec > 60:
                    eta_str = f"{eta_sec/60:.0f}m"
                else:
                    eta_str = f"{eta_sec:.0f}s"
                pct = done_total / total_samples * 100
                print(
                    f"  [{new_scenes}/{remaining_scenes} scenes] "
                    f"[{done_total}/{total_samples} samples ({pct:.1f}%)] "
                    f"{avg_per_scene:.1f}s/scene, ETA {eta_str}",
                    flush=True,
                )

        elapsed_total = time.time() - start_time
        elapsed_str = f"{elapsed_total/3600:.1f}h" if elapsed_total > 3600 else f"{elapsed_total/60:.1f}m"

        print(f"\n  === G1 결과 요약 ===")
        print(f"  새로 처리: {new_scenes} scenes, {new_samples} 샘플 ({elapsed_str})")
        if new_scenes > 0:
            print(f"  평균 속도: {elapsed_total/new_scenes:.1f}s/scene, {elapsed_total/max(new_samples,1):.2f}s/sample")
        print(f"  이전 완료: {skipped_scenes} scenes, {skipped_samples} 샘플")
        print(f"  전체 완료: {skipped_scenes + new_scenes}/{total_scenes} scenes ({(skipped_scenes+new_scenes)/total_scenes*100:.1f}%)")
        if failed_scenes > 0:
            print(f"  렌더 없음(스킵): {failed_scenes} scenes")

    total_processed = skipped_samples + new_samples
    print(f"  Unknown contact_parts: {unknown_count} ({unknown_count/max(new_samples,1)*100:.1f}% of new)")

    if new_samples > 0 and unknown_count / new_samples > 0.3:
        print("  ⚠️ Unknown 비율이 높습니다! 프롬프트나 이미지 품질 확인 필요")


# ============================================================
# G1-offline: MLLM 없이 규칙 기반 추출 (fallback)
# ============================================================
def extract_semantics_rule_based(cfg):
    """
    MLLM 없이 규칙 기반으로 semantic 추출 (테스트/디버깅용)
    DexGYS의 guidance 텍스트에서 키워드 기반 추출
    """
    print("\n" + "=" * 60)
    print("[G1-Fallback] 규칙 기반 Semantic 추출")
    print("=" * 60)

    semantics_dir = Path(cfg['paths']['semantics'])
    semantics_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"

    # 키워드 매핑
    intention_keywords = {
        "use": ["use", "utilize", "operate", "employ"],
        "pick_up": ["pick up", "grab", "take", "lift", "hold"],
        "hand_over": ["hand over", "pass", "give", "deliver"],
        "pour": ["pour", "dispense", "empty"],
        "open": ["open", "unscrew", "twist open"],
        "close": ["close", "shut", "seal"],
        "cut": ["cut", "slice", "chop"],
        "press": ["press", "push", "squeeze"],
    }

    part_keywords = {
        "handle": ["handle", "grip", "knob"],
        "body": ["body", "main", "barrel", "shaft"],
        "lid": ["lid", "cap", "cover", "top"],
        "rim": ["rim", "edge", "brim", "lip"],
        "base": ["base", "bottom", "foot"],
        "neck": ["neck", "spout", "nozzle"],
        "blade": ["blade", "cutting edge"],
        "trigger": ["trigger", "button", "switch"],
    }

    direction_keywords = {
        "from_right": ["from the right", "right side", "from right"],
        "from_left": ["from the left", "left side", "from left"],
        "from_above": ["from above", "from top", "top down", "from the top"],
        "from_front": ["from the front", "from front", "facing"],
        "from_behind": ["from behind", "from the back", "from back"],
        "from_side": ["from the side", "lateral", "sideways"],
    }

    processed = 0
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            all_entries = list(reader)

        semantics = []
        for entry in tqdm(all_entries, desc=f"  {meta_file.stem}"):
            guidance = entry.get('guidance', '').lower()

            # 의도 추출
            intention = "unknown"
            for intent, keywords in intention_keywords.items():
                if any(kw in guidance for kw in keywords):
                    intention = intent
                    break

            # 접촉 부위 추출
            contact = "unknown"
            for part, keywords in part_keywords.items():
                if any(kw in guidance for kw in keywords):
                    contact = part
                    break

            # 방향 추출
            direction = "unknown"
            for dirn, keywords in direction_keywords.items():
                if any(kw in guidance for kw in keywords):
                    direction = dirn
                    break

            semantic = {
                "sample_id": entry['sample_id'],
                "scene_id": f"scene_{entry.get('obj_id', 'unknown')}",
                "guidance_original": entry.get('guidance', ''),
                "object_category": "inferred",
                "intention": intention,
                "contact_parts": contact,
                "grasp_direction": direction,
                "normalized_command": guidance.strip(),
                "reasoning": "rule-based extraction",
            }
            semantics.append(semantic)
            processed += 1

        # 저장 (obj_id별로)
        obj_groups = defaultdict(list)
        for s in semantics:
            obj_groups[s['scene_id']].append(s)

        for scene_id, scene_semantics in obj_groups.items():
            output_path = semantics_dir / f"{scene_id}.jsonl"
            with jsonlines.open(output_path, mode='w') as writer:
                for s in scene_semantics:
                    writer.write(s)

    print(f"  규칙 기반 추출 완료: {processed} 샘플")


# ============================================================
# G2: Semantic Group 생성
# ============================================================
def generate_semantic_groups(cfg):
    """
    Semantic 속성이 같은 grasps를 그룹으로 묶기

    Group key = (scene/obj, intention, contact_parts, grasp_direction)
    """
    print("\n" + "=" * 60)
    print("[G2] Semantic Group 생성")
    print("=" * 60)

    semantics_dir = Path(cfg['paths']['semantics'])

    # 모든 semantic 로드
    all_semantics = []
    for sem_file in semantics_dir.glob("*.jsonl"):
        with jsonlines.open(sem_file) as reader:
            for entry in reader:
                all_semantics.append(entry)

    print(f"  총 semantic entries: {len(all_semantics)}")

    # 그룹핑
    groups = defaultdict(list)
    for entry in all_semantics:
        # contact_parts가 리스트일 수 있으므로 문자열로 변환
        cp = entry.get('contact_parts', 'unknown')
        if isinstance(cp, list):
            cp = ", ".join(str(x) for x in cp)
        gd = entry.get('grasp_direction', 'unknown')
        if isinstance(gd, list):
            gd = ", ".join(str(x) for x in gd)

        group_key = (
            entry.get('scene_id', 'unknown'),
            str(entry.get('intention', 'unknown')),
            str(cp),
            str(gd),
        )
        groups[group_key].append(entry['sample_id'])

    # 그룹 통계
    group_sizes = [len(v) for v in groups.values()]
    print(f"  그룹 수: {len(groups)}")
    if group_sizes:
        print(f"  그룹 크기: mean={np.mean(group_sizes):.1f}, min={min(group_sizes)}, max={max(group_sizes)}")
    else:
        print("  ⚠️ semantic entries가 없습니다. G1(MLLM 분석)이 먼저 실행되어야 합니다.")
        print("    확인사항:")
        print("    1. Phase F 렌더링 결과가 renders/ 폴더에 있는지")
        print("    2. G1 실행 시 semantics/ 폴더에 JSON이 생성되었는지")
        return

    # 너무 작은 그룹 통합 규칙
    MIN_GROUP_SIZE = 2
    small_groups = {k: v for k, v in groups.items() if len(v) < MIN_GROUP_SIZE}
    if small_groups:
        print(f"  ⚠️ 크기 < {MIN_GROUP_SIZE} 그룹: {len(small_groups)}개")
        print(f"    → 상위 그룹(scene+intention만)으로 통합 시도")

        # 통합: (scene, intention)만으로 re-group
        merged_groups = defaultdict(list)
        for key, sample_ids in groups.items():
            if len(sample_ids) < MIN_GROUP_SIZE:
                # 상위 키로 통합
                merged_key = (key[0], key[1], "merged", "merged")
                merged_groups[merged_key].extend(sample_ids)
            else:
                merged_groups[key] = sample_ids

        groups = merged_groups
        group_sizes = [len(v) for v in groups.values()]
        print(f"  통합 후 그룹 수: {len(groups)}")
        print(f"  통합 후 크기: mean={np.mean(group_sizes):.1f}, min={min(group_sizes)}, max={max(group_sizes)}")

    # 저장
    semantic_groups = {}
    for idx, (key, sample_ids) in enumerate(groups.items()):
        group_id = f"group_{idx:05d}"
        semantic_groups[group_id] = {
            "group_id": group_id,
            "scene_id": key[0],
            "intention": key[1],
            "contact_parts": key[2],
            "grasp_direction": key[3],
            "sample_ids": sample_ids,
            "num_grasps": len(sample_ids),
        }

    output_path = Path(cfg['paths']['processed']) / "semantic_groups.json"
    with open(output_path, 'w') as f:
        json.dump(semantic_groups, f, indent=2)

    print(f"  Semantic groups 저장: {output_path}")
    return semantic_groups


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase G: MLLM Semantic 추출 + Group 생성")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "extract", "extract_rule", "group"],
                       default="all")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--no_vllm", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.no_vllm:
        args.use_vllm = False

    if args.step in ["all", "extract"]:
        extract_all_semantics(cfg, max_samples=args.max_samples, use_vllm=args.use_vllm)

    if args.step == "extract_rule":
        extract_semantics_rule_based(cfg)

    if args.step in ["all", "group"]:
        generate_semantic_groups(cfg)

    print("\n" + "=" * 60)
    print("Phase G 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
