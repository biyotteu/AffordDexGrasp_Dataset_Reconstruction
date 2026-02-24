#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파이프라인 공통 유틸리티
- Phase 간 전제조건 검증
- Subprocess 실행 및 로깅 헬퍼
- 공통 설정 로드
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import yaml


# ============================================================
# Config 로드
# ============================================================
def load_config(config_path="configs/pipeline_config.yaml"):
    """파이프라인 설정 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# Phase 전제조건 검증
# ============================================================
class PhaseValidator:
    """각 Phase의 전제조건(이전 Phase 출력물)을 검증"""

    def __init__(self, cfg):
        self.cfg = cfg

    def validate_phase_b(self):
        """Phase B 전제조건: Phase A 출력물 확인"""
        issues = []
        dexgys_dir = Path(self.cfg['paths']['dexgys'])
        oakink_dir = Path(self.cfg['paths']['oakink'])

        if not dexgys_dir.exists():
            issues.append(f"DexGYS 데이터 없음: {dexgys_dir}")
        if not oakink_dir.exists():
            issues.append(f"OakInk 데이터 없음: {oakink_dir}")

        return issues

    def validate_phase_c(self):
        """Phase C 전제조건: Phase B 출력물 확인"""
        issues = []
        meta_dir = Path(self.cfg['paths']['processed']) / "dexgys_meta"

        if not meta_dir.exists():
            issues.append(f"메타 데이터 없음: {meta_dir}")
        else:
            meta_files = list(meta_dir.glob("*_meta.jsonl"))
            if not meta_files:
                issues.append("메타 JSONL 파일 없음")

        index_path = Path(self.cfg['paths']['processed']) / "obj_mesh_index.json"
        if not index_path.exists():
            issues.append(f"Mesh index 없음: {index_path}")

        return issues

    def validate_phase_d(self):
        """Phase D 전제조건: Phase B + C 출력물 확인"""
        issues = self.validate_phase_c()

        splits_dir = Path(self.cfg['paths']['splits'])
        if not splits_dir.exists() or not list(splits_dir.glob("*.json")):
            issues.append(f"Split 데이터 없음: {splits_dir}")

        return issues

    def validate_phase_e(self):
        """Phase E 전제조건: mesh index 확인"""
        issues = []
        index_path = Path(self.cfg['paths']['processed']) / "obj_mesh_index.json"
        if not index_path.exists():
            issues.append(f"Mesh index 없음: {index_path}")
        return issues

    def validate_phase_f(self):
        """Phase F 전제조건: Phase D scene 출력물 확인"""
        issues = []
        scenes_dir = Path(self.cfg['paths']['scenes'])

        if not scenes_dir.exists():
            issues.append(f"Scene 디렉토리 없음: {scenes_dir}")
        else:
            scene_dirs = [d for d in scenes_dir.iterdir()
                         if d.is_dir() and (d / "job.json").exists()]
            if not scene_dirs:
                issues.append("Scene job 파일 없음")

        return issues

    def validate_phase_g(self):
        """Phase G 전제조건: Phase F 렌더링 출력물 확인"""
        issues = []
        renders_dir = Path(self.cfg['paths']['renders'])

        if not renders_dir.exists():
            issues.append(f"렌더링 디렉토리 없음: {renders_dir}")
        else:
            render_dirs = [d for d in renders_dir.iterdir()
                          if d.is_dir() and (d / "rgb_cam0.png").exists()]
            if not render_dirs:
                issues.append("렌더링된 scene 없음")

        return issues

    def validate_phase_h(self):
        """Phase H 전제조건: 메타 + 메쉬 + semantic groups 확인"""
        issues = []

        # 메타 데이터
        meta_dir = Path(self.cfg['paths']['processed']) / "dexgys_meta"
        if not meta_dir.exists():
            issues.append(f"메타 데이터 없음: {meta_dir}")

        # Mesh index
        index_path = Path(self.cfg['paths']['processed']) / "obj_mesh_index.json"
        if not index_path.exists():
            issues.append(f"Mesh index 없음: {index_path}")

        # ShadowHand MJCF (DexGraspNet 우선)
        dexgraspnet_dir = Path(self.cfg['shadow_hand'].get('dexgraspnet_mjcf', ''))
        mjcf_dir = Path(self.cfg['paths']['mjcf'])
        if not dexgraspnet_dir.exists() and not mjcf_dir.exists():
            issues.append("ShadowHand MJCF 모델 없음 (DexGraspNet 또는 menagerie)")

        # Semantic groups (선택적이지만 권장)
        sem_dir = Path(self.cfg['paths']['semantics'])
        groups_file = sem_dir / "semantic_groups.json"
        if not groups_file.exists():
            issues.append(f"Semantic groups 없음 (Phase G 먼저 실행 권장): {groups_file}")

        return issues

    def validate_phase_i(self):
        """Phase I 전제조건: 주요 출력물 확인"""
        issues = []

        # 핵심 데이터
        for check_name, check_path in [
            ("메타 데이터", Path(self.cfg['paths']['processed']) / "dexgys_meta"),
            ("Splits", Path(self.cfg['paths']['splits'])),
        ]:
            if not Path(check_path).exists():
                issues.append(f"{check_name} 없음: {check_path}")

        return issues

    def validate(self, phase):
        """지정된 Phase의 전제조건 검증"""
        validators = {
            'b': self.validate_phase_b,
            'c': self.validate_phase_c,
            'd': self.validate_phase_d,
            'e': self.validate_phase_e,
            'f': self.validate_phase_f,
            'g': self.validate_phase_g,
            'h': self.validate_phase_h,
            'i': self.validate_phase_i,
        }

        validator = validators.get(phase.lower())
        if validator is None:
            return []  # Phase A는 전제조건 없음

        issues = validator()

        if issues:
            print(f"\n  ⚠️ Phase {phase.upper()} 전제조건 미충족:")
            for i, issue in enumerate(issues, 1):
                print(f"    {i}. {issue}")
            print()

        return issues


# ============================================================
# Subprocess 실행 헬퍼
# ============================================================
def run_subprocess(cmd, timeout=300, log_dir=None, task_name=None):
    """
    Subprocess를 실행하고 결과를 처리하는 헬퍼.

    Args:
        cmd: 실행할 명령어 리스트
        timeout: 타임아웃 (초)
        log_dir: 실패 로그 저장 디렉토리 (None이면 로그 저장 안 함)
        task_name: 작업 식별자 (로그 파일명에 사용)

    Returns:
        dict: {
            "success": bool,
            "returncode": int,
            "stdout": str,
            "stderr": str,
            "error": str or None,
        }
    """
    result_info = {
        "success": False,
        "returncode": -1,
        "stdout": "",
        "stderr": "",
        "error": None,
    }

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        result_info["returncode"] = result.returncode
        result_info["stdout"] = result.stdout or ""
        result_info["stderr"] = result.stderr or ""
        result_info["success"] = (result.returncode == 0)

        if result.returncode != 0:
            # stderr에서 실제 에러 추출
            stderr_lines = result.stderr.strip().split('\n') if result.stderr else []
            tb_start = -1
            for li, line in enumerate(stderr_lines):
                if 'Traceback' in line:
                    tb_start = li
            if tb_start >= 0:
                result_info["error"] = '\n'.join(stderr_lines[tb_start:])
            else:
                result_info["error"] = '\n'.join(stderr_lines[-10:])

    except subprocess.TimeoutExpired:
        result_info["error"] = f"타임아웃 ({timeout}초)"

    except FileNotFoundError as e:
        result_info["error"] = f"명령어를 찾을 수 없음: {e}"

    except Exception as e:
        result_info["error"] = f"예외 발생: {e}"

    # 실패 로그 저장
    if not result_info["success"] and log_dir and task_name:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{task_name}.log"
        with open(log_file, 'w', encoding='utf-8') as lf:
            lf.write(f"=== {task_name} ===\n")
            lf.write(f"시간: {datetime.now().isoformat()}\n")
            lf.write(f"명령어: {' '.join(str(c) for c in cmd)}\n")
            lf.write(f"종료 코드: {result_info['returncode']}\n\n")
            lf.write(f"=== STDOUT ===\n{result_info['stdout']}\n\n")
            lf.write(f"=== STDERR ===\n{result_info['stderr']}\n")

    return result_info


def run_blenderproc(worker_script, args_dict, timeout=300, log_dir=None, task_name=None):
    """
    BlenderProc worker를 실행하는 헬퍼.

    Args:
        worker_script: worker 스크립트 경로
        args_dict: {"--arg_name": "value", ...} 형태의 인자 딕셔너리
        timeout: 타임아웃 (초)
        log_dir: 실패 로그 저장 디렉토리
        task_name: 작업 식별자

    Returns:
        run_subprocess 결과 딕셔너리
    """
    cmd = ["blenderproc", "run", str(worker_script)]
    for key, value in args_dict.items():
        cmd.extend([key, str(value)])

    return run_subprocess(cmd, timeout=timeout, log_dir=log_dir, task_name=task_name)
