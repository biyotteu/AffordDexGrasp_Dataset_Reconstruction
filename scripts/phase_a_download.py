#!/usr/bin/env python3
"""
Phase A: 원천 데이터 확보 & 로더 정합
- A1: DexGYS 라벨 다운로드/정합
- A2: OakInk 메쉬 + metaV2 정합
- A3: ShadowHand MJCF 준비
"""

import os
import sys
import json
import zipfile
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import yaml
import numpy as np

# ============================================================
# Configuration
# ============================================================
def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# A1: DexGYS 다운로드 및 정합
# ============================================================
def download_dexgys(cfg):
    """HuggingFace에서 DexGYS 데이터셋 다운로드"""
    dexgys_dir = Path(cfg['paths']['dexgys'])
    dexgys_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[A1] DexGYS 데이터셋 다운로드")
    print("=" * 60)

    # huggingface_hub를 사용하여 다운로드
    from huggingface_hub import hf_hub_download, snapshot_download

    repo_id = cfg['dexgys']['hf_repo']

    # 전체 스냅샷 다운로드
    print(f"  HuggingFace repo: {repo_id}")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dexgys_dir),
    )
    print(f"  다운로드 완료: {local_dir}")

    # 파일 확인
    train_file = dexgys_dir / cfg['dexgys']['train_file']
    test_file = dexgys_dir / cfg['dexgys']['test_file']

    if not train_file.exists():
        # 다른 이름으로 존재할 수 있음 - 검색
        json_files = list(dexgys_dir.rglob("*.json"))
        print(f"  발견된 JSON 파일들: {[str(f) for f in json_files]}")
        # 가장 적합한 파일 매핑
        for f in json_files:
            if 'train' in f.name.lower():
                shutil.copy(f, train_file)
                print(f"  Train 파일 매핑: {f.name} → {train_file.name}")
            elif 'test' in f.name.lower():
                shutil.copy(f, test_file)
                print(f"  Test 파일 매핑: {f.name} → {test_file.name}")

    return validate_dexgys(cfg)


def validate_dexgys(cfg):
    """DexGYS 데이터 검증: dex_grasp 28차원 분해 확인"""
    dexgys_dir = Path(cfg['paths']['dexgys'])
    stats = {"train": {}, "test": {}}

    for split_name, filename in [("train", cfg['dexgys']['train_file']),
                                  ("test", cfg['dexgys']['test_file'])]:
        filepath = dexgys_dir / filename
        if not filepath.exists():
            # 대안 파일명 시도
            alt_files = list(dexgys_dir.glob(f"*{split_name}*.json"))
            if alt_files:
                filepath = alt_files[0]
                print(f"  대안 파일 사용: {filepath}")
            else:
                print(f"  ⚠️ {split_name} 파일 없음: {filepath}")
                continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        # 데이터 구조 분석
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            # 키 구조 확인
            print(f"  {split_name} 최상위 키: {list(data.keys())[:10]}")
            if 'data' in data:
                samples = data['data']
            else:
                samples = list(data.values()) if not isinstance(list(data.values())[0], (str, int, float)) else []
        else:
            samples = []

        if len(samples) > 0:
            sample = samples[0] if isinstance(samples, list) else samples
            print(f"\n  [{split_name}] 샘플 수: {len(samples) if isinstance(samples, list) else 'dict형'}")
            print(f"  [{split_name}] 첫 번째 샘플 키: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")

            # dex_grasp 28차원 검증
            if isinstance(sample, dict):
                for key in ['dex_grasp', 'grasp', 'hand_pose', 'grasp_pose']:
                    if key in sample:
                        grasp = np.array(sample[key])
                        print(f"  [{split_name}] {key} shape: {grasp.shape}")
                        if grasp.shape[-1] == 28 or (len(grasp.shape) > 1 and grasp.shape[-1] == 28):
                            t = grasp[..., :3]
                            r = grasp[..., 3:6]
                            q = grasp[..., 6:28]
                            print(f"    ✅ t(3): {t.shape}, axis-angle(3): {r.shape}, qpos(22): {q.shape}")
                        break

                # guidance / language 확인
                for key in ['guidance', 'language', 'text', 'instruction', 'command']:
                    if key in sample:
                        print(f"  [{split_name}] {key} 예시: '{sample[key][:100] if isinstance(sample[key], str) else sample[key]}'")
                        break

                # category / object ID 확인
                for key in ['cate_id', 'category', 'cat_id', 'obj_id', 'object_id']:
                    if key in sample:
                        print(f"  [{split_name}] {key}: {sample[key]}")

        stats[split_name] = {
            "file": str(filepath),
            "num_samples": len(samples) if isinstance(samples, list) else -1,
        }

    return stats


# ============================================================
# A2: OakInk 메쉬 + metaV2 정합
# ============================================================
def download_oakink(cfg):
    """OakInk shape 데이터 다운로드 (correct repo: oakink/OakInk-v1)"""
    oakink_dir = Path(cfg['paths']['oakink'])
    oakink_dir.mkdir(parents=True, exist_ok=True)
    shape_dir = oakink_dir / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[A2] OakInk mesh + metaV2 download")
    print("=" * 60)

    from huggingface_hub import snapshot_download, hf_hub_download

    # Correct HuggingFace repo ID: oakink/OakInk-v1
    REPO_ID = "oakink/OakInk-v1"
    SHAPE_FILES = [
        "shape/metaV2.zip",
        "shape/OakInkObjectsV2.zip",
        "shape/OakInkVirtualObjectsV2.zip",
        "shape/oakink_shape_v2.zip",
    ]

    print(f"  HuggingFace repo: {REPO_ID}")
    print(f"  Target: {shape_dir}")

    for remote_file in SHAPE_FILES:
        local_name = Path(remote_file).name
        local_path = shape_dir / local_name

        if local_path.exists():
            print(f"  [skip] {local_name} already exists")
            continue

        print(f"  Downloading {local_name} ...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_file,
                local_dir=str(oakink_dir),
            )
            # hf_hub_download preserves subfolder structure
            actual_path = oakink_dir / remote_file
            if actual_path.exists() and actual_path != local_path:
                shutil.move(str(actual_path), str(local_path))
            print(f"  OK: {local_name}")
        except Exception as e:
            print(f"  FAILED: {local_name} - {e}")

    # Unzip
    for zip_name in ["metaV2.zip", "OakInkObjectsV2.zip",
                      "OakInkVirtualObjectsV2.zip", "oakink_shape_v2.zip"]:
        zip_path = shape_dir / zip_name
        if zip_path.exists():
            extract_dir = shape_dir / zip_name.replace('.zip', '')
            if not extract_dir.exists():
                print(f"  Extracting: {zip_name} ...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(shape_dir)
                print(f"  OK: {zip_name}")
            else:
                print(f"  [skip] {zip_name} already extracted")
        else:
            print(f"  WARNING: {zip_name} not found at {zip_path}")

    return validate_oakink(cfg)


def validate_oakink(cfg):
    """OakInk 메쉬 구조 검증: obj_id → mesh 1:1 매핑 확인"""
    oakink_dir = Path(cfg['paths']['oakink']) / "shape"

    stats = {
        "metaV2_exists": False,
        "objects_v2_exists": False,
        "virtual_objects_v2_exists": False,
        "total_real_objects": 0,
        "total_virtual_objects": 0,
        "mesh_formats": set(),
    }

    # metaV2 확인
    meta_dir = oakink_dir / "metaV2"
    if meta_dir.exists():
        stats["metaV2_exists"] = True
        for json_file in meta_dir.glob("*.json"):
            print(f"  metaV2 파일: {json_file.name}")
            with open(json_file) as f:
                meta = json.load(f)
                print(f"    키 수: {len(meta)}")
                if len(meta) > 0:
                    first_key = list(meta.keys())[0]
                    print(f"    예시: {first_key} → {meta[first_key]}")

    # OakInkObjectsV2 확인
    obj_dir = oakink_dir / "OakInkObjectsV2"
    if obj_dir.exists():
        stats["objects_v2_exists"] = True
        obj_dirs = [d for d in obj_dir.iterdir() if d.is_dir()]
        stats["total_real_objects"] = len(obj_dirs)
        print(f"  Real objects: {len(obj_dirs)}")

        # align_ds 메쉬 확인
        if obj_dirs:
            sample_obj = obj_dirs[0]
            align_ds = sample_obj / "align_ds"
            if align_ds.exists():
                meshes = list(align_ds.glob("*"))
                for m in meshes[:3]:
                    stats["mesh_formats"].add(m.suffix)
                    print(f"    메쉬 예시: {m}")

    # VirtualObjectsV2 확인
    vobj_dir = oakink_dir / "OakInkVirtualObjectsV2"
    if vobj_dir.exists():
        stats["virtual_objects_v2_exists"] = True
        vobj_dirs = [d for d in vobj_dir.iterdir() if d.is_dir()]
        stats["total_virtual_objects"] = len(vobj_dirs)
        print(f"  Virtual objects: {len(vobj_dirs)}")

    stats["mesh_formats"] = list(stats["mesh_formats"])
    return stats


def build_obj_mesh_index(cfg):
    """obj_id -> mesh path index (core for loader alignment)

    OakInk metaV2 JSON structure:
      object_id.json:         {oakink_id: {name, class, attr, from, scale}}
      virtual_object_id.json: {oakink_id: {name, class, attr, from, scale}}

    Examples:
      object_id.json:         "O01000" -> {"name": "omo_cleaner", ...}
      virtual_object_id.json: "s10101" -> {"name": "mug_s101", ...}

    DexGYS obj_id uses OakInk IDs directly: "C90001", "o42123", "s10105", etc.

    Mapping chain:
      DexGYS obj_id (e.g. "s10101")
        -> OakInk metaV2 lookup -> name (e.g. "mug_s101")
        -> mesh at OakInkObjectsV2/{name}/align_ds/*.obj
           or OakInkVirtualObjectsV2/{name}/align_ds/*.obj
    """
    oakink_dir = Path(cfg['paths']['oakink']) / "shape"
    meta_dir = oakink_dir / "metaV2"

    obj_mesh_index = {}
    name_to_id = {}  # reverse lookup: folder_name -> oakink_id

    def find_mesh(obj_name, search_dirs):
        """Find mesh file for an object by name in given base directories."""
        for base_dir in search_dirs:
            for mesh_subdir in ["align_ds", "align", "."]:
                if mesh_subdir == ".":
                    mesh_dir = base_dir / obj_name
                else:
                    mesh_dir = base_dir / obj_name / mesh_subdir
                if not mesh_dir.exists():
                    continue
                mesh_files = list(mesh_dir.glob("*.obj")) + list(mesh_dir.glob("*.ply"))
                if mesh_files:
                    return str(mesh_files[0]), str(mesh_dir)
        return None, None

    # All possible mesh directories
    real_dirs = [oakink_dir / "OakInkObjectsV2"]
    virtual_dirs = [oakink_dir / "OakInkVirtualObjectsV2"]
    all_dirs = real_dirs + virtual_dirs

    # Strategy 1: Load metaV2 object_id.json
    # Format: {oakink_id: {name: obj_name, class: ..., ...}}
    obj_id_file = meta_dir / "object_id.json"
    if obj_id_file.exists():
        with open(obj_id_file) as f:
            obj_id_map = json.load(f)
        print(f"  object_id.json: {len(obj_id_map)} entries")

        for oakink_id, info in obj_id_map.items():
            obj_name = info['name']
            mesh_path, mesh_dir = find_mesh(obj_name, real_dirs)
            if mesh_path:
                obj_mesh_index[oakink_id] = {
                    "type": "real",
                    "name": obj_name,
                    "mesh_path": mesh_path,
                    "mesh_dir": mesh_dir,
                }
                name_to_id[obj_name] = oakink_id
    else:
        print(f"  WARNING: {obj_id_file} not found")

    # Strategy 2: Load metaV2 virtual_object_id.json
    vobj_id_file = meta_dir / "virtual_object_id.json"
    if vobj_id_file.exists():
        with open(vobj_id_file) as f:
            vobj_id_map = json.load(f)
        print(f"  virtual_object_id.json: {len(vobj_id_map)} entries")

        for oakink_id, info in vobj_id_map.items():
            obj_name = info['name']
            mesh_path, mesh_dir = find_mesh(obj_name, virtual_dirs)
            # Also try real dirs as fallback
            if not mesh_path:
                mesh_path, mesh_dir = find_mesh(obj_name, real_dirs)
            if mesh_path:
                obj_mesh_index[oakink_id] = {
                    "type": "virtual",
                    "name": obj_name,
                    "mesh_path": mesh_path,
                    "mesh_dir": mesh_dir,
                }
                name_to_id[obj_name] = oakink_id
    else:
        print(f"  WARNING: {vobj_id_file} not found")

    # Strategy 3: Scan folders not yet indexed (catches any missed objects)
    for base_dir in all_dirs:
        if not base_dir.exists():
            continue
        obj_type = "virtual" if "Virtual" in base_dir.name else "real"
        for obj_folder in base_dir.iterdir():
            if not obj_folder.is_dir():
                continue
            folder_name = obj_folder.name
            # Skip if already indexed by name
            if folder_name in name_to_id:
                continue
            mesh_path, mesh_dir = find_mesh(folder_name, [base_dir])
            if mesh_path:
                # Index by folder name as fallback key
                obj_mesh_index[folder_name] = {
                    "type": obj_type,
                    "name": folder_name,
                    "mesh_path": mesh_path,
                    "mesh_dir": mesh_dir,
                }

    # Cross-reference with DexGYS to report coverage
    dexgys_dir = Path(cfg['paths']['dexgys'])
    dexgys_obj_ids = set()
    for jf in dexgys_dir.glob("*.json"):
        try:
            with open(jf) as f:
                data = json.load(f)
            samples = data if isinstance(data, list) else data.get('data', [])
            if isinstance(samples, list):
                for s in samples:
                    if isinstance(s, dict) and 'obj_id' in s:
                        dexgys_obj_ids.add(str(s['obj_id']))
        except:
            pass

    matched = dexgys_obj_ids & set(obj_mesh_index.keys())
    unmatched = dexgys_obj_ids - set(obj_mesh_index.keys())

    # Save index
    proc_dir = Path(cfg['paths']['processed'])
    proc_dir.mkdir(parents=True, exist_ok=True)
    index_path = proc_dir / "obj_mesh_index.json"
    with open(index_path, 'w') as f:
        json.dump(obj_mesh_index, f, indent=2)

    print(f"\n  Mesh index: {len(obj_mesh_index)} objects -> {index_path}")
    print(f"  DexGYS unique obj_ids: {len(dexgys_obj_ids)}")
    print(f"  Matched: {len(matched)} ({len(matched)/max(len(dexgys_obj_ids),1)*100:.1f}%)")
    if unmatched:
        print(f"  Unmatched: {len(unmatched)} (samples: {list(unmatched)[:5]}...)")
    else:
        print(f"  All DexGYS obj_ids matched!")

    # Verify paths exist
    missing = [oid for oid, info in obj_mesh_index.items()
               if not Path(info['mesh_path']).exists()]
    if missing:
        print(f"  WARNING: {len(missing)} meshes not found on disk")
    else:
        print(f"  All mesh paths verified OK")

    return obj_mesh_index


# ============================================================
# A3: ShadowHand MJCF 준비
# ============================================================
def download_shadowhand(cfg):
    """ShadowHand MJCF download.

    The Grasp-as-You-Say repo does NOT include mjcf files.
    They must be downloaded from PKU mirror or MuJoCo Menagerie.

    Method 1: git clone mujoco_menagerie (Google DeepMind) - most reliable
    Method 2: Manual download from PKU mirror
    """
    mjcf_dir = Path(cfg['paths']['mjcf'])
    mjcf_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[A3] ShadowHand MJCF download")
    print("=" * 60)

    # Check if already downloaded
    existing_xml = list(mjcf_dir.rglob("*.xml"))
    if existing_xml:
        print(f"  Already have {len(existing_xml)} XML file(s), skipping download")
        return validate_shadowhand(cfg)

    # Method 1: MuJoCo Menagerie (Google DeepMind) - has shadow_hand MJCF + STL meshes
    print("  Method 1: git clone mujoco_menagerie (shadow_hand only)...")
    temp_clone = Path("_temp_menagerie")
    try:
        if temp_clone.exists():
            shutil.rmtree(temp_clone, ignore_errors=True)

        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            str(temp_clone)
        ], check=True, capture_output=True, text=True, timeout=120)

        subprocess.run(
            ["git", "sparse-checkout", "set", "shadow_hand"],
            cwd=str(temp_clone), check=True, capture_output=True, text=True
        )

        src = temp_clone / "shadow_hand"
        if src.exists() and any(src.rglob("*.xml")):
            shutil.copytree(src, mjcf_dir, dirs_exist_ok=True)
            print("  OK: ShadowHand from mujoco_menagerie")
            shutil.rmtree(temp_clone, ignore_errors=True)
            return validate_shadowhand(cfg)
        else:
            print("  shadow_hand folder not found after sparse checkout")
            shutil.rmtree(temp_clone, ignore_errors=True)
    except Exception as e:
        print(f"  sparse clone failed: {e}")
        shutil.rmtree(temp_clone, ignore_errors=True)

    # Method 1b: Full clone fallback
    print("  Method 1b: full clone of mujoco_menagerie...")
    try:
        if temp_clone.exists():
            shutil.rmtree(temp_clone, ignore_errors=True)

        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/google-deepmind/mujoco_menagerie.git",
            str(temp_clone)
        ], check=True, capture_output=True, text=True, timeout=300)

        src = temp_clone / "shadow_hand"
        if src.exists():
            shutil.copytree(src, mjcf_dir, dirs_exist_ok=True)
            print("  OK: ShadowHand copied from mujoco_menagerie")
        else:
            print("  shadow_hand folder not found")

        shutil.rmtree(temp_clone, ignore_errors=True)
    except Exception as e:
        print(f"  full clone failed: {e}")
        shutil.rmtree(temp_clone, ignore_errors=True)

    # Check if we got it
    if list(mjcf_dir.rglob("*.xml")):
        return validate_shadowhand(cfg)

    # Method 2: Manual download instructions
    print()
    print("  === AUTOMATIC DOWNLOAD FAILED ===")
    print("  Please download ShadowHand MJCF manually:")
    print()
    print("  Option A (recommended):")
    print("    git clone https://github.com/google-deepmind/mujoco_menagerie.git")
    print(f"    Then copy mujoco_menagerie/shadow_hand/* to {mjcf_dir}")
    print()
    print("  Option B (PKU mirror, as specified by Grasp-as-You-Say):")
    print(f"    URL: {cfg['shadow_hand']['mjcf_url']}")
    print(f"    Download 'mjcf' folder and place at: {mjcf_dir}")
    print()
    print("  After downloading, re-run: python scripts/phase_a_download.py --step shadowhand")

    return validate_shadowhand(cfg)


def validate_shadowhand(cfg):
    """ShadowHand MJCF 검증"""
    mjcf_dir = Path(cfg['paths']['mjcf'])

    stats = {
        "mjcf_exists": False,
        "meshes_exist": False,
        "xml_files": [],
    }

    # XML 파일 찾기
    xml_files = list(mjcf_dir.rglob("*.xml"))
    stats["xml_files"] = [str(f) for f in xml_files]
    stats["mjcf_exists"] = len(xml_files) > 0

    # 메쉬 파일 찾기
    mesh_files = list(mjcf_dir.rglob("*.stl")) + list(mjcf_dir.rglob("*.obj"))
    stats["meshes_exist"] = len(mesh_files) > 0

    print(f"  MJCF XML 파일: {len(xml_files)}개")
    print(f"  메쉬 파일: {len(mesh_files)}개")

    if xml_files:
        print(f"  ✅ ShadowHand MJCF 준비 완료")
    else:
        print(f"  ⚠️ ShadowHand MJCF 파일 없음 - 수동 다운로드 필요")

    return stats


# ============================================================
# A3b: DexGraspNet ShadowHand 원본 모델 (Phase H용)
# ============================================================
def download_dexgraspnet_hand(cfg):
    """DexGraspNet의 shadow_hand_wrist_free.xml + meshes 다운로드.

    DexGYS/DexGraspNet의 grasp parameter(28D)는 이 모델 기준으로 생성됨.
    - Root body = robot0:palm (forearm/wrist 없음)
    - 22 DOF finger joints
    - mujoco_menagerie 모델과 다른 geometry/reference frame
    Phase H에서 정확한 GT affordance 생성을 위해 이 모델이 필요.
    """
    dexgraspnet_dir = Path(cfg['shadow_hand'].get('dexgraspnet_mjcf', 'data/mjcf_dexgraspnet'))
    dexgraspnet_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("[A3b] DexGraspNet ShadowHand 원본 모델 다운로드")
    print("=" * 60)

    # 이미 있는지 확인
    target_xml = dexgraspnet_dir / cfg['shadow_hand'].get('dexgraspnet_xml', 'shadow_hand_wrist_free.xml')
    meshes_dir = dexgraspnet_dir / "meshes"
    if target_xml.exists() and meshes_dir.exists() and list(meshes_dir.glob("*.obj")):
        print(f"  Already have DexGraspNet hand model, skipping")
        print(f"    XML: {target_xml}")
        print(f"    Meshes: {len(list(meshes_dir.glob('*.obj')))} OBJ files")
        return validate_dexgraspnet_hand(cfg)

    # 방법 1: shallow clone (sparse-checkout 없이)
    temp_clone = Path("_temp_dexgraspnet")
    clone_success = False
    try:
        if temp_clone.exists():
            shutil.rmtree(temp_clone, ignore_errors=True)

        print("  Cloning DexGraspNet (shallow)...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/PKU-EPIC/DexGraspNet.git",
            str(temp_clone)
        ], check=True, capture_output=True, text=True, timeout=300)

        # 복사
        src_mjcf = temp_clone / "grasp_generation" / "mjcf"
        if src_mjcf.exists():
            shutil.copytree(src_mjcf, dexgraspnet_dir, dirs_exist_ok=True)
            print(f"  OK: MJCF copied to {dexgraspnet_dir}")
            clone_success = True
        else:
            print(f"  ⚠️ grasp_generation/mjcf not found in clone")

        # hand_model.py, rot6d.py도 참고용으로 복사
        utils_src = temp_clone / "grasp_generation" / "utils"
        if utils_src.exists():
            ref_dir = dexgraspnet_dir / "_reference"
            ref_dir.mkdir(exist_ok=True)
            for py_file in ["hand_model.py", "rot6d.py"]:
                src_py = utils_src / py_file
                if src_py.exists():
                    shutil.copy(src_py, ref_dir / py_file)

        shutil.rmtree(temp_clone, ignore_errors=True)

    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Clone failed: {e}")
        shutil.rmtree(temp_clone, ignore_errors=True)

    # 방법 2: GitHub raw URL로 개별 파일 다운로드
    if not clone_success:
        print("  Trying GitHub raw download...")
        import urllib.request
        base_url = "https://raw.githubusercontent.com/PKU-EPIC/DexGraspNet/main/grasp_generation/mjcf"

        # XML 파일 다운로드
        xml_name = cfg['shadow_hand'].get('dexgraspnet_xml', 'shadow_hand_wrist_free.xml')
        xml_url = f"{base_url}/{xml_name}"
        try:
            print(f"  Downloading {xml_name}...")
            urllib.request.urlretrieve(xml_url, dexgraspnet_dir / xml_name)
            print(f"  OK: {xml_name}")
        except Exception as e:
            print(f"  ⚠️ XML download failed: {e}")

        # 메쉬 파일 다운로드 (DexGraspNet 표준 메쉬 목록)
        mesh_names = [
            "F1.obj", "F2.obj", "F3.obj",
            "TH1_z.obj", "TH2_z.obj", "TH3_z.obj",
            "palm.obj", "knuckle.obj", "lfmetacarpal.obj", "wrist.obj",
        ]
        meshes_dir = dexgraspnet_dir / "meshes"
        meshes_dir.mkdir(exist_ok=True)

        for mesh_name in mesh_names:
            mesh_url = f"{base_url}/meshes/{mesh_name}"
            try:
                urllib.request.urlretrieve(mesh_url, meshes_dir / mesh_name)
            except Exception as e:
                print(f"    ⚠️ {mesh_name}: {e}")

        obj_count = len(list(meshes_dir.glob("*.obj")))
        if obj_count > 0:
            print(f"  OK: {obj_count} meshes downloaded")
            clone_success = True

    if not clone_success:
        print(f"\n  === 수동 다운로드 방법 ===")
        print(f"  git clone --depth 1 https://github.com/PKU-EPIC/DexGraspNet.git /tmp/DexGraspNet")
        print(f"  cp -r /tmp/DexGraspNet/grasp_generation/mjcf/* {dexgraspnet_dir}/")
        print(f"  rm -rf /tmp/DexGraspNet")

    return validate_dexgraspnet_hand(cfg)


def validate_dexgraspnet_hand(cfg):
    """DexGraspNet hand model 검증"""
    dexgraspnet_dir = Path(cfg['shadow_hand'].get('dexgraspnet_mjcf', 'data/mjcf_dexgraspnet'))
    xml_name = cfg['shadow_hand'].get('dexgraspnet_xml', 'shadow_hand_wrist_free.xml')

    stats = {
        "xml_exists": False,
        "meshes_count": 0,
    }

    xml_path = dexgraspnet_dir / xml_name
    stats["xml_exists"] = xml_path.exists()

    meshes_dir = dexgraspnet_dir / "meshes"
    if meshes_dir.exists():
        obj_files = list(meshes_dir.glob("*.obj"))
        stats["meshes_count"] = len(obj_files)

    if stats["xml_exists"]:
        print(f"  ✅ DexGraspNet hand model 준비 완료")
        print(f"    XML: {xml_path}")
        print(f"    Meshes: {stats['meshes_count']} OBJ files")
    else:
        print(f"  ⚠️ {xml_path} 없음 - 수동 다운로드 필요")

    return stats


def test_sample_loading(cfg):
    """A단계 최종 체크포인트: 샘플 1개 로드 → object PC(4096pts) 생성 테스트"""
    print("\n" + "=" * 60)
    print("[A-CHECK] 샘플 로딩 테스트")
    print("=" * 60)

    import trimesh

    # DexGYS에서 첫 샘플 로드
    dexgys_dir = Path(cfg['paths']['dexgys'])
    train_files = list(dexgys_dir.glob("*train*.json"))
    if not train_files:
        print("  ⚠️ DexGYS train 파일 없음 - 스킵")
        return False

    with open(train_files[0]) as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get('data', list(data.values()))
    if not samples or not isinstance(samples, list):
        print("  ⚠️ 샘플 데이터 구조 불명 - 스킵")
        return False

    sample = samples[0]

    # obj_id 추출
    obj_id = None
    for key in ['obj_id', 'object_id', 'oid']:
        if key in sample:
            obj_id = str(sample[key])
            break

    if obj_id is None:
        print("  ⚠️ obj_id 필드 없음")
        return False

    # mesh 인덱스에서 찾기
    index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
    if not index_path.exists():
        print("  ⚠️ mesh index 없음 - build_obj_mesh_index 먼저 실행")
        return False

    with open(index_path) as f:
        mesh_index = json.load(f)

    if obj_id not in mesh_index:
        print(f"  ⚠️ obj_id={obj_id} 메쉬 없음")
        return False

    mesh_path = mesh_index[obj_id]['mesh_path']
    print(f"  obj_id={obj_id}, mesh={mesh_path}")

    # 메쉬 로드 및 포인트 클라우드 생성
    mesh = trimesh.load(mesh_path)
    points, _ = trimesh.sample.sample_surface(mesh, cfg['affordance']['num_object_points'])

    print(f"  ✅ Object PC 생성: {points.shape} (목표: {cfg['affordance']['num_object_points']}pts)")
    return True


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase A: 원천 데이터 확보")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "dexgys", "oakink", "shadowhand", "dexgraspnet", "index", "test"],
                       default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "dexgys"]:
        download_dexgys(cfg)

    if args.step in ["all", "oakink"]:
        download_oakink(cfg)

    if args.step in ["all", "shadowhand"]:
        download_shadowhand(cfg)

    if args.step in ["all", "dexgraspnet"]:
        download_dexgraspnet_hand(cfg)

    if args.step in ["all", "index"]:
        build_obj_mesh_index(cfg)

    if args.step in ["all", "test"]:
        test_sample_loading(cfg)

    print("\n" + "=" * 60)
    print("Phase A 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
