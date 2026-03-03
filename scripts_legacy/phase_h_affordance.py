#!/usr/bin/env python3
"""
Phase H: GT Affordance 생성 (논문 정의)
- H1: Hand surface points 생성 (FK → link mesh → surface sampling)
- H2: Group union contact-distance map
- H3: Gaussian smoothing (σ = avg NN distance)
- H4: Score 변환 (distance → 0~1 score)

논문 핵심 공식:
  1) d_i^g = min_{h ∈ H_g} ||p_i - h||   (각 그래스프별 최소 거리)
  2) d_i = min_g d_i^g                      (그룹 union: 모든 그래스프의 최소)
  3) σ = avg nearest neighbor distance       (포인트 클라우드의 평균 NN 거리)
  4) Gaussian smoothing: 이웃 포인트의 가중 평균
  5) a_i = exp(-d_i / σ)                    (0~1 score로 변환)

Hand Model:
  DexGraspNet shadow_hand_wrist_free.xml 사용 (DexGYS grasp와 동일 모델)
  - Root body = robot0:palm (forearm/wrist 없음)
  - 22 DOF finger joints
  - t = palm 위치, R = palm 회전 (object frame 기준)

Requirements:
  pip install pytorch_kinematics mujoco trimesh scipy
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import jsonlines
from tqdm import tqdm
from scipy.spatial import cKDTree


def load_config(config_path="configs/pipeline_config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================
# H1: Hand Surface Points 생성
# ============================================================
class ShadowHandFK:
    """
    ShadowHand Forward Kinematics + Surface Point Sampling

    DexGraspNet shadow_hand_wrist_free.xml 기반:
    - Root = palm, 22 DOF, t/R로 palm pose 직접 지정
    - mujoco_menagerie 모델은 fallback으로만 사용

    입력: grasp(t, r, q) = translation(3) + axis-angle(3) + joint_angles(22)
    출력: hand surface points H_g (Nh × 3) in object coordinates
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_surface_points = cfg['affordance']['num_hand_surface_points']
        self.chain = None
        self.link_meshes = {}
        self.use_dexgraspnet = False
        self.mesh_scale = 1.0

    def load(self):
        """ShadowHand 모델 로드 (DexGraspNet 우선, menagerie fallback)"""
        print("  [H1] ShadowHand FK 로드")

        # 1) DexGraspNet 원본 모델 (추천)
        dexgraspnet_dir = Path(self.cfg['shadow_hand'].get('dexgraspnet_mjcf', 'data/mjcf_dexgraspnet'))
        dexgraspnet_xml = self.cfg['shadow_hand'].get('dexgraspnet_xml', 'shadow_hand_wrist_free.xml')
        xml_path = dexgraspnet_dir / dexgraspnet_xml

        if xml_path.exists():
            if self._load_pk_model(xml_path, model_type="dexgraspnet"):
                self.use_dexgraspnet = True
                return

        # 2) Fallback: mujoco_menagerie
        mjcf_dir = Path(self.cfg['paths']['mjcf'])
        xml_path = self._find_shadowhand_xml(mjcf_dir)
        if xml_path:
            print(f"    ⚠️ DexGraspNet 모델 없음 → menagerie fallback (정확도 저하)")
            print(f"    → python scripts/phase_a_download.py --step dexgraspnet 실행 권장")
            if self._load_pk_model(xml_path, model_type="menagerie"):
                return

        print("    ⚠️ FK 로드 실패. Fallback hand approximation 사용")

    def _sanitize_mjcf_for_pk(self, xml_content):
        """pytorch_kinematics용 최소 MJCF 생성.

        pk는 MuJoCo 스키마의 일부만 지원하므로,
        FK에 필요한 요소만 남기고 나머지는 전부 제거:
          - <compiler> (angle, meshdir)
          - <asset> → <mesh> 만
          - <worldbody> (body/joint/geom/inertial)
        """
        import xml.etree.ElementTree as ET

        root = ET.fromstring(xml_content)

        # FK에 불필요한 최상위 태그 전부 제거
        for tag in ['option', 'size', 'visual', 'default', 'contact',
                     'tendon', 'actuator', 'sensor']:
            for elem in root.findall(tag):
                root.remove(elem)

        # <asset>에서 mesh만 남기고 texture/material 제거
        asset = root.find('asset')
        if asset is not None:
            for child in list(asset):
                if child.tag != 'mesh':
                    asset.remove(child)

        # compiler 정리: meshdir을 로컬 구조에 맞게 수정
        compiler = root.find('compiler')
        if compiler is not None:
            for attr in list(compiler.attrib.keys()):
                if attr not in ('angle', 'meshdir'):
                    del compiler.attrib[attr]
            # DexGraspNet 원본은 meshdir="./mjcf/meshes/" 이지만
            # 우리 로컬 구조는 meshes/ 가 xml과 같은 레벨
            if 'meshdir' in compiler.attrib:
                old_meshdir = compiler.attrib['meshdir']
                if 'mjcf/meshes' in old_meshdir or 'mjcf\\meshes' in old_meshdir:
                    compiler.attrib['meshdir'] = './meshes/'

        # worldbody 정리: site, light 제거 + 불필요 속성 제거
        pk_keep_attrs = {
            'body': {'name', 'pos', 'quat', 'euler', 'axisangle', 'xyaxes', 'zaxis'},
            'joint': {'name', 'pos', 'axis', 'range', 'type', 'limited', 'damping',
                      'stiffness', 'armature', 'frictionloss', 'ref'},
            'geom': {'name', 'type', 'size', 'pos', 'quat', 'mesh', 'group',
                     'fromto', 'euler', 'axisangle', 'rgba', 'density', 'mass'},
            'inertial': {'pos', 'quat', 'mass', 'diaginertia', 'fullinertia', 'euler'},
        }

        def clean_element(elem):
            # site, light 제거
            for tag in ('site', 'light'):
                for child in list(elem.findall(tag)):
                    elem.remove(child)
            # 속성 정리
            if elem.tag in pk_keep_attrs:
                allowed = pk_keep_attrs[elem.tag]
                for attr in list(elem.attrib.keys()):
                    if attr not in allowed:
                        del elem.attrib[attr]
            for child in list(elem):
                clean_element(child)

        worldbody = root.find('worldbody')
        if worldbody is not None:
            clean_element(worldbody)

        result = ET.tostring(root, encoding='unicode')
        return result

    def _load_pk_model(self, xml_path, model_type="dexgraspnet"):
        """pytorch_kinematics로 MJCF 모델 로드"""
        try:
            import pytorch_kinematics as pk
            import torch
            import os

            print(f"    모델: {xml_path.name} ({model_type})")

            old_cwd = os.getcwd()
            os.chdir(xml_path.parent)
            try:
                with open(xml_path.name) as f:
                    xml_content = f.read()

                # DexGraspNet XML은 pk가 모르는 태그가 많아서 sanitize 필요
                # menagerie XML은 그대로 사용 (default class 등이 필요)
                if model_type == "dexgraspnet":
                    xml_content = self._sanitize_mjcf_for_pk(xml_content)

                self.chain = pk.build_chain_from_mjcf(xml_content)
                print(f"    FK chain 로드 성공")
                print(f"    Joints: {self.chain.n_joints}")
                self.use_pk = True
                self.xml_path = xml_path
                self.model_type = model_type
            finally:
                os.chdir(old_cwd)

            # Joint 이름 출력
            joint_names = self.chain.get_joint_parameter_names()
            print(f"    Joint names: {joint_names[:6]}... ({len(joint_names)}개)")

            # Zero pose에서 link 구조 확인
            q_zero = torch.zeros(1, self.chain.n_joints)
            transforms_zero = self.chain.forward_kinematics(q_zero)
            link_names = list(transforms_zero.keys())
            print(f"    Links: {len(link_names)}개")

            return True

        except Exception as e:
            print(f"    pytorch_kinematics 로드 실패: {e}")
            return False

    def _find_shadowhand_xml(self, mjcf_dir):
        """ShadowHand MJCF XML 파일 찾기 (menagerie fallback용)"""
        xml_files = list(mjcf_dir.rglob("*.xml"))
        if not xml_files:
            return None
        for xf in xml_files:
            name_lower = xf.stem.lower()
            if any(kw in name_lower for kw in ["shadow", "hand"]):
                return xf
        # 내용으로 검색
        for xf in xml_files:
            try:
                content = xf.read_text(errors='ignore')[:2000]
                if content.lower().count('<joint') >= 15:
                    return xf
            except:
                pass
        return xml_files[0] if xml_files else None

    def load_link_meshes(self):
        """링크 메쉬 로드 (surface sampling용)"""
        import trimesh
        import re

        if not hasattr(self, 'xml_path'):
            print(f"    ⚠️ XML 없음, 메쉬 로드 스킵")
            return

        xml_dir = self.xml_path.parent

        # MJCF에서 mesh scale 파싱
        try:
            content = self.xml_path.read_text(errors='ignore')
            scale_matches = re.findall(r'<mesh\s[^>]*scale\s*=\s*"([^"]+)"', content, re.IGNORECASE)
            if scale_matches:
                self.mesh_scale = float(scale_matches[0].strip().split()[0])
                print(f"    MJCF mesh scale: {self.mesh_scale}")
        except:
            pass

        # 메쉬 디렉토리 검색
        mesh_dirs = set()
        for dirname in ["meshes", "assets"]:
            for d in xml_dir.rglob(dirname):
                if d.is_dir():
                    mesh_dirs.add(d)
        mesh_dirs.add(xml_dir)

        # 메쉬 로드
        for mesh_dir in mesh_dirs:
            for ext in ["*.obj", "*.stl", "*.ply"]:
                for mesh_file in mesh_dir.glob(ext):
                    name = mesh_file.stem
                    if name in self.link_meshes:
                        continue
                    try:
                        mesh = trimesh.load(str(mesh_file), force='mesh')
                        if self.mesh_scale != 1.0:
                            mesh.apply_scale(self.mesh_scale)
                        self.link_meshes[name] = mesh
                    except:
                        pass

        print(f"    Link meshes: {len(self.link_meshes)}개")
        if self.link_meshes:
            names = list(self.link_meshes.keys())
            print(f"    메쉬 목록: {names[:8]}{'...' if len(names) > 8 else ''}")

    def _match_link_mesh(self, link_name):
        """FK link name → mesh 매칭"""
        if not self.link_meshes:
            return None

        # robot0: prefix 제거, rh_ prefix 제거
        name = link_name.lower()
        for prefix in ['robot0:', 'rh_']:
            name = name.replace(prefix, '')

        # exact match
        if name in self.link_meshes:
            return self.link_meshes[name]

        # DexGraspNet mesh naming: F1, F2, F3, TH1_z, TH2_z, TH3_z, knuckle, palm, wrist, lfmetacarpal
        # menagerie mesh naming: f_proximal, f_distal_pst, th_proximal, th_middle, palm, wrist, lf_metacarpal

        if self.model_type == "dexgraspnet":
            # DexGraspNet: proximal→F3, middle→F2, distal→F1 (finger)
            #              thproximal→TH3, thmiddle/thhub→TH2, thdistal→TH1 (thumb)
            if 'proximal' in name:
                if 'th' in name:
                    return self.link_meshes.get('TH3_z') or self.link_meshes.get('th_proximal')
                return self.link_meshes.get('F3') or self.link_meshes.get('f_proximal')
            if 'middle' in name:
                if 'th' in name:
                    return self.link_meshes.get('TH2_z') or self.link_meshes.get('th_middle')
                return self.link_meshes.get('F2') or self.link_meshes.get('f_middle') or self.link_meshes.get('F3')
            if 'distal' in name:
                if 'th' in name:
                    return self.link_meshes.get('TH1_z') or self.link_meshes.get('th_distal')
                return self.link_meshes.get('F1') or self.link_meshes.get('f_distal_pst')
            if 'knuckle' in name:
                return self.link_meshes.get('knuckle') or self.link_meshes.get('F3')
            if 'metacarpal' in name:
                return self.link_meshes.get('lfmetacarpal') or self.link_meshes.get('lf_metacarpal')
            if 'thbase' in name or 'thhub' in name:
                return self.link_meshes.get('TH2_z') or self.link_meshes.get('th_proximal')
        else:
            # menagerie fallback
            if 'proximal' in name:
                if 'th' in name:
                    return self.link_meshes.get('th_proximal')
                return self.link_meshes.get('f_proximal')
            if 'middle' in name:
                if 'th' in name:
                    return self.link_meshes.get('th_middle')
                return self.link_meshes.get('f_proximal')
            if 'distal' in name:
                return self.link_meshes.get('f_distal_pst') or self.link_meshes.get('f_distal')
            if 'knuckle' in name:
                return self.link_meshes.get('f_proximal')
            if 'metacarpal' in name:
                return self.link_meshes.get('lf_metacarpal')
            if 'thbase' in name or 'thhub' in name:
                return self.link_meshes.get('th_proximal')

        # 부분 문자열 매칭
        for mesh_name, mesh in self.link_meshes.items():
            if mesh_name.lower() in name or name in mesh_name.lower():
                return mesh
        return None

    def compute_hand_surface_points(self, translation, rotation_aa, joint_angles):
        """
        그래스프 파라미터 → object coordinate hand surface points

        DexGraspNet 모델: t = palm position, R = palm rotation (object frame)
        menagerie 모델: t에서 palm까지 offset 보정 필요

        Args:
            translation: (3,) palm 위치 (object frame)
            rotation_aa: (3,) axis-angle 회전
            joint_angles: (22,) 관절 각도

        Returns:
            points: (Nh, 3) hand surface points in object space
        """
        import trimesh
        from scipy.spatial.transform import Rotation

        t = np.array(translation)
        r = np.array(rotation_aa)
        q = np.array(joint_angles)
        R_global = Rotation.from_rotvec(r).as_matrix()

        if hasattr(self, 'use_pk') and self.use_pk:
            return self._compute_pk(t, R_global, q)
        else:
            return self._compute_fallback(t, R_global, q)

    def _compute_pk(self, t, R_global, q):
        """pytorch_kinematics 기반 FK"""
        import torch
        import trimesh

        n_chain_joints = self.chain.n_joints

        # Joint padding: DexGYS 22 DOF → chain joints
        if len(q) < n_chain_joints:
            q_padded = np.zeros(n_chain_joints)
            q_padded[n_chain_joints - len(q):] = q
            q = q_padded
        elif len(q) > n_chain_joints:
            q = q[:n_chain_joints]

        q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        transforms = self.chain.forward_kinematics(q_tensor)

        # --- Reference point 보정 ---
        # DexGraspNet 모델: root=palm → t가 직접 palm 위치 → 보정 불필요
        # menagerie 모델: root=forearm → palm 위치까지 offset 필요
        t_corrected = t
        if not self.use_dexgraspnet:
            ref_pos = np.zeros(3)
            for link_name, tf in transforms.items():
                ln = link_name.lower()
                if 'palm' in ln:
                    ref_pos = tf.get_matrix()[0, :3, 3].numpy()
                    break
            if np.allclose(ref_pos, 0):
                for link_name, tf in transforms.items():
                    if 'wrist' in link_name.lower():
                        ref_pos = tf.get_matrix()[0, :3, 3].numpy()
                        break
            t_corrected = t - R_global @ ref_pos

        # Surface point sampling
        all_points = []
        skip_keywords = ['forearm', 'mounting', 'world', 'grasp_site']

        for link_name, tf in transforms.items():
            if any(kw in link_name.lower() for kw in skip_keywords):
                continue

            link_pos = tf.get_matrix()[0, :3, 3].numpy()
            link_rot = tf.get_matrix()[0, :3, :3].numpy()

            matched_mesh = self._match_link_mesh(link_name)
            if matched_mesh is not None:
                n_pts = max(20, self.num_surface_points // 15)
                pts, _ = trimesh.sample.sample_surface(matched_mesh, n_pts)
                pts_world = (R_global @ (link_rot @ pts.T + link_pos.reshape(3, 1))).T + t_corrected
                all_points.append(pts_world)
            else:
                world_pos = R_global @ link_pos + t_corrected
                all_points.append(world_pos.reshape(1, 3))

        if all_points:
            points = np.concatenate(all_points, axis=0)
            if len(points) > self.num_surface_points:
                idx = np.random.choice(len(points), self.num_surface_points, replace=False)
                points = points[idx]
            elif len(points) < self.num_surface_points:
                idx = np.random.choice(len(points), self.num_surface_points, replace=True)
                points = points[idx]
            return points
        else:
            return self._compute_fallback(t, R_global, q)

    def _compute_fallback(self, t, R_global, q):
        """Fallback: 간단한 hand 근사 (FK 없이)"""
        palm_center = t.copy()
        finger_dirs = np.array([
            [0, 0.04, 0.08], [-0.02, 0.02, 0.1], [0, 0, 0.1],
            [0.02, -0.02, 0.1], [0.04, -0.04, 0.08],
        ])
        all_points = [palm_center + np.random.randn(self.num_surface_points // 4, 3) * 0.02]
        for i, fdir in enumerate(finger_dirs):
            finger_q = q[i * 4:(i + 1) * 4] if len(q) >= (i + 1) * 4 else [0] * 4
            bend = sum(np.abs(finger_q)) / 4.0
            finger_tip = palm_center + R_global @ (fdir * (1.0 - bend * 0.5))
            all_points.append(finger_tip + np.random.randn(self.num_surface_points // 5, 3) * 0.008)
        points = np.concatenate(all_points, axis=0)
        idx = np.random.choice(len(points), self.num_surface_points,
                               replace=len(points) < self.num_surface_points)
        return points[idx]


# ============================================================
# H2: Group Union Contact-Distance Map
# ============================================================
def compute_distance_map(object_points, hand_surface_points):
    """d_i^g = min_{h ∈ H_g} ||p_i - h||"""
    tree = cKDTree(hand_surface_points)
    distances, _ = tree.query(object_points, k=1)
    return distances


def compute_group_union_distance(object_points, all_hand_points_list):
    """d_i = min_g d_i^g"""
    M = len(object_points)
    union_distances = np.full(M, np.inf)
    for hand_points in all_hand_points_list:
        if len(hand_points) == 0:
            continue
        d_g = compute_distance_map(object_points, hand_points)
        union_distances = np.minimum(union_distances, d_g)
    return union_distances


# ============================================================
# H3: Gaussian Smoothing
# ============================================================
def compute_avg_nn_distance(points, k=5):
    """σ = avg nearest neighbor distance"""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k+1)
    return np.mean(distances[:, 1:])


def gaussian_smooth_affordance(points, distances, sigma, knn=20):
    """Gaussian 가중 평균으로 smoothing"""
    tree = cKDTree(points)
    nn_dists, nn_indices = tree.query(points, k=knn)
    smoothed = np.zeros(len(points))
    for i in range(len(points)):
        weights = np.exp(-0.5 * (nn_dists[i] / sigma) ** 2)
        weights /= weights.sum() + 1e-10
        smoothed[i] = np.sum(weights * distances[nn_indices[i]])
    return smoothed


# ============================================================
# H4: Score 변환
# ============================================================
def distance_to_score(distances, sigma, method="gaussian"):
    """
    Distance → 0~1 affordance score 변환

    논문 공식: a_i = exp(-d_i / σ)
    σ = avg NN distance of object point cloud
    """
    if method == "gaussian":
        scores = np.exp(-distances / sigma)
    elif method == "linear":
        max_d = distances.max() + 1e-10
        scores = 1.0 - distances / max_d
    elif method == "inverse":
        scores = 1.0 / (1.0 + distances / sigma)
    else:
        scores = np.exp(-distances / sigma)

    return np.clip(scores, 0.0, 1.0)


# ============================================================
# 전체 GT Affordance 생성 파이프라인
# ============================================================
def generate_affordance_gt(cfg, max_groups=None):
    """전체 GT Affordance 생성"""
    print("=" * 60)
    print("[H] GT Affordance 생성")
    print("=" * 60)

    # --- 초기화 ---
    hand_fk = ShadowHandFK(cfg)
    hand_fk.load()
    hand_fk.load_link_meshes()

    # Semantic groups 로드
    groups_path = Path(cfg['paths']['processed']) / "semantic_groups.json"
    if not groups_path.exists():
        print("  ⚠️ semantic_groups.json 없음. Phase G 먼저 실행")
        return

    with open(groups_path) as f:
        semantic_groups = json.load(f)

    # 메타 데이터 로드
    meta_dir = Path(cfg['paths']['processed']) / "dexgys_meta"
    all_meta = {}
    for meta_file in meta_dir.glob("*_meta.jsonl"):
        with jsonlines.open(meta_file) as reader:
            for entry in reader:
                all_meta[entry['sample_id']] = entry

    # Mesh index
    index_path = Path(cfg['paths']['processed']) / "obj_mesh_index.json"
    mesh_index = {}
    if index_path.exists():
        with open(index_path) as f:
            mesh_index = json.load(f)

    # 출력 디렉토리
    affordance_dir = Path(cfg['paths']['processed']) / "affordance_gt"
    affordance_dir.mkdir(parents=True, exist_ok=True)

    score_method = cfg['affordance']['score_method']
    num_obj_points = cfg['affordance']['num_object_points']

    groups_list = list(semantic_groups.items())
    if max_groups:
        groups_list = groups_list[:max_groups]

    total_groups = 0
    failed_groups = 0

    for group_id, group_info in tqdm(groups_list, desc="  Affordance GT"):
        sample_ids = group_info['sample_ids']
        scene_id = group_info['scene_id']

        # --- Object Point Cloud: 메쉬에서 직접 샘플링 ---
        import trimesh as _trimesh
        obj_points = None

        # 방법 1: obj_mesh_index
        for sid in sample_ids:
            if sid in all_meta:
                obj_id = all_meta[sid].get('obj_id')
                if obj_id and obj_id in mesh_index:
                    try:
                        mesh = _trimesh.load(mesh_index[obj_id]['mesh_path'], force='mesh')
                        obj_points, _ = _trimesh.sample.sample_surface(mesh, num_obj_points)
                    except Exception as e:
                        if total_groups == 0:
                            print(f"    메쉬 로드 실패 ({obj_id}): {e}")
                break

        # 방법 2: job.json
        if obj_points is None:
            job_file = Path(cfg['paths']['scenes']) / scene_id / "job.json"
            if job_file.exists():
                try:
                    with open(job_file) as f:
                        job = json.load(f)
                    mesh = _trimesh.load(job['mesh_path'], force='mesh')
                    obj_points, _ = _trimesh.sample.sample_surface(mesh, num_obj_points)
                except:
                    pass

        if obj_points is None or len(obj_points) == 0:
            failed_groups += 1
            continue

        # 포인트 수 맞추기
        if len(obj_points) > num_obj_points:
            idx = np.random.choice(len(obj_points), num_obj_points, replace=False)
            obj_points = obj_points[idx]
        elif len(obj_points) < num_obj_points:
            idx = np.random.choice(len(obj_points), num_obj_points, replace=True)
            obj_points = obj_points[idx]

        # --- 각 Grasp의 Hand Surface Points 생성 ---
        all_hand_points = []
        for sid in sample_ids:
            if sid not in all_meta:
                continue
            meta = all_meta[sid]
            t = np.array(meta['translation'])
            r = np.array(meta['rotation_aa'])
            q = np.array(meta['joint_angles'])
            hand_pts = hand_fk.compute_hand_surface_points(t, r, q)
            all_hand_points.append(hand_pts)

        if not all_hand_points:
            failed_groups += 1
            continue

        # --- H2: Group Union Distance ---
        union_distances = compute_group_union_distance(obj_points, all_hand_points)

        # --- H3: Gaussian Smoothing ---
        sigma = compute_avg_nn_distance(obj_points, k=5)
        smoothed_distances = gaussian_smooth_affordance(
            obj_points, union_distances, sigma,
            knn=cfg['affordance']['knn_neighbors']
        )

        # --- H4: Score 변환 ---
        affordance_scores = distance_to_score(smoothed_distances, sigma, method=score_method)

        # --- 디버그 (첫 그룹만) ---
        if total_groups == 0:
            print(f"\n    [DEBUG] obj center={obj_points.mean(axis=0)}")
            print(f"    [DEBUG] hand0 center={all_hand_points[0].mean(axis=0)}")
            print(f"    [DEBUG] distance: {np.linalg.norm(obj_points.mean(axis=0) - all_hand_points[0].mean(axis=0)):.4f}m")
            print(f"    [DEBUG] union_dist: min={union_distances.min():.6f}, max={union_distances.max():.6f}")
            print(f"    [DEBUG] sigma={sigma:.6f}")
            print(f"    [DEBUG] scores: min={affordance_scores.min():.4f}, max={affordance_scores.max():.4f}, mean={affordance_scores.mean():.4f}")
            high_pct = (affordance_scores > 0.5).mean() * 100
            print(f"    [DEBUG] high affordance (>0.5): {high_pct:.1f}%")

        # --- 저장 ---
        output = {
            "group_id": group_id,
            "scene_id": scene_id,
            "num_points": len(obj_points),
            "sigma": float(sigma),
            "score_method": score_method,
            "num_grasps_in_group": len(all_hand_points),
            "model_type": hand_fk.model_type if hasattr(hand_fk, 'model_type') else "fallback",
        }

        np.savez_compressed(
            affordance_dir / f"{group_id}.npz",
            object_points=obj_points.astype(np.float32),
            affordance_scores=affordance_scores.astype(np.float32),
            raw_distances=union_distances.astype(np.float32),
            smoothed_distances=smoothed_distances.astype(np.float32),
        )

        with open(affordance_dir / f"{group_id}_meta.json", 'w') as f:
            json.dump(output, f, indent=2)

        total_groups += 1

    print(f"\n  완료: {total_groups} groups")
    print(f"  실패: {failed_groups} groups")
    print(f"  저장: {affordance_dir}")


def visualize_affordance_sample(cfg, group_id=None):
    """Affordance GT 시각화 (검증용)"""
    print("\n  [검증] Affordance GT 시각화")

    affordance_dir = Path(cfg['paths']['processed']) / "affordance_gt"
    npz_files = sorted(affordance_dir.glob("*.npz"))
    if not npz_files:
        print("    ⚠️ Affordance GT 파일 없음")
        return

    target = affordance_dir / f"{group_id}.npz" if group_id else npz_files[0]
    if not target.exists():
        target = npz_files[0]

    data = np.load(target)
    points = data['object_points']
    scores = data['affordance_scores']

    print(f"    Group: {target.stem}")
    print(f"    Points: {points.shape}")
    print(f"    Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"    Score mean: {scores.mean():.4f}")
    print(f"    High affordance (>0.5): {(scores > 0.5).sum()} points ({(scores > 0.5).mean()*100:.1f}%)")
    print(f"    Peak affordance (>0.8): {(scores > 0.8).sum()} points ({(scores > 0.8).mean()*100:.1f}%)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        sc = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=scores, cmap='hot', s=2, alpha=0.8)
        plt.colorbar(sc, ax=ax1, label='Affordance Score')
        ax1.set_title(f'Affordance GT: {target.stem}')

        ax2 = fig.add_subplot(122)
        ax2.hist(scores, bins=50, edgecolor='black')
        ax2.set_xlabel('Affordance Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Score Distribution')

        viz_path = affordance_dir / f"{target.stem}_viz.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    시각화 저장: {viz_path}")
    except ImportError:
        print("    matplotlib 없음 - 시각화 스킵")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase H: GT Affordance 생성")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--step", choices=["all", "generate", "visualize"], default="all")
    parser.add_argument("--max_groups", type=int, default=None)
    parser.add_argument("--group_id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.step in ["all", "generate"]:
        generate_affordance_gt(cfg, max_groups=args.max_groups)

    if args.step in ["all", "visualize"]:
        visualize_affordance_sample(cfg, group_id=args.group_id)

    print("\n" + "=" * 60)
    print("Phase H 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
