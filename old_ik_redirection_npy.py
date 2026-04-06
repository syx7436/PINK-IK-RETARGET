import numpy as np
import os
import sys
import time
import joblib
import pinocchio as pin
import pink
import pink.tasks
from pink.limits import ConfigurationLimit
from scipy.spatial.transform import Rotation as sRot
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
class H1Config:
    FACE_KEYPOINTS = ['nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear']
    SMPL_BONE_ORDER = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
    ] + FACE_KEYPOINTS

    H1_NODE_NAMES = [
        'base_link', 'left_hip_linkage', 'left_thigh', 'left_knee_linkage',
        'left_calf', 'left_ankle', 'left_foot', 'right_hip_linkage',
        'right_thigh', 'right_knee_linkage', 'right_calf', 'right_ankle', 'right_foot',
        'waist', 'waist_gearbox', 'chest', 'left_shoulder_linkage', 'left_upper_arm',
        'left_elbow_linkage', 'left_force_arm', 'left_hand', 'right_shoulder_linkage', 'right_upper_arm',
        'right_elbow_linkage', 'right_force_arm', 'right_hand', 'neck', 'neck_linkage',
        'head', 'left_ear', 'right_ear',
    ]

    # [CRITICAL] 物理关节顺序 (Legs First)
    H1_JOINT_NAMES = [
        # Left Leg (0-5)
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_pitch_joint', 'left_ankle_yaw_joint', 'left_ankle_pitch_joint',
        # Right Leg (6-11)
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_pitch_joint', 'right_ankle_yaw_joint', 'right_ankle_pitch_joint',
        # Waist (12-14)
        'waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint',
        # Left Arm (15-19)
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        'left_elbow_pitch_joint', 'left_wrist_yaw_joint',
        # Right Arm (20-24)
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
        'right_elbow_pitch_joint', 'right_wrist_yaw_joint',
        # Neck (25-27)
        'neck_yaw_joint', 'neck_roll_joint', 'neck_pitch_joint',
    ]

    JOINT_LIMITS = np.array([
        # Left Leg
        [-1.57, 1.57], [-0.79, 0.79], [-1.05, 1.05], [-2.09, 0.0], [-0.87, 0.87], [-0.79, 0.79],
        # Right Leg
        [-1.57, 1.57], [-0.79, 0.79], [-1.05, 1.05], [-2.09, 0.0], [-0.87, 0.87], [-0.79, 0.79],
        # Waist
        [-1.57, 1.57], [-0.79, 0.79], [-0.52, 0.52],
        # Left Arm
        [-3.14, 2.09], [-3.14, 0.0], [-1.05, 1.05], [-2.62, 0.0], [-1.22, 1.22],
        # Right Arm
        [-2.09, 3.14], [0.0, 3.14], [-1.05, 1.05], [0.0, 2.62], [-1.22, 1.22],
        # Neck
        [-1.05, 1.05], [-0.52, 0.52], [-0.79, 0.79],
    ], dtype=np.float32)

    # (H1 link 名, SMPL 骨骼名) 追踪对
    TRACKING_PAIRS = [
        ("left_force_arm",  "left_elbow"),
        ("right_force_arm", "right_elbow"),
        ("left_hand",       "left_wrist"),
        ("right_hand",      "right_wrist"),
        ("left_calf",       "left_knee"),
        ("right_calf",      "right_knee"),
        ("left_foot",       "left_ankle"),
        ("right_foot",      "right_ankle"),
        ("head",            "head"),
    ]

    STAND_JOINT_ANGLES = np.array([
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        -1.57079632679, 0, 0, 0, 0, 1.57079632679, 0, 0,
        0, 0, 0, 0,
    ], dtype=np.float32)
    USER_DATA_DICT = dict(zip(H1_JOINT_NAMES, STAND_JOINT_ANGLES))


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _normalize(vec):
    """沿最后一维归一化，防止除零。shape: [..., 3]"""
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6)


def weighted_moving_filter(data, window_size=7):
    """线性加权滑动均值滤波。"""
    w = np.arange(1, window_size + 1, dtype=float)
    w /= w.sum()
    result = np.zeros_like(data)
    for j in range(window_size):
        shift = window_size - 1 - j
        if shift == 0:
            result += w[j] * data
        else:
            result[shift:] += w[j] * data[:-shift]
            result[:shift] += w[j] * data[0:1]
    return result


# ---------------------------------------------------------------------------
# IK Solver
# ---------------------------------------------------------------------------
class H1PinkSolver:
    # 机器人物理骨骼长度（米）
    ROBOT_DIMS = {
        'thigh':     0.0832,
        'calf':      0.1105,
        'upper_arm': 0.07579,
        'forearm':   0.04739,
        'neck2head': 0.022487329,
    }
    SMPL_SCALE = 0.24658347237104405  # SMPL 坐标 -> H1 世界坐标缩放系数
    TARGET_FPS = 90
    SOURCE_FPS = 20
    MAX_STEP   = 0.01  # 每帧 IK 目标最大位移（防止跳变）

    def __init__(self, urdf_path):
        self._init_pinocchio(urdf_path)
        self._init_ik_tasks()
        self._cache_joint_indices()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_pinocchio(self, urdf_path):
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=[os.path.dirname(urdf_path)],
        )
        self.model = self.robot.model
        self.data  = self.robot.data

        # 设置站立初始姿态
        self.q_ref = pin.neutral(self.model)
        for name, value in H1Config.USER_DATA_DICT.items():
            if self.model.existJointName(name):
                q_idx = self.model.joints[self.model.getJointId(name)].idx_q
                self.q_ref[q_idx] = value
            else:
                print(f"警告: URDF 中找不到关节 {name}，已忽略")
        self.configuration = pink.Configuration(self.model, self.data, self.q_ref)

        # 覆写关节限位
        for idx, name in enumerate(H1Config.H1_JOINT_NAMES):
            if self.model.existJointName(name):
                q_idx = self.model.joints[self.model.getJointId(name)].idx_q
                self.model.lowerPositionLimit[q_idx] = H1Config.JOINT_LIMITS[idx, 0]
                self.model.upperPositionLimit[q_idx] = H1Config.JOINT_LIMITS[idx, 1]
        self.config_limit = ConfigurationLimit(self.model)

    def _init_ik_tasks(self):
        """创建并缓存所有 pink IK 任务。"""
        self.tasks    = []
        self.ik_tasks = {}

        # 胸部朝向任务（只追旋转，用于压制身体后仰）
        self.chest_task = pink.tasks.FrameTask("chest", position_cost=0.0, orientation_cost=50.0)
        self.chest_task.set_target(self.configuration.get_transform_frame_to_world("chest"))
        self.tasks.append(self.chest_task)

        # 末端执行器 + 中间关节追踪任务
        for link_name, smpl_name in H1Config.TRACKING_PAIRS:
            if "hand" in link_name:
                pos_cost, rot_cost = 10.0, 0.0
            elif "foot" in link_name:
                pos_cost, rot_cost = 10.0, 1.0
            elif "force_arm" in link_name or "calf" in link_name:
                pos_cost, rot_cost = 5.0, 0.0
            elif "head" in link_name:
                pos_cost, rot_cost = 1.0, 5.0
            else:
                pos_cost, rot_cost = 1.0, 0.0

            if self.model.existFrame(link_name):
                task = pink.tasks.FrameTask(link_name, position_cost=pos_cost, orientation_cost=rot_cost)
                self.tasks.append(task)
                self.ik_tasks[smpl_name] = task
            else:
                print(f"跳过: URDF 中找不到 Link [{link_name}]")

        # 姿态正则化（权重极低，作为软约束防止关节乱飘）
        self.posture_task = pink.tasks.PostureTask(self.model)
        self.posture_task.set_target(self.q_ref)
        self.posture_task.cost = 0.1
        self.tasks.append(self.posture_task)

    def _cache_joint_indices(self):
        """预计算所有需要用到的 SMPL_BONE_ORDER 索引，避免在循环中重复查找。"""
        self.smpl_indices = {
            smpl_name: H1Config.SMPL_BONE_ORDER.index(smpl_name)
            for _, smpl_name in H1Config.TRACKING_PAIRS
        }
        self.bone_idx = {
            name: H1Config.SMPL_BONE_ORDER.index(name)
            for name in [
                'left_hip', 'left_knee', 'left_ankle',
                'right_hip', 'right_knee', 'right_ankle',
                'left_shoulder', 'left_elbow', 'left_wrist',
                'right_shoulder', 'right_elbow', 'right_wrist',
                'neck', 'head',
            ]
        }
        self.chest_idx = {
            name: H1Config.SMPL_BONE_ORDER.index(name)
            for name in ['spine3', 'left_collar', 'right_collar']
        }

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def process_motion(self, target_motion, head_rot_mats=None, visualize=False):
        """
        将 SMPL 动作序列重定向到 H1 机器人关节角度。

        Args:
            target_motion:  [T, J, 3] SMPL 关节位置（米）
            head_rot_mats:  [T, 3, 3] 头部旋转矩阵（可选）
            visualize:      是否在结果中附带可视化数据

        Returns:
            dict: dof [T_out, 28]（弧度），fps，帧数等
        """
        start = time.time()

        smpl_joints, smpl_raw_seq, head_rot_mats = self._preprocess(target_motion, head_rot_mats)
        smpl_joints, head_rot_interp, N_orig = self._upsample_to_90fps(
            smpl_joints, head_rot_mats, smpl_raw_seq
        )
        dof_raw, vis_data = self._solve_ik(smpl_joints, head_rot_interp, visualize)
        sim_dof = self._compute_dof(dof_raw, smpl_joints)

        print(f"Total time: {time.time() - start:.2f}s")
        return self._build_result(sim_dof, N_orig, len(smpl_joints), vis_data)

    # ------------------------------------------------------------------
    # 流程步骤（私有）
    # ------------------------------------------------------------------
    def _preprocess(self, target_motion, head_rot_mats):
        """
        1. 坐标系旋转（SMPL -> H1 朝向）
        2. 以 pelvis 为中心缩放到机器人比例
        3. 平移到机器人 waist 高度
        4. 消除逐帧 yaw 漂移（对齐第0帧方向）
        5. Savitzky-Golay 平滑
        """
        rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_matrix()

        smpl_raw_seq = target_motion @ rot.T                          # [T, J, 3]
        if head_rot_mats is not None:
            head_rot_mats = rot @ head_rot_mats @ rot.T              # [T, 3, 3]

        smpl_joints  = (smpl_raw_seq - smpl_raw_seq[:, 0:1, :]) * self.SMPL_SCALE
        waist_pos    = self.configuration.get_transform_frame_to_world("waist").translation
        smpl_joints += waist_pos.reshape(1, 1, 3)

        smpl_joints  = self._align_yaw(smpl_joints)
        smpl_joints  = savgol_filter(smpl_joints, window_length=9, polyorder=2, axis=0)

        return smpl_joints, smpl_raw_seq, head_rot_mats

    def _align_yaw(self, smpl_joints):
        """绕 Z 轴旋转每帧，使髋部朝向始终与第0帧一致。"""
        idx_l    = self.bone_idx['left_hip']
        idx_r    = self.bone_idx['right_hip']
        hip_vecs = smpl_joints[:, idx_r] - smpl_joints[:, idx_l]  # [N, 3]
        hip_vecs[:, 2] = 0

        yaw_ref  = np.arctan2(hip_vecs[0, 1], hip_vecs[0, 0])
        yaw_all  = np.arctan2(hip_vecs[:, 1], hip_vecs[:, 0])
        rot_mats = sRot.from_euler('z', yaw_ref - yaw_all).as_matrix()  # [N, 3, 3]

        root_pos = smpl_joints[:, 0:1, :]
        centered = smpl_joints - root_pos
        return np.einsum('nij,nkj->nki', rot_mats, centered) + root_pos

    def _upsample_to_90fps(self, smpl_joints, head_rot_mats, smpl_raw_seq):
        """
        三次样条插值，将 20fps 扩帧到 90fps。
        同步插值头部旋转（如有），并检测行走动作以决定是否保留 yaw。
        """
        N_orig = len(smpl_joints)
        N_new  = int((N_orig - 1) * (self.TARGET_FPS / self.SOURCE_FPS)) + 1
        t_orig = np.linspace(0, 1, N_orig)
        t_new  = np.linspace(0, 1, N_new)

        smpl_joints = interp1d(t_orig, smpl_joints, axis=0, kind='cubic')(t_new)

        head_rot_interp = None
        if head_rot_mats is not None:
            rotvec = np.unwrap(sRot.from_matrix(head_rot_mats).as_rotvec(), axis=0)

            max_disp = np.max(np.linalg.norm(np.diff(smpl_raw_seq[:, 0], axis=0), axis=-1))
            print(f"根节点最大位移: {max_disp:.3f}")
            if max_disp > 0.05:
                print("检测到走路类动作，去掉 head yaw")
                rotvec[:, 2] = 0
            else:
                print("原地动作，保留完整 head 旋转")

            rotvec_new      = interp1d(t_orig, rotvec, axis=0, kind='cubic')(t_new)
            head_rot_interp = sRot.from_rotvec(rotvec_new).as_matrix()

        return smpl_joints, head_rot_interp, N_orig

    def _solve_ik(self, smpl_joints, head_rot_interp, visualize):
        """
        主 IK 循环：每帧计算骨骼重定向目标，调用 quadprog QP 求解。
        返回原始 q 数组以及可选的可视化数据。
        """
        N    = len(smpl_joints)
        dt   = 1.0 / self.TARGET_FPS
        eye3 = np.eye(3)
        self.last_targets = {}

        dof_results = []
        vis_data    = {"smpl": [], "h1": []} if visualize else None

        print(f"Starting IK for {N} frames...")
        for i in range(N):
            # 从当前机器人正运动学读取肢体锚点
            anchors = {
                'l_shoulder': self.configuration.get_transform_frame_to_world("left_upper_arm").translation.copy(),
                'r_shoulder': self.configuration.get_transform_frame_to_world("right_upper_arm").translation.copy(),
                'l_thigh':    self.configuration.get_transform_frame_to_world("left_thigh").translation.copy(),
                'r_thigh':    self.configuration.get_transform_frame_to_world("right_thigh").translation.copy(),
                'neck':       self.configuration.get_transform_frame_to_world("neck_linkage").translation.copy(),
            }

            # 骨骼重定向：保持 SMPL 方向，替换为机器人骨骼长度
            retargeted = self.apply_bone_retargeting_single(smpl_joints[i:i+1], anchors)
            if visualize:
                vis_data["smpl"].append(retargeted[0])

            # 限速：防止目标点在相邻帧间跳变
            limited_targets = self._limit_target_speed(retargeted[0])

            # 胸部朝向（用于 chest_task 和 head 世界旋转）
            chest_rot = self._compute_chest_rotation(smpl_joints[i])

            # 设置所有追踪任务目标（每帧只需设一次）
            for smpl_name, task in self.ik_tasks.items():
                if smpl_name == "waist":
                    continue
                if smpl_name == "head" and head_rot_interp is not None:
                    task.set_target(pin.SE3(chest_rot @ head_rot_interp[i], limited_targets[smpl_name]))
                else:
                    task.set_target(pin.SE3(eye3, limited_targets[smpl_name]))

            # chest 位置跟随机器人当前状态（位置不强制追，只追旋转）
            chest_pos = self.configuration.get_transform_frame_to_world("chest").translation.copy()
            self.chest_task.set_target(pin.SE3(chest_rot, chest_pos))

            velocity = pink.solve_ik(
                self.configuration, self.tasks, dt,
                solver="quadprog", damping=1e-3,
                limits=[self.config_limit],
            )
            self.configuration.integrate_inplace(velocity, dt)
            dof_results.append(self.configuration.q.copy())

            if visualize:
                vis_data["h1"].append([
                    self.configuration.get_transform_frame_to_world(link).translation.copy()
                    if self.model.existFrame(link) else np.zeros(3)
                    for link in H1Config.H1_NODE_NAMES
                ])

        return np.array(dof_results), vis_data

    def _limit_target_speed(self, retargeted_frame):
        """限制每个追踪目标相邻帧间的最大位移，平滑目标轨迹。"""
        limited = {}
        for smpl_name in self.ik_tasks:
            if smpl_name == "waist":
                continue
            target = retargeted_frame[self.smpl_indices[smpl_name]].copy()
            if smpl_name in self.last_targets:
                delta = target - self.last_targets[smpl_name]
                norm  = np.linalg.norm(delta)
                if norm > self.MAX_STEP:
                    target = self.last_targets[smpl_name] + delta / norm * self.MAX_STEP
            limited[smpl_name] = target
        self.last_targets = {k: v.copy() for k, v in limited.items()}
        return limited

    def _compute_chest_rotation(self, smpl_frame):
        """从 spine3 和 collar 位置构造胸部旋转矩阵（forward-right-up 坐标系）。"""
        p_spine3   = smpl_frame[self.chest_idx['spine3']]
        p_l_collar = smpl_frame[self.chest_idx['left_collar']]
        p_r_collar = smpl_frame[self.chest_idx['right_collar']]

        up      = _normalize((p_l_collar + p_r_collar) / 2.0 - p_spine3)
        right   = _normalize(p_l_collar - p_r_collar)
        forward = _normalize(np.cross(right, up))
        right   = _normalize(np.cross(up, forward))
        return np.column_stack([forward, right, up])

    def _compute_dof(self, dof_raw, smpl_joints):
        """
        1. 加权滑动滤波 + unwrap
        2. 从 Pinocchio q 向量提取命名关节角度
        3. 用 SMPL 几何直接覆写脚踝 pitch（IK 对该关节精度不足）
        """
        print("Applying Weighted Moving Filter...")
        dof_raw = weighted_moving_filter(dof_raw, window_size=7)
        dof_raw = np.unwrap(dof_raw, axis=0)

        N = len(dof_raw)
        print("Applying Sim2Real Offsets...")
        sim_dof = np.zeros((N, len(H1Config.H1_JOINT_NAMES)))
        for target_idx, name in enumerate(H1Config.H1_JOINT_NAMES):
            if not self.model.existJointName(name):
                print(f"Joint not found in Pinocchio: {name}")
                continue
            joint = self.model.joints[self.model.getJointId(name)]
            sim_dof[:, target_idx] = np.rad2deg(dof_raw[:, joint.idx_q])

        sim_dof[:, 5]  = np.rad2deg(self._ankle_pitch(smpl_joints, 'left'))
        sim_dof[:, 11] = np.rad2deg(self._ankle_pitch(smpl_joints, 'right'))

        return sim_dof

    def _ankle_pitch(self, smpl_joints, side):
        """从 SMPL 关节位置估算脚踝 pitch，以第0帧为中性角（站立≈0）。"""
        knee  = smpl_joints[:, self.bone_idx[f'{side}_knee']]
        ankle = smpl_joints[:, self.bone_idx[f'{side}_ankle']]
        foot  = smpl_joints[:, H1Config.SMPL_BONE_ORDER.index(f'{side}_foot')]

        shin  = _normalize(ankle - knee)
        toes  = _normalize(foot  - ankle)
        angle = np.arccos(np.clip((shin * toes).sum(axis=-1), -1.0, 1.0))
        pitch = angle - angle[0]

        lo, hi = H1Config.JOINT_LIMITS[5]
        return np.clip(pitch, lo, hi)

    def _build_result(self, sim_dof, N_orig, N, vis_data):
        result = {
            "dof":               np.deg2rad(sim_dof),
            "fps":               self.TARGET_FPS,
            "input_frames":      N_orig,
            "output_frames":     N,
            "root_trans_offset": np.zeros((N, 3)),
            "root_rot":          np.tile(np.array([0, 0, 0, 1]), (N, 1)),
        }
        if vis_data is not None:
            result["smpl_joints_target"] = np.array(vis_data["smpl"])
            result["h1_joint_pos"]       = np.array(vis_data["h1"])
        return result

    # ------------------------------------------------------------------
    # 骨骼重定向
    # ------------------------------------------------------------------
    def apply_bone_retargeting_single(self, smpl_joints, anchors):
        """
        对单帧 SMPL 骨骼进行重定向：保留骨骼方向，替换为机器人骨骼长度。
        anchors 中的肩/髋/颈位置来自当前帧的机器人正运动学结果，
        确保躯干运动（弯腰、侧倾）能正确传递到四肢。

        Args:
            smpl_joints: [1, J, 3]
            anchors:     dict，键: l_shoulder, r_shoulder, l_thigh, r_thigh, neck

        Returns:
            new_joints:  [1, J, 3]
        """
        j  = smpl_joints        # 原始关节位置（只读方向）
        nj = smpl_joints.copy() # 重定向后的结果
        b  = self.bone_idx
        d  = self.ROBOT_DIMS

        # 头部
        nj[:, b['neck']] = anchors['neck']
        nj[:, b['head']] = anchors['neck'] + _normalize(j[:, b['head']] - j[:, b['neck']]) * d['neck2head']

        # 左腿（hip -> knee -> ankle，链式传递）
        nj[:, b['left_hip']]   = anchors['l_thigh']
        nj[:, b['left_knee']]  = anchors['l_thigh']         + _normalize(j[:, b['left_knee']]  - j[:, b['left_hip']])   * d['thigh']
        nj[:, b['left_ankle']] = nj[:, b['left_knee']]      + _normalize(j[:, b['left_ankle']] - j[:, b['left_knee']])  * d['calf']

        # 右腿
        nj[:, b['right_hip']]   = anchors['r_thigh']
        nj[:, b['right_knee']]  = anchors['r_thigh']        + _normalize(j[:, b['right_knee']]  - j[:, b['right_hip']])  * d['thigh']
        nj[:, b['right_ankle']] = nj[:, b['right_knee']]    + _normalize(j[:, b['right_ankle']] - j[:, b['right_knee']]) * d['calf']

        # 左臂（shoulder -> elbow -> wrist，链式传递）
        nj[:, b['left_shoulder']] = anchors['l_shoulder']
        nj[:, b['left_elbow']]    = anchors['l_shoulder']   + _normalize(j[:, b['left_elbow']]  - j[:, b['left_shoulder']]) * d['upper_arm']
        nj[:, b['left_wrist']]    = nj[:, b['left_elbow']]  + _normalize(j[:, b['left_wrist']]  - j[:, b['left_elbow']])    * d['forearm']

        # 右臂
        nj[:, b['right_shoulder']] = anchors['r_shoulder']
        nj[:, b['right_elbow']]    = anchors['r_shoulder']  + _normalize(j[:, b['right_elbow']]  - j[:, b['right_shoulder']]) * d['upper_arm']
        nj[:, b['right_wrist']]    = nj[:, b['right_elbow']] + _normalize(j[:, b['right_wrist']]  - j[:, b['right_elbow']])   * d['forearm']

        return nj


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_data_all_npy(data_path, batch_id):
    data            = np.load(data_path, allow_pickle=True).item()
    motion          = data['motion'][batch_id].transpose(2, 0, 1)  # [22, 3, T] -> [T, 22, 3]
    head_rot        = data['head_rot_mats'][batch_id]
    left_wrist_rot  = data['left_wrist_rot_mats'][batch_id]
    right_wrist_rot = data['right_wrist_rot_mats'][batch_id]
    return motion, head_rot, left_wrist_rot, right_wrist_rot


if __name__ == "__main__":
    urdf_path   = "/home/syx/tw-robot/retarget/urdf/Assembly.urdf"
    data_path   = "/home/syx/tw-robot/retarget/input/results.npy"
    output_path = "/home/syx/tw-robot/retarget/output/result.pkl"

    amass_data, head_rot, _, _ = load_data_all_npy(data_path, 14)
    if amass_data is not None:
        solver = H1PinkSolver(urdf_path)
        result = solver.process_motion(amass_data, head_rot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(result, output_path)
        print(f"Saved to {output_path}")
    else:
        print("Failed to load Amass Data")
