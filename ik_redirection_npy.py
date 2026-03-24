import numpy as np
import os
import sys
import time
import joblib
import torch
import pinocchio as pin
import pink
import pink.tasks
from pink.limits import ConfigurationLimit
from scipy.spatial.transform import Rotation as sRot
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as sRot
from scipy.interpolate import interp1d

# 配置参数
class H1Config:
    DEVICE = torch.device("cpu") # IK 通常在 CPU 上跑得很快，不需要 CUDA
    FACE_KEYPOINTS = ['nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear']
    SMPL_BONE_ORDER = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
        "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
        "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
    ] + FACE_KEYPOINTS
    H1_NODE_NAMES = ['base_link', 'left_hip_linkage', 'left_thigh', 'left_knee_linkage', 
                     'left_calf', 'left_ankle', 'left_foot', 'right_hip_linkage',
                     'right_thigh', 'right_knee_linkage', 'right_calf', 'right_ankle', 'right_foot',
                    'waist', 'waist_gearbox', 'chest', 'left_shoulder_linkage', 'left_upper_arm',
                    'left_elbow_linkage', 'left_force_arm', 'left_hand', 'right_shoulder_linkage', 'right_upper_arm',
                    'right_elbow_linkage', 'right_force_arm', 'right_hand', 'neck', 'neck_linkage',
                    'head', 'left_ear', 'right_ear']
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
        'neck_yaw_joint', 'neck_roll_joint', 'neck_pitch_joint'
    ]

    # H1 关节限位
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
        [-1.05, 1.05], [-0.52, 0.52], [-0.79, 0.79]  
    ], dtype=np.float32)
    
    TRACKING_PAIRS = [
        ("left_force_arm", "left_elbow"),   
        ("right_force_arm", "right_elbow"),

        ("left_hand", "left_wrist"),
        ("right_hand", "right_wrist"),
        
        # 腿部
        ("left_calf", "left_knee"),
        ("right_calf", "right_knee"),
        
        ("left_foot", "left_ankle"),  
        ("right_foot", "right_ankle"),

        ("head", "head"),
    ]

    STAND_JOINT_ANGLES = np.array([
        0, 0, 0, 0, 0,  0,  0 ,  0,
        0, 0,  0, 0, 0, 0 ,  0,  0,
        -1.57079632679,  0, 0,  0, 0,  1.57079632679, 0,  0,
        0, 0, 0, 0
    ], dtype=np.float32)
    USER_DATA_DICT = dict(zip(H1_JOINT_NAMES, STAND_JOINT_ANGLES))

# 加权滑动滤波
def weighted_moving_filter(data, window_size=7):
    w = np.arange(1, window_size + 1, dtype=float)
    w /= w.sum()
    result = np.zeros_like(data)
    for j in range(window_size):
        weight = w[j]
        shift = window_size - 1 - j
        if shift == 0:
            result += weight * data
        else:
            result[shift:] += weight * data[:-shift]
            result[:shift] += weight * data[0:1]  # 边界用第一帧填充
    return result

class H1PinkSolver:
    def __init__(self, urdf_path):
        self.device = H1Config.DEVICE
        # 初始化 Pinocchio 模型
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, 
            package_dirs=[os.path.dirname(urdf_path)],
            # root_joint=pin.JointModelFreeFlyer()
            )
        self.model = self.robot.model
        self.data = self.robot.data
        self.q_ref = pin.neutral(self.model) 
        for name, value in H1Config.USER_DATA_DICT.items():
            if self.model.existJointName(name):
                # 获取该关节在 q 向量中的起始索引
                joint_id = self.model.getJointId(name)
                q_idx = self.model.joints[joint_id].idx_q
                # 赋值
                self.q_ref[q_idx] = value
            else:
                print(f" 警告: URDF 中找不到关节 {name}，已忽略")
        self.configuration = pink.Configuration(self.model, self.data, self.q_ref)

        self.robot_dims = {
            'thigh': 0.0832,      # 大腿 (Hip -> Knee)
            'calf': 0.1105,       # 小腿 (Knee -> Ankle)
            'upper_arm': 0.07579,  # 大臂 (Shoulder -> Elbow)
            'forearm': 0.04739,    # 小臂 (Elbow -> Wrist)
            'neck2head': 0.022487329,
        }

        self.robot_l_shoulder = self.configuration.get_transform_frame_to_world("left_upper_arm").translation
        self.robot_r_shoulder = self.configuration.get_transform_frame_to_world("right_upper_arm").translation
        self.robot_l_thigh = self.configuration.get_transform_frame_to_world("left_thigh").translation
        self.robot_r_thigh = self.configuration.get_transform_frame_to_world("right_thigh").translation

        self.scale = 0.24658347237104405

        self.tasks = []
        self.ik_tasks = {} # 存储 Task 句柄以便后续更新 Target
        # 在 posture_task 之前加
        self.chest_task = pink.tasks.FrameTask(
            "chest", 
            position_cost=0.0,      # 不追位置，只追朝向
            orientation_cost=50.0   # 朝向权重要高，压制后仰
        )
        self.chest_task.set_target(
            self.configuration.get_transform_frame_to_world("chest")
        )
        self.tasks.append(self.chest_task)

        for link_name, smpl_name in H1Config.TRACKING_PAIRS:
            # 默认权重
            pos_cost = 1.0
            rot_cost = 0.0
            # --- 针对不同部位调整权重 ---
            # 1. 手和脚 (End Effectors)：必须精准
            if "hand" in link_name:
                pos_cost = 10.0  # 提高位置权重
                rot_cost = 0.0   # 旋转通常不追，靠手腕关节自己解算
            elif "foot" in link_name:
                pos_cost = 10.0  # 提高位置权重
                rot_cost = 1.0   # 旋转通常不追，靠手腕关节自己解算
            # 2. 膝盖和手肘 (Middle Joints)：主要用于引导方向
            elif "upper_arm" in link_name or "calf" in link_name:
                pos_cost = 5.0   # 权重低一点，允许有误差（因为臂长不一样）
                rot_cost = 0.0
            elif "force_arm" in link_name:
                pos_cost = 5.0   # 权重低一点，允许有误差（因为臂长不一样）
                rot_cost = 0.0
            # 3. 头部/脖子：主要追旋转，位置次要
            elif "head" in link_name:
                pos_cost = 1.0   # 位置不重要，跟着脊柱走
                rot_cost = 5.0   # 关键：旋转要跟住！让头别乱歪
            # 创建任务
            if self.model.existFrame(link_name):
                task = pink.tasks.FrameTask(link_name, position_cost=pos_cost, orientation_cost=rot_cost)
                self.tasks.append(task)
                self.ik_tasks[smpl_name] = task
            else:
                print(f"跳过: URDF 中找不到 Link [{link_name}]")
            
        self.posture_task = pink.tasks.PostureTask(self.model)
        self.posture_task.set_target(self.q_ref)
        self.posture_task.cost = 0.1 # 权重很低，作为软约束
        self.tasks.append(self.posture_task)

        limits = H1Config.JOINT_LIMITS
        for idx, name in enumerate(H1Config.H1_JOINT_NAMES):
            if self.model.existJointName(name):
                joint_id = self.model.getJointId(name)
                joint = self.model.joints[joint_id]
                q_idx = joint.idx_q
                self.model.lowerPositionLimit[q_idx] = limits[idx, 0]
                self.model.upperPositionLimit[q_idx] = limits[idx, 1]
        self.config_limit = ConfigurationLimit(self.model)

        self.smpl_indices = {name: H1Config.SMPL_BONE_ORDER.index(name) for _, name in H1Config.TRACKING_PAIRS}
        self.last_targets = {}

        self.bone_idx = {
            name: H1Config.SMPL_BONE_ORDER.index(name)
            for name in [
                'left_hip','left_knee','left_ankle',
                'right_hip','right_knee','right_ankle',
                'left_shoulder','left_elbow','left_wrist',
                'right_shoulder','right_elbow','right_wrist',
                'neck','head'
            ]
        }
        self.chest_idx = {
            name: H1Config.SMPL_BONE_ORDER.index(name)
            for name in ['spine3','left_collar','right_collar']
        }

    def process_motion(self, target_motion, head_rot_mats=None):
        start_time = time.time()
        N = len(target_motion)
        target_motion = torch.from_numpy(target_motion)
        quat = [0.5, 0.5, 0.5, 0.5] 
        r = sRot.from_quat(quat)
        rot_matrix_np = r.as_matrix()
        if head_rot_mats is not None:
            head_rot_mats = rot_matrix_np @ head_rot_mats @ rot_matrix_np.T
        rot_matrix = torch.from_numpy(r.as_matrix()).float().to(target_motion.device)
        target_motion = torch.matmul(target_motion, rot_matrix.t())
        smpl_raw_seq = target_motion.detach().cpu().numpy()
        _, num_joints, _ = smpl_raw_seq.shape
        pos_smpl_root = smpl_raw_seq[:, 0:1, :]
        scales = np.full(num_joints, self.scale)
        scales = scales.reshape(1, num_joints, 1)
        smpl_joints_scaled = (smpl_raw_seq - pos_smpl_root) * scales
        T_waist_init = self.configuration.get_transform_frame_to_world("waist")
        waist_pos = T_waist_init.translation
        waist_offset = waist_pos.reshape(1, 1, 3)
        smpl_joints_scaled = smpl_joints_scaled + waist_offset
        T_chest = self.configuration.get_transform_frame_to_world("chest")
        # # ========== 朝向对齐预处理 ==========
        idx_l_hip = H1Config.SMPL_BONE_ORDER.index('left_hip')
        idx_r_hip = H1Config.SMPL_BONE_ORDER.index('right_hip')

        # 第0帧髋部向量作为标准朝向参考
        hip_vec_0 = smpl_joints_scaled[0, idx_r_hip] - smpl_joints_scaled[0, idx_l_hip]
        hip_vec_0[2] = 0  # 投影到XY水平面，去掉Z
        yaw_0 = np.arctan2(hip_vec_0[1], hip_vec_0[0])

        for i in range(len(smpl_joints_scaled)):
            hip_vec = smpl_joints_scaled[i, idx_r_hip] - smpl_joints_scaled[i, idx_l_hip]
            hip_vec[2] = 0
            yaw_i = np.arctan2(hip_vec[1], hip_vec[0])

            # 消除相对第0帧的yaw变化，绕Z轴旋转
            delta_yaw = yaw_0 - yaw_i
            rot_mat = sRot.from_euler('z', delta_yaw).as_matrix()  # 绕Z轴

            # 以pelvis为中心旋转整帧
            root_pos = smpl_joints_scaled[i, 0:1, :]   # [1, 3]
            centered = smpl_joints_scaled[i] - root_pos  # [J, 3]
            smpl_joints_scaled[i] = centered @ rot_mat.T + root_pos

        smpl_joints_scaled = savgol_filter(
            smpl_joints_scaled,
            window_length=9,
            polyorder=2,
            axis=0
        )
        # 插值扩帧 5倍
        N_orig = len(smpl_joints_scaled)
        aim_fps = 90
        N_new = int((N_orig - 1) * (aim_fps / 20)) + 1
        t_orig = np.linspace(0, 1, N_orig)
        t_new  = np.linspace(0, 1, N_new)
        interpolator = interp1d(t_orig, smpl_joints_scaled, axis=0, kind='cubic')
        smpl_joints_scaled = interpolator(t_new)  # [N_new, J, 3]

        if head_rot_mats is not None:
            # 转成旋转向量
            rotvec = sRot.from_matrix(head_rot_mats).as_rotvec()  # [T_orig, 3]
            rotvec = np.unwrap(rotvec, axis=0)  # 消除符号跳变
            # rotvec[:, 2] = 0  # Z轴是yaw，清零
            root_displacement = np.max(np.linalg.norm(
                np.diff(smpl_raw_seq[:, 0, :], axis=0), axis=-1))
            print(f"根节点最大位移: {root_displacement:.3f}")
            if root_displacement > 0.05:  # 注意：smpl_raw_seq已经归一化缩放，阈值要小一些
                print("检测到走路类动作，去掉head yaw")
                rotvec[:, 2] = 0
            else:
                print("原地动作，保留完整head旋转")
            interp_rv = interp1d(t_orig, rotvec, axis=0, kind='cubic')
            rotvec_new = interp_rv(t_new)  # [N_new, 3]
            head_rot_mats_interp = sRot.from_rotvec(rotvec_new).as_matrix()  # [N_new, 3, 3]
        else:
            head_rot_mats_interp = None

        N = N_new
        # --- IK 求解循环 ---
        dt = 1.0 / 90 # 20fps -> 0.05s
        solver = pink.solve_ik
        dof_results = []

        h1_cartesian_seq = []
        smpl_joints_target_seq = []

        print(f"Starting IK for {N} frames...")
        substeps = 2
        max_step = 0.01
        self.last_targets = {}
        for i in range(N):
            robot_l_shoulder = self.configuration.get_transform_frame_to_world(
                "left_upper_arm").translation.copy()
            robot_r_shoulder = self.configuration.get_transform_frame_to_world(
                "right_upper_arm").translation.copy()
            robot_l_thigh = self.configuration.get_transform_frame_to_world(
                "left_thigh").translation.copy()
            robot_r_thigh = self.configuration.get_transform_frame_to_world(
                "right_thigh").translation.copy()
            robot_neck = self.configuration.get_transform_frame_to_world("neck_linkage").translation.copy()
            smpl_frame = smpl_joints_scaled[i:i+1]
            retargeted_frame = self.apply_bone_retargeting_single(
                smpl_frame,
                self.robot_dims,
                robot_l_shoulder,
                robot_r_shoulder,
                robot_l_thigh,
                robot_r_thigh, 
                robot_neck
            ) 
            smpl_joints_target_seq.append(retargeted_frame[0])
            limited_targets = {}
            for smpl_name, task in self.ik_tasks.items():
                if smpl_name in ["waist"]:
                    continue
                target_absolute = retargeted_frame[0, self.smpl_indices[smpl_name]].copy()
                if smpl_name in self.last_targets:
                    delta = target_absolute - self.last_targets[smpl_name]
                    norm = np.linalg.norm(delta)
                    if norm > max_step:
                        target_absolute = self.last_targets[smpl_name] + delta / norm * max_step
                limited_targets[smpl_name] = target_absolute

            # 每帧只更新一次 last_targets
            self.last_targets = {k: v.copy() for k, v in limited_targets.items()}
            p_spine3   = smpl_joints_scaled[i, self.chest_idx['spine3']]
            p_l_collar = smpl_joints_scaled[i, self.chest_idx['left_collar']]
            p_r_collar = smpl_joints_scaled[i, self.chest_idx['right_collar']]
            mid_collar = (p_l_collar + p_r_collar) / 2.0
            up    = mid_collar - p_spine3
            right = p_l_collar - p_r_collar
            up    /= np.linalg.norm(up)    + 1e-6
            right /= np.linalg.norm(right) + 1e-6
            forward = np.cross(right, up)
            forward /= np.linalg.norm(forward) + 1e-6
            right   = np.cross(up, forward)
            right   /= np.linalg.norm(right) + 1e-6
            rot = np.column_stack([forward, right, up])
            chest_rot_target = rot  # 缓存起来
            # if i % 5 == 0:
            #     chest_yaw = np.arctan2(chest_rot_target[1,0], chest_rot_target[0,0])
            #     print(f"帧{i} chest_yaw: {np.rad2deg(chest_yaw):.1f}度")

            for _ in range(substeps):
                for smpl_name, task in self.ik_tasks.items():
                    if smpl_name in ["waist"]:
                        continue
                    if smpl_name == "head" and head_rot_mats_interp is not None:
                        # 世界朝向 = 躯干朝向 * 头部局部旋转
                        head_rot_world = chest_rot_target @ head_rot_mats_interp[i]
                        # task.set_target(pin.SE3(head_rot_mats_interp[i], limited_targets[smpl_name]))
                        task.set_target(pin.SE3(head_rot_world, limited_targets[smpl_name]))
                    else:
                        task.set_target(pin.SE3(np.eye(3), limited_targets[smpl_name]))
                
                # 位置用机器人chest当前真实位置（不强制追位置）
                chest_pos = self.configuration.get_transform_frame_to_world("chest").translation.copy()
                self.chest_task.set_target(pin.SE3(chest_rot_target, chest_pos))
            
                # 求解 IK
                velocity = solver(
                    self.configuration, 
                    self.tasks, 
                    dt / substeps, 
                    solver="quadprog",
                    damping=1e-3,
                    limits=[self.config_limit]
                    )
                # 更新状态
                self.configuration.integrate_inplace(velocity, dt / substeps)
            curr_q = self.configuration.q.copy()
            dof_results.append(curr_q)

            current_frame_pos = []
            for link_name in H1Config.H1_NODE_NAMES:
                if self.model.existFrame(link_name):
                    # 获取该 Link 在世界坐标系下的变换矩阵 (SE3)，并只取平移部分 (translation)
                    pos = self.configuration.get_transform_frame_to_world(link_name).translation.copy()
                    current_frame_pos.append(pos)
                else:
                    current_frame_pos.append(np.zeros(3)) # 防止报错补0
            h1_cartesian_seq.append(current_frame_pos)

        dof_results = np.array(dof_results)
        # sigma 控制平滑程度
        print("Applying Weighted Moving Filter...")
        dof_results = weighted_moving_filter(dof_results, window_size=7)
        dof_results = np.unwrap(dof_results, axis=0)
        # --- 后处理 (Sim2Real Offset) ---
        print("Applying Sim2Real Offsets...")

        sim_dof_deg = np.zeros((N, len(H1Config.H1_JOINT_NAMES)))

        for target_idx, name in enumerate(H1Config.H1_JOINT_NAMES):
            if not self.model.existJointName(name):
                print(f"Joint not found in Pinocchio: {name}")
                continue
            joint_id = self.model.getJointId(name)
            joint = self.model.joints[joint_id]
            q_idx = joint.idx_q
            nq = joint.nq
            val = dof_results[:, q_idx : q_idx + nq]
            sim_dof_deg[:, target_idx] = np.rad2deg(val[:,0])

        # 计算左脚踝pitch（index 5）
        idx_l_knee  = self.bone_idx['left_knee']
        idx_l_ankle = self.bone_idx['left_ankle']
        idx_l_foot  = H1Config.SMPL_BONE_ORDER.index('left_foot')
        idx_r_knee  = self.bone_idx['right_knee']
        idx_r_ankle = self.bone_idx['right_ankle']
        idx_r_foot  = H1Config.SMPL_BONE_ORDER.index('right_foot')

        # 注意：这里用插值后的smpl_joints_scaled，已经包含了waist_offset
        # 左腿
        shin_l = smpl_joints_scaled[:, idx_l_ankle] - smpl_joints_scaled[:, idx_l_knee]   # 小腿方向 [N,3]
        foot_l = smpl_joints_scaled[:, idx_l_foot]  - smpl_joints_scaled[:, idx_l_ankle]  # 脚掌方向 [N,3]

        shin_l /= np.linalg.norm(shin_l, axis=-1, keepdims=True) + 1e-6
        foot_l /= np.linalg.norm(foot_l, axis=-1, keepdims=True) + 1e-6

        cos_l = np.clip(np.sum(shin_l * foot_l, axis=-1), -1.0, 1.0)  # [N]
        angle_l = np.arccos(cos_l)          # 绝对夹角，竖直站立时约为π
        # 右腿
        shin_r = smpl_joints_scaled[:, idx_r_ankle] - smpl_joints_scaled[:, idx_r_knee]
        foot_r = smpl_joints_scaled[:, idx_r_foot]  - smpl_joints_scaled[:, idx_r_ankle]

        shin_r /= np.linalg.norm(shin_r, axis=-1, keepdims=True) + 1e-6
        foot_r /= np.linalg.norm(foot_r, axis=-1, keepdims=True) + 1e-6

        cos_r = np.clip(np.sum(shin_r * foot_r, axis=-1), -1.0, 1.0)
        angle_r = np.arccos(cos_r)
        
        neutral_angle_l = np.arccos(cos_l[0]) 
        neutral_angle_r = np.arccos(cos_r[0])
        ankle_pitch_l = angle_l - neutral_angle_l     # 以π为基准，站立时≈0，跖屈为正，背屈为负
        ankle_pitch_r = angle_r - neutral_angle_r
        lo, hi = H1Config.JOINT_LIMITS[5]   # [-0.79, 0.79]
        ankle_pitch_l = np.clip(ankle_pitch_l, lo, hi)
        ankle_pitch_r = np.clip(ankle_pitch_r, lo, hi)
        sim_dof_deg[:, 5]  = np.rad2deg(ankle_pitch_l)
        sim_dof_deg[:, 11] = np.rad2deg(ankle_pitch_r)
        print("Applying start transition...")
        transition_frames = 20  # 100fps下=0.2秒，可调
        stand_deg = np.rad2deg(H1Config.STAND_JOINT_ANGLES)  # [28] 站立姿态角度
        first_frame = sim_dof_deg[0]  # [28] 动作第一帧
        transition = np.zeros((transition_frames, sim_dof_deg.shape[1]))
        for i in range(transition_frames):
            alpha = i / transition_frames  # 0→1 线性插值
            transition[i] = (1 - alpha) * stand_deg + alpha * first_frame

        N_total = len(sim_dof_deg)
        print(f"Total time: {time.time() - start_time:.2f}s")
        return {
            "dof": np.deg2rad(sim_dof_deg),
            "fps": 90,
            "input_frames": N_orig,   # 输入总帧数（插值前）
            "output_frames": N,       # 输出总帧数（插值后）
            "root_trans_offset": np.zeros((N, 3)),
            "root_rot": np.tile(np.array([0,0,0,1]), (N,1)),

            "smpl_joints_target": np.array(smpl_joints_target_seq),
            "h1_joint_pos": np.array(h1_cartesian_seq)
        }
    
    def apply_bone_retargeting_single(
            self,
            smpl_joints,        # [1, J, 3] 当前帧 SMPL 关节坐标
            robot_dims,         # 机器人物理尺寸字典
            robot_l_shoulder,   # [3] 当前帧机器人左肩真实位置
            robot_r_shoulder,   # [3] 当前帧机器人右肩真实位置
            robot_l_thigh,      # [3] 当前帧机器人左髋真实位置
            robot_r_thigh,      # [3] 当前帧机器人右髋真实位置
            robot_neck
        ):
        """
        对单帧 SMPL 骨骼进行重定向：保留方向，替换长度。
        anchor 点（肩膀/髋部）使用机器人当前帧的真实位置，
        这样躯干运动（弯腰、侧倾等）能被正确传递到四肢。

        Returns:
            new_joints: [1, J, 3]
        """
        new_joints = smpl_joints.copy() 
        # 获取关节索引 (请确保这些名称与 Config 中的一致)
        idx_l_hip = self.bone_idx['left_hip']
        idx_l_knee = self.bone_idx['left_knee']
        idx_l_ankle = self.bone_idx['left_ankle']
        
        idx_r_hip = self.bone_idx['right_hip']
        idx_r_knee = self.bone_idx['right_knee']
        idx_r_ankle = self.bone_idx['right_ankle']

        idx_l_shoulder = self.bone_idx['left_shoulder']
        idx_l_elbow = self.bone_idx['left_elbow']
        idx_l_wrist = self.bone_idx['left_wrist']

        idx_r_shoulder = self.bone_idx['right_shoulder']
        idx_r_elbow = self.bone_idx['right_elbow']
        idx_r_wrist = self.bone_idx['right_wrist']

        idx_neck = self.bone_idx['neck']
        idx_head = self.bone_idx['head']
        def get_direction(vec):
            """归一化向量，shape: [..., 3]"""
            norm = np.linalg.norm(vec, axis=-1, keepdims=True)
            return vec / (norm + 1e-6)

        vec_head = smpl_joints[:, idx_head] - smpl_joints[:, idx_neck]
        dir_head = get_direction(vec_head)
        new_joints[:, idx_neck] = robot_neck                                              # anchor
        new_joints[:, idx_head] = robot_neck + dir_head * robot_dims['neck2head']
        # === 开始链式修正 ===
        # --- 1. 左腿 ---
        # 修正左膝盖 (基于左髋 -> 膝盖的方向)
        # 注意：我们假设 Hip 的位置是准确的(由全局缩放决定)，不动 Hip，只动 Knee
        vec_l_thigh = smpl_joints[:, idx_l_knee] - smpl_joints[:, idx_l_hip]
        dir_l_thigh = get_direction(vec_l_thigh)
        new_joints[:, idx_l_hip] = robot_l_thigh
        new_joints[:, idx_l_knee] = robot_l_thigh + dir_l_thigh * robot_dims['thigh']

        # 修正左脚踝 (基于 新的左膝盖 -> 脚踝的方向)
        vec_l_calf = smpl_joints[:, idx_l_ankle] - smpl_joints[:, idx_l_knee] # 取旧方向
        dir_l_calf = get_direction(vec_l_calf)
        new_joints[:, idx_l_ankle] = new_joints[:, idx_l_knee] + dir_l_calf * robot_dims['calf'] # 用新膝盖位置

        # --- 2. 右腿 ---
        # 修正右膝盖
        vec_r_thigh = smpl_joints[:, idx_r_knee] - smpl_joints[:, idx_r_hip]
        dir_r_thigh = get_direction(vec_r_thigh)
        new_joints[:, idx_r_hip] = robot_r_thigh
        new_joints[:, idx_r_knee] = robot_r_thigh + dir_r_thigh * robot_dims['thigh']

        # 修正右脚踝
        vec_r_calf = smpl_joints[:, idx_r_ankle] - smpl_joints[:, idx_r_knee]
        dir_r_calf = get_direction(vec_r_calf)
        new_joints[:, idx_r_ankle] = new_joints[:, idx_r_knee] + dir_r_calf * robot_dims['calf']

        # --- 3. 左臂 ---
        # 修正左肘 (基于 Shoulder)
        vec_l_arm = smpl_joints[:, idx_l_elbow] - smpl_joints[:, idx_l_shoulder]
        dir_l_arm = get_direction(vec_l_arm)
        new_joints[:, idx_l_shoulder] = robot_l_shoulder
        new_joints[:, idx_l_elbow] = robot_l_shoulder + dir_l_arm * robot_dims['upper_arm']
        # 修正左手腕 (基于 新的左肘)
        vec_l_forearm = smpl_joints[:, idx_l_wrist] - smpl_joints[:, idx_l_elbow]
        dir_l_forearm = get_direction(vec_l_forearm)
        new_joints[:, idx_l_wrist] = new_joints[:, idx_l_elbow] + dir_l_forearm * robot_dims['forearm']

        # --- 4. 右臂 ---
        # 修正右肘
        vec_r_arm = smpl_joints[:, idx_r_elbow] - smpl_joints[:, idx_r_shoulder]
        dir_r_arm = get_direction(vec_r_arm)
        new_joints[:, idx_r_shoulder] = robot_r_shoulder
        new_joints[:, idx_r_elbow] = robot_r_shoulder + dir_r_arm * robot_dims['upper_arm']

        # 修正右手腕
        vec_r_forearm = smpl_joints[:, idx_r_wrist] - smpl_joints[:, idx_r_elbow]
        dir_r_forearm = get_direction(vec_r_forearm)
        new_joints[:, idx_r_wrist] = new_joints[:, idx_r_elbow] + dir_r_forearm * robot_dims['forearm']

        return new_joints


def load_data(data_path):
    entry_data = np.load(open(data_path, "rb"), allow_pickle=True) 
    return entry_data

def load_data_all_npy(data_path, batch_id):
    data = np.load(data_path, allow_pickle=True).item()
    motion = data['motion'][batch_id]          # [22, 3, T]
    motion = motion.transpose(2, 0, 1)         # [T, 22, 3]
    head_rot       = data['head_rot_mats'][batch_id]        # [T, 3, 3]
    left_wrist_rot = data['left_wrist_rot_mats'][batch_id]  # [T, 3, 3]
    right_wrist_rot= data['right_wrist_rot_mats'][batch_id] # [T, 3, 3]
    return motion, head_rot, left_wrist_rot, right_wrist_rot

if __name__ == "__main__":
    urdf_path = "/home/syx/tw-robot/retarget/urdf/Assembly.urdf"
    data_path = "/home/syx/tw-robot/retarget/input/results.npy"
    output_path = "/home/syx/tw-robot/retarget/output/result.pkl"
    
    amass_data,head_rot,_,_ = load_data_all_npy(data_path,14)
    # print(head_rot.shape)
    if amass_data is not None:
        solver = H1PinkSolver(urdf_path)
        result = solver.process_motion(amass_data, head_rot)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(result, output_path)
        print(f"Saved to {output_path}")
    else:
        print("Failed to load Amass Data")