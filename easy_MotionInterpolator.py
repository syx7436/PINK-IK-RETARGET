"""
两段动作之间的平滑过渡（基于 Pinocchio 几何碰撞检测 + 动态中继姿态）

改进点（相比旧版）：
- 移除 torch / PHC 依赖，改用 Pinocchio + hpp-fcl 做自碰撞检测
- 碰撞检测使用 URDF 真实碰撞几何体，不再依赖手工维护的点距离对
- 安全中继姿态由 Pinocchio neutral + 站立肩部角度动态生成
- 碰撞检测改为沿过渡轨迹逐帧采样，而非只检测完整过渡序列
"""

import joblib
import numpy as np
import pinocchio as pin
import os
import time


# ---------------------------------------------------------------------------
# H1 关节顺序（与 pkl 中 dof 数组对应，单位: 弧度）
# ---------------------------------------------------------------------------
H1_JOINT_NAMES = [
    # Left Leg (0-5)
    'left_hip_pitch_joint',   'left_hip_roll_joint',   'left_hip_yaw_joint',
    'left_knee_pitch_joint',  'left_ankle_yaw_joint',  'left_ankle_pitch_joint',
    # Right Leg (6-11)
    'right_hip_pitch_joint',  'right_hip_roll_joint',  'right_hip_yaw_joint',
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

# 站立安全姿态：肩部内收（pitch ±90°），其余归零
# 此配置作为动态中继姿态，经验证在 H1 上不会发生自碰撞
_RELAY_DOF = np.zeros(len(H1_JOINT_NAMES), dtype=np.float64)
_RELAY_DOF[15] = -1.5707963  # left_shoulder_pitch
_RELAY_DOF[20] =  1.5707963  # right_shoulder_pitch


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time  = self.start_time

    def start(self, name="section"):
        self.section_name = name
        self.last_time = time.time()
        print(f"\n--- Start [{name}] ---")

    def step(self, msg):
        now = time.time()
        print(f"[{self.section_name}] +{now - self.last_time:.4f}s : {msg}")
        self.last_time = now

    def total(self):
        print(f"\n=== Total Time: {time.time() - self.start_time:.4f}s ===\n")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class HumanoidMotionInterpolator:
    """
    两段动作的碰撞感知过渡。

    Args:
        urdf_path:       H1 URDF 文件路径
        transition_time: 最大过渡时长（秒），默认 1.0
        path_samples:    碰撞检测沿路径的采样帧数，默认 20
    """

    def __init__(self, urdf_path: str, path_samples: int = 20):
        self.path_samples = path_samples
        self._init_pinocchio(urdf_path)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _init_pinocchio(self, urdf_path: str):
        """加载运动学模型与碰撞几何模型，构建 dof→q 下标映射。"""
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, package_dirs=[os.path.dirname(urdf_path)]
        )
        self.model = self.robot.model
        self.data  = self.robot.data

        # dof 数组下标 → pinocchio q 向量下标
        self.dof_to_qidx = []
        for name in H1_JOINT_NAMES:
            if self.model.existJointName(name):
                self.dof_to_qidx.append(
                    self.model.joints[self.model.getJointId(name)].idx_q
                )
            else:
                print(f"警告: URDF 中找不到关节 {name}")
                self.dof_to_qidx.append(None)

        self._load_geometry_collision(urdf_path)
        self._build_relay_dof()

    def _load_geometry_collision(self, urdf_path: str):
        """从 URDF 加载几何碰撞模型（需要 hpp-fcl）。"""
        geom_model = pin.GeometryModel()
        pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION,
            geom_model, package_dirs=[os.path.dirname(urdf_path)]
        )
        geom_model.addAllCollisionPairs()
        self.geom_model = geom_model
        self.geom_data  = pin.GeometryData(geom_model)
        print(f"几何碰撞检测已启用: {len(geom_model.collisionPairs)} 碰撞对")

    def _build_relay_dof(self):
        """计算并验证中继安全姿态（站立肩部内收）。"""
        self.relay_dof = _RELAY_DOF.copy()
        q = self._dof_to_q(self.relay_dof)
        if self._check_config_collision(q):
            print("警告: 默认中继姿态检测到碰撞，退回 pinocchio neutral")
            self.relay_dof = np.zeros(len(H1_JOINT_NAMES), dtype=np.float64)
        else:
            print("中继姿态验证通过（无碰撞）")

    # ------------------------------------------------------------------
    # 碰撞检测
    # ------------------------------------------------------------------
    def _dof_to_q(self, dof: np.ndarray) -> np.ndarray:
        """28-DOF 数组（弧度）→ pinocchio q 向量。"""
        q = pin.neutral(self.model).copy()
        for i, idx in enumerate(self.dof_to_qidx):
            if idx is not None:
                q[idx] = dof[i]
        return q

    def _check_config_collision(self, q: np.ndarray) -> bool:
        """检测单个配置是否自碰撞。"""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(
            self.model, self.data, self.geom_model, self.geom_data
        )
        pin.computeCollisions(
            self.geom_model, self.geom_data, stop_at_first_collision=True
        )
        return any(
            self.geom_data.collisionResults[i].isCollision()
            for i in range(len(self.geom_model.collisionPairs))
        )

    def _check_path_collision(self, dof_start: np.ndarray, dof_end: np.ndarray) -> bool:
        """沿 dof_start → dof_end 的五次多项式路径采样，检测是否存在碰撞帧。"""
        for w in self._quintic_weights(self.path_samples):
            dof = (1.0 - w) * dof_start + w * dof_end
            if self._check_config_collision(self._dof_to_q(dof)):
                return True
        return False

    # ------------------------------------------------------------------
    # 插值工具
    # ------------------------------------------------------------------
    @staticmethod
    def _quintic_weights(n: int) -> np.ndarray:
        """返回 n 个均匀采样的五次多项式权重（0→1），起止速度/加速度为 0。"""
        t = np.linspace(0.0, 1.0, n)
        return 6*t**5 - 15*t**4 + 10*t**3

    @staticmethod
    def _quintic_interp(q_start: np.ndarray, q_end: np.ndarray, num_frames: int) -> np.ndarray:
        """五次多项式插值，起止速度/加速度均为 0。"""
        w = HumanoidMotionInterpolator._quintic_weights(num_frames)[:, None]
        return (1.0 - w) * q_start + w * q_end

    @staticmethod
    def _smooth_boundary(dof_full: np.ndarray, window: int = 7) -> np.ndarray:
        """对过渡前后各 2*window 帧做滑动均值平滑，消除拼接处抖动。"""
        T, _ = dof_full.shape
        half  = window // 2
        result = dof_full.copy()
        for i in list(range(window * 2)) + list(range(T - window * 2, T)):
            lo = max(0, i - half)
            hi = min(T, i + half + 1)
            result[i] = dof_full[lo:hi].mean(axis=0)
        return result

    # ------------------------------------------------------------------
    # FPS 重采样
    # ------------------------------------------------------------------
    def _unify_fps(self, action_A, action_B, index_A, index_B, target_fps):
        """将两段动作统一到同一 fps（取两者与 target_fps 的最大值）。"""
        work_fps = max(target_fps, action_A["fps"], action_B["fps"])
        if action_A["fps"] != work_fps:
            action_A, index_A = self.resample_action_to_fps(action_A, work_fps, index_A)
        if action_B["fps"] != work_fps:
            action_B, index_B = self.resample_action_to_fps(action_B, work_fps, index_B)
        return action_A, action_B, index_A, index_B, work_fps

    def resample_action_to_fps(self, action: dict, target_fps: int, index=None):
        """
        将 action 的 dof / root_trans_offset / root_rot 插值到 target_fps。
        若提供 index，同步映射到新帧序号并返回。
        """
        src_fps = action["fps"]
        dof     = action["dof"]
        T_old   = dof.shape[0]

        duration = (T_old - 1) / src_fps
        T_new    = int(np.round(duration * target_fps)) + 1
        t_old    = np.linspace(0.0, duration, T_old)
        t_new    = np.linspace(0.0, duration, T_new)

        dof_new = np.stack([np.interp(t_new, t_old, dof[:, j])
                            for j in range(dof.shape[1])], axis=1)

        index_new = None
        if index is not None:
            index_new = int(np.argmin(np.abs(t_new - t_old[index])))

        new_action = dict(action)
        new_action.update(fps=target_fps, dof=dof_new,
                          root_trans_offset=np.zeros((T_new, 3), dtype=np.float32),
                          root_rot=np.tile([0, 0, 0, 1], (T_new, 1)).astype(np.float32))
        return new_action, index_new

    # ------------------------------------------------------------------
    # 过渡时长估算
    # ------------------------------------------------------------------
    def _compute_transition_time(
        self,
        dof_s, dof_e,
        min_time: float,
        max_time: float,
    ) -> float:
        """根据关节角差异估算过渡时长，结果在 [min_time, max_time] 内。"""
        diff = min(np.linalg.norm(dof_s - dof_e) / 2.0, 1.0)
        return min_time + diff * (max_time - min_time)

    # ------------------------------------------------------------------
    # 核心过渡生成
    # ------------------------------------------------------------------
    def _make_relay_action(self, fps: int) -> dict:
        """生成单帧中继动作（站立安全姿态）。"""
        return {
            "fps":               fps,
            "dof":               self.relay_dof[np.newaxis].copy(),
            "root_trans_offset": np.zeros((1, 3), dtype=np.float32),
            "root_rot":          np.array([[0, 0, 0, 1]], dtype=np.float32),
        }

    def concatenate_actions(
        self,
        action_A: dict,
        action_B: dict,
        is_smooth:       bool  = True,
        transition_time: float = None,
        target_fps:      int   = 20,
        index_A:         int   = None,
        index_B:         int   = None,
        min_time:        float = 0.5,
        max_time:        float = 1.0,
    ) -> dict:
        """
        直接拼接两段动作（不做碰撞检测）。
        transition_time=None 时根据 min_time/max_time 自动估算。
        """
        if index_A is None: index_A = len(action_A["dof"]) - 1
        if index_B is None: index_B = 0

        action_A, action_B, index_A, index_B, work_fps = self._unify_fps(
            action_A, action_B, index_A, index_B, target_fps
        )

        if transition_time is None:
            transition_time = self._compute_transition_time(
                action_A["dof"][index_A], action_B["dof"][index_B], min_time, max_time,
            )

        num_frames = max(2, int(transition_time * work_fps))
        T_dof = self._quintic_interp(action_A["dof"][index_A],
                                     action_B["dof"][index_B], num_frames)

        dof_full = np.concatenate([action_A["dof"][:index_A+1],
                                   T_dof[1:],
                                   action_B["dof"][index_B+1:]], axis=0)
        if is_smooth:
            dof_full = self._smooth_boundary(dof_full)

        N = len(dof_full)
        return {
            "fps":               work_fps,
            "dof":               dof_full,
            "root_trans_offset": np.zeros((N, 3), dtype=np.float32),
            "root_rot":          np.tile([0, 0, 0, 1], (N, 1)).astype(np.float32),
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(
        self,
        action_A:  dict,
        action_B:  dict,
        is_smooth: bool  = True,
        target_fps: int  = 20,
        min_time:  float = 0.5,
        max_time:  float = 1.0,
    ) -> dict:
        """
        带碰撞感知的动作过渡：action_A 最后一帧 → action_B 第一帧。

        min_time / max_time: 过渡时长的下界和上界（秒），
            实际时长根据两段动作差异在此范围内自动估算。
        """
        index_A = len(action_A["dof"]) - 1
        index_B = 0

        action_A, action_B, index_A, index_B, work_fps = self._unify_fps(
            action_A, action_B, index_A, index_B, target_fps
        )

        dof_start = action_A["dof"][index_A]
        dof_end   = action_B["dof"][index_B]

        def calc_time(a, ia, b, ib):
            return self._compute_transition_time(
                a["dof"][ia], b["dof"][ib], min_time, max_time,
            )

        if not self._check_path_collision(dof_start, dof_end):
            print("无碰撞，直接过渡")
            return self.concatenate_actions(
                action_A, action_B, is_smooth, calc_time(action_A, index_A, action_B, index_B),
                work_fps, index_A, index_B,
            )

        print("检测到碰撞，使用动态中继姿态过渡")
        relay = self._make_relay_action(work_fps)

        part1 = self.concatenate_actions(
            action_A, relay, is_smooth, calc_time(action_A, index_A, relay, 0),
            work_fps, index_A, 0,
        )
        part2 = self.concatenate_actions(
            relay, action_B, is_smooth, calc_time(relay, 0, action_B, index_B),
            work_fps, 0, index_B,
        )

        return {
            "fps":               work_fps,
            "dof":               np.concatenate([part1["dof"],
                                                 part2["dof"][1:]], axis=0),
            "root_trans_offset": np.concatenate([part1["root_trans_offset"],
                                                 part2["root_trans_offset"][1:]], axis=0),
            "root_rot":          np.concatenate([part1["root_rot"],
                                                 part2["root_rot"][1:]], axis=0),
        }


# ---------------------------------------------------------------------------
# 入口示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    timer = Timer()
    timer.start("初始化")

    urdf_path = "/home/syx/tw-robot/retarget/Interpolation/data/Assembly/Assembly.SLDASM.urdf"
    interpolator = HumanoidMotionInterpolator(urdf_path)
    timer.step("加载插值器完成")

    action_A = joblib.load("/home/syx/tw-robot/retarget/Interpolation/input/right-back.pkl")
    print("动作A帧数:", len(action_A["dof"]))
    action_B = joblib.load("/home/syx/tw-robot/retarget/Interpolation/input/right-front.pkl")
    print("动作B帧数:", len(action_B["dof"]))
    timer.step("导入文件完成")

    full_action = interpolator.run(
        action_A,
        action_B,
        is_smooth=True,
        target_fps=20,
        min_time=0.5,  # 过渡时长下界（秒）
        max_time=1.0,  # 过渡时长上界（秒），实际时长根据动作差异自动估算
    )
    print("合并后总帧数:", len(full_action["dof"]))
    timer.step("生成合并轨迹完成")

    joblib.dump(full_action, "/home/syx/tw-robot/retarget/output/result.pkl")
    timer.total()
