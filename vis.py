"""
vis.py —— 机器人动作可视化脚本

功能：
    从 .pkl 文件中读取重定向后的机器人动作数据（DOF序列、根节点位姿等），
    在 Isaac Gym 中逐帧播放机器人动作，并叠加显示：
      - 绿色线框球：H1 机器人关节坐标
      - 蓝色线框球 + 黄色骨架线：SMPL 目标关节点
      - 红色线框盒：障碍物
"""

import os
import sys
import time

sys.path.append(os.getcwd())

from isaacgym import gymapi, gymutil, gymtorch

import joblib
import numpy as np
import torch

# ============================================================
# 全局配置
# ============================================================

DATA_PATH = 'data/output/result_for_ik.pkl'
URDF_ROOT = 'data/urdf'
URDF_FILE = 'Assembly.urdf'

# URDF Mesh 与物理质心之间的固定视觉偏移量（经验值）
URDF_VISUAL_OFFSET = torch.tensor([0.0, 0.0, 0.0008981])

# H1 关节自定义顺序（与 .pkl 中 dof 数组的列顺序对应）
CUSTOM_DOF_ORDER = [
    # 左腿（6个关节）
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_pitch_joint',
    'left_ankle_yaw_joint',
    'left_ankle_pitch_joint',
    # 右腿（6个关节）
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_pitch_joint',
    'right_ankle_yaw_joint',
    'right_ankle_pitch_joint',
    # 腰部（3个关节）
    'waist_yaw_joint',
    'waist_pitch_joint',
    'waist_roll_joint',
    # 左臂（5个关节）
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint',
    'left_wrist_yaw_joint',
    # 右臂（5个关节）
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint',
    'right_wrist_yaw_joint',
    # 头部（3个关节）
    'neck_yaw_joint',
    'neck_roll_joint',
    'neck_pitch_joint',
]

# SMPL 骨架父子连接对（用于绘制黄色骨架线）
SMPL_SKELETON_PAIRS = [
    (0, 1), (0, 2), (0, 3),       # 根节点 -> 左髋、右髋、脊柱1
    (1, 4), (2, 5),               # 髋 -> 膝
    (3, 6),                       # 脊柱1 -> 脊柱2
    (4, 7), (5, 8),               # 膝 -> 踝
    (6, 9),                       # 脊柱2 -> 脊柱3
    (7, 10), (8, 11),             # 踝 -> 脚
    (9, 12), (9, 13), (9, 14),    # 脊柱3 -> 颈、左锁骨、右锁骨
    (12, 15),                     # 颈 -> 头
    (13, 16), (14, 17),           # 锁骨 -> 肩
    (16, 18), (17, 19),           # 肩 -> 肘
    (18, 20), (19, 21),           # 肘 -> 腕
]


# ============================================================
# 数据加载
# ============================================================

def load_motion_data(data_path):
    """
    从 .pkl 文件加载重定向后的动作数据。

    Returns:
        dict，包含：
            root_trans_offset   [T, 3]  —— 根节点世界坐标（含偏移）
            dof                 [T, 28] —— 各关节角度（rad）
            root_rot            [T, 4]  —— 根节点四元数（xyzw）
            fps                 float   —— 动画帧率
            h1_joint_pos        [T, J, 3] 或 None —— H1 关节世界坐标
            smpl_joints_target  [T, 24, 3] 或 None —— SMPL 目标关节点
            obstacles           list    —— 障碍物列表，每项 [cx,cy,cz,lx,ly,lz]
            scale               float   —— SMPL 到机器人的缩放比例
    """
    transition = joblib.load(data_path)
    print(f"[数据加载] DOF 序列形状: {transition['dof'].shape}")
    return {
        "root_trans_offset":  transition["root_trans_offset"],
        "dof":                transition["dof"],
        "root_rot":           transition["root_rot"],
        "fps":                transition["fps"],
        "h1_joint_pos":       transition.get("h1_joint_pos", None),
        "smpl_joints_target": transition.get("smpl_joints_target", None),
        "obstacles":          transition.get("obstacles", []),
        "scale":              transition.get("scale", 1.0),
    }


# ============================================================
# Isaac Gym 初始化
# ============================================================

def create_sim(args, fps):
    """创建 Isaac Gym 模拟环境，返回 gym 句柄和 sim 对象。"""
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / fps        # 与动画帧率保持一致
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 6
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if not args.use_gpu_pipeline:
        print("[警告] 强制使用 CPU 管线。")

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params
    )
    if sim is None:
        print("[错误] 创建模拟环境失败。")
        quit()

    return gym, sim


def create_viewer(gym, sim):
    """创建可视化窗口，返回 viewer 句柄。"""
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("[错误] 创建视图失败。")
        quit()

    # 设置初始相机位置和朝向
    cam_pos    = gymapi.Vec3(1.2, -0.5, 0)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    return viewer


def setup_ground_plane(gym, sim):
    """定义地面平面参数（仅配置法向量，不启用碰撞，避免与显式根节点状态设置冲突）。"""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)


def load_robot_asset(gym, sim):
    """加载机器人 URDF 资产，返回 asset 对象。"""
    asset_options = gymapi.AssetOptions()
    asset_options.use_mesh_materials = True  # 显示材质贴图

    print(f"[资产加载] 加载 '{URDF_FILE}' 来自 '{URDF_ROOT}'")
    asset = gym.load_asset(sim, URDF_ROOT, URDF_FILE, asset_options)
    return asset


def create_envs(gym, sim, asset, num_envs=1):
    """
    创建仿真环境并在其中放置机器人 Actor。

    Returns:
        envs, actor_handles —— 环境列表和 Actor 句柄列表
    """
    spacing = 5
    env_lower = gymapi.Vec3(-spacing,  spacing, 0)
    env_upper = gymapi.Vec3( spacing,  spacing, spacing)

    envs, actor_handles = [], []
    for i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_envs)

        # 机器人初始位姿（原点，无旋转）
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
        envs.append(env)
        actor_handles.append(actor_handle)

        # 初始化 DOF 状态为全零
        num_dofs = gym.get_asset_dof_count(asset)
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    print(f"[环境] 创建了 {num_envs} 个环境，机器人 DOF 数: {num_dofs}")
    return envs, actor_handles


# ============================================================
# 可视化几何体构建
# ============================================================

def build_obstacle_geoms(obstacles):
    """
    根据障碍物数据构建红色线框盒几何体列表。

    Args:
        obstacles: list，每项格式为 [cx, cy, cz, lx, ly, lz]

    Returns:
        list of (geom, pose) 元组
    """
    geom_list = []
    for obs in obstacles:
        center = obs[:3]   # 中心坐标 (x, y, z)
        size   = obs[3:]   # 尺寸     (lx, ly, lz)
        geom = gymutil.WireframeBoxGeometry(size[0], size[1], size[2], None, color=(1, 0, 0))
        pose = gymapi.Transform(gymapi.Vec3(*center), r=None)
        geom_list.append((geom, pose))

    if geom_list:
        print(f"[障碍物] 共加载 {len(geom_list)} 个障碍物。")
    return geom_list


# ============================================================
# 主程序
# ============================================================

def main():
    # ---------- 加载动作数据 ----------
    amass_data = load_motion_data(DATA_PATH)
    N   = amass_data["dof"].shape[0]   # 总帧数
    fps = amass_data["fps"]
    dt  = 1.0 / fps
    print(f"[数据] 总帧数: {N},  帧率: {fps} fps")

    # ---------- 解析命令行参数 ----------
    class AssetDesc:
        def __init__(self, file_name, flip_visual_attachments=False):
            self.file_name = file_name
            self.flip_visual_attachments = flip_visual_attachments

    asset_descriptors = [AssetDesc(URDF_FILE, False)]

    args = gymutil.parse_arguments(
        description="Joint monkey: Animate degree-of-freedom ranges",
        custom_parameters=[
            {"name": "--asset_id",    "type": int,   "default": 0,   "help": "资产ID"},
            {"name": "--speed_scale", "type": float, "default": 1.0, "help": "动画播放速度"},
            {"name": "--show_axis",   "action": "store_true",        "help": "显示DOF轴"},
        ]
    )

    if not (0 <= args.asset_id < len(asset_descriptors)):
        print(f"[错误] 无效的 asset_id，有效范围: 0 ~ {len(asset_descriptors) - 1}")
        quit()

    # ---------- 初始化 Isaac Gym ----------
    gym, sim = create_sim(args, fps)
    setup_ground_plane(gym, sim)
    viewer  = create_viewer(gym, sim)
    asset   = load_robot_asset(gym, sim)
    envs, actor_handles = create_envs(gym, sim, asset, num_envs=1)

    env          = envs[0]
    actor_handle = actor_handles[0]

    # ---------- 计算 DOF 重排映射 ----------
    # Isaac Gym 中关节顺序可能与我们自定义的顺序不同，需要做列重排
    dof_names         = gym.get_actor_dof_names(env, actor_handle)
    dof_order_mapping = [CUSTOM_DOF_ORDER.index(name) for name in dof_names]

    # ---------- 将动作序列转移到设备 ----------
    gym.prepare_sim(sim)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    urdf_visual_offset = URDF_VISUAL_OFFSET.to(device)

    dof_seq              = torch.from_numpy(amass_data["dof"]).float().to(device)             # [N, 28]
    root_rot_seq         = torch.from_numpy(amass_data["root_rot"]).float().to(device)        # [N, 4]
    root_trans_seq       = torch.from_numpy(amass_data["root_trans_offset"]).float().to(device)  # [N, 3]

    # 生成环境 ID 张量（用于 indexed set）
    env_ids = torch.arange(len(envs)).int().to(args.sim_device)

    # ---------- 加载可选可视化数据 ----------
    h1_joint_pos_seq = amass_data.get("h1_joint_pos")    # [N, J, 3] 或 None
    smpl_target_seq  = amass_data.get("smpl_joints_target")  # [N, 24, 3] 或 None

    if h1_joint_pos_seq is not None:
        print(f"[可视化] H1 关节坐标序列形状: {h1_joint_pos_seq.shape}")
    if smpl_target_seq is not None:
        print(f"[可视化] SMPL 目标点序列形状: {smpl_target_seq.shape}")

    # ---------- 构建可视化几何体 ----------
    # 绿色小球：H1 机器人关节
    joint_sphere_geom   = gymutil.WireframeSphereGeometry(radius=0.01,  num_lats=6, num_lons=6, color=(0, 1, 0))
    # 红色极小球：关节中心标记
    center_marker_geom  = gymutil.WireframeSphereGeometry(radius=0.001, num_lats=4, num_lons=4, color=(1, 0, 0))
    # 蓝色小球：SMPL 目标关节
    smpl_sphere_geom    = gymutil.WireframeSphereGeometry(radius=0.013, num_lats=6, num_lons=6, color=(0, 0, 1))

    # 障碍物线框盒（红色）
    obstacle_geoms = build_obstacle_geoms(amass_data.get("obstacles", []))

    # ============================================================
    # 主循环：逐帧播放动作
    # ============================================================
    frame_idx  = 0
    time_step  = 0.0

    while not gym.query_viewer_has_closed(viewer):

        # ---------- 清空上一帧的线框绘制 ----------
        gym.clear_lines(viewer)

        # ---------- 绘制障碍物（红色线框盒） ----------
        for geom, pose in obstacle_geoms:
            gymutil.draw_lines(geom, gym, viewer, env, pose)

        # ---------- 绘制 H1 机器人关节点（绿色 + 红色中心标记） ----------
        if h1_joint_pos_seq is not None:
            current_joints = h1_joint_pos_seq[frame_idx]   # [J, 3]
            for j in range(31):
                pos  = current_joints[j]
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                pose.r = gymapi.Quat(0, 0, 0, 1)
                gymutil.draw_lines(joint_sphere_geom,  gym, viewer, env, pose)
                gymutil.draw_lines(center_marker_geom, gym, viewer, env, pose)

        # ---------- 绘制 SMPL 目标关节点（蓝色球 + 黄色骨架线） ----------
        if smpl_target_seq is not None:
            current_smpl = smpl_target_seq[frame_idx]   # [24, 3]

            # A. 蓝色关节球（跳过 22、23 号手部关节）
            for j in range(22):
                if j in (22, 23):
                    continue
                pos  = current_smpl[j]
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                pose.r = gymapi.Quat(0, 0, 0, 1)
                gymutil.draw_lines(smpl_sphere_geom, gym, viewer, env, pose)

            # B. 黄色骨架连线
            for (p_idx, c_idx) in SMPL_SKELETON_PAIRS:
                start = current_smpl[p_idx]
                end   = current_smpl[c_idx]
                gym.add_lines(
                    viewer, env, 1,
                    [float(start[0]), float(start[1]), float(start[2]),
                     float(end[0]),   float(end[1]),   float(end[2])],
                    [1, 1, 0]   # 黄色
                )

        # ---------- 设置机器人根节点状态 ----------
        root_trans  = root_trans_seq[frame_idx].unsqueeze(0)   # [1, 3]
        root_rot    = root_rot_seq[frame_idx].unsqueeze(0)     # [1, 4]
        zeros_vel   = torch.zeros_like(root_trans)

        # 叠加视觉偏移（修正 Mesh 与物理质心的位置差）
        corrected_trans = root_trans + urdf_visual_offset

        # Isaac Gym Actor 根状态格式：[pos(3), quat(4), lin_vel(3), ang_vel(3)]
        root_states = torch.cat(
            [corrected_trans, root_rot, zeros_vel, zeros_vel], dim=-1
        ).repeat(len(envs), 1)   # [num_envs, 13]

        gym.set_actor_root_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids)
        )

        # ---------- 设置机器人关节角度 ----------
        dof_pos      = dof_seq[frame_idx].unsqueeze(0)                # [1, 28]
        dof_pos_isaac = dof_pos[:, dof_order_mapping]                 # 按 URDF 顺序重排
        zeros_vel_dof = torch.zeros_like(dof_pos_isaac)

        # DOF 状态格式：[pos, vel]，展开为 [num_envs, num_dofs, 2]
        dof_state = torch.stack(
            [dof_pos_isaac, zeros_vel_dof], dim=-1
        ).squeeze(0).repeat(len(envs), 1)

        gym.set_dof_state_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(dof_state),
            gymtorch.unwrap_tensor(env_ids),
            len(env_ids)
        )

        # ---------- 推进物理模拟并渲染 ----------
        gym.simulate(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)   # 同步渲染帧率与物理帧率

        time_step += dt
        frame_idx = (frame_idx + 1) % N   # 循环播放

    # ---------- 清理资源 ----------
    print("可视化结束。")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()