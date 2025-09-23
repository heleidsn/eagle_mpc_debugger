# MPC状态消息
Header header

# 状态向量
float64[] state          # 完整状态向量
float64[] state_ref      # 参考状态向量
float64[] state_ref_next  # 参考状态向量（下一个时刻）
float64[] state_error    # 状态误差

# MPC debug information
int32 mpc_time_step      # 迭代位置
int32 mpc_iter_num          # 迭代次数
float64 solving_time     # 求解时间
float64 mpc_final_cost   # final cost
float64 mpc_start_cost   # final cost with reference

float64[] u_mpc           # MPC控制指令

# L1 debug information
float64[] u_ad           # L1控制指令

float64[] z_ref          # 参考状态
float64[] z_hat          # 估计状态
float64[] z_real         # 实际状态

float64[] sig_hat        # 估计扰动
float64[] z_tilde        # 状态误差
float64[] z_tilde_ref        # 与参考状态的误差
float64[] z_tilde_tracking   # for tracking error

# tracking controller
float64[] u_tracking     # 轨迹跟踪控制指令

# gripper information
float64[] gripper_position     # gripper position in world frame [x, y, z]
float64[] gripper_orientation  # gripper orientation in world frame [qx, qy, qz, qw]
float64[] gripper_euler        # gripper orientation in world frame [roll, pitch, yaw] in radians

float64 gripper_tracking_error # gripper tracking error
float64[] gripper_tracking_error_xyz # gripper tracking error in x, y, z
float64[] gripper_position_ref # gripper position reference in world frame [x, y, z]
float64[] gripper_euler_ref # gripper orientation reference in world frame [roll, pitch, yaw] in radians