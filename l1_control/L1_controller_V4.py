'''
Author: Lei He
Date: 2024-09-07 17:26:46
LastEditTime: 2025-05-13 17:13:28
Description: L1 adaptive controller with 9 state variables
Version 3: Used for full actuation
0916： fixed the bug in adaptive_law_new, change tau_body=u_b + u_ad_all + self.sig_hat_b
1010： add more functions for adaptive law
Github: https://github.com/heleidsn
'''


import numpy as np

from numpy import linalg as LA
from scipy import linalg as sLA
from scipy.spatial.transform import Rotation as R
import pinocchio as pin


class L1AdaptiveControllerAll:
    '''
    description: 使用所有的状态变量进行控制 也就是z=q
    param {*} self
    return {*}
    '''    
    def __init__(self, dt, int_model, int_model_data, robot_model, debug=False, flag_using_z_ref=False):
        self.dt = dt
        self.flag_using_z_ref = flag_using_z_ref
        
        self.robot_model = robot_model
        self.robot_model_data = self.robot_model.createData()
        
        self.int_model = int_model
        self.int_model_data = int_model_data
          
        self.debug = debug
        
        self.state_dim = self.robot_model.nq + self.robot_model.nv  # number of state using quaternion
        self.state_dim_euler = self.state_dim - 1                   # number of state using euler angle
        self.control_dim = self.robot_model.nv                      # number of control input
        
        # tunning parameters
        self.a_s = np.ones(self.state_dim_euler) * -1   #! Hurwitz matrix
        # self.filter_time_const = np.array([0.001, 0.001, 0.001])  # 注意这里的单位为s， dt = 0.005s
        self.filter_time_const = np.ones(3) * 0.001  # 注意这里的单位为s， dt = 0.005s  tc = 0.001 alpha = 0.833  tc = 0.005 alpha = 0.5 
        
    def init_controller(self):
        '''
        description: 初始化所有控制器参数
        return {*}
        ''' 
        self.current_state = np.zeros(self.state_dim)  # 当前状态, used for update dynamics
        self.z_ref_all = np.zeros(self.state_dim)    # 参考状态
        
        self.z_hat = np.zeros(self.state_dim_euler)     # 估计的状态向量 Euler
        self.z_ref = np.zeros(self.state_dim_euler)     # 参考状态向量 Euler
        self.z_real = np.zeros(self.state_dim_euler) # 状态测量值  in rad
        
        self.sig_hat = np.zeros(self.state_dim_euler)  # 估计的扰动
        
        self.z_tilde = np.zeros(self.state_dim_euler)  # 对于速度状态估计的误差
        self.z_tilde_ref = np.zeros(self.state_dim_euler)  # 与参考状态的误差
        self.z_tilde_tracking = np.zeros(self.state_dim_euler)  # for tracking error
        
        self.u_ad = np.zeros(self.control_dim)  # L1控制器的输出
        self.u_mpc = np.zeros(self.control_dim)  # MPC控制器的输出
        self.u_tracking = np.zeros(self.control_dim)  # 跟踪控制器的输出
        
        self.sig_f_prev = np.zeros(3)
        self.sig_t_prev = np.zeros(3)
        self.sig_t_arm_prev = np.zeros(3)
        
        self.A_s = np.diag(self.a_s) # diagonal Hurwitz matrix, same as that used in [3]
        self.expm_A_s_dt = sLA.expm(self.A_s * self.dt)
              
    def update_z_tilde(self):
        '''
        description: 更新z_tilde, 以角度形式，需要18个变量
        return {*}
        '''       
        self.z_tilde_ref      = self.z_hat.copy()     - self.z_ref.copy()
        self.z_tilde          = self.z_hat.copy()     - self.z_real.copy()
        # self.z_tilde_tracking = self.z_real.copy() - self.z_ref.copy()
        
        # update tracking error
        self.tracking_error_angle =  (self.get_state_angle_single_rad(self.z_ref_all) - self.get_state_angle_single_rad(self.current_state))[:9] 
        self.tracking_error_velocity = self.z_ref_all[10:] - self.current_state[10:]
    
    def update_z_hat(self):
        '''
        description: update z_hat using pinocchio model
                     different from L1AdaptiveControllerNew, the z_hat is calculated using pinocchio model, similar to the dynamic model
        return {*}
        ''' 
        #! 方法1：使用积分模型进行预测 无法加入 self.A_s @ self.z_tilde
          
        #! 重点 使用四元数进行积分
        # 假设你已经有了当前状态q和速度v，以及控制输入u
        q = self.current_state[:self.robot_model.nq]
        v = self.current_state[self.robot_model.nq:]
        u = self.u_mpc + self.u_ad + self.sig_hat[9:]  # 假设只用u_b进行控制
        dt = self.dt

        # 创建模型和数据
        model = self.robot_model
        # data = model.createData()
        data = self.robot_model_data
        
        if self.flag_using_z_ref:
            A_mul_z_tilde = self.A_s @ self.z_tilde_ref
        else:
            A_mul_z_tilde = self.A_s @ self.z_tilde

        # 计算加速度
        a = pin.aba(model, data, q, v, u) + A_mul_z_tilde[9:]

        # 提取当前的线速度和角速度
        # linear_velocity_body = v[:3]  # 机体坐标系下的线速度
        # angular_velocity = v[3:]  # 角速度
        
        linear_velocity_body_hat = self.z_hat[9:12].copy()  # 机体坐标系下的线速度
        angular_velocity_hat = self.z_hat[12:].copy()  # 角速度

        # 更新线速度和角速度
        linear_velocity_next_body = linear_velocity_body_hat + a[:3] * dt  # 更新机体坐标系下的线速度
        angular_velocity_next = angular_velocity_hat + a[3:] * dt  # 更新角速度 (包含arm)
        
        # 给速度环也加上干扰
        linear_velocity_next_body_dist = linear_velocity_next_body + A_mul_z_tilde[:3]
        angular_velocity_next_dist = angular_velocity_next + A_mul_z_tilde[3:9]
        
        temp_method = 1
        if temp_method == 1:  # using disturbance velocity
            # 更新位置和姿态
            # 使用四元数进行姿态更新
            quat = q[3:7]  # 当前四元数
            rotation = R.from_quat(quat)  # 创建旋转对象
            omega = angular_velocity_next_dist[:3]  # 更新后的角速度
            delta_rotation = R.from_rotvec(omega * dt)  # 计算旋转增量
            new_rotation = rotation * delta_rotation  # 更新旋转
            new_quat = new_rotation.as_quat()  # 转换回四元数

            # 将机体坐标系下的线速度转换到世界坐标系
            linear_velocity_next_world = rotation.apply(linear_velocity_next_body_dist)  # 转换到世界坐标系
            
            # 更新arm位置
            arm_position = q[7:10]
            arm_position_next = arm_position + angular_velocity_next_dist[3:] * dt

            # 更新位置
            position = q[:3]  # 当前的位置
            position_next = position + linear_velocity_next_world * dt  # 更新位置

            # 组合新的状态
            q_next = np.hstack((position_next, new_quat, arm_position_next))  # 新的状态
            v_next = np.hstack((linear_velocity_next_body_dist, angular_velocity_next_dist))  # 新的速度，注意都是在机体坐标系下定
        else: 
            # 更新位置和姿态
            # 使用四元数进行姿态更新
            quat = q[3:7]  # 当前四元数
            rotation = R.from_quat(quat)  # 创建旋转对象
            omega = angular_velocity_next[:3]  # 更新后的角速度
            delta_rotation = R.from_rotvec(omega * dt)  # 计算旋转增量
            new_rotation = rotation * delta_rotation  # 更新旋转
            new_quat = new_rotation.as_quat()  # 转换回四元数

            # 将机体坐标系下的线速度转换到世界坐标系
            linear_velocity_next_world = rotation.apply(linear_velocity_next_body)  # 转换到世界坐标系
            
            # 更新arm位置
            arm_position = q[7:10]
            arm_position_next = arm_position + angular_velocity_next[3:] * dt

            # 更新位置
            position = q[:3]  # 当前的位置
            position_next = position + linear_velocity_next_world * dt  # 更新位置

            # 组合新的状态
            q_next = np.hstack((position_next, new_quat, arm_position_next))  # 新的状态
            v_next = np.hstack((linear_velocity_next_body, angular_velocity_next))  # 新的速度，注意都是在机体坐标系下定义

        # 现在q_next和v_next是下一步的状态和速度
        self.z_hat = self.get_state_angle_single_rad(np.hstack((q_next, v_next)))

    def rk4_integration(self, q, v, u, dt, A_mul_z_tilde):
        '''
        description: 使用RK4方法更新状态
        return: 更新后的q和v
        '''
        def state_derivative(q, v, u):
            # 计算加速度
            a = pin.aba(self.robot_model, self.robot_model_data, q, v, u) + A_mul_z_tilde[9:0]
            
            linear_velocity_body_hat = self.z_hat[9:12].copy()  # 机体坐标系下的线速度
            angular_velocity_hat = self.z_hat[12:].copy()  # 角速度
            
            # 更新线速度和角速度
            linear_velocity_next_body = linear_velocity_body_hat + a[:3] * dt  # 更新机体坐标系下的线速度
            angular_velocity_next = angular_velocity_hat + a[3:] * dt  # 更新角速度 (包含arm)
            
            # 给速度环也加上干扰
            linear_velocity_next_body_dist = linear_velocity_next_body + A_mul_z_tilde[:3]
            angular_velocity_next_dist = angular_velocity_next + A_mul_z_tilde[3:9]
            
            v = np.hstack([linear_velocity_body_hat, angular_velocity_hat])
            
            return v, a  # 返回速度和加速度
        
        def get_next_state(q, linear_velocity_next_body_dist, angular_velocity_next_dist):
            # 更新位置和姿态
            # 使用四元数进行姿态更新
            quat = q[3:7]  # 当前四元数
            rotation = R.from_quat(quat)  # 创建旋转对象
            omega = angular_velocity_next_dist[:3]  # 更新后的角速度
            delta_rotation = R.from_rotvec(omega * dt)  # 计算旋转增量
            new_rotation = rotation * delta_rotation  # 更新旋转
            new_quat = new_rotation.as_quat()  # 转换回四元数

            # 将机体坐标系下的线速度转换到世界坐标系
            linear_velocity_next_world = rotation.apply(linear_velocity_next_body_dist)  # 转换到世界坐标系
            
            # 更新arm位置
            arm_position = q[7:10]
            arm_position_next = arm_position + angular_velocity_next_dist[3:] * dt

            # 更新位置
            position = q[:3]  # 当前的位置
            position_next = position + linear_velocity_next_world * dt  # 更新位置

            # 组合新的状态
            q_next = np.hstack((position_next, new_quat, arm_position_next))  # 新的状态
            v_next = np.hstack((linear_velocity_next_body_dist, angular_velocity_next_dist))  # 新的速度，注意都是在机体坐标系下定义
            
            return q_next, v_next

        # 计算RK4步骤
        k1_v, k1_a = state_derivative(q, v, u)
        k1_q = get_next_state(q, k1_v, k1_a, dt)   #  这里得到的

        k2_v, k2_a = state_derivative(q + 0.5 * dt * k1_q, v + 0.5 * dt * k1_a, u)
        # k2_q = v + 0.5 * dt * k1_a
        k2_q = get_next_state(q, k2_v, k2_a, dt)

        k3_v, k3_a = state_derivative(q + 0.5 * dt * k2_q, v + 0.5 * dt * k2_a, u)
        k3_q = get_next_state(q, k3_v, k3_a, dt)

        k4_v, k4_a = state_derivative(q + dt * k3_q, v + dt * k3_a, u)
        # k4_q = v + dt * k3_a
        
        # 1. 利用中间状态得到一个新的导数，包含速度和加速度两个回路
        a_new = (k1_a + 2*k2_a + 2*k3_a + k4_a)/6
        v_new = (k1_v + 2*k2_v + 2*k3_v + k4_v)/6
        
        # 2. 利用新的导数和dt得到位置和速度
        q_next, v_next = get_next_state(q, v_new, a_new, dt)

        return q_next, v_next 
    
    def update_sig_hat_all_v2(self):
        '''
        description: 更新对于扰动的估计，由于对矩阵进行了增广，
        return {*}
        '''
        debug_sig_hat = False
        
        # 1. get inertia matrix        
        q = self.current_state[:self.robot_model.nq]
        M = pin.crba(self.robot_model, self.robot_model_data, q)  # get inertia matrix according to the current state
        
        # 3. calculate sigma_hat (body frame)
        #! 对B_bar_inv进行增广
        M_aug = np.zeros((18, 18))
        M_aug[:9, :9] = np.diag(np.ones(9))
        M_aug[-9:, -9:] = M.copy()

        B_bar_inv = M_aug.copy()
        
        #! 将以下计算改为18维
        PHI = np.matmul(LA.inv(self.A_s), (self.expm_A_s_dt - np.identity(18)))
        PHI_inv = LA.inv(PHI)
        # if self.flag_using_z_ref:
        #     mu = np.matmul(self.expm_A_s_dt, self.z_tilde_ref)
        # else:
        #     mu  = np.matmul(self.expm_A_s_dt, self.z_tilde)    #! z_tself.A_s @ self.z_tilde[9:]ilde = z_hat - z in rad 之前应该是角度
        
        mu  = np.matmul(self.expm_A_s_dt, self.z_tilde)
        PHI_inv_mul_mu = np.matmul(PHI_inv, mu)

        sigma_hat_disturb  = -np.matmul(B_bar_inv, PHI_inv_mul_mu)
        
        #! 加入对于tracking error的反馈
        sigma_hat_tracking = np.zeros(18)
        # sigma_hat_tracking[11] = -pos_tracking_error[2] * 100
        # sigma_hat_tracking[-3:] = pos_tracking_error[-3:] * np.array([20, 0, 0])
        
        if debug_sig_hat:
            #! for z_ref testing
            print('========================sig_hat========================')
            print('z_real:       ', self.get_state_angle_single_rad(self.current_state))
            print('z_hat:        ', self.z_hat)
            print('z_tilde_pose: ', self.z_tilde[:9])
            print('z_tilde_velo: ', self.z_tilde[9:])
            print('sig_hat_pose: ', sigma_hat_disturb[:9])
            print('sig_hat_velo: ', sigma_hat_disturb[9:])
            print('======================================================')

        self.sig_hat = sigma_hat_disturb.copy() + sigma_hat_tracking.copy()
    
    def get_u_l1_all_v2(self):
        '''
        description: u_ad = -sig_hat_filtered
        return {*}
        '''         
        # 1. get the output of the low-pass filter
        sig_hat_original = self.sig_hat.copy()[9:]
        # print('sig_hat_original: ', sig_hat_original)
        sig_hat_original_new = sig_hat_original.copy()
        # sig_hat_original_new[-3:] = sig_hat_original[-3:] + self.z_tilde_tracking[6:9].copy() * np.array([50, 0, 0])
        
        f_l1_lpf = self.low_pass_filter(self.filter_time_const[0], sig_hat_original_new[:3], self.sig_f_prev)
        self.sig_f_prev = f_l1_lpf
        
        t_l1_lpf = self.low_pass_filter(self.filter_time_const[1], sig_hat_original_new[3:6], self.sig_t_prev)
        self.sig_t_prev = t_l1_lpf
        
        t_l1_lpf_arm = self.low_pass_filter(self.filter_time_const[2], sig_hat_original_new[6:], self.sig_t_arm_prev)
        self.sig_t_arm_prev = t_l1_lpf_arm
        
        sig_hat_filtered = np.concatenate((f_l1_lpf, t_l1_lpf, t_l1_lpf_arm))
        
        self.u_ad = -sig_hat_filtered
        
        # limitation
        flag_using_limit = False
        if flag_using_limit:
            min_values = np.array([0, 0, -50, -5, -5, -5, -5, -5, -1])
            max_values = -min_values
            
            clipped_array = np.where(self.u_ad < min_values, min_values, self.u_ad)
            clipped_array = np.where(clipped_array > max_values, max_values, clipped_array)
            
            self.u_ad = clipped_array
        
        # 加入对于tracking error的反馈
        if self.flag_using_z_ref:
            error_p = np.array([0, 0, 0, 0, 0, 0, 20, 3, 0.5]) # 位置反馈的权重  error_p = np.array([0, 0, 0, 0, 0, 0, 1.8, 3, 0.03]) # 位置反馈的权重
            self.u_tracking = self.tracking_error_angle * error_p * 0.5 + self.tracking_error_velocity * 0
            
            return self.u_ad + self.u_tracking
        else:
            return self.u_ad
    
    def low_pass_filter(self, time_const, curr_i, prev_i):
        '''
        description: 一阶低通滤波器 
        time_const:  t_c = 1  dt = 0.005 alpha = 0.005/(0.005+1) = 0.005
                     t_c = 0.001 dt = 0.005 alpha = 0.005/(0.005+0.001) = 0.833
                     t_c = 0.005 dt = 0.005 alpha = 0.005/(0.005+0.005) = 0.5
        return {*}
        '''        
        
        alpha       = self.dt / (self.dt + time_const)
        y_filter    = (1 - alpha) * prev_i + alpha * curr_i
        
        return y_filter
    
    def get_state_angle_single_deg(self, state_buffer_np):
        # 对state进行处理，将四元数转换成欧拉角
        quat = state_buffer_np[3:7]
        rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
        
        state_buffer_angle = np.hstack((state_buffer_np[:3], euler_angles, state_buffer_np[7:]))
        
        # 将弧度转换成角度
        state_buffer_angle[3:9] = np.rad2deg(state_buffer_angle[3:9])
        state_buffer_angle[12:] = np.rad2deg(state_buffer_angle[12:])
        
        return state_buffer_angle
    
    def get_state_angle_single_rad(self, state_buffer_np):
        # 对state进行处理，将四元数转换成欧拉角
        quat = state_buffer_np[3:7]
        rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
        
        state_buffer_angle = np.hstack((state_buffer_np[:3], euler_angles, state_buffer_np[7:]))
        
        return state_buffer_angle
    
    def transfer_state_to_deg(self, state):
        state_deg = state.copy()
        state_deg[3:9] = np.rad2deg(state[3:9])
        state_deg[12:] = np.rad2deg(state[12:])
        
        return state_deg