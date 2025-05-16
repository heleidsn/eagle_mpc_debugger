'''
Author: Lei He
Date: 2024-09-07 17:26:46
LastEditTime: 2025-04-30 15:00:01
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


class L1AdaptiveController_V1:
    '''
    description: 尝试使用state_ref来进行控制，利用z_hat和state_ref的差来进行控制
    param {*} self
    return {*}
    '''    
    def __init__(self, dt, robot_model, As_coef, filter_time_constant, using_z_real=False):
        self.dt = dt
        
        self.as_matrix_coef   = As_coef
        self.filter_time_coef = filter_time_constant
        
        self.flag_using_z_ref = False
        self.robot_model = robot_model
        self.model = self.robot_model
        
        self.using_z_real = using_z_real
        
        self.model_data = self.robot_model.createData()
        
        self.state_dim = self.robot_model.nq + self.robot_model.nv  # number of state using quaternion
        self.state_dim_euler = int((self.state_dim - 1)/2 )                  # number of state using euler angle
        self.control_dim = self.robot_model.nv                      # number of control input
        
        self.debug = False
        
        if self.robot_model.nq == 7:
            self.use_arm = False
            self.filter_num = 2   # no arm
        else:
            self.filter_num = 3   # with arm
            self.use_arm = True
            self.arm_joint_num = self.robot_model.nq - 7  # number of arm joints
            
        self.a_s = np.ones(self.state_dim_euler) * self.as_matrix_coef
        self.filter_time_const = np.ones(self.filter_num) * self.filter_time_coef
        
    def init_controller(self):
        '''
        description: 初始化所有控制器参数
        return {*}
        '''
        self.z_ref_all = np.zeros(self.state_dim)      # 参考状态向量
        self.current_state = np.zeros(self.state_dim)  # 当前状态
                
        self.z_hat = np.zeros(self.state_dim_euler)    # 估计的状态向量
        self.z_real = np.zeros(self.state_dim_euler)    # 参考状态向量
        self.z_ref = np.zeros(self.state_dim_euler)    # 参考状态向量

        self.sig_hat = np.zeros(self.state_dim_euler)  # 估计的扰动
        self.sig_hat_local = np.zeros(self.state_dim_euler)  # 估计的扰动在机体坐标系下
        self.sig_hat_tracking = np.zeros(self.state_dim_euler)  # 用来追踪参考状态的扰动
        
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
        description: 更新z_tilde和z_ref
        return {*}
        '''       
        # update current state and z_tilde 
        self.z_tilde     = self.z_hat - self.z_real
        self.z_tilde_ref = self.z_hat - self.z_ref
        
        # get tracking error in deg, only velocity tracking error
        # self.tracking_error_angle = (self.get_state_angle_single_rad(self.z_ref_all) - self.get_state_angle_single_rad(self.current_state))[:9] 
        # self.tracking_error_velocity = self.z_ref - self.current_state[10:]
        
    def get_z_hat_dot(self):
        '''
        description: 这里假设只用u_b进行控制，然后使用z_hat来对估计误差进行old修正，乘以系数A_s
        return {*}
        '''        
        # z  = self.current_state[self.model.nq:]  # 状态向量
        # z_tilde = z_hat - z  # 估计误差self.data
        
        tau_body = self.u_mpc + self.u_ad + self.sig_hat + self.u_tracking   # 假设只用u_b进行控制
        if self.flag_using_z_ref:
            z_hat_dot = pin.aba(self.model, self.model_data, self.current_state[:self.model.nq], self.current_state[self.model.nq:], tau_body) + self.A_s @ self.z_tilde_ref
        else:
            z_hat_dot = pin.aba(self.model, self.model_data, self.current_state[:self.model.nq], self.current_state[self.model.nq:], tau_body) + self.A_s @ self.z_tilde
        # z_hat_dot = pin.aba(self.model, self.data, current_state[:self.model.nq], current_state[self.model.nq:], tau_body) + self.A_s @ z_tilde
        return z_hat_dot
    
    def update_z_hat(self):
        # update predictor
        z_hat_dot = self.get_z_hat_dot()   # in body frame
        
        if self.using_z_real:
            self.z_hat = self.z_real + self.dt * z_hat_dot     # in body frame
        else:
            self.z_hat = self.z_hat + self.dt * z_hat_dot
    
    def update_sig_hat_v1(self):
        '''
        description: 更新对于扰动的估计
        return {*}
        '''
        # 1. get inertia matrix        
        q = self.current_state[:self.model.nq]
        M = pin.crba(self.model, self.model_data, q)  # get inertia matrix
        
        # 2. get rotation matrix to transform the disturbance to body frame
        quat = self.current_state[3:7]  # 获取四元数
        rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
        Rb = rotation.as_matrix()  # 从四元数得到旋转矩阵
        
        # 3. calculate sigma_hat (body frame)
        B_bar_inv = M.copy()
        PHI = np.matmul(LA.inv(self.A_s), (self.expm_A_s_dt - np.identity(self.state_dim_euler)))
        PHI_inv = LA.inv(PHI)
        mu  = np.matmul(self.expm_A_s_dt, self.z_tilde)
        PHI_inv_mul_mu = np.matmul(PHI_inv, mu)
        sigma_hat  = -np.matmul(B_bar_inv, PHI_inv_mul_mu)  # 奇怪，为什么z_hat - z 要乘以B_bar，最后两列才是扰动呀
        
        # get z tracking error
        sigma_hat_tracking_error = np.matmul(self.A_s, self.z_real - self.z_ref)  # using rad
        # print('sigma_hat_tracking_error: ', sigma_hat_tracking_error)
        temp = -np.matmul(B_bar_inv, sigma_hat_tracking_error)
        
        # print('temp: ', temp)
        
        weight = np.array([0, 0, 0, 0, 0, 0])
        self.sig_hat_tracking = temp * weight
        # print('self.sig_hat_tracking: ', self.sig_hat_tracking[-2])


        #! changing to with or without reference tracking
        if self.flag_using_z_ref:
            self.sig_hat = sigma_hat.copy() + self.sig_hat_tracking.copy()
            # self.sig_hat = sigma_hat.copy()
        else:
            self.sig_hat = sigma_hat.copy()
        
        # 4. get disturbance in local frame
        sigma_hat_force_body = sigma_hat[:3]  # 
        sigma_hat_force_local = np.matmul(Rb.T, sigma_hat_force_body[:3])  # 需要确认Rb是否需要转置
        self.sig_hat_local = np.concatenate((sigma_hat_force_local, sigma_hat[3:]))
    
    def update_u_ad(self):
        '''
        description: 在目前的版本中
        return {*}
        '''        
        
        # 1. get the output of the low-pass filter
        sig_hat_original = self.sig_hat.copy()
        # print('sig_hat_original: ', sig_hat_original)
        sig_hat_original_new = sig_hat_original.copy()
        # sig_hat_original_new[-3:] = sig_hat_original[-3:] + self.z_tilde_tracking[6:9].copy() * np.array([50, 0, 0])
        
        f_l1_lpf = self.low_pass_filter(self.filter_time_const[0], sig_hat_original_new[:3], self.sig_f_prev)
        self.sig_f_prev = f_l1_lpf
        
        t_l1_lpf = self.low_pass_filter(self.filter_time_const[1], sig_hat_original_new[3:6], self.sig_t_prev)
        self.sig_t_prev = t_l1_lpf
        
        if self.use_arm:
            t_l1_lpf_arm = self.low_pass_filter(self.filter_time_const[2], sig_hat_original_new[6:], self.sig_t_arm_prev)
            self.sig_t_arm_prev = t_l1_lpf_arm
            
            sig_hat_filtered = np.concatenate((f_l1_lpf, t_l1_lpf, t_l1_lpf_arm))
        else:
            sig_hat_filtered = np.concatenate((f_l1_lpf, t_l1_lpf))
        
        self.u_ad = -sig_hat_filtered
        
        self.u_ad[:2] = np.zeros(2)  # 不对x, y方向的力进行控制
        
        # limitation
        flag_using_limit = True
        if flag_using_limit:
            min_values = np.array([0, 0, -10, -1, -1, -1])
            max_values = -min_values
            
            clipped_array = np.where(self.u_ad < min_values, min_values, self.u_ad)
            clipped_array = np.where(clipped_array > max_values, max_values, clipped_array)
            
            self.u_ad = clipped_array
        
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
    
    def get_state_angle_single_rad(self, state_buffer_np):
        # 对state进行处理，将四元数转换成欧拉角
        quat = state_buffer_np[3:7]
        rotation = R.from_quat(quat)  # 创建旋转对象，注意传入四元数的顺序为 [x, y, z, w]
        euler_angles = rotation.as_euler('xyz', degrees=False)  # 将四元数转换为欧拉角
        
        state_buffer_angle = np.hstack((state_buffer_np[:3], euler_angles, state_buffer_np[7:]))
        
        # # 将弧度转换成角度
        # state_buffer_angle[3:9] = np.rad2deg(state_buffer_angle[3:9])
        # state_buffer_angle[12:] = np.rad2deg(state_buffer_angle[12:])
        
        return state_buffer_angle
