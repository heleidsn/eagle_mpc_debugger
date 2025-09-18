'''
Author: Lei He
Date: 2024-09-07 17:26:46
LastEditTime: 2025-09-18 10:07:25
Description: L1 adaptive controller with 9 state variables
Version 3: Used for full actuation
0916： fixed the bug in adaptive_law_new, change tau_body=u_b + u_ad_all + self.sig_hat_b
1010： add more functions for adaptive law
Github: https://github.com/heleidsn
'''


import numpy as np
import time
import rospy

from numpy import linalg as LA
from scipy import linalg as sLA
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from scipy.linalg import solve

    
class L1AdaptiveControllerAll:
    '''
    description: 使用所有的状态变量进行控制 也就是z=q
    param {*} self
    return {*}
    '''    
    def __init__(self, dt, robot_model, As_coef, filter_time_constant):
        self.dt = dt
        
        self.as_matrix_coef   = As_coef  # Hurwitz matrix
        
        self.step_debug = False
        
        self.using_vel_disturbance = False
        self.using_full_state = False         # choose whether to use full state or not
        self.flag_using_z_ref = False
        
        self.using_z_real = False
        
        self.robot_model = robot_model
        self.robot_model_data = self.robot_model.createData()
        
        self.state_dim = self.robot_model.nq + self.robot_model.nv  # number of state using quaternion
        self.state_dim_euler = self.state_dim - 1                   # number of state using euler angle
        self.control_dim = self.robot_model.nv                      # number of control input
        
        if self.robot_model.nq == 7:
            self.use_arm = False
            self.filter_num = 2   # no arm
            self.arm_joint_num = 0
        else:
            self.filter_num = 3   # with arm
            self.use_arm = True
            self.arm_joint_num = self.robot_model.nq - 7  # number of arm joints
        
        # tunning parameters
        self.a_s = np.ones(self.state_dim_euler) * self.as_matrix_coef
        self.filter_time_const = filter_time_constant
        
        # Pre-compute matrices for optimization
        self.A_s = np.diag(self.a_s)  # diagonal Hurwitz matrix
        self.expm_A_s_dt = sLA.expm(self.A_s * self.dt)
        
        # Pre-compute PHI inverse elements since A_s is diagonal
        # For a diagonal matrix A, exp(A*dt) is a diagonal matrix with exp(a_ii*dt)
        # And (exp(A*dt) - I) is diagonal with (exp(a_ii*dt) - 1)
        # Therefore PHI = A^(-1)(exp(A*dt) - I) is diagonal with (exp(a_ii*dt) - 1)/a_ii
        expm_minus_I = self.expm_A_s_dt - np.identity(self.state_dim_euler)
        self.PHI_diag = expm_minus_I.diagonal() / self.a_s
        self.PHI_inv_diag = 1.0 / self.PHI_diag  # Pre-compute the inverse elements
        
        # Pre-compute M_aug template
        self.M_aug_template = np.zeros((self.state_dim_euler, self.state_dim_euler))
        self.M_aug_template[:self.control_dim, :self.control_dim] = np.diag(np.ones(self.control_dim))
        
    def init_controller(self):
        '''
        description: 初始化所有控制器参数
        return {*}
        ''' 
        self.current_state = np.zeros(self.state_dim)  # 当前状态, used for update dynamics
        self.z_ref_all = np.zeros(self.state_dim)    # 参考状态
        
        self.z_hat = np.zeros(self.state_dim_euler)     # 估计的状态向量 Euler
        self.z_ref = np.zeros(self.state_dim_euler)     # 参考状态向量 Euler
        self.z_real = np.zeros(self.state_dim_euler)    # 状态测量值  in rad
        
        self.sig_hat = np.zeros(self.state_dim_euler)  # 估计的扰动
        
        self.z_tilde = np.zeros(self.state_dim_euler)  # 对于速度状态估计的误差
        self.z_tilde_ref = np.zeros(self.state_dim_euler)  # 与参考状态的误差
        self.z_tilde_tracking = np.zeros(self.state_dim_euler)  # for tracking error
        
        self.u_ad = np.zeros(self.control_dim)  # L1控制器的输出
        self.u_mpc = np.zeros(self.control_dim)  # MPC控制器的输出
        self.u_tracking = np.zeros(self.control_dim)  # 跟踪控制器的输出
        
        self.sig_f_prev = np.zeros(3)
        self.sig_t_prev = np.zeros(3)
        
        if self.use_arm:
            self.sig_t_arm_prev = np.zeros(self.arm_joint_num) 
        
        self.A_s = np.diag(self.a_s) # diagonal Hurwitz matrix, same as that used in [3]
        self.expm_A_s_dt = sLA.expm(self.A_s * self.dt)
              
    def update_z_tilde(self):
        '''
        description: update z_tilde for all 18 states
        return {*}
        '''
        self.z_tilde          = self.z_hat.copy() - self.z_real.copy()  # default z_tilde
        if not self.using_full_state:
            self.z_tilde[:self.control_dim] = np.zeros(self.control_dim)  # 只使用速度进行估计       
        self.z_tilde_ref      = self.z_hat.copy() - self.z_ref.copy()
        self.z_tilde_tracking = self.z_real.copy() - self.z_ref.copy()
        
        # update tracking error
        self.tracking_error = self.z_ref - self.z_real
        self.tracking_error_angle =  self.tracking_error[:self.control_dim]  # 角度误差
        self.tracking_error_velocity = self.tracking_error[self.control_dim:]  # 速度误差
            
    def update_z_hat_vel(self):
        '''
        description: Only update z_hat for velocity, used for non-full state L1
        return {*}
        '''        
        # update predictor
        tau_body = self.u_mpc + self.u_ad + self.sig_hat[self.control_dim:] + self.u_tracking   # 假设只用u_b进行控制

        if self.step_debug:
            print('--------------------------------')
            print('u_mpc   : ', self.u_mpc)
            print('u_ad    : ', self.u_ad)
            print('sig_hat : ', self.sig_hat[self.control_dim:])
            print('tau_body: ', tau_body)
            print('dt      : ', self.dt)
            print('--------------------------------')
        
        z_hat_prev = self.z_hat.copy()
        
        model = self.robot_model
        data = self.robot_model_data
        q = self.current_state[:self.robot_model.nq]    # state
        v = self.current_state[self.robot_model.nq:]    # velocity
        
        z_hat_dot_without_disturb = pin.aba(model, data, q, v, tau_body)   # get a using the current state
        z_hat_dot_disturb = self.A_s[self.control_dim:, self.control_dim:] @ self.z_tilde[self.control_dim:]  # get a using the current state
        
        z_hat_dot_vel = z_hat_dot_without_disturb.copy() + z_hat_dot_disturb.copy()
        
        self.z_hat[self.control_dim:] = z_hat_prev[self.control_dim:].copy() + self.dt * z_hat_dot_vel.copy()
        
        if self.step_debug:
            print('--------------------------------')
            print('q                   : ', q)
            print('v                   : ', v)
            print('tau_body            : ', tau_body)
            print('z_hat_dot_wo_disturb: ', z_hat_dot_without_disturb[-2:])
            print('z_hat_dot_disturb   : ', z_hat_dot_disturb[-2:])
            print('z_hat_dot_vel       : ', z_hat_dot_vel[-2:]) 
            print('z_hat               : ', self.z_hat[self.control_dim:])
            print('z_tilde             : ', self.z_tilde[self.control_dim:])
            print('--------------------------------')
        
    def update_sig_hat_all_v2_new(self):
        """Optimized version that uses pre-computed matrices and diagonal structure"""
        q = self.current_state[:self.robot_model.nq]
        t0 = time.time()
        M = pin.crba(self.robot_model, self.robot_model_data, q)
        t1 = time.time()

        # Use pre-computed template and update only the necessary part
        M_aug = self.M_aug_template.copy()
        M_aug[-self.control_dim:, -self.control_dim:] = M

        t2 = time.time()

        # Fast computation using pre-computed diagonal elements and element-wise operations
        mu = np.matmul(self.expm_A_s_dt, self.z_tilde)
        # Since PHI is diagonal, PHI_inv @ mu is element-wise multiplication
        PHI_inv_mul_mu = self.PHI_inv_diag * mu
        
        t4 = time.time()
        sigma_hat_disturb = -np.matmul(M_aug, PHI_inv_mul_mu)
        
        # print('mu: ', mu)
        # print('PHI_inv_mul_mu: ', PHI_inv_mul_mu)
        # print('sigma_hat_disturb: ', sigma_hat_disturb)

        # Use direct weight multiplication instead of matrix operations
        # weight = np.array([0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 50, 1, 1, 1, 1, 1])
        weight = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.09])
        # self.sig_hat = -1 * weight * self.z_tilde.copy()
        self.sig_hat = sigma_hat_disturb.copy()
        
        # add constrain to sig_hat
        self.sig_hat = np.clip(self.sig_hat, -20, 20)

        t5 = time.time()
        # rospy.loginfo(f"timing (ms): crba: {(t1-t0)*1000:.3f}  aug: {(t2-t1)*1000:.3f}  phi_mul: {(t4-t2)*1000:.3f}  sigma_hat: {(t5-t4)*1000:.3f}")
        
        if self.step_debug:
            print('--------------------------------')
            print('sig_hat: ', self.sig_hat[self.control_dim:])
            print('--------------------------------')
    
    def update_u_ad(self):
        '''
        description: u_ad = -sig_hat_filtered
        return {*}
        '''         
        # 1. get the output of the low-pass filter
        sig_hat_original = self.sig_hat.copy()[self.control_dim:]
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
        
        # limitation
        flag_using_limit = True
        if flag_using_limit:
            if self.use_arm:
                min_values = np.array([0, 0, -10, -1, -1, -1, -1.0, -1.0])
            else:
                min_values = np.array([0, 0, -10, -1, -1, -1])
            max_values = -min_values
            
            clipped_array = np.where(self.u_ad < min_values, min_values, self.u_ad)
            clipped_array = np.where(clipped_array > max_values, max_values, clipped_array)
            
            self.u_ad = clipped_array
            
        if self.step_debug:
            print('--------------------------------')
            print('u_ad: ', self.u_ad)
            print('--------------------------------')
    
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