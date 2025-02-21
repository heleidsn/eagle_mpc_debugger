'''
Author: Lei He
Date: 2024-09-10 16:21:20
LastEditTime: 2024-11-01 14:25:13
Description: 对控制器的输出进行转换
Github: https://github.com/heleidsn
'''
import numpy as np
import math
import pinocchio as pin


def thrustToForceTorque(control_thrust, tau_f):
    # transfer thrust to Fz, Mx, My, Mz according to the thrust matrix(tau_f)
    n_rotor = tau_f.shape[1]
    control_ft = tau_f @ control_thrust[:n_rotor]
    
    return control_ft

def thrustToForceTorqueAll(control_thrust, tau_f):
    # transfer thrust to Fz, Mx, My, Mz according to the thrust matrix(tau_f)
    # with arm controller
    n_rotor = tau_f.shape[1]
    control_ft = tau_f @ control_thrust[:n_rotor].copy()
    
    # print(np.linalg.inv(tau_f) @ tau_f)  # 由于tau_f不满秩，所以这里只能求伪逆 但是如何从伪逆中得到原矩阵呢？
    
    control_thrust_new = np.linalg.pinv(tau_f) @ control_ft
    control_ft_all = np.concatenate((control_ft, control_thrust[n_rotor:]))
    
    return control_ft_all

def thrustToForceTorqueAll_array(control_thrust_array, tau_f):
    # transfer thrust to Fz, Mx, My, Mz according to the thrust matrix(tau_f)
    # with arm controller
    n_rotor = tau_f.shape[1]
    temp = control_thrust_array[:, :n_rotor].copy().T
    control_ft = (tau_f @ temp).T
    temp1 = control_ft  # 400, 6
    temp2 = control_thrust_array[:, n_rotor:]
    control_ft_all = np.concatenate((control_ft, control_thrust_array[:, n_rotor:]), axis=1)
    
    return control_ft_all

def forceTorqueToThrust(control_ft, tau_f):
    # transfer Fz, Mx, My, Mz to thrust according to the thrust matrix(tau_f)
    n_rotor = tau_f.shape[1]
    control_thrust = np.linalg.pinv(tau_f) @ control_ft[:n_rotor].copy()
    
    return control_thrust

def add_control_disturbance(control, time_step):
    # control[0] += 2.5
    # control[1] -= 2.5
    control[2] += 0
    # control[3] += 0.5
    # control[4] += 1.0
    # control[5] += 1.5
    # control[6] += 0.15
    # control[7] -= 0.05
    # control[8] += 0.05 * math.sin(0.002 * i)
    # control[8] += 0.2
    control[8] += 0.05 * math.sin(0.001 * time_step)
    # control[6] += 0.2
    
    return control

def get_link_position(robot_model, robot_data, link_id, state_buffer):
    
    pin.framesForwardKinematics(robot_model, robot_data, state_buffer[-1][:robot_model.nq])
    gripper_position = robot_data.oMf[link_id].translation

    return gripper_position

def get_link_position_and_velocity(robot_model, robot_data, link_name):
    # get link position in world frame
    frame_id = robot_model.getFrameId(link_name)
    
    # frame_velocity = pin.getFrameVelocity(robot_model, robot_data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    
    # print(frame_velocity)
    
    return robot_data.oMi[frame_id].translation

def get_gripper_position_plan(robot_model, robot_data, link_name, traj_state_ref):
    # get link position in world frame
    position_gripper_buffer = []
    for i in range(traj_state_ref.shape[0]):
        q = traj_state_ref[i, :robot_model.nq]
        pin.framesForwardKinematics(robot_model, robot_data, q)
        gripper_position = robot_data.oMf[robot_model.getFrameId(link_name)].translation
        # print(gripper_position)
        position_gripper_buffer.append(gripper_position.copy())
    
    return position_gripper_buffer