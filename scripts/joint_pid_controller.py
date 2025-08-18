#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-01-XX
Description: Joint PID Controller Node for Eagle MPC Debugger
Modified for 4 joints with P+PID cascade control and velocity limits
'''

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import Trigger, TriggerResponse
import threading
import time
import dynamic_reconfigure.client
import dynamic_reconfigure.server
from eagle_mpc_debugger.cfg import JointPIDConfig

class JointPIDController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('joint_pid_controller', anonymous=False)
        
        # PID parameters for each joint - will be updated by dynamic_reconfigure
        self.pid_params = {
            'joint_1': {'kp': 5.0, 'ki': 0, 'kd': 0.0, 'kp_outer': 10.0, 'max_velocity': 0.5, 'max_effort': 0.3},
            'joint_2': {'kp': 5.0, 'ki': 0, 'kd': 0.0, 'kp_outer': 10.0, 'max_velocity': 0.5, 'max_effort': 0.3},
            'joint_3': {'kp': 5.0, 'ki': 0, 'kd': 0.0, 'kp_outer': 10.0, 'max_velocity': 0.5, 'max_effort': 0.3},
            'joint_4': {'kp': 5.0, 'ki': 0, 'kd': 0.0, 'kp_outer': 10.0, 'max_velocity': 0.5, 'max_effort': 0.3}
        }
        
        # Initialize dynamic_reconfigure server
        self.dyn_server = dynamic_reconfigure.server.Server(JointPIDConfig, self.dynamic_reconfigure_callback)
        
        # Initialize PID state variables for both outer (position) and inner (velocity) loops
        self.pid_state = {}
        for joint_name in self.pid_params.keys():
            self.pid_state[joint_name] = {
                'outer': {
                    'error_prev': 0.0,
                    'error_integral': 0.0,
                    'last_time': None
                },
                'inner': {
                    'error_prev': 0.0,
                    'error_integral': 0.0,
                    'last_time': None
                }
            }
        
        # Current and desired joint states
        self.current_joint_states = {}
        self.current_joint_velocities = {}
        self.desired_joint_states = {'joint_1': 0, 'joint_2': 0, 'joint_3': 0, 'joint_4': -1}
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
        
        # Control flag
        self.controller_active = False
        self.control_rate = 200.0  # Hz
        
        # Publishers and Subscribers    
        # Individual joint command publishers (effort control)
        self.joint1_pub = rospy.Publisher('/arm_controller/joint_1_controller/command', Float64, queue_size=10)
        self.joint2_pub = rospy.Publisher('/arm_controller/joint_2_controller/command', Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher('/arm_controller/joint_3_controller/command', Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher('/arm_controller/joint_4_controller/command', Float64, queue_size=10)
        
        self.current_state_sub = rospy.Subscriber('/arm_controller/joint_states', JointState, self.joint_state_callback)
        self.desired_state_sub = rospy.Subscriber('/arm_controller/desired_joint_states', JointState, self.desired_state_callback)
        
        # Services
        self.start_service = rospy.Service('/start_joint_pid_control', Trigger, self.start_control)
        self.stop_service = rospy.Service('/stop_joint_pid_control', Trigger, self.stop_control)
        self.reset_service = rospy.Service('/reset_joint_pid_control', Trigger, self.reset_control)
        
        # Control thread
        self.control_thread = None
        self.control_thread_running = False
        
        rospy.loginfo("Joint PID Controller initialized with P+PID cascade control for 4 joints")
        
    def dynamic_reconfigure_callback(self, config, level):
        """Callback for dynamic_reconfigure parameter updates"""
        try:
            rospy.loginfo("Received dynamic reconfigure update")
            
            # Update PID parameters from dynamic_reconfigure
            for joint_name in self.joint_names:
                if joint_name in self.pid_params:
                    # Get parameters from config
                    kp_param = joint_name + "_p"
                    ki_param = joint_name + "_i"
                    kd_param = joint_name + "_d"
                    kp_outer_param = joint_name + "_p_outer"
                    max_vel_param = joint_name + "_max_velocity"
                    max_effort_param = joint_name + "_max_effort"
                    
                    if hasattr(config, kp_param):
                        self.pid_params[joint_name]['kp'] = getattr(config, kp_param)
                    if hasattr(config, ki_param):
                        self.pid_params[joint_name]['ki'] = getattr(config, ki_param)
                    if hasattr(config, kd_param):
                        self.pid_params[joint_name]['kd'] = getattr(config, kd_param)
                    if hasattr(config, kp_outer_param):
                        self.pid_params[joint_name]['kp_outer'] = getattr(config, kp_outer_param)
                    if hasattr(config, max_vel_param):
                        self.pid_params[joint_name]['max_velocity'] = getattr(config, max_vel_param)
                    if hasattr(config, max_effort_param):
                        self.pid_params[joint_name]['max_effort'] = getattr(config, max_effort_param)
                    
                    rospy.loginfo(f"Updated {joint_name} PID: Kp={self.pid_params[joint_name]['kp']:.3f}, "
                                f"Ki={self.pid_params[joint_name]['ki']:.3f}, "
                                f"Kd={self.pid_params[joint_name]['kd']:.3f}, "
                                f"Kp_outer={self.pid_params[joint_name]['kp_outer']:.3f}, "
                                f"Max_vel={self.pid_params[joint_name]['max_velocity']:.3f}, "
                                f"Max_effort={self.pid_params[joint_name]['max_effort']:.3f}")
            
            return config
        except Exception as e:
            rospy.logerr(f"Error in dynamic_reconfigure callback: {str(e)}")
            return config
        
    def joint_state_callback(self, msg):
        """Callback for current joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_states[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]
                
    def desired_state_callback(self, msg):
        """Callback for desired joint states"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.desired_joint_states[name] = msg.position[i]
                
    def calculate_pid_output(self, joint_name, error, error_derivative, error_integral, kp, ki, kd):
        """Calculate PID control output"""
        return kp * error + ki * error_integral + kd * error_derivative
        
    def calculate_cascade_control(self, joint_name, current_pos, desired_pos, current_vel, dt):
        """Calculate P+PID cascade control output for a joint"""
        if joint_name not in self.pid_params:
            return 0.0
            
        # Get PID parameters
        kp_outer = self.pid_params[joint_name]['kp_outer']
        kp = self.pid_params[joint_name]['kp']
        ki = self.pid_params[joint_name]['ki']
        kd = self.pid_params[joint_name]['kd']
        max_velocity = self.pid_params[joint_name]['max_velocity']
        max_effort = self.pid_params[joint_name]['max_effort']
        
        # Get PID state
        outer_state = self.pid_state[joint_name]['outer']
        inner_state = self.pid_state[joint_name]['inner']
        
        # Outer loop (position control) - P controller
        position_error = desired_pos - current_pos
        position_error = np.arctan2(np.sin(position_error), np.cos(position_error))  # Normalize to [-pi, pi]
        
        # Calculate desired velocity from outer loop
        desired_velocity = kp_outer * position_error
        
        # Apply velocity limit
        desired_velocity = np.clip(desired_velocity, -max_velocity, max_velocity)
        
        # Inner loop (velocity control) - PID controller
        velocity_error = desired_velocity - current_vel
        
        # Calculate integral term for inner loop
        inner_state['error_integral'] += velocity_error * dt
        
        # Anti-windup: limit integral term
        max_integral = 10.0
        inner_state['error_integral'] = np.clip(inner_state['error_integral'], -max_integral, max_integral)
        
        # Calculate derivative term for inner loop
        velocity_error_derivative = (velocity_error - inner_state['error_prev']) / dt if dt > 0 else 0.0
        
        # Calculate PID output for inner loop (effort command)
        effort = self.calculate_pid_output(joint_name, velocity_error, velocity_error_derivative, 
                                         inner_state['error_integral'], kp, ki, kd)
        
        # Apply effort limit to prevent excessive torque
        effort = np.clip(effort, -max_effort, max_effort)
        
        # Update previous error for inner loop
        inner_state['error_prev'] = velocity_error
        
        return effort
        
    def control_loop(self):
        """Main control loop"""
        rate = rospy.Rate(self.control_rate)
        
        while self.control_thread_running and not rospy.is_shutdown():
            if self.controller_active and self.joint_names:
                try:
                    # Calculate control outputs for each joint
                    for joint_name in self.joint_names:
                        if (joint_name in self.current_joint_states and 
                            joint_name in self.desired_joint_states and
                            joint_name in self.current_joint_velocities):
                            
                            current_pos = self.current_joint_states[joint_name]
                            desired_pos = self.desired_joint_states[joint_name]
                            current_vel = self.current_joint_velocities[joint_name]
                            
                            # Calculate time step
                            current_time = time.time()
                            if self.pid_state[joint_name]['inner']['last_time'] is None:
                                dt = 1.0 / self.control_rate
                            else:
                                dt = current_time - self.pid_state[joint_name]['inner']['last_time']
                            
                            self.pid_state[joint_name]['inner']['last_time'] = current_time
                            self.pid_state[joint_name]['outer']['last_time'] = current_time
                            
                            # Calculate cascade control output
                            effort = self.calculate_cascade_control(joint_name, current_pos, desired_pos, current_vel, dt)
                            
                            # Publish individual joint command
                            effort_msg = Float64()
                            effort_msg.data = effort
                            
                            if joint_name == 'joint_1':
                                self.joint1_pub.publish(effort_msg)
                            elif joint_name == 'joint_2':
                                self.joint2_pub.publish(effort_msg)
                            elif joint_name == 'joint_3':
                                self.joint3_pub.publish(effort_msg)
                            elif joint_name == 'joint_4':
                                self.joint4_pub.publish(effort_msg)
                        
                except Exception as e:
                    rospy.logerr(f"Error in control loop: {str(e)}")
                    
            rate.sleep()
            
    def start_control(self, req):
        """Start PID control"""
        try:
            if not self.controller_active:
                self.controller_active = True
                
                # Start control thread if not already running
                if not self.control_thread_running:
                    self.control_thread_running = True
                    self.control_thread = threading.Thread(target=self.control_loop)
                    self.control_thread.daemon = True
                    self.control_thread.start()
                    
                rospy.loginfo("Joint PID control started")
                return TriggerResponse(success=True, message="Joint PID control started")
            else:
                return TriggerResponse(success=False, message="Joint PID control already active")
        except Exception as e:
            return TriggerResponse(success=False, message=f"Failed to start control: {str(e)}")
            
    def stop_control(self, req):
        """Stop PID control"""
        try:
            self.controller_active = False
            rospy.loginfo("Joint PID control stopped")
            return TriggerResponse(success=True, message="Joint PID control stopped")
        except Exception as e:
            return TriggerResponse(success=False, message=f"Failed to stop control: {str(e)}")
            
    def reset_control(self, req):
        """Reset PID control state"""
        try:
            # Reset PID state variables for both outer and inner loops
            for joint_name in self.pid_state.keys():
                self.pid_state[joint_name] = {
                    'outer': {
                        'error_prev': 0.0,
                        'error_integral': 0.0,
                        'last_time': None
                    },
                    'inner': {
                        'error_prev': 0.0,
                        'error_integral': 0.0,
                        'last_time': None
                    }
                }
                
            rospy.loginfo("Joint PID control state reset")
            return TriggerResponse(success=True, message="Joint PID control state reset")
        except Exception as e:
            return TriggerResponse(success=False, message=f"Failed to reset control: {str(e)}")
            
    def update_pid_params(self, joint_name, kp, ki, kd, kp_outer, max_velocity, max_effort):
        """Update PID parameters for a joint"""
        if joint_name in self.pid_params:
            self.pid_params[joint_name]['kp'] = kp
            self.pid_params[joint_name]['ki'] = ki
            self.pid_params[joint_name]['kd'] = kd
            self.pid_params[joint_name]['kp_outer'] = kp_outer
            self.pid_params[joint_name]['max_velocity'] = max_velocity
            self.pid_params[joint_name]['max_effort'] = max_effort
            rospy.loginfo(f"Updated PID parameters for {joint_name}: Kp={kp}, Ki={ki}, Kd={kd}, Kp_outer={kp_outer}, Max_vel={max_velocity}, Max_effort={max_effort}")
            
    def run(self):
        """Main run method"""
        rospy.loginfo("Joint PID Controller node started")
        
        # Wait for joint state messages
        rospy.loginfo("Waiting for joint state messages...")
        rospy.wait_for_message('/arm_controller/joint_states', JointState, timeout=10.0)
        rospy.loginfo("Joint state messages received")
        
        # Spin
        rospy.spin()
        
        # Cleanup
        self.control_thread_running = False
        if self.control_thread:
            self.control_thread.join()

if __name__ == '__main__':
    try:
        controller = JointPIDController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Joint PID Controller error: {str(e)}") 