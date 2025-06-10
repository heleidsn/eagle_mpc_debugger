#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Test script for arm joint1 step response
'''

import rospy
import numpy as np
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import deque

class ArmStepResponseTest:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('arm_step_response_test', anonymous=False, log_level=rospy.INFO)
        
        # Get parameters
        self.test_duration = rospy.get_param('~test_duration', 10.0)  # seconds
        self.step_amplitude = rospy.get_param('~step_amplitude', 0.05)  # Nm
        self.sampling_rate = rospy.get_param('~sampling_rate', 100.0)  # Hz
        self.switch_interval = rospy.get_param('~switch_interval', 2.0)  # seconds
        
        # Initialize data storage using deque with maxlen
        max_samples = int(self.test_duration * self.sampling_rate)
        self.time_data = deque(maxlen=max_samples)
        self.position_data = deque(maxlen=max_samples)
        self.velocity_data = deque(maxlen=max_samples)
        self.effort_data = deque(maxlen=max_samples)
        self.command_data = deque(maxlen=max_samples)
        
        # Initialize publishers and subscribers
        self.arm_control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        self.arm_state_sub = rospy.Subscriber('/joint_states', JointState, self.arm_state_callback)
        
        # Initialize test state
        self.test_started = False
        self.test_finished = False
        self.start_time = None
        self.last_switch_time = None
        self.current_amplitude = self.step_amplitude  # Start with positive amplitude
        
        # Create timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.sampling_rate), self.control_callback)
        
        rospy.loginfo("Arm step response test initialized")
        rospy.loginfo(f"Test parameters: duration={self.test_duration}s, amplitude={self.step_amplitude}Nm, switch_interval={self.switch_interval}s")
        
    def arm_state_callback(self, msg):
        """Callback for joint states"""
        if not self.test_started or self.test_finished:
            return
            
        # Get current time
        current_time = (rospy.Time.now() - self.start_time).to_sec()
        
        # Store data
        self.time_data.append(current_time)
        self.position_data.append(msg.position[-1])  # joint1 position
        self.velocity_data.append(msg.velocity[-1])  # joint1 velocity
        self.effort_data.append(msg.effort[-1])      # joint1 effort
        self.command_data.append(self.current_amplitude * 1000)  # Store current command
        
        # Check if test duration is reached
        if current_time >= self.test_duration:
            self.test_finished = True
            self.save_and_plot_data()
            
    def control_callback(self, event):
        """Timer callback for control loop"""
        if not self.test_started:
            self.test_started = True
            self.start_time = rospy.Time.now()
            self.last_switch_time = self.start_time
            rospy.loginfo("Test started")
            return
            
        if self.test_finished:
            return
            
        # Check if it's time to switch amplitude
        current_time = rospy.Time.now()
        if (current_time - self.last_switch_time).to_sec() >= self.switch_interval:
            self.current_amplitude = -self.current_amplitude
            self.last_switch_time = current_time
            rospy.loginfo(f"Switching amplitude to: {self.current_amplitude}")
            
        # Create joint state message
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['joint_1', 'joint_2']
        
        # Set effort command
        joint_msg.position = [0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0]
        joint_msg.effort = [self.current_amplitude, 0.0]  # Step command for joint1
        
        # Publish command
        self.arm_control_pub.publish(joint_msg)
        
    def save_and_plot_data(self):
        """Save test data and generate plots"""
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert deques to numpy arrays
        time_array = np.array(self.time_data)
        position_array = np.array(self.position_data)
        velocity_array = np.array(self.velocity_data)
        effort_array = np.array(self.effort_data)
        command_array = np.array(self.command_data)
        
        # Print array lengths for debugging
        rospy.loginfo(f"Array lengths - Time: {len(time_array)}, Position: {len(position_array)}, "
                     f"Velocity: {len(velocity_array)}, Effort: {len(effort_array)}, "
                     f"Command: {len(command_array)}")
        
        # Ensure all arrays have the same length
        min_length = min(len(time_array), len(position_array), len(velocity_array), 
                        len(effort_array), len(command_array))
        
        time_array = time_array[:min_length]
        position_array = position_array[:min_length]
        velocity_array = velocity_array[:min_length]
        effort_array = effort_array[:min_length]
        command_array = command_array[:min_length]
        
        # Save raw data
        data = np.column_stack((
            time_array,
            position_array,
            velocity_array,
            effort_array,
            command_array
        ))
        
        np.savetxt(
            os.path.join(data_dir, f'step_response_{timestamp}.csv'),
            data,
            delimiter=',',
            header='time,position,velocity,effort,command',
            comments=''
        )
        
        # Generate plots
        plt.figure(figsize=(12, 8))
        
        # Position plot
        plt.subplot(311)
        plt.plot(time_array, position_array, 'b-', label='Position')
        plt.grid(True)
        plt.ylabel('Position (rad)')
        plt.legend()
        
        # Velocity plot
        plt.subplot(312)
        plt.plot(time_array, velocity_array, 'g-', label='Velocity')
        plt.grid(True)
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        
        # Effort plot
        plt.subplot(313)
        plt.plot(time_array, effort_array, 'r-', label='Effort')
        plt.plot(time_array, command_array, 'k--', label='Command')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Effort (Nm)')
        plt.legend()
        
        # Add vertical lines to show command switches
        for t in time_array[::int(self.switch_interval * self.sampling_rate)]:
            plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'step_response_{timestamp}.png'))
        plt.close()
        
        rospy.loginfo(f"Test data and plots saved to {data_dir}")
        rospy.loginfo(f"Data points collected: {min_length}")
        rospy.loginfo(f"Command range: [{min(command_array):.3f}, {max(command_array):.3f}]")
        
if __name__ == '__main__':
    try:
        test = ArmStepResponseTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.") 