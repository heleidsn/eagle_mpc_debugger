#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-03-19
Description: Test script for arm joint1 PID position control
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

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-0.4, 0.4), control_frequency=100.0, smoothing_factor=0.9):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Add smoothing parameters
        self.prev_output = 0.0
        self.smoothing_factor = smoothing_factor  # Lower value means more smoothing
        
        # Fixed time step based on sampling rate
        self.dt = 1.0 / control_frequency  # Default to 100Hz, will be updated in main class
        
    def compute(self, setpoint, measurement, current_time):
        # Calculate error
        error = setpoint - measurement
        
        # Calculate integral term with anti-windup
        self.integral += error * self.dt
        # Limit integral term to prevent windup
        max_integral = 0.1 / self.ki if self.ki != 0 else 0.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # Calculate derivative term with smoothing
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        # Calculate PID output
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Apply smoothing
        smoothed_output = (1 - self.smoothing_factor) * self.prev_output + self.smoothing_factor * output
        self.prev_output = smoothed_output
        
        return smoothed_output

class ArmPIDControlTest:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('arm_pid_control_test', anonymous=False, log_level=rospy.INFO)
        
        # Get parameters
        self.test_duration = rospy.get_param('~test_duration', 10.0)  # seconds
        self.sampling_rate = rospy.get_param('~sampling_rate', 100.0)  # Hz
        self.switch_interval = rospy.get_param('~switch_interval', 2.0)  # seconds
        
        # Control mode
        self.control_mode = rospy.get_param('~control_mode', 'torque')  # 'torque' or 'position'
        
        # PID parameters
        self.kp = rospy.get_param('~kp', 0.6)
        self.ki = rospy.get_param('~ki', 0.2)
        self.kd = rospy.get_param('~kd', 0.1)
        
        # Initialize PID controller with fixed dt
        self.pid = PIDController(self.kp, self.ki, self.kd, control_frequency=self.sampling_rate, smoothing_factor=0.9)
        self.pid.dt = 1.0 / self.sampling_rate  # Set fixed dt based on sampling rate
        
        # Initialize data storage using deque with maxlen
        max_samples = int(self.test_duration * self.sampling_rate)
        self.time_data = deque(maxlen=max_samples)
        self.position_data = deque(maxlen=max_samples)
        self.velocity_data = deque(maxlen=max_samples)
        self.effort_data = deque(maxlen=max_samples)
        self.command_data = deque(maxlen=max_samples)
        self.setpoint_data = deque(maxlen=max_samples)
        self.response_delay = deque(maxlen=max_samples)  # Store response delay
        
        # Initialize raw data storage for interpolation
        self.raw_time_data = []
        self.raw_position_data = []
        self.raw_velocity_data = []
        self.raw_effort_data = []
        
        # Initialize publishers and subscribers
        self.arm_control_pub = rospy.Publisher('/desired_joint_states', JointState, queue_size=10)
        self.arm_state_sub = rospy.Subscriber('/joint_states', JointState, self.arm_state_callback)
        
        # Initialize test state
        self.test_started = False
        self.test_finished = False
        self.start_time = None
        self.last_switch_time = None
        self.current_setpoint = 0.0  # Start with zero setpoint
        self.last_setpoint = None
        self.setpoint_switch_time = None
        self.response_threshold = 0.1  # Threshold for response detection
        self.initial_delay = 1.0  # Initial delay before starting setpoint switching
        
        # Create timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.sampling_rate), self.control_callback)
        
        rospy.loginfo("Arm PID control test initialized")
        rospy.loginfo(f"Test parameters: duration={self.test_duration}s, switch_interval={self.switch_interval}s")
        rospy.loginfo(f"Control mode: {self.control_mode}")
        rospy.loginfo(f"PID parameters: kp={self.kp}, ki={self.ki}, kd={self.kd}")
        
    def arm_state_callback(self, msg):
        """Callback for joint states"""
        if not self.test_started or self.test_finished:
            return
            
        # Get current time
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.start_time).to_sec()
        
        # Get current position
        current_position = msg.position[-1]  # joint1 position
        
        # Calculate response delay when setpoint changes
        if self.last_setpoint is not None and self.last_setpoint != self.current_setpoint:
            if self.setpoint_switch_time is not None:
                # Check if position has moved beyond threshold
                if abs(current_position - self.last_setpoint) > self.response_threshold:
                    delay = elapsed_time - self.setpoint_switch_time
                    self.response_delay.append(delay)
                    rospy.loginfo(f"Response delay: {delay:.3f}s")
                    self.setpoint_switch_time = None
        
        # Store raw data for interpolation
        self.raw_time_data.append(elapsed_time)
        self.raw_position_data.append(current_position)
        self.raw_velocity_data.append(msg.velocity[-1])
        self.raw_effort_data.append(msg.effort[-1])
        
        # Check if test duration is reached
        if elapsed_time >= self.test_duration:
            self.test_finished = True
            self.save_and_plot_data()
            
    def control_callback(self, event):
        """Timer callback for control loop"""
        if not self.test_started:
            self.test_started = True
            self.start_time = rospy.Time.now()
            self.last_switch_time = self.start_time
            self.last_setpoint = self.current_setpoint
            rospy.loginfo("Test started")
            return
            
        if self.test_finished:
            return
            
        # Get current time since start
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.start_time).to_sec()
        
        # Check if we're in the final second of the test
        if elapsed_time >= (self.test_duration - 2.0):
            if self.current_setpoint != 0.0:
                self.last_setpoint = self.current_setpoint
                self.current_setpoint = 0.0
                self.last_switch_time = current_time
                self.setpoint_switch_time = elapsed_time
                rospy.loginfo(f"Final phase: Switching setpoint to 0.0 at time {elapsed_time:.2f}s")
        # Normal switching logic before the final second
        elif elapsed_time >= self.initial_delay:
            time_since_last_switch = (current_time - self.last_switch_time).to_sec()
            if time_since_last_switch >= self.switch_interval:
                self.last_setpoint = self.current_setpoint
                if self.current_setpoint == 0.0:
                    self.current_setpoint = 1.0  # First switch after initial delay
                else:
                    self.current_setpoint = -self.current_setpoint  # Subsequent switches
                self.last_switch_time = current_time
                self.setpoint_switch_time = elapsed_time
                rospy.loginfo(f"Switching setpoint to: {self.current_setpoint} at time {elapsed_time:.2f}s")
            
        # Get current position from the last received joint state
        if len(self.raw_position_data) > 0:
            # Interpolate position data
            current_position = np.interp(elapsed_time, self.raw_time_data, self.raw_position_data)
            
            # Create joint state message
            joint_msg = JointState()
            joint_msg.header.stamp = current_time
            joint_msg.name = ['joint_1', 'joint_2']
            
            if self.control_mode == 'torque':
                # Use PID controller to compute control output
                control_output = self.pid.compute(self.current_setpoint, current_position, current_time)
                # Set effort command
                joint_msg.position = [0.0, 0.0]
                joint_msg.velocity = [0.0, 0.0]
                joint_msg.effort = [control_output, 0.0]
                
                # Store the actual command being sent
                self.command_data.append(control_output*1000)  # Store command in mNm
                
                # Store interpolated data
                self.time_data.append(elapsed_time)
                self.position_data.append(current_position)
                self.velocity_data.append(np.interp(elapsed_time, self.raw_time_data, self.raw_velocity_data))
                self.effort_data.append(np.interp(elapsed_time, self.raw_time_data, self.raw_effort_data))
                self.setpoint_data.append(self.current_setpoint)
            else:  # position control
                joint_msg.position = [self.current_setpoint, 0.0]
                joint_msg.velocity = [0.0, 0.0]
                joint_msg.effort = [0.0, 0.0]
            
            # Publish command
            self.arm_control_pub.publish(joint_msg)
        
    def save_and_plot_data(self):
        """Save test data and generate plots"""
        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with PID parameters
            pid_params = f"kp{self.kp:.2f}_ki{self.ki:.2f}_kd{self.kd:.2f}"
            filename_base = f'pid_control_{self.control_mode}_{timestamp}_{pid_params}'
            
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Convert deques to numpy arrays
            time_array = np.array(self.time_data)
            position_array = np.array(self.position_data)
            velocity_array = np.array(self.velocity_data)
            effort_array = np.array(self.effort_data)
            command_array = np.array(self.command_data)
            setpoint_array = np.array(self.setpoint_data)
            response_delay_array = np.array(self.response_delay)
            
            # Print array lengths for debugging
            rospy.loginfo(f"Array lengths - Time: {len(time_array)}, Position: {len(position_array)}, "
                         f"Velocity: {len(velocity_array)}, Effort: {len(effort_array)}, "
                         f"Command: {len(command_array)}, Setpoint: {len(setpoint_array)}")
            
            # Calculate average response delay
            if len(response_delay_array) > 0:
                avg_delay = np.mean(response_delay_array)
                rospy.loginfo(f"Average response delay: {avg_delay:.3f}s")
            
            # Ensure all arrays have the same length
            min_length = min(len(time_array), len(position_array), len(velocity_array), 
                            len(effort_array), len(command_array), len(setpoint_array))
            
            time_array = time_array[:min_length]
            position_array = position_array[:min_length]
            velocity_array = velocity_array[:min_length]
            effort_array = effort_array[:min_length]
            command_array = command_array[:min_length]
            setpoint_array = setpoint_array[:min_length]
            
            # Save raw data
            data = np.column_stack((
                time_array,
                position_array,
                velocity_array,
                effort_array,
                command_array,
                setpoint_array
            ))
            
            np.savetxt(
                os.path.join(data_dir, f'{filename_base}.csv'),
                data,
                delimiter=',',
                header='time,position,velocity,effort,command,setpoint',
                comments=''
            )
            
            # Generate plots
            plt.figure(figsize=(12, 10))
            
            # Position plot
            plt.subplot(311)
            plt.plot(time_array, position_array, 'b-', label='Position')
            plt.plot(time_array, setpoint_array, 'r--', label='Setpoint')
            plt.grid(True)
            plt.ylabel('Position (rad)')
            plt.legend()
            
            # Velocity plot
            plt.subplot(312)
            plt.plot(time_array, velocity_array, 'g-', label='Velocity')
            plt.grid(True)
            plt.ylabel('Velocity (rad/s)')
            plt.legend()
            
            # Combined Effort and Command plot
            plt.subplot(313)
            plt.plot(time_array, effort_array, 'r-', label='Effort')
            plt.plot(time_array, command_array, 'k--', label='Command')
            plt.grid(True)
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.legend()
            
            # Add vertical lines to show setpoint switches
            for t in time_array[::int(self.switch_interval * self.sampling_rate)]:
                plt.axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(data_dir, f'{filename_base}.png')
            plt.savefig(plot_path)
            plt.close()
            
            rospy.loginfo(f"Test data and plots saved to {data_dir}")
            rospy.loginfo(f"Data points collected: {min_length}")
            rospy.loginfo(f"Command range: [{min(command_array):.3f}, {max(command_array):.3f}]")
            
            # Verify that the plot file was created
            if os.path.exists(plot_path):
                rospy.loginfo(f"Plot saved successfully: {plot_path}")
            else:
                rospy.logwarn("Failed to save plot file!")
            
            # Shutdown ROS node and exit
            rospy.loginfo("Test completed, shutting down...")
            rospy.signal_shutdown("Test completed")
            
        except Exception as e:
            rospy.logerr(f"Error during data saving: {str(e)}")
            raise
        
if __name__ == '__main__':
    try:
        test = ArmPIDControlTest()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
    finally:
        rospy.loginfo("Program exited.") 