#!/usr/bin/env python3
'''
Author: Lei He
Date: 2024-12-19
Description: UAV takeoff, move to target position, and land script
'''

import rospy
import numpy as np
import time
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import SetMode, SetModeRequest, CommandBool, CommandBoolRequest
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from std_msgs.msg import Float32
from gazebo_msgs.msg import ModelStates

class UAVTakeoffMoveLand:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('uav_takeoff_move_land', anonymous=False, log_level=rospy.INFO)
        
        # Get parameters
        self.control_rate = rospy.get_param('~control_rate', 50.0)  # Hz
        self.odom_source = rospy.get_param('~odom_source', 'mavros')  # mavros, gazebo
        self.use_simulation = rospy.get_param('~use_simulation', True)
        
        # Mission parameters
        self.takeoff_height = rospy.get_param('~takeoff_height', 1.0)  # meters
        self.target_x = rospy.get_param('~target_x', 2.0)  # meters
        self.target_y = rospy.get_param('~target_y', 2.0)  # meters
        self.target_z = rospy.get_param('~target_z', 1.0)  # meters
        self.target_yaw = rospy.get_param('~target_yaw', 0.0)  # radians
        
        # Position tolerance for reaching waypoints
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.1)  # meters
        self.yaw_tolerance = rospy.get_param('~yaw_tolerance', 0.1)  # radians
        
        # State variables
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        self.mission_state = "WAITING_FOR_READY"  # WAITING_FOR_READY, TAKEOFF, HOVER, MOVE, LAND, COMPLETE
        self.waypoint_reached = False
        self.waypoint_timeout = 30.0  # seconds
        self.waypoint_start_time = None
        self.mission_started = False
        self.setpoint_sent = False  # Track if we've started sending setpoints
        
        # Initialize subscribers
        self.mav_state_sub = rospy.Subscriber('/mavros/state', State, self.mav_state_callback)
        
        if self.odom_source == 'mavros':
            self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
        else:
            self.odom_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gazebo_odom_callback)
        
        # Initialize publishers
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.pose_pub = rospy.Publisher('/reference/pose', PoseStamped, queue_size=10)
        
        # Initialize service clients
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        
        # Wait for services to be available
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        
        # Initialize control timer
        self.control_timer = rospy.Timer(rospy.Duration(1.0/self.control_rate), self.control_callback)
        
        # Mission status timer
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.status_callback)
        
        rospy.loginfo("UAV Takeoff Move Land node initialized")
        rospy.loginfo(f"Target position: x={self.target_x}, y={self.target_y}, z={self.target_z}")
        rospy.loginfo(f"Takeoff height: {self.takeoff_height} meters")
        rospy.loginfo("Waiting for UAV to be armed and in OFFBOARD mode...")
        rospy.loginfo("Please use RC to arm UAV and switch to OFFBOARD mode")
        rospy.loginfo("Sending setpoints to enable OFFBOARD mode...")
        
    def mav_state_callback(self, msg):
        """Callback for MAV state"""
        self.current_state = msg
        
    def odom_callback(self, msg):
        """Callback for MAVROS odometry"""
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose
        self.current_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
    def gazebo_odom_callback(self, msg):
        """Callback for Gazebo model states"""
        try:
            uav_index = msg.name.index('s500')
            pose = msg.pose[uav_index]
            twist = msg.twist[uav_index]
            
            self.current_pose.header.stamp = rospy.Time.now()
            self.current_pose.header.frame_id = "map"
            self.current_pose.pose = pose
            
            self.current_velocity = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z
            ])
        except ValueError:
            rospy.logwarn("UAV model not found in Gazebo model states")
            
    def get_current_position(self):
        """Get current position as numpy array"""
        return np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])
        
    def get_current_yaw(self):
        """Get current yaw angle from quaternion"""
        quat = self.current_pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        return yaw
        
    def distance_to_target(self, target_pos):
        """Calculate distance to target position"""
        current_pos = self.get_current_position()
        return np.linalg.norm(current_pos - target_pos)
        
    def yaw_distance_to_target(self, target_yaw):
        """Calculate yaw distance to target"""
        current_yaw = self.get_current_yaw()
        yaw_diff = target_yaw - current_yaw
        
        # Normalize to [-pi, pi]
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
            
        return abs(yaw_diff)
        
    def is_waypoint_reached(self, target_pos, target_yaw=None):
        """Check if waypoint is reached"""
        pos_distance = self.distance_to_target(target_pos)
        pos_reached = pos_distance < self.position_tolerance
        
        if target_yaw is not None:
            yaw_distance = self.yaw_distance_to_target(target_yaw)
            yaw_reached = yaw_distance < self.yaw_tolerance
            return pos_reached and yaw_reached
        else:
            return pos_reached
            
    def send_setpoint(self, x, y, z, yaw=0.0, frame_id="LOCAL_NED"):
        """Send position setpoint to UAV"""
        setpoint = PositionTarget()
        setpoint.header.stamp = rospy.Time.now()
        setpoint.header.frame_id = frame_id
        
        # Set position
        setpoint.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        setpoint.type_mask = (PositionTarget.IGNORE_VX + 
                             PositionTarget.IGNORE_VY + 
                             PositionTarget.IGNORE_VZ + 
                             PositionTarget.IGNORE_AFX + 
                             PositionTarget.IGNORE_AFY + 
                             PositionTarget.IGNORE_AFZ + 
                             PositionTarget.IGNORE_YAW_RATE)
        
        setpoint.position.x = x
        setpoint.position.y = y
        setpoint.position.z = z
        setpoint.yaw = yaw
        
        self.setpoint_pub.publish(setpoint)
        
        # Also publish as PoseStamped for visualization
        pose_msg = PoseStamped()
        pose_msg.header = setpoint.header
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        
        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, yaw)
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)
        
    def send_land_command(self):
        """Send land command to UAV"""
        rospy.loginfo("Sending land command...")
        
        try:
            response = self.set_mode_client(custom_mode='AUTO.LAND')
            if response.mode_sent:
                rospy.loginfo("Successfully sent AUTO.LAND command")
                return True
            else:
                rospy.logerr("Failed to send AUTO.LAND command")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
            

            
    def control_callback(self, event):
        """Main control loop"""
        if not self.current_state.connected:
            rospy.logwarn("UAV not connected")
            return
            
        # State machine for mission execution
        if self.mission_state == "WAITING_FOR_READY":
            # Always send setpoints to enable OFFBOARD mode
            if not self.setpoint_sent:
                rospy.loginfo("Starting to send setpoints for OFFBOARD mode...")
                self.setpoint_sent = True
            
            # Send current position as setpoint to enable OFFBOARD mode
            current_pos = self.get_current_position()
            self.send_setpoint(current_pos[0], current_pos[1], current_pos[2])
            
            # Wait for UAV to be armed and in OFFBOARD mode
            if self.current_state.armed and self.current_state.mode == "OFFBOARD":
                if not self.mission_started:
                    rospy.loginfo("UAV is armed and in OFFBOARD mode, starting mission...")
                    self.mission_started = True
                    self.mission_state = "TAKEOFF"
                    self.waypoint_start_time = rospy.Time.now()
            else:
                # Log current status
                if not self.current_state.armed:
                    rospy.loginfo_throttle(5, "Waiting for UAV to be armed...")
                elif self.current_state.mode != "OFFBOARD":
                    rospy.loginfo_throttle(5, f"Waiting for OFFBOARD mode, current mode: {self.current_state.mode}")
                
        elif self.mission_state == "TAKEOFF":
            # Takeoff to specified height
            target_pos = np.array([0, 0, self.takeoff_height])
            self.send_setpoint(target_pos[0], target_pos[1], target_pos[2])
            
            if self.is_waypoint_reached(target_pos):
                rospy.loginfo(f"Reached takeoff height: {self.takeoff_height}m")
                self.mission_state = "HOVER"
                self.waypoint_start_time = rospy.Time.now()
            else:
                # Check timeout
                if (rospy.Time.now() - self.waypoint_start_time).to_sec() > self.waypoint_timeout:
                    rospy.logerr("Takeoff timeout, aborting mission")
                    self.mission_state = "LAND"
                    
        elif self.mission_state == "HOVER":
            # Hover at takeoff height for a few seconds
            target_pos = np.array([0, 0, self.takeoff_height])
            self.send_setpoint(target_pos[0], target_pos[1], target_pos[2])
            
            hover_time = 3.0  # seconds
            if (rospy.Time.now() - self.waypoint_start_time).to_sec() > hover_time:
                rospy.loginfo("Hover complete, moving to target position...")
                self.mission_state = "MOVE"
                self.waypoint_start_time = rospy.Time.now()
                
        elif self.mission_state == "MOVE":
            # Move to target position
            target_pos = np.array([self.target_x, self.target_y, self.target_z])
            self.send_setpoint(target_pos[0], target_pos[1], target_pos[2], self.target_yaw)
            
            if self.is_waypoint_reached(target_pos, self.target_yaw):
                rospy.loginfo(f"Reached target position: {target_pos}")
                self.mission_state = "HOVER_AT_TARGET"
                self.waypoint_start_time = rospy.Time.now()
            else:
                # Check timeout
                if (rospy.Time.now() - self.waypoint_start_time).to_sec() > self.waypoint_timeout:
                    rospy.logerr("Move timeout, starting landing")
                    self.mission_state = "LAND"
                    
        elif self.mission_state == "HOVER_AT_TARGET":
            # Hover at target position for a few seconds
            target_pos = np.array([self.target_x, self.target_y, self.target_z])
            self.send_setpoint(target_pos[0], target_pos[1], target_pos[2], self.target_yaw)
            
            hover_time = 3.0  # seconds
            if (rospy.Time.now() - self.waypoint_start_time).to_sec() > hover_time:
                rospy.loginfo("Hover at target complete, starting landing...")
                self.mission_state = "LAND"
                
        elif self.mission_state == "LAND":
            # Land the UAV
            if self.send_land_command():
                self.mission_state = "COMPLETE"
            else:
                # If auto land fails, try manual landing
                rospy.logwarn("Auto land failed, trying manual landing...")
                self.send_setpoint(0, 0, 0)  # Send zero setpoint for manual landing
                
        elif self.mission_state == "COMPLETE":
            # Mission completed
            rospy.loginfo("Mission completed successfully!")
            self.control_timer.shutdown()
            
    def status_callback(self, event):
        """Status callback for logging"""
        if self.current_state.connected:
            current_pos = self.get_current_position()
            current_yaw = self.get_current_yaw()
            
            rospy.loginfo(f"Mission State: {self.mission_state}")
            rospy.loginfo(f"Position: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}]")
            rospy.loginfo(f"Yaw: {math.degrees(current_yaw):.1f}°")
            rospy.loginfo(f"Mode: {self.current_state.mode}, Armed: {self.current_state.armed}")
            
    def run(self):
        """Main run function"""
        rospy.loginfo("Starting UAV Takeoff Move Land mission...")
        rospy.loginfo("Press Ctrl+C to abort")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Mission aborted by user")
            # Try to land safely
            self.send_land_command()
        except Exception as e:
            rospy.logerr(f"Mission failed with error: {e}")
            # Try to land safely
            self.send_land_command()

if __name__ == "__main__":
    try:
        uav_mission = UAVTakeoffMoveLand()
        uav_mission.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Node failed with error: {e}") 