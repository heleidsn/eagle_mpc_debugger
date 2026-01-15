#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import PositionTarget, AttitudeTarget
from geometry_msgs.msg import Vector3
from mav_msgs.msg import Actuators

def actuator_control_publisher():
    

    pub = rospy.Publisher('/mavros/actuator_control', ActuatorControl, queue_size=10)
    rate = rospy.Rate(100)  # 50 Hz

    # # Switch to OFFBOARD mode
    # rospy.wait_for_service('/mavros/set_mode')
    # set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    # set_mode_srv(custom_mode="OFFBOARD")

    # # Arm vehicle
    # rospy.wait_for_service('/mavros/cmd/arming')
    # arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    # arm_srv(True)

    while not rospy.is_shutdown():
        msg = ActuatorControl()
        msg.header.stamp = rospy.Time.now()
        msg.group_mix = 0
        msg.controls = [0.0]*8
        msg.controls[0] = 0.0   # roll
        msg.controls[1] = 0.0   # pitch
        msg.controls[2] = 0.0   # yaw
        msg.controls[3] = 0.65   # thrust (50%)
        pub.publish(msg)

        rate.sleep()
        
def setpoint_raw_publisher():
    # rospy.init_node('actuator_control_node')

    pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)
    rate = rospy.Rate(50)  # 50 Hz

    # # Switch to OFFBOARD mode
    # rospy.wait_for_service('/mavros/set_mode')
    # set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    # set_mode_srv(custom_mode="OFFBOARD")

    # # Arm vehicle
    # rospy.wait_for_service('/mavros/cmd/arming')
    # arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    # arm_srv(True)

    while not rospy.is_shutdown():
        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        msg.body_rate = Vector3(0.0, 0.0, 0.0)
        msg.thrust = 0.2
        pub.publish(msg)

        rate.sleep()
        
def motor_control_publisher():
    rospy.init_node('motor_control_node')
    
    pub = rospy.Publisher("/s500/gazebo/command/motor_speed", Actuators, queue_size=1)
    rate = rospy.Rate(500)
    msg = Actuators()
    msg.angular_velocities = [1000.0, 1000.0, 1000.0, 1000.0]  # rad/s
    
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('actuator_control_node')
        # setpoint_raw_publisher()
        actuator_control_publisher()
        # motor_control_publisher()
    except rospy.ROSInterruptException:
        pass
