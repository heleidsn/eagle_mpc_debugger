#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import ActuatorControl
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import PositionTarget, AttitudeTarget
from geometry_msgs.msg import Vector3

def actuator_control_publisher():
    rospy.init_node('actuator_control_node')

    pub = rospy.Publisher('/mavros/actuator_control', ActuatorControl, queue_size=10)
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
        msg = ActuatorControl()
        msg.header.stamp = rospy.Time.now()
        msg.group_mix = 0
        msg.controls = [0.0]*8
        msg.controls[0] = 0.0   # roll
        msg.controls[1] = 0.0   # pitch
        msg.controls[2] = 0.0   # yaw
        msg.controls[3] = 0.5   # thrust (50%)
        pub.publish(msg)

        rate.sleep()
        
def setpoint_raw_publisher():
    rospy.init_node('actuator_control_node')

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

if __name__ == '__main__':
    try:
        setpoint_raw_publisher()
    except rospy.ROSInterruptException:
        pass
