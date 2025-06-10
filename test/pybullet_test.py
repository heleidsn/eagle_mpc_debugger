import pybullet as p

p.connect(p.GUI)
robot_id = p.loadURDF("/home/helei/catkin_eagle_mpc/src/example_robot_data/robots/s500_description/s500_uam/urdf/s500_uam.urdf")
p.setJointMotorControl2(robot_id, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=1.0)