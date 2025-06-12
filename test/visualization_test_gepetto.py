import example_robot_data
import crocoddyl
import time
import gepetto
from pinocchio.visualize import GepettoVisualizer
import numpy as np
from scipy.spatial.transform import Rotation as R

robot_name = 's500_uam_simple'  # 's500_uam_simple' or 'hexacopter370_flying_arm_3'
robot = example_robot_data.load(robot_name)

print(robot)
print(example_robot_data.__file__)

rate = -1
freq = 1
cameraTF = [-0.03, 4.4, 2.3, 0, 0.7071, 0, 0.7071]

gepetto.corbaserver.Client()

# display = crocoddyl.GepettoDisplay(robot, rate, freq, cameraTF, floor=False)
viz = GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)
viz.loadViewerModel()

roll, pitch, yaw = 1, 1, 0

# get quaternion from roll, pitch, yaw
quat = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

# get robot initial state
viz.display(np.array([0, 0, 0, quat[0], quat[1], quat[2], quat[3], 1, 1]))
# viz.display(robot.q0)
print(robot.q0)

print('display robot with initial state')
