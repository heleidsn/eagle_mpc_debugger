import example_robot_data
import crocoddyl
import time
import gepetto
from pinocchio.visualize import GepettoVisualizer
import numpy as np

robot = example_robot_data.load('s500_uam')

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

# get robot initial state
# viz.display(np.array([1, 1, 1, 0, 0, 0, 1]))
viz.display(robot.q0)
print(robot.q0)

print('display robot with initial state')
