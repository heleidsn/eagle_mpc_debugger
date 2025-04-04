from scipy.spatial.transform import Rotation as R
import numpy as np

vel_body_world = np.array([1.0, 0.0, 0.0])
rot = R.from_euler('xyz', [0, 50, 0], degrees=True)  # body to world

quat_from_rot = rot.as_quat()
print(quat_from_rot)
quat = np.array([0.0, 0.0, 0.0, 1.0])
rot_2 = R.from_quat([quat_from_rot[0], quat_from_rot[1], quat_from_rot[2], quat_from_rot[3]])

R_mat = rot_2.as_matrix()
R_mat_inv = R_mat.T
vel_world = R_mat @ vel_body_world
print(vel_world)

test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(test_array[6:9])