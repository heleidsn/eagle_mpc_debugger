import os
import signal
import sys
import time
import example_robot_data
import numpy as np
import pinocchio
import crocoddyl
import matplotlib.pyplot as plt

signal.signal(signal.SIGINT, signal.SIG_DFL)

# 1️⃣ 加载模型
hector = example_robot_data.load("hector")
robot_model = hector.model

target_pos = np.array([1.0, 0.0, 1.0])
target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)
state = crocoddyl.StateMultibody(robot_model)

# 2️⃣ 定义致动器
d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
ps = [
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([d_cog, 0, 0])), cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([0, d_cog, 0])), cm / cf, crocoddyl.ThrusterType.CW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([-d_cog, 0, 0])), cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(pinocchio.SE3(np.eye(3), np.array([0, -d_cog, 0])), cm / cf, crocoddyl.ThrusterType.CW),
]
actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
nu = actuation.nu

# 3️⃣ 定义 cost
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0.1]*3 + [1000.0]*3 + [1000.0]*robot_model.nv)
)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("base_link"),
    pinocchio.SE3(target_quat.matrix(), target_pos),
    nu,
)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)

runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
terminalCostModel.addCost("goalPose", goalTrackingCost, 3.0)

# 4️⃣ 定义 running & terminal 模型
dt = 3e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt
)
# 注意 terminal 的 dt=0
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.0
)

# 5️⃣ 构建 Problem
x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
T = 33
problem = crocoddyl.ShootingProblem(x0, [runningModel]*T, terminalModel)

# 6️⃣ 初始 guess
xs_init = [x0.copy() for _ in range(T+1)]
us_init = [np.zeros(nu) for _ in range(T)]

# 7️⃣ Solver
solver = crocoddyl.SolverFDDP(problem)

# === 关键：添加 plot 回调 ===
log = crocoddyl.CallbackLogger()
# plot = crocoddyl.CallbackPlot()
solver.setCallbacks([log])

solver.solve(xs_init, us_init, maxiter=100, isFeasible=False)

# 8️⃣ 手动绘图（可选）
plt.figure(figsize=(10,4))
plt.semilogy(log.costs, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.legend()
plt.title("Cost Convergence")
plt.show()
