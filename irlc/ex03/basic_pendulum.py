# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
import numpy as np
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from gymnasium.spaces import Box

class BasicPendulumModel(ControlModel): 
    def sym_f(self, x, u, t=None):
        g = 9.82
        l = 1
        m = 2
        theta_dot = x[1]  # Parameterization: x = [theta, theta']
        theta_dot_dot = g / l * sym.sin(x[0]) + 1 / (m * l ** 2) * u[0]
        return [theta_dot, theta_dot_dot]

    def get_cost(self) -> SymbolicQRCost:
        return SymbolicQRCost(Q=np.eye(2), R=np.eye(1))

    def u_bound(self) -> Box:
        return Box(np.asarray([-10]), np.asarray([10]))

    def x0_bound(self) -> Box:
        return Box(np.asarray( [np.pi, 0] ), np.asarray( [np.pi, 0])) 

if __name__ == "__main__":
    p = BasicPendulumModel() 
    print(p) 

    from irlc.ex04.discrete_control_model import DiscreteControlModel
    model = BasicPendulumModel()
    discrete_pendulum = DiscreteControlModel(model, dt=0.5)  # Using a discretization time step: 0.5 seconds.
    x0 = model.x0_bound().low  # Get the initial state: x0 = [np.pi, 0].
    u0 = [0]  # No action. Note the action must be a list.
    x1 = discrete_pendulum.f(x0, u0)
    print(x1)
    print("Now, lets compute the Euler step manually to confirm")
    x1_manual = x0 + 0.5 * model.f(x0, u0, 0)
    print(x1_manual)
