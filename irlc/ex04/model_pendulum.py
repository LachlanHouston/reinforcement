# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from irlc.ex04.discrete_control_model import DiscreteControlModel
import gymnasium as gym
from gymnasium.spaces.box import Box
from irlc.ex04.control_environment import ControlEnvironment
import numpy as np

"""
SEE: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""
class PendulumModel(ControlModel):
    state_labels= [r"$\theta$", r"$\frac{d \theta}{dt}$"]
    action_labels = ['Torque $u$']
    x_upright, x_down = np.asarray([0.0, 0.0]), np.asarray([np.pi, 0.0])

    def __init__(self, l=1., m=.8, friction=0.0, max_torque=6.0, transform_coordinates=False): 
        self.l, self.m, self.max_torque = l, m, max_torque
        assert not transform_coordinates
        super().__init__() 
        self.friction = friction
        self._u_prev = None                        # For rendering
        self.cp_render = {}
        assert friction == 0.0

    def sym_f(self, x, u, t=None): 
        l, m = self.l, self.m
        g = 9.82
        theta_dot = x[1]  # Parameterization: x = [theta, theta']
        theta_dot_dot =  g/l * sym.sin(x[0]) + 1/(m*l**2) * u[0]
        return [theta_dot, theta_dot_dot]

    def get_cost(self) -> SymbolicQRCost:
        return SymbolicQRCost(R=np.ones((1, 1)), Q=np.eye(2))  

    def tF_bound(self) -> Box: 
        return Box(0.5, 4, shape=(1,))

    def t0_bound(self) -> Box:
        return Box(0, 0, shape=(1,))

    def x_bound(self) -> Box:
        return Box(np.asarray( [-2 * np.pi, -np.inf]), np.asarray( [2 * np.pi, np.inf]) )

    def u_bound(self) -> Box:
        return Box(np.asarray([-self.max_torque]), np.asarray([self.max_torque]))

    def x0_bound(self) -> Box:
        return Box(np.asarray( [np.pi, 0] ), np.asarray( [np.pi, 0]))

    def xF_bound(self) -> Box:
        return Box(np.asarray([0, 0]), np.asarray([0, 0]))         

    # def close(self):
    #     if self.cp_render is not None:
    #         self.cp_render.close()

    # def render(self, x, render_mode="human"):
    #     if self.cp_render is None:
    #         self.cp_render = gym.make("Pendulum-v1", render_mode=render_mode)  # environment only used for rendering
    #         self.cp_render.max_time_limit = 10000
    #         self.cp_render.reset()
    #
    #     self.cp_render.unwrapped.last_u = float(self._u_prev) if self._u_prev is not None else self._u_prev
    #     self.cp_render.unwrapped.state = np.asarray(x)
    #     return self.cp_render.render()


    def close(self):
        for r in self.cp_render.values():
            r.close()

    def render(self, x, render_mode="human"):
        if render_mode not in self.cp_render: # is None or self.cp_render[1] != render_mode:
            # if self.cp_render is not None:
            #     self.cp_render.close()

            self.cp_render[render_mode] = gym.make("Pendulum-v1", render_mode=render_mode)  # environment only used for rendering. Change to v1 in gym 0.26.
            # self.cp_render[render_mode].render_mode = render_mode
            self.cp_render[render_mode].max_time_limit = 10000
            self.cp_render[render_mode].reset()
        self.cp_render[render_mode].unwrapped.state = np.asarray(x)  # environment is wrapped
        self.cp_render[render_mode].unwrapped.last_u = self._u_prev[0] if self._u_prev is not None else None
        return self.cp_render[render_mode].render()

class SinCosPendulumModel(PendulumModel):
    def phi_x(self, x):
        theta, theta_dot = x[0], x[1]
        return [sym.sin(theta), sym.cos(theta), theta_dot]

    def phi_x_inv(self, x):
        sin_theta, cos_theta, theta_dot = x[0], x[1], x[2]
        theta = sym.atan2(sin_theta, cos_theta)  # Obtain angle theta from sin(theta),cos(theta)
        return [theta, theta_dot]

    def phi_u(self, u):
        return [sym.atanh(u[0] / self.max_torque)]

    def phi_u_inv(self, u):
        return [sym.tanh(u[0]) * self.max_torque]

    def u_bound(self) -> Box:
        return Box(np.asarray([-np.inf]), np.asarray([np.inf]))

def _pendulum_cost(model):
    from irlc.ex04.discrete_control_cost import DiscreteQRCost
    Q = np.eye(model.state_size)
    Q[0, 1] = Q[1, 0] = model.l
    Q[0, 0] = Q[1, 1] = model.l ** 2
    Q[2, 2] = 0.0
    R = np.array([[0.1]]) * 10
    c0 = DiscreteQRCost(Q=np.zeros((model.state_size,model.state_size)), R=R)
    c0 = c0 + c0.goal_seeking_cost(Q=Q, x_target=model.x_upright)
    c0 = c0 + c0.goal_seeking_terminal_cost(xN_target=model.x_upright) * 1000
    return c0 * 2


class DiscreteSinCosPendulumModel(DiscreteControlModel): 
    state_labels =  ['$\sin(\\theta)$', '$\cos(\\theta)$', '$\\dot{\\theta}$'] # Check if this escape character works.
    action_labels = ['Torque $u$']

    def __init__(self, dt=0.02, cost=None, **kwargs): 
        model = SinCosPendulumModel(**kwargs) 
        self.max_torque = model.max_torque
        # self.transform_actions = transform_actions  
        super().__init__(model=model, dt=dt, cost=cost) 
        self.x_upright = np.asarray(self.phi_x(model.x_upright))
        self.l = model.l # Pendulum length
        if cost is None:  
            cost = _pendulum_cost(self)
        self.cost = cost  


class ThetaPendulumEnvironment(ControlEnvironment):
    def __init__(self, Tmax=5, render_mode=None):
        dt = 0.02
        discrete_model = DiscreteControlModel(PendulumModel(), dt=dt)
        super().__init__(discrete_model, Tmax=Tmax, render_mode=render_mode)

class GymSinCosPendulumEnvironment(ControlEnvironment): 
    def __init__(self, *args, Tmax=5, supersample_trajectory=False, render_mode=None, **kwargs): 
        discrete_model = DiscreteSinCosPendulumModel(*args, **kwargs) 
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(discrete_model.action_size,), dtype=float)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(discrete_model.state_size,), dtype=float)
        super().__init__(discrete_model, Tmax=Tmax, supersample_trajectory=supersample_trajectory, render_mode=render_mode) 

if __name__ == "__main__":
    model = SinCosPendulumModel(l=1, m=1)
    print(str(model))
    print(f"Pendulum with l={model.l}, m={model.m}") 
    x = [1,2]
    u = [0] # Input state/action.
    # x_dot = ...
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute dx/dt = f(x, u, t=0) here using the model-class defined above")
    # x_dot_numpy = ...
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute dx/dt = f(x, u, t=0) here using numpy-expressions you write manually.")

    print(f"Using model-class: dx/dt = f(x, u, t) = {x_dot}")
    print(f"Using numpy: dx/dt = f(x, u, t) = {x_dot_numpy}") 
