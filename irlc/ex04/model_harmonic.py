# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.model_linear_quadratic import LinearQuadraticModel
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
import numpy as np
from irlc.utils.graphics_util_pygame import UpgradedGraphicsUtil

"""
Simulate a Harmonic oscillator governed by equations:

d^2 x1 / dt^2 = -k/m x1 + u(x1, t)

where x1 is the position and u is our externally applied force (the control)
k is the spring constant and m is the mass. See:

https://en.wikipedia.org/wiki/Simple_harmonic_motion#Dynamics

for more details.
In the code, we will re-write the equations as:

dx/dt = f(x, u),   u = u_fun(x, t)

where x = [x1,x2] is now a vector and f is a function of x and the current control.
here, x1 is the position (same as x in the first equation) and x2 is the velocity.

The function should return ts, xs, C

where ts is the N time points t_0, ..., t_{N-1}, xs is a corresponding list [ ..., [x_1(t_k),x_2(t_k)], ...] and C is the cost.
"""

class HarmonicOscilatorModel(LinearQuadraticModel):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    """
    See: https://books.google.dk/books?id=tXZDAAAAQBAJ&pg=PA147&lpg=PA147&dq=boeing+747+flight+0.322+model+longitudinal+flight&source=bl&ots=L2RpjCAWiZ&sig=ACfU3U2m0JsiHmUorwyq5REcOj2nlxZkuA&hl=en&sa=X&ved=2ahUKEwir7L3i6o3qAhWpl4sKHQV6CdcQ6AEwAHoECAoQAQ#v=onepage&q=boeing%20747%20flight%200.322%20model%20longitudinal%20flight&f=false
    """
    def __init__(self, k=1., m=1., drag=0.0, Q=None, R=None):
        self.k = k
        self.m = m
        A = [[0, 1],
             [-k/m, 0]]

        B = [[0], [1/m]]
        d = [[0], [drag/m]]

        A, B, d = np.asarray(A), np.asarray(B), np.asarray(d)
        if Q is None:
            Q = np.eye(2)
        if R is None:
            R = np.eye(1)
        self.viewer = None
        super().__init__(A=A, B=B, Q=Q, R=R, d=d)

    def render(self, x, render_mode="human"):
        """ Render the environment. You don't have to understand this code.  """
        if self.viewer is None:
            self.viewer = HarmonicViewer(xstar=0) # target: x=0.
        self.viewer.update(x)
        import time
        time.sleep(0.05)
        return self.viewer.blit(render_mode=render_mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class DiscreteHarmonicOscilatorModel(DiscreteControlModel): 
    def __init__(self, dt=0.1, discretization_method=None, **kwargs):
        model = HarmonicOscilatorModel(**kwargs)
        super().__init__(model=model, dt=dt, discretization_method=discretization_method)
        # self.cost = model.cost.discretize(dt=dt) 


class HarmonicOscilatorEnvironment(ControlEnvironment): 
    def __init__(self, Tmax=80, supersample_trajectory=False, render_mode=None, **kwargs):
        model = DiscreteHarmonicOscilatorModel(**kwargs)
        self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax, render_mode=render_mode, supersample_trajectory=supersample_trajectory) 

    def _get_initial_state(self) -> np.ndarray:
        return np.asarray([1, 0])

class HarmonicViewer(UpgradedGraphicsUtil):
    def __init__(self, xstar = 0):
        self.xstar = xstar
        width = 1100
        self.scale = width / 6
        self.dw = self.scale * 0.1
        super().__init__(screen_width=width, xmin=-width / 2, xmax=width / 2, ymin=-width / 5, ymax=width / 5, title='Harmonic Osscilator')

    def render(self):
        self.draw_background(background_color=(255, 255, 255))
        dw = self.dw
        self.rectangle(color=(0,0,0), x=-dw//2, y=-dw//2, width=dw, height=dw)
        xx = np.linspace(0, 1)
        y = np.sin(xx * 2 * np.pi * 5) * 0.1*self.scale * 0.5

        for i in range(len(xx) - 1):
            self.line("asfasf", here=(xx[i] * self.x[0] * self.scale, y[i]), there=(xx[i + 1] * self.x[0] * self.scale, y[i+1]),
                      color=(0,0,0), width=2)
        self.circle("asdf", pos=( self.x[0] * self.scale, 0), r=dw, fillColor=(0,0,0))
        self.circle("asdf", pos=( self.x[0] * self.scale, 0), r=dw*0.9, fillColor=(int(.7 * 255),) * 3)

    def update(self, x):
        self.x = x

if __name__ == "__main__":
    from irlc import train
    env = HarmonicOscilatorEnvironment(render_mode='human')
    # train(env, NullAgent(env), num_episodes=1, max_steps=200)
    # env.close()
