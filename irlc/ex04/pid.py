# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
from irlc import savepdf
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex04.locomotive import LocomotiveEnvironment

class PID:
    def __init__(self, dt, Kp, Ki, Kd, target):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt          # discretization time
        self.target = target  # target, in our case just a number.
        self.I = 0            # Internal variables for integral/derivative terms; use these or define your own.
        self.e_prior = 0      # Previous value of the error. Used in the derivative term. Remember to update it in the pi-function.

    def reset(self):
        self.I = 0
        self.e_prior = 0

    def pi(self, x):
        """
        Policy for the PID class. x is always a scalar (float) and the output u is a scalar.
        Should implement (Her24, Algorithm 19)

        :param x: Input state (float)
        :return: Action to take (float)
        """
        # TODO: 4 lines missing.
        raise NotImplementedError("Compute u here.")
        return u


def pid_explicit():
    env = LocomotiveEnvironment(m=70, slope=0, dt=0.05, Tmax=15) 
    pid = PID(dt=0.05, Kp=40, Kd=0, Ki=0, target=0)
    # Compute the first action using PID control:
    print(f"When x_0 = 1 then the first action is u_0 = {pid.pi(x=1)} (and should be u_0 = -40.0)") 

    x0, _ = env.reset()
    x = [x0]
    for _ in range(200): # Simulate for 200 steps, i.e. 0.05 * 200 seconds.
        x_cur = x[-1] # x is the last state [position, velocity]. Note that you only need to pass position to your PID controller.
        # TODO: 1 lines missing.
        raise NotImplementedError("Compute action here using the pid class.")
        u = np.clip(u, -100, 100) # clip actions.
        xp_, reward, done, truncated, _ = env.step(u)
        x.append(xp_)

    x = np.stack(x)
    plt.plot(x[:,0], 'k-', label="PID state trajectory")
    savepdf("pid_basic")
    plt.show()

if __name__ == "__main__":
    pid_explicit()
