# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex04.locomotive import LocomotiveEnvironment
from irlc.ex04.pid_car import PID
from irlc import Agent, train
from irlc import savepdf
from irlc.ex04.control_environment import ControlEnvironment

class PIDLocomotiveAgent(Agent):
    def __init__(self, env: ControlEnvironment, dt, Kp=1.0, Ki=0.0, Kd=0.0, target=0):
        # self.pid = PID(dt=...)
        # TODO: 1 lines missing.
        raise NotImplementedError("Make a pid instance here.")
        super().__init__(env)

    def pi(self, x, k, info=None):
        # TODO: 1 lines missing.
        raise NotImplementedError("Get the correct action using self.pid.pi(...). Same as previous exercise")
        u = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0]) # Clip actions to ensure u is in the action space
        return np.asarray([u]) # Must return actions as numpy ndarrays.

def fixplt():
    plt.legend()
    plt.grid('on')
    plt.box(False)
    # plt.ylim([-dd, dd])
    plt.xlabel('Time/seconds')
    plt.ylabel('$x(t)$')

def pid_locomotive():
    dt = .08
    m = 70
    Tmax=15

    env = LocomotiveEnvironment(m=m, slope=0, dt=dt, Tmax=Tmax, render_mode='human')
    Kp = 40
    agent = PIDLocomotiveAgent(env, dt=dt, Kp=Kp, Ki=0, Kd=0, target=0)
    stats, traj = train(env, agent, return_trajectory=True)
    plt.plot(traj[0].time, traj[0].state[:, 0], '-', label=f"$K_p={40}$")
    fixplt()
    savepdf('pid_locomotive_Kp')
    plt.show()

    # Now include a derivative term:
    Kp = 40
    for Kd in [10, 50, 100]:
        agent = PIDLocomotiveAgent(env, dt=dt, Kp=Kp, Ki=0, Kd=Kd, target=0)
        stats, traj = train(env, agent, return_trajectory=True)
        plt.plot(traj[0].time, traj[0].state[:, 0], '-', label=f"$K_p={Kp}, K_d={Kd}$")
    fixplt()
    savepdf('pid_locomotive_Kd')
    plt.show()
    env.close()

    # Derivative test: Include a slope term. For fun, let's also change the target.
    env = LocomotiveEnvironment(m=m, slope=2, dt=dt, Tmax=20, target=1, render_mode='human')
    for Ki in [0, 10]:
        agent = PIDLocomotiveAgent(env, dt=dt, Kp=40, Ki=Ki, Kd=50, target=1)
        stats, traj = train(env, agent, return_trajectory=True)
        x = traj[0].state
        tt = traj[0].time
        plt.plot(tt, x[:, 0], '-', label=f"$K_p={Kp}, K_i={Ki}, K_d={Kd}$")
    fixplt()
    savepdf('pid_locomotive_Ki')
    plt.show()
    env.close()

if __name__ == '__main__':
    pid_locomotive()
