# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(32)
from irlc import Agent, savepdf
from irlc.ex04.pid import PID
from irlc.ex01.agent import train
from irlc.ex05.model_cartpole import CartpoleEnvironment

class PIDCartpoleAgent(Agent):
    def __init__(self, env, dt, Kp=1.0, Ki=0.0, Kd=0.0, target=0, balance_to_x0=True):
        """ Balance_to_x0 = True implies the agent should also try to get the cartpole to x=0 (i.e. center).
        If balance_to_x0 = False implies it is good enough for the agent to get the cart upright.
        """
        self.pid = PID(dt=dt, Kp = Kp, Ki=Ki, Kd=Kd, target=target)
        self.balance_to_x0 = balance_to_x0
        super().__init__(env)

    def pi(self, x, k, info=None): 
        """ Compute action using self.pid. You have to think about the inputs as they will depend on
        whether balance_to_x0 is true or not.  """
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")
        return u


def get_offbalance_cart(waiting_steps=30):
    env = CartpoleEnvironment(Tmax=10, render_mode='human')

    env.reset()
    env.state[0] = 0
    env.state[1] = 0
    env.state[2] = 0  # Upright, but leaning slightly to one side.
    env.state[3] = 0
    for _ in range(waiting_steps):  # Simulate the environment for 30 steps to get things out of balance.
        env.step(1)
    return env

def plot_trajectory(trajectory):
    t = trajectory
    plt.plot(t.time, t.state[:,2], label="Stick angle $\\theta$" )
    plt.plot(t.time, t.state[:,0], label="Cart location $x$")
    plt.xlabel("Time")
    plt.legend()

if __name__ == "__main__":
    """
    First task: Bring the balance upright from a slightly off-center position. 
    For this task, we do not care about the x-position, only the angle theta which should be 0 (upright)
    """
    env = get_offbalance_cart(20)
    agent = PIDCartpoleAgent(env, env.dt, ...)
    # TODO: 1 lines missing.
    raise NotImplementedError("Define your agent here (including parameters)")
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
    plot_trajectory(trajectories[0])
    savepdf("pid_cartpoleA")
    plt.show()

    """
    Second task: We will now also try to bring the cart towards x=0.
    """
    env = get_offbalance_cart(20)
    agent = PIDCartpoleAgent(env, env.dt, ...)
    # TODO: 1 lines missing.
    raise NotImplementedError("Define your agent here (including parameters)")
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
    plot_trajectory(trajectories[0])
    savepdf("pid_cartpoleB")
    plt.show()

    """
    Third task: Bring the cart upright theta=0 and to the center x=0, but starting from a more challenging position. 
    """
    env = get_offbalance_cart(30)
    agent = PIDCartpoleAgent(env, env.dt, ...)
    # TODO: 1 lines missing.
    raise NotImplementedError("Define your agent here (including parameters)")
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
    plot_trajectory(trajectories[0])
    savepdf("pid_cartpoleC")
    plt.show()
