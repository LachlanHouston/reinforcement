# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(32)
from irlc import Agent, savepdf
from irlc.ex04.pid import PID
from irlc.ex01.agent import train

class PIDPendulumAgent(Agent):
    def __init__(self, env, dt, Kp=1.0, Ki=0.0, Kd=0.0, target_angle=0):
        """ Balance_to_x0 = True implies the agent should also try to get the cartpole to x=0 (i.e. center).
        If balance_to_x0 = False implies it is good enough for the agent to get the cart upright.
        """
        self.pid = PID(dt=dt, Kp = Kp, Ki=Ki, Kd=Kd, target=target_angle)
        super().__init__(env)

    def pi(self, x, k, info=None): 
        """ Compute action using self.pid. YCartpoleou have to think about the inputs as they will depend on
        whether balance_to_x0 is true or not.  """
        # TODO: 2 lines missing.
        raise NotImplementedError("Implement function body")
        return u


def get_offbalance_pendulum(waiting_steps=30):
    from irlc.ex04.model_pendulum import ThetaPendulumEnvironment
    env = ThetaPendulumEnvironment(Tmax=10, render_mode='human')

    env.reset()
    env.state[0] = 0
    env.state[1] = 0
    for _ in range(waiting_steps):  # Simulate the environment for 30 steps to get things out of balance.
        env.step(1)
    return env

def plot_trajectory(trajectory):
    t = trajectory
    plt.plot(t.time, t.state[:,0], label="Angle $\\theta$" )
    plt.plot(t.time, t.state[:,1], label="Angular speed $\\cdot{\\theta}$")
    plt.xlabel("Time")
    plt.legend()


target_angle = np.pi/6 # The target angle for the second task in the pendulum problem.
if __name__ == "__main__":
    """
    First task: Bring the balance upright from a slightly off-center position. 
    For this task, we do not care about the x-position, only the angle theta which should be 0 (upright)
    """
    env = get_offbalance_pendulum(30)
    ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
    #----------------------------------------------------------------------------------------------------------------------------
    # agent = PIDPendulumAgent(env, dt=env.??????????????????????????????????????
    raise NotImplementedError("Define your agent here (including parameters)")
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
    plot_trajectory(trajectories[0])
    savepdf("pid_pendulumA")
    plt.show()

    """
    Second task: We will now try to get to a target angle of target_angle=np.pi/6.    
    """
    env = get_offbalance_pendulum(30)
    ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
    #----------------------------------------------------------------------------------------------------------------------------
    # agent = PIDPendulumAgent(env, dt=env.dt,?????????????????????????????????????????
    raise NotImplementedError("Define your agent here (include the target_angle parameter to the agent!)")
    _, trajectories = train(env, agent, num_episodes=1, return_trajectory=True, reset=False)  # Note reset=False to maintain initial conditions.
    env.close()
    plot_trajectory(trajectories[0])
    print("Final state is x(t_F) =", trajectories[0].state[-1], f"goal [{target_angle:.2f}, 0]")
    savepdf("pid_pendulumB")
    plt.show()
