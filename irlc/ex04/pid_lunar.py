# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
For information about the Apollo 11 lunar lander see:
https://eli40.com/lander/02-debrief/

For code for the Gym LunarLander environment see:

https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

This particular controller is inspired by:

https://github.com/wfleshman/PID_Control/blob/master/pid.py

However, I had better success with different parameters for the PID controller.
"""
# import gym
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from irlc import train
from irlc.ex04.pid import PID
from irlc import Agent
from irlc.ex04 import speech
from irlc import savepdf
from gymnasium.envs.box2d.lunar_lander import FPS

class ApolloLunarAgent(Agent):
    def __init__(self, env, dt, Kp_altitude=18, Kd_altitude=13, Kp_angle=-18, Kd_angle=-18):
        """ Set up PID parameters for the two controllers (one controlling the altitude, another the angle of the lander) """
        self.Kp_altitude = Kp_altitude
        self.Kd_altitude = Kd_altitude
        self.Kp_angle = Kp_angle
        self.Kd_angle = Kd_angle
        self.error_angle = []
        self.error_altitude = []
        self.dt = dt
        super().__init__(env)

    def pi(self, x, k, info=None):
        """ From documentation: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
             x (list): The state. Attributes:
              x[0] is the horizontal coordinate
              x[1] is the vertical coordinate
              x[2] is the horizontal speed
              x[3] is the vertical speed
              x[4] is the angle
              x[5] is the angular speed
              x[6] 1 if first leg has contact, else 0
              x[7] 1 if second leg has contact, else 0

              Your implementation should follow what happens in:

              https://github.com/wfleshman/PID_Control/blob/master/pid.py

              I.e. you have to compute the target for the angle and altitude as done in the code (and explained in the documentation.
              Note the target for the PID controllers is 0.
        """
        if k == 0:
            """ At time t=0 we set up the two PID controllers. You don't have to change these lines. """
            self.pid_alt = PID(dt=self.dt, Kp=self.Kp_altitude, Kd=self.Kd_altitude, Ki=0, target=0)
            self.pid_ang = PID(dt=self.dt, Kp=self.Kp_angle, Kd=self.Kd_angle, Ki=0, target=0)

        """ Compute the PID control signals using two calls to the PID controllers such as: """
        # alt_adj = self.pid_alt.pi(...)
        # ang_adj = self.pid_ang.pi(...)
        """ You need to specify the inputs to the controllers. Look at the code in the link above and implement a comparable control rule. 
        The inputs you give to the controller will be simple functions of the coordinates of x, i.e. x[0], x[1], and so on.
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute the alt_adj and ang_adj as in the gitlab repo (see code comment).")

        u = np.array([alt_adj, ang_adj])
        u = np.clip(u, -1, +1)

        # If the legs are on the ground we made it, kill engines
        if (x[6] or x[7]):
            u[:] = 0
        # Record stats.
        self.error_altitude.append(self.pid_alt.e_prior)
        self.error_angle.append(self.pid_ang.e_prior)
        return u

def get_lunar_lander(env):
    dt = 1/FPS # Get time discretization from environment.
    spars = ['Kp_altitude', 'Kd_altitude', 'Kp_angle', 'Kd_angle']
    def x2pars(x2):
        return {spars[i]: x2[i] for i in range(4)}
    x_opt = np.asarray([52.23302414, 34.55938593, -80.68722976, -38.04571655])
    agent = ApolloLunarAgent(env, dt=dt, **x2pars(x_opt))
    return agent

def lunar_single_mission():
    env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    env._max_episode_steps = 1000  # We don't want it to time out.

    agent = get_lunar_lander(env)
    stats, traj = train(env, agent, return_trajectory=True, num_episodes=1)
    env.close()
    if traj[0].reward[-1] == 100:
        print("A small step for man, a giant leap for mankind!")
    elif traj[0].reward[-1] == -100:
        print(speech)
    else:
        print("Environment timed out and the lunar module is just kind of floating around")

    states = np.stack(traj[0].state)
    plt.plot(states[:, 0], label='x')
    plt.plot(states[:, 1], label='y')
    plt.plot(states[:, 2], label='vx')
    plt.plot(states[:, 3], label='vy')
    plt.plot(states[:, 4], label='theta')
    plt.plot(states[:, 5], label='vtheta')
    plt.legend()
    plt.grid()
    plt.ylim(-1.1, 1.1)
    plt.title('PID Control')
    plt.ylabel('Value')
    plt.xlabel('Steps')
    savepdf("pid_lunar_trajectory")
    plt.show(block=False)

def lunar_average_performance():
    env = gym.make('LunarLanderContinuous-v2', render_mode=None) # Set render_mode = 'human' to see what it does.
    env._max_episode_steps = 1000  # To avoid the environment timing out after just 200 steps

    agent = get_lunar_lander(env)
    stats, traj = train(env, agent, return_trajectory=True, num_episodes=20)
    env.close()

    n_won = sum([np.sum(t.reward[-1] == 100) for t in traj])
    n_lost = sum([np.sum(t.reward[-1] == -100) for t in traj])
    print("Successfull landings: ", n_won, "of 20")
    print("Unsuccessfull landings: ", n_lost, "of 20")

if __name__ == "__main__":
    lunar_single_mission() 
    lunar_average_performance() 
