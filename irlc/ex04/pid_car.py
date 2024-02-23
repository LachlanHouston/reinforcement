# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from irlc import savepdf
from irlc.ex04.pid import PID
from irlc import Agent
from irlc.ex04.control_environment import ControlEnvironment

class PIDCarAgent(Agent):
    def __init__(self, env: ControlEnvironment, v_target=0.5, use_both_x5_x3=True):
        """
        Define two pid controllers: One for the angle, the other for the velocity.

        self.pid_angle = PID(dt=self.discrete_model.dt, Kp=x, ...)
        self.pid_velocity = PID(dt=self.discrete_model.dt, Kp=z, ...)

        I did not use Kd/Ki, however you need to think a little about the targets.
        """
        # self.pid_angle = ...
        ## TODO: Half of each line of code in the following 2 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # self.pid_angle = PID(dt=env.discrete_m??????????????????????????????????????
        # self.pid_velocity = PID(dt=env.discrete_mod???????????????????????????????????????????
        raise NotImplementedError("Define PID controllers here.")
        self.use_both_x5_x3 = use_both_x5_x3 # Using both x3+x5 seems to make it a little easier to get a quick lap time, but you can just use x5 to begin with.
        super().__init__(env)

    def pi(self, x, k, info=None):
        """
        Call PID controller. The steering angle controller should initially just be based on
        x[5] (distance to the centerline), but you can later experiment with a linear combination of x5 and x3 as input.

        Hints:
            - To control the velocity, you should use x[0], the velocity of the car in the direction of the car.
            - Remember to start out with a low value of v_target, then tune the controller and look at the animation.
            - You can access the pid controllers as self.pid_angle(x_input)
            - Remember the function must return a 2d numpy ndarray.
        """

        # TODO: 2 lines missing.
        raise NotImplementedError("Compute action here. No clipping necesary.")
        return u


if __name__ == "__main__":
    from irlc.ex01.agent import train
    from irlc.car.car_model import CarEnvironment
    import matplotlib.pyplot as plt

    env = CarEnvironment(noise_scale=0,Tmax=30, max_laps=1, render_mode='human')
    agent = PIDCarAgent(env, v_target=1, use_both_x5_x3=True) # I recommend lowering v_target to make the problem simpler.

    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()
    t = trajectories[0]
    plt.clf()
    plt.plot(t.state[:,0], label="velocity" )
    plt.plot(t.state[:,5], label="s (distance to center)" )
    plt.xlabel("Time/seconds")
    plt.legend()
    savepdf("pid_car_agent")
    plt.show()
