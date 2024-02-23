# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
import numpy as np
from irlc import train, Agent, savepdf
import matplotlib.pyplot as plt
from irlc.ex03.kuramoto import KuramotoModel, f


def fk(x,u):
    """ Computes the discrete (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,i.e.

    x_{k+1} = f_k(x,u).

    Look at dmodel.f for inspiration. As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a numpy ndarray.
    """
    dmodel = DiscreteControlModel(KuramotoModel(), dt=0.5)  # this is how we discretize the Kuramoto model.
    # TODO: 1 lines missing.
    raise NotImplementedError("Compute Euler discretized dynamics here using the dmodel.")
    return f_euler

def dfk_dx(x,u):
    """ Computes the derivative of the (Euler 1-step integrated) version of the Kuromoto update with discretization time dt=0.5,
    i.e. if

    .. math::

        x_{k+1} = f_k(x,u)

    this function should return

    .. math::

        \frac{\partial f_k}{\partial x }

    (i.e. the Jacobian with respect to x) as a numpy matrix.
    Look at dmodel.f for inspiration, and note it has an input argument that is relevant.
    As usual, use a debugger and experiment. Note you have to specify input arguments as lists,
    and the function should return a two-dimensional numpy ndarray.

    """
    dmodel = DiscreteControlModel(KuramotoModel(), dt=0.5)
    # the function dmodel.f accept various parameters. Perhaps their name can give you an idea?
    # TODO: 1 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    return f_euler_derivative


if __name__ == "__main__":
    # Part 1: Making a model
    cmodel = KuramotoModel()
    print(cmodel)
    # Computing f_k
    dmodel = DiscreteControlModel(KuramotoModel(), dt=0.5) 
    print(dmodel) # This will print details about the discrete model.  

    print("The Euler-discretized version, f_k(x,u) = x + Delta f(x,u), is") 
    print("f_k(x=0,u=0) =", fk([0], [0]))
    print("f_k(x=1,u=0.3) =", fk([1], [0.3]))

    # Computing df_k / dx (The Jacobian).
    print("The derivative of the Euler discretized version wrt. x is:")
    print("df_k/dx(x=0,u=0) =", dfk_dx([0], [0])) 

    # Part 2: The environment and simulation:
    env = ControlEnvironment(dmodel, Tmax=20) # An environment that runs for 20 seconds. 
    u = 1.3 # Action to take in each time step.

    ts_step = []  # Current time (according to the environment, i.e. in increments of dt.
    xs_step = []  # x_k using the env.step-function in the enviroment.

    x, _ = env.reset()       # Get starting state.
    ts_step.append(env.time) # env.time keeps track of the clock-time in the environment.
    xs_step.append(x)        # Initialize with first state

    # Use
    # > next_x, cost, terminated, truncated, metadata = env.step([u])
    # to simulate a single step.
    for _ in range(10000):
        # TODO: 1 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        xs_step.append(next_x)
        ts_step.append(env.time) # This is how you get the current time (in seconds) from the environment.

        if terminated: # Obtain 'terminated' from the step-function. It will be true when Tmax=20 seconds has passed.
            break

    x0 = cmodel.x0_bound().low  # Get the starting state x0. We exploit that the bound on x0 is an equality constraint.
    xs_rk4, us_rk4, ts_rk4 = cmodel.simulate(x0, u_fun=u, t0=0, tF=20, N_steps=100)

    plt.plot(ts_rk4, xs_rk4, 'k-', label='RK4 (nearly exact)')
    plt.plot(ts_step, xs_step, 'ro', label='RK4 (step-function in environment)')

    # Use the train-function to plot the result of simulating a random agent.
    stats, trajectories = train(env, Agent(env), return_trajectory=True) 
    plt.plot(trajectories[0].time, trajectories[0].state, label='x(t) when using a random action sequence from agent') 
    plt.legend()
    savepdf('kuramoto_step')
    plt.show(block=False)
    print("The total cost obtained using random actions", -stats[0]['Accumulated Reward'])
