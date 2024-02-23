# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import gymnasium as gym
import numpy as np
from irlc.ex03.control_model import ensure_policy
from irlc.ex04.discrete_control_model import DiscreteControlModel


class ControlEnvironment(gym.Env):
    """
    Helper class to convert a discretized model into an environment.
    See the ``__init__`` function for how to create a new environment using this class. Once an environment has been
    created, you can use it as any other gym environment:

    .. runblock:: pycon

        >>> from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        >>> env = GymSinCosPendulumEnvironment(Tmax=4) # Specify we want it to run for a maximum of 4 seconds
        >>> env.reset() # Reset both the time and state variable
        >>> u = env.action_space.sample()
        >>> next_state, cost, done, truncated, info = env.step(u)
        >>> print("Current state: ", env.state)
        >>> print("Current time", env.time)

    In this example, tell the environment to terminate after 4 seconds using ``Tmax`` (after which ``done=True``)

    .. Note::
        The ``step``-method will use the (nearly exact) RK4 method to integrate the enviorent over a timespan of ``dt``,
        and **not** use the approximate  ``model.f(x_k,u_k, k)``-method in the discrete environment which is based on
        Euler discretization.
        This is the correct behavior since we want the environment to reflect what happens in the real world and not
        our apprixmation method.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    action_space = None
    observation_space = None

    def __init__(self, discrete_model: DiscreteControlModel, Tmax=None, supersample_trajectory=False, render_mode=None):
        """
        Creates a new instance. You should use this in conjunction with a discrete model to build a new class. An example:

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import DiscreteSinCosPendulumModel
            >>> from irlc.ex04.control_environment import ControlEnvironment
            >>> from gymnasium.spaces import Box
            >>> import numpy as np
            >>> class MyGymSinCosEnvironment(ControlEnvironment):
            ...     def __init__(self, Tmax=5):
            ...         discrete_model = DiscreteSinCosPendulumModel()
            ...         self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
            ...         self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)
            ...         super().__init__(discrete_model, Tmax=Tmax)
            >>>
            >>> env = MyGymSinCosEnvironment()
            >>> env.reset()
            >>> env.step(env.action_space.sample())

        :param discrete_model: The discrete model the environment is based on
        :param Tmax: Time in seconds until the environment terminates (``step`` returns ``done=True``)
        :param supersample_trajectory: Used to create nicer (smooth) trajectories. Don't worry about it.
        :param render_mode: If ``human`` the environment will be rendered (inherited from ``Env``)
        """
        self.dt = discrete_model.dt  # Discretization time
        self.state = None            # the current state
        self.time = 0                # Current global time index
        self.discrete_model = discrete_model
        self.Tmax = Tmax

        # Try to guess action/observation spaces unless they are already defined.
        if self.observation_space is None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(discrete_model.state_size,) )

        if self.action_space is None:
            u_bound = self.discrete_model.continuous_model.u_bound()
            self.action_space = gym.spaces.Box(low=np.asarray(self.discrete_model.phi_u(u_bound.low)),
                                               high=np.asarray(self.discrete_model.phi_u(u_bound.high)),
                                               dtype=np.float64,
                                               )
        self.state_labels = discrete_model.state_labels
        self.action_labels = discrete_model.action_labels
        self.supersample_trajectory = supersample_trajectory
        self.render_mode = render_mode


    def step(self, u):
        """
        This works similar to the gym ``Env.step``-function. ``u`` is an action in the action-space,
        and the environment will then assume we (constantly) apply the action ``u`` from the current time step, :math:`t_k`, until
        the next time step :math:`t_{k+1} = t_k + \Delta`, where :math:`\Delta` is equal to ``env.model.dt``.

        During this period, the next state is computed using the relatively exact RK4 simulation, and the incurred cost will be
        computed using Riemann integration.

        .. math::
            \int_{t_k}^{t_k+\Delta} c(x(t), u(t)) dt

        .. Note::
            The gym environment requires that we return a cost. The reward will therefore be equal to minus the (integral) of the cost function.

            In case the environment terminates, the reward will include the terminal cost. :math:`c_F`.

        :param u: The action we apply :math:`u`
        :return:
            - ``state`` - the state we arrive in
            - ``reward`` - (minus) the total (integrated) cost incurred in this time period.
            - ``done`` - ``True`` if the environment has finished, i.e. we reached ``env.Tmax``.
            - ``truncated`` - ``True`` if the environment was forced to terminated prematurely. Assume it is ``False`` and ignore it.
            - ``info`` - A dictionary of potential extra information. Includes ``info['time_seconds']``, which is the current time after the step function has completed.
        """

        def clip_action(self, u):
            return np.clip(u, a_max=self.action_space.high, a_min=self.action_space.low)

        u = clip_action(self, u)
        self.discrete_model.continuous_model._u_prev = u # for rendering.
        if not ((self.action_space.low <= u).all() and (u <= self.action_space.high).all()): #  u not in self.action_space:
            raise Exception("Action", u, "not contained in action space", self.action_space)
        # N=20 is a bit arbitrary; should probably be a parameter to the environment.
        xx, uu, tt = self.discrete_model.simulate2(x0=self.state, policy=ensure_policy(u), t0=self.time, tF=self.time + self.discrete_model.dt, N=20)
        self.state = xx[-1]
        self.time = tt[-1]
        cc = [self.discrete_model.cost.c(x, u, k=None) for x, u in zip(xx[:-1], uu[:-1])]
        done = False
        if self.time + self.discrete_model.dt/2 > self.Tmax:
            cc[-1] += self.discrete_model.cost.cN(xx[-1])
            done = True
        info = {'dt': self.discrete_model.dt, 'time_seconds': self.time}  # Allow the train() function to figure out the simulation time step size
        if self.supersample_trajectory:   # This is only for nice visualizations.
            from irlc.ex01.agent import Trajectory
            traj = Trajectory(time=tt, state=xx.T, action=uu.T, reward=np.asarray(cc), env_info=[])
            info['supersample'] = traj # Supersample the trajectory
        reward = -sum(cc)  # To be compatible with openai gym we return the reward as -cost.
        if not ( (self.observation_space.low <= self.state).all() and (self.state <= self.observation_space.high).all() ): #self.state not in self.observation_space:
            print("> state", self.state)
            print("> observation space", self.observation_space)
            raise Exception("State no longer in observation space", self.state)
        if self.render_mode == "human": # as in gym's carpole
            self.render()

        return self.state, reward, done, False, info

    def reset(self):
        """
        Reset the environment to the initial state. This will by default be `the value computed using `self.discrete_model.reset()``.

        :return:
            - ``state`` - The initial state the environment has been reset to
            - ``info`` - A dictionary with extra information, in this case that time begins at 0 seconds.
        """
        self.state = self._get_initial_state()
        self.time = 0 # Reset internal time (seconds)
        if self.render_mode == "human":
            self.render()
        return self.state, {'time_seconds': self.time}

    def _get_initial_state(self) -> np.ndarray:
        # This helper function returns an initial state. It will be used by the reset() function, and it is this function
        # you should overwrite if you want to reset to a state which is not implied by the bounds.
        if (self.discrete_model.continuous_model.x0_bound().low == self.discrete_model.continuous_model.x0_bound().high).all():
            return np.asarray(self.discrete_model.phi_x(self.discrete_model.continuous_model.x0_bound().low))
        else:
            raise Exception("Since bounds do not agree I cannot return initial state.")

    def render(self):
        return self.discrete_model.render(x=self.state, render_mode=self.render_mode)

    def close(self):
        self.discrete_model.close()
