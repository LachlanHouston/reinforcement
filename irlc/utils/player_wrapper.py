# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from gymnasium import logger
from irlc.ex01.agent import Agent
import time
import sys
import gymnasium as gym
import os

try:
    # Imports that may not be availble:
    # Using this backend apparently clash with scientific mode. Not sure why it was there in the first place so
    # disabling it for now.
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pygame
except ImportError as e:
    logger.warn('failed to set matplotlib backend, plotting will not work: %s' % str(e))
    plt = None


class AgentWrapper(Agent):
    """Wraps the environment to allow a modular transformation.

    This class is the base class for all wrappers. The subclass could override
    some methods to change the behavior of the original environment without touching the
    original code.

    .. note::

        Don't forget to call ``super().__init__(env)`` if the subclass overrides :meth:`__init__`.

    """
    def __init__(self, agent, env):
        # print("AgentWrapper is deprecated. ")
        self.agent = agent
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def pi(self, state, k, info=None):
        return self.agent.pi(state, k, info)
        # return self.env.step(action)

    def train(self, *args, **kwargs):
        return self.agent.train(*args, **kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.agent)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.agent.unwrapped

PAUSE_KEY = ord('p')
SPACEBAR = "_SPACE_BAR_PRESSED_"
class PlayWrapperPygame(AgentWrapper):
    def __init__(self, agent : Agent, env : gym.Env, keys_to_action=None, autoplay=False):
        super().__init__(agent, env)
        if keys_to_action is None:
            if hasattr(env, 'get_keys_to_action'):
                keys_to_action = env.get_keys_to_action()
            elif hasattr(env.env, 'get_keys_to_action'):
                keys_to_action = env.env.get_keys_to_action()
            elif hasattr(env.unwrapped, 'get_keys_to_action'):
                keys_to_action = env.unwrapped.get_keys_to_action()
            else:
                print(env.spec.id +" does not have explicit key to action mapping, please specify one manually")
                assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                              "please specify one manually"
                # keys_to_action = dict()
        self.keys_to_action = keys_to_action
        self.env = env
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.human_agent_action = -1
        self.human_demand_autoplay = autoplay
        # Now fix the train function
        train2 = agent.train
        def train_(s, a, r, sp, done, info1, info2):
            train2(s, a, r, sp, done, info1, info2)
            env.render()

        agent.train = train_
        env.agent = agent

    # space bar: 0x0020
    def key_press(self,key, mod):
        if key == 0xff0d: self.human_wants_restart = True
        if key == PAUSE_KEY:
            self.human_demand_autoplay = not self.human_demand_autoplay
            a = -1
        else:
            a = self.keys_to_action.get((key,), -1)

        if a == -1 and hasattr(self.env, 'keypress'):
            self.env.keypress(key)

        if key == 0x0020:
            a = SPACEBAR
        self.human_agent_action = a

    def key_release(self,key, mod):
        pass

    # def _get_viewer(self):
    #     return None
    #     return self.env.viewer if hasattr(self.env, 'viewer') else self.env.unwrapped.viewer

    # def setup(self):
    #     # print("In play wrapper - setup")
    #     # print(self._get_viewer())
    #     # return
    #     return
    #     viewer = self._get_viewer()
    #     if viewer is not None:
    #         viewer.window.on_key_press = self.key_press
    #         viewer.window.on_key_release = self.key_release


    def pi(self,state, k, info=None):
        pi_action = super().pi(state, k, info) # make sure super class pi method is called in case it has side effects.
        # self.setup()
        # If unpaused, don't use events given by keyboard until pause is hit again.
        a = None
        while True:
            # Get pygame events:
            # for event in pygame.event.get():
            #     # get the pressed key
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # print("Want to quit")
                    if hasattr(self, 'env'):
                        self.env.close()
                    time.sleep(0.1)
                    pygame.display.quit()
                    time.sleep(0.1)
                    pygame.quit()
                    time.sleep(0.1)
                    # print("Laila tov!")
                    sys.exit()


                # checking if keydown event happened or not
                if event.type == pygame.KEYDOWN:
                    # if keydown event happened
                    # than printing a string to output
                    # print("A key has been pressed", event)
                    # if event.key == pygame.K_LEFT:
                    #     print("LEFT!")
                    # print(event.key, event.unicode)
                    # Determine if event is one environment should handle.

                    if event.key == pygame.K_SPACE:
                        # Got space, autoplay.
                        a = pi_action
                        break
                    elif (event.key,) in self.keys_to_action:
                        a = self.keys_to_action[(event.key,)]
                        if info is not None and 'mask' in info:
                            # Consider refactoring the environment later.
                            from irlc.utils.common import DiscreteTextActionSpace

                            if isinstance(self.env.action_space, DiscreteTextActionSpace):
                                aint = self.env.action_space.actions.index(a)
                            else:
                                aint = a

                            if info['mask'][aint] == 0:
                                # The action was masked. This means that this action is unavailable, and we should select another.
                                # The default is to select one of the available actions from the mask.
                                a = info['mask'].argmax()
                                if isinstance(self.env.action_space, DiscreteTextActionSpace):
                                    a = self.env.action_space.actions[a]



                        else:
                            break
                    elif event.unicode == 'p':
                        # unpause
                        self.human_demand_autoplay = not self.human_demand_autoplay
                        break
                    else:
                        # try to pass event on to the game.
                        if hasattr(self.env, 'keypress'):
                            self.env.keypress(event)
            # now broke and got event.
            if self.human_demand_autoplay:
                a = pi_action

            if a is not None:
                # return a # We don't  are if action is not in action-space.
                # if hasattr(self.env, 'A') and a not in self.env.A(state):
                #     print(f"Got action {a} not available in action space {self.env.A(state)}")
                #     a = self.env.A(state)[-1] # Last because of the gym environment.
                # else:
                #     return a
                try:
                    from irlc.pacman.gamestate import GameState
                    if isinstance(state, GameState):
                        if a not in state.A():
                            a = "Stop"
                except Exception as e:
                    pass

                return a
            # viewer = self._get_viewer()
            time.sleep(0.1)
            # if viewer is not None:
            #     viewer.window.dispatch_events()
            # a = self.human_agent_action
            # if a == SPACEBAR or self.human_demand_autoplay:
            #     # Just do what the agent wanted us to do
            #     action_okay = True
            #     a = pi_action
            # elif hasattr(self.env, 'P'):
            #     if len(self.env.P[state]) == 1 and a != -1:
            #         a = next(iter(self.env.P[state]))
            #     action_okay = a in self.env.P[state]
            # elif self.env.action_space is not None:
            #     action_okay = self.env.action_space.contains(a)
            # else:
            #     action_okay = a != -1
            # if action_okay:
            #     self.human_agent_action = -1
            #     break
        # print("In keyboard wrapper, returning action", a)
        # return a


def interactive(env : gym.Env, agent: Agent, autoplay=False) -> (gym.Env, Agent):
    """
    This function is used for visualizations. It can

    - Allow you to input keyboard commands to an environment
    - Allow you to save results
    - Visualize reinforcement-learning agents in the gridworld environment.

    by adding a single extra line ``env, agent = interactive(env,agent)``.
    The following shows an example:

        >>> from irlc.gridworld.gridworld_environments import BookGridEnvironment
        >>> from irlc import train, Agent, interactive
        >>> env = BookGridEnvironment(render_mode="human", zoom=0.8) # Pass render_mode='human' for visualization.
        >>> env, agent = interactive(env, Agent(env))               # Make the environment interactive. Note that it needs an agent.
        >>> train(env, agent, num_episodes=2)                     # You can train and use the agent and environment as usual.
        >>> env.close()

    It also enables you to visualize the environment at a matplotlib figure or save it as a pdf file using ``env.plot()`` and ``env.savepdf('my_file.pdf)``.

    All demos and figures in the notes are made using this function.

    :param env: A gym environment (an instance of the ``Env`` class)
    :param agent: An agent (an instance of the ``Agent`` class)
    :param autoplay: Whether the simulation should be unpaused automatically
    :return: An environment and agent which have been slightly updated to make them interact with each other. You can use them as usual with the ``train``-function.
    """
    from PIL import Image # Let's put this one here in case we run the code in headless mode.

    agent = PlayWrapperPygame(agent, env, autoplay=autoplay)

    def plot():
        env.render_mode, rmt = 'rgb_array', env.render_mode
        frame = env.render()
        env.render_mode = rmt
        im = Image.fromarray(frame)
        plt.imshow(im)
        plt.axis('off')
        plt.axis('off')
        plt.tight_layout()

    def savepdf(file):
        env.render_mode, rmt = 'rgb_array', env.render_mode
        frame = env.render()
        env.render_mode = rmt

        im = Image.fromarray(frame)
        snapshot_base = file
        if snapshot_base.endswith(".png"):
            sf = snapshot_base[:-4]
            fext = 'png'
        else:
            fext = 'pdf'
            if snapshot_base.endswith(".pdf"):
                sf = snapshot_base[:-4]
            else:
                sf = snapshot_base

        sf = f"{sf}.{fext}"
        dn = os.path.dirname(sf)
        if len(dn) > 0 and not os.path.isdir(dn):
            os.makedirs(dn)
        print("Saving snapshot of environment to", os.path.abspath(sf))
        if fext == 'png':
            im.save(sf)
            from irlc import _move_to_output_directory
            _move_to_output_directory(sf)
        else:
            plt.figure(figsize=(16, 16))
            plt.imshow(im)
            plt.axis('off')
            plt.tight_layout()
            from irlc import savepdf
            savepdf(sf, verbose=True)
            plt.show()
    env.plot = plot
    env.savepdf = savepdf
    return env, agent


def main():
    from irlc.ex11.q_agent import QAgent

    from irlc.gridworld.gridworld_environments import BookGridEnvironment  
    from irlc import train, Agent
    env = BookGridEnvironment(render_mode="human", zoom=0.8)  # Pass render_mode='human' for visualization.
    env, agent = interactive(env, Agent(env))  # Make th
    env.reset()     # We always need to call reset
    env.plot()      # Plot the environment.
    env.close()  

    # Interaction with a random agent.
    from irlc.gridworld.gridworld_environments import BookGridEnvironment 
    from irlc import train, Agent
    env = BookGridEnvironment(render_mode="human", zoom=0.8) # Pass render_mode='human' for visualization.
    env, agent = interactive(env, Agent(env))               # Make the environment interactive. Note that it needs an agent.
    train(env, agent, num_episodes=100)                      # You can train and use the agent and environment as usual. 
    env.close()

    # Second example: plotting.


    a = 234
    # from irlc.utils.berkley import BerkleyBookGridEnvironment
    # from irlc.ex11.sarsa_agent import SarsaAgent
    # from irlc.ex01.agent import train
    # from irlc.utils.berkley import VideoMonitor
    # env = BerkleyBookGridEnvironment(adaptor='gym')
    # agent = SarsaAgent(env, gamma=0.95, alpha=0.5)
    # """
    # agent = PlayWrapper(agent, env)

    # env = VideoMonitor(env, agent=agent, video_file="videos/SarsaGridworld.mp4", fps=30, continious_recording=True,
    #                    label="SADSF",
    #                    monitor_keys=("Q",))
    # """
    # env.reset()
    # env.render()
    # train(env, agent, num_episodes=3)
    # env.close()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='MontezumaRevengeNoFrameskip-v4', help='Define Environment')
    # args = parser.parse_args()
    # env = gym.make(args.env)
    # play(env, zoom=4, fps=60)

if __name__ == "__main__":


    main()
