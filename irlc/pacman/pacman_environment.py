# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import pygame
from irlc.pacman.gamestate import Directions, ClassicGameRules
from irlc.pacman.layout import getLayout
from irlc.pacman.pacman_text_display import PacmanTextDisplay
from irlc.pacman.pacman_graphics_display import PacmanGraphics, FirstPersonPacmanGraphics
from irlc.pacman.pacman_utils import PacAgent, RandomGhost
from irlc.pacman.layout import Layout
import gymnasium as gym
from gymnasium import RewardWrapper
from irlc.utils.common import ExplicitActionSpace, DiscreteTextActionSpace

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

very_small_maze = """
%%%%%%
%P. .%
%  %%%
%%%%%%
"""

very_small_haunted_maze = """
%%%%%%
%P. .%
% %%%%
%   G%
%%%%%%
"""


class PacmanEnvironment(gym.Env):
    _unpack_search_state = True  # A hacky fix to set the search state.
    """
    A fairly messy pacman environment class. I do not recommend reading this code.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    # def A(self, state):
    #     """
    #     Return a list of actions available in the given state. This function should be considered deprecated.
    #     """
    #     raise Exception("HArd deprecation.")
    #     return state.A()

    def __init__(self, layout_str=None, render_mode=None, animate_movement=None, layout='mediumGrid', zoom=2.0, num_ghosts=4, frames_per_second=30, ghost_agent=None,
                 method_str='', allow_all_actions=False, verbose=False):
        self.metadata['video_frames_per_second'] = frames_per_second
        self.ghosts = [ghost_agent(i+1) if ghost_agent is not None else RandomGhost(i+1) for i in range(num_ghosts)]
        if animate_movement is None:
            animate_movement = render_mode =='human'
        if animate_movement:
            render_mode = 'human'

        # from irlc.utils.
        # self.action_space = ExplicitActionSpace(self) # Wrapper environments copy the action space.

        from irlc.pacman.gamestate import Directions
        self.action_space = DiscreteTextActionSpace(seed=None, actions=[Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST, Directions.STOP])


        # Load level layout
        if layout_str is not None:
            self.layout = Layout([line.strip() for line in layout_str.strip().splitlines()])
        else:
            self.layout = getLayout(layout)
            if self.layout is None:
                raise Exception("Layout file not found", layout)
        self.rules = ClassicGameRules(30)
        self.options_frametime = 1/frames_per_second
        self.game = None

        # Setup displays.
        self.first_person_graphics = False
        self.animate_movement = animate_movement
        self.options_zoom = zoom
        self.text_display = PacmanTextDisplay(1 / frames_per_second)
        self.graphics_display = None

        # temporary variables for animation/visualization. Don't remove.
        self.visitedlist = None
        self.ghostbeliefs = None
        self.path = None
        self.render_mode = render_mode
        self.method = method_str

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        :param seed:
        :param options:
        :return:
        """
        self.game = self.rules.newGame(self.layout, PacAgent(index=0), self.ghosts, quiet=True, catchExceptions=False)
        self.game.numMoves = 0
        if self.render_mode == 'human':
            self.render()
        return self.state, {'mask': self.action_space._make_mask(self.state.A()) }


    def close(self):
        if self.graphics_display is not None:
            self.graphics_display.close()
            return

    @property
    def state(self):
        if self.game is None:
            return None
        return self.game.state.deepCopy()

    def get_keys_to_action(self):
        return {(pygame.K_LEFT,): Directions.WEST,
                (pygame.K_RIGHT,): Directions.EAST,
                (pygame.K_UP,): Directions.NORTH,
                (pygame.K_DOWN,): Directions.SOUTH,
                (pygame.K_s,): Directions.STOP,
                }

    def step(self, action):
        r_ = self.game.state.getScore()
        done = False

        if action not in self.state.A():
            # if action not in self.A(self.state):
            raise Exception(f"Agent tried {action=} available actions {self.state.A()}")

        # Let player play `action`, then let the ghosts play their moves in sequence.
        for agent_index in range(len(self.game.agents)):
            a = self.game.agents[agent_index].getAction(self.game.state) if agent_index > 0 else action
            self.game.state = self.game.state.f(a)
            self.game.rules.process(self.game.state, self.game)

            if self.graphics_display is not None and self.animate_movement and agent_index == 0:
                self.graphics_display.update(self.game.state, animate=self.animate_movement, ghostbeliefs=self.ghostbeliefs, path=self.path, visitedlist=self.visitedlist)

            done = self.game.gameOver or self.game.state.is_won() or self.game.state.is_lost()
            if done:
                break
        reward = self.game.state.getScore() - r_
        return self.state, reward, done, False, {'mask': self.action_space._make_mask(self.state.A())}

    def render(self):
        if hasattr(self, 'agent'):
            path = self.agent.__dict__.get('path', None)
            ghostbeliefs = self.agent.__dict__.get('ghostbeliefs', None)
            visitedlist = self.agent.__dict__.get('visitedlist', None)
        else:
            path, ghostbeliefs, visitedlist = None, None, None

        # Initialize graphics adaptor.
        if self.graphics_display is None and self.render_mode in ["human", 'rgb_array']:
            if self.first_person_graphics:
                self.graphics_display = FirstPersonPacmanGraphics(self.game.state, self.options_zoom, showGhosts=True, frameTime=self.options_frametime, ghostbeliefs=self.ghostbeliefs)
                # self.graphics_display.ghostbeliefs = self.ghostbeliefs
            else:
                self.graphics_display = PacmanGraphics(self.game.state, self.options_zoom, frameTime=self.options_frametime, method=self.method)

        if self.render_mode in ["human", 'rgb_array']:
            # if self.graphics_display is None:
            #     if self.first_person_graphics:
            #         self.graphics_display = FirstPersonPacmanGraphics(self.options_zoom, showGhosts=True,
            #                                                           frameTime=self.options_frametime)
            #         self.graphics_display.ghostbeliefs = self.ghostbeliefs
            #     else:
            #         self.graphics_display = PacmanGraphics(self.options_zoom, frameTime=self.options_frametime)

            if not hasattr(self.graphics_display, 'viewer'):
                self.graphics_display.initialize(self.game.state.data)

            # We save these because the animation code may need it in step()
            self.visitedlist = visitedlist
            self.path = path
            self.ghostbeliefs = ghostbeliefs
            self.graphics_display.master_render(self.game.state, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)

            return self.graphics_display.blit(render_mode=self.render_mode)
            # return self.graphics_display.viewer.render(return_rgb_array=self.render_mode == "rgb_array")

        elif self.render_mode in ['ascii']:
            return self.text_display.draw(self.game.state)
        else:
            raise Exception("Bad video mode", self.render_mode)

    @property
    def viewer(self):
        if self.graphics_display is not None and hasattr(self.graphics_display, 'viewer'):
            return self.graphics_display.viewer
        else:
            return None


class PacmanWinWrapper(RewardWrapper):
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        if self.env.game.state.is_won():
            reward = 1
        else:
            reward = 0
        return observation, reward, done, truncated, info


if __name__ == "__main__":
    # from irlc import VideoMonitor
    import time
    # from irlc.utils.player_wrapper_pygame import PlayWrapperPygame
    # from irlc.utils.player_wrapper import PlayWrapper
    from irlc.ex01.agent import Agent, train
    from irlc import interactive

    # from irlc.pacman.pacman_environment import PacmanEnvironment
    # from irlc import Agent
    # env = PacmanEnvironment()
    # s, info = env.reset()
    # agent = Agent(env)
    # agent.pi(s, k=0, info=info)  # get a random action
    # agent.pi(s, k=0)  # If info is not specified, all actions are assumed permissible.


    env = PacmanEnvironment(layout='mediumClassic', animate_movement=True, render_mode='human')
    agent = Agent(env)
    # agent = PlayWrapperPygame(agent, env)
    env, agent = interactive(env, agent)

    # env = VideoMonitor(env)
    # experiment = "experiments/pacman_q"
    # if True:
    #     agent = Agent(env)
    #     agent = PlayWrapper(agent, env)
    train(env, agent, num_episodes=1)
    # env.unwrapped.close()
    time.sleep(0.1)
    env.close()
# 230 174, 159
