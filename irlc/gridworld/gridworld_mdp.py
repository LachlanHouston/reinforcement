# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from irlc.ex09.mdp import MDP


class GridworldMDP(MDP):
    TERMINAL = "Terminal state"
    NORTH = 0   # These are the four available actions.
    EAST = 1
    SOUTH = 2
    WEST = 3
    actions2labels = {NORTH: 'North',
                      SOUTH: 'South',
                      EAST: 'East',
                      WEST: 'West'}  # This dictionary is useful for labelling purposes but otherwise serve no purpose.

    def __init__(self, grid, living_reward=0.0, noise=0.0):
        self.grid = {}
        self.height = len(grid)
        self.width = len(grid[0])
        initial_state = None
        for dy, line in enumerate(grid):
            y = self.height - dy - 1
            for x, el in enumerate(line):
                self.grid[x, y] = el
                if el == 'S':
                    initial_state = (x, y)
        self.noise = noise
        self.living_reward = living_reward
        super().__init__(initial_state=initial_state)

    def A(self, state):
        """
        Returns list of valid actions available in 'state'.

        You can try to go into walls (but will state in your location)
        and when you are on the exit-squares (i.e., the ones with numbers), you have a single action available
        'North' which will take you to the terminal square.
        """
        return (self.NORTH,) if type(self.grid[state]) in [int, float] else (self.NORTH, self.EAST, self.SOUTH, self.WEST)

    def is_terminal(self, state):
        return state == self.TERMINAL

    def Psr(self, state, action):
        if type(self.grid[state]) in [float, int]:
            return {(self.TERMINAL, self.grid[state]): 1.}

        probabilities = defaultdict(float)
        for a, pr in [(action, 1-self.noise),  ((action - 1) % 4, self.noise/2), ((action + 1) % 4, self.noise/2)]:
            sp = self.f(state, a)
            r = self.grid[state] if type(self.grid[state]) in [int, float] else self.living_reward
            probabilities[(sp, r)] += pr
        return probabilities

    def f(self, state, action):
        x, y = state
        nxt = {self.NORTH: (x,   y+1),
               self.WEST:  (x-1, y),
               self.EAST:  (x+1, y),
               self.SOUTH: (x,   y-1)}
        return nxt[action] if self._legal(nxt[action]) else state

    def _legal(self, state):
        return state in self.grid and self.grid[state] != "#"


class FrozenGridMDP(GridworldMDP):
    def __init__(self, grid, is_slippery=True, living_reward=0):
        self.is_slippery = is_slippery
        super().__init__(grid, noise=2/3 if is_slippery else 0, living_reward=living_reward)
