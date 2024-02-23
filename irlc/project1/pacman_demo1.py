# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.project1.pacman import east, datadiscs, SS1tiny, SS2tiny
from irlc import interactive, savepdf, Agent, train
import matplotlib
matplotlib.use('qtagg')

count = """
%%%%
%P %
%..%
%%%%
"""


if __name__ == "__main__":
    # Example interaction with an environment:
    # Instantiate the map 'east' and get a GameState instance: 
    env = PacmanEnvironment(layout_str=east, render_mode='human')
    x, info = env.reset() # x is a irlc.pacman.gamestate.GameState object. See irlc/pacman/gamestate.py for the definition if you are curious.
    print("Start configuration of board:")
    print(x)
    env.close() # If you use render_mode = 'human', I recommend you use env.close() at the end of the code to free up graphics resources.
    # The GameState object `x` has a handful of useful functions. The important ones are:
    # x.A()       # Action space
    # x.f(action) # State resulting in taking action 'action' in state 'x'
    # x.players() # Number of agents on board (at least 1)
    # x.player()  # Whose turn it is (player = 0 is us)
    # x.isWin()   # True if we have won
    # x.isLose()  # True if we have lost
    # You can check if two GameState objects x1 and x2 are the same by simply doing x1 == x2. 
    # There are other functions in the GameState class, but I advise against using them.
    from irlc.pacman.pacman_environment import PacmanEnvironment, datadiscs
    env = PacmanEnvironment(layout_str=datadiscs, render_mode='human')
    env.reset()
    savepdf('pacman_east', env=env)
    env.close()

    env = PacmanEnvironment(layout_str=datadiscs, render_mode='human')
    env.reset()
    savepdf('pacman_datadiscs', env=env)
    env.close()

    env = PacmanEnvironment(layout_str=SS1tiny, render_mode='human')
    env.reset()
    savepdf('pacman_SS1tiny', env=env)
    env.close()

    env = PacmanEnvironment(layout_str=SS2tiny, render_mode='human')
    env.reset()
    savepdf('pacman_SS2tiny', env=env)
    env.close()
