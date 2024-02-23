# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc.project1.pacman import east, datadiscs, SS1tiny, SS2tiny
from irlc import interactive, savepdf, Agent, train

if __name__ == "__main__":
    env = PacmanEnvironment(layout_str=datadiscs, render_mode='human')
    env, agent = interactive(env, Agent(env))
    stats, trajectory = train(env, agent, num_episodes=1)
    print("First state was\n", trajectory[0].state[0])
    env.close()
