# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc import Agent, train, savepdf


# Maze layouts can be specified using a string.
layout = """
%%%%%%%%%%
%P.......%
%.%%%%%%.%
%.%    %.%
%.%    %.%
%.%    %.%
%.%    %.%
%.%%%%%%.%
%........%
%%%%%%%%%%
"""

# This is our first agent. Note it inherits from the Agent class. Use <ctrl>+click in pycharm to navigate to code definitions --
# this is a very useful habbit when you work with other peoples code in general, and object-oriented code in particular.
class GoAroundAgent(Agent):
    def pi(self, x, k, info=None): 
        """ Collect all dots in the maze in the smallest amount of time.
        This function should return an action, check the output of the code below to see what actions you can potentially
        return.
        Remember Pacman only have to solve this single maze, so don't make the function general.

        Hints:
            - Insert a breakpoint in the function. Try to write self.env and self.env.action_space.actions in the interpreter. Where did self.env get set?
            - Remember that k is the current step number.
            - Ignore the info dictionary; you can probably also ignore the state x.
            - The function should return a string (the actions are strings such as 'North')
        """
        # TODO: 7 lines missing.
        raise NotImplementedError("Implement function body")
        return 'West'

if __name__ == "__main__":
    # Create an environment with the given layout. animate_movement is just for a nicer visualization.
    env = PacmanEnvironment(layout_str=layout, render_mode='human')
    # This creates a visualization (Note this makes the environment slower) which can help us see what Pacman does
    # This create the GoAroundAgent-instance
    agent = GoAroundAgent(env)
    # Uncomment the following line to input actions instead of the agent using the keyboard:
    # env, agent = interactive(env, agent)
    s, info = env.reset() # Reset (and start) the environment

    savepdf("pacman_roundabout.pdf", env=env) # Saves a snapshot of the start layout
    # The next two lines display two ways to get the available actions. The 'canonical' way using the
    # env.action_space, and a way particular to Pacman by using the s.A() function on the state.
    # You can read more about the functions in the state in project 1.
    # print("Available actions at start:", env.action_space.actions) # This will list the available actions. 
    print("Alternative way of getting actions:", s.A())  # See also project description

    # Simulate the agent for one episode
    stats, _ = train(env, agent, num_episodes=1)
    # Print your obtained score.
    print("Your obtained score was", stats[0]['Accumulated Reward'])
    env.close()  # When working with visualizations, call env.close() to close windows it may have opened. "
