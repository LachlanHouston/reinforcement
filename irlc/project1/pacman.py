# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from collections import defaultdict
from typing import Any
from irlc import train
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
from irlc.ex02.dp_agent import DynamicalProgrammingAgent
from irlc.pacman.pacman_environment import PacmanEnvironment

east = """ 
%%%%%%%%
% P   .%
%%%%%%%% """ 

east2 = """
%%%%%%%%
%    P.%
%%%%%%%% """

SS2tiny = """
%%%%%%
%.P  %
% GG.%
%%%%%%
"""

SS0tiny = """
%%%%%%
%.P  %
%   .%
%%%%%%
"""

SS1tiny = """
%%%%%%
%.P  %
%  G.%
%%%%%%
"""

datadiscs = """
%%%%%%%
%    .%
%.P%% %
%.   .%
%%%%%%%
"""

# TODO: 30 lines missing.
class PacManDPModel(DPModel):
    def __init__(self, x, N=3):
        super().__init__(N=N)
        self.future_states = get_future_states(x, N)

    def A(self, x, k):
        return x.A()
    
    def S(self, k):
        return self.future_states[k]
    
    def g(self, x, u, w, k):
        return 1 if not x.is_won() else 0
    
    def f(self, x, u, w, k):
        return x.f(u)
    
    def Pw(slf, x, u, k):
        return p_next(x, u)
    
    def gN(self, x):
        return 0
    
class PacmanDPModelWithGhosts(PacManDPModel): 
    # Override some functions.

    def g(self, x, u, w, k): # Cost function g_k(x,u,w) 
        return 0

    def f(self, x, u, w, k): # Dynamics f_k(x,u,w)
        return w

    def gN(self, x): 
        return -1 if x.is_won() else 0

def p_next(x, u): 
    """ Given the agent is in GameState x and takes action u, the game will transition to a new state xp.
    The state xp will be random when there are ghosts. This function should return a dictionary of the form

    {..., xp: p, ...}

    of all possible next states xp and their probability -- you need to compute this probability.

    Hints:
        * In the above, xp should be a GameState, and p will be a float. These are generated using the functions in the GameState x.
        * Start simple (zero ghosts). Then make it work with one ghosts, and then finally with any number of ghosts.
        * Remember the ghosts move at random. I.e. if a ghost has 3 available actions, it will choose one with probability 1/3
        * The slightly tricky part is that when there are multiple ghosts, different actions by the individual ghosts may lead to the same final state
        * Check the probabilities sum to 1. This will be your main way of debugging your code and catching issues relating to the previous point.
    """
    # TODO: 8 lines missing.
    x = x.f(u)  # Pacman's turn (deterministic)
    states = defaultdict(float)  # Defaults to 0.0
    
    # Multiple ghosts call recursively until no ghosts left.
    def next_ghost(x, prob, ghosts):
        if ghosts == 0:         # Terminal state, no ghosts left
            states[x] += prob
            return
        
        actions = x.A()
        prob /= len(actions)
        for u in actions:
            next_ghost(x.f(u), prob, ghosts-1)
    
    next_ghost(x, 1.0, x.players() - 1)
    return states


def go_east(map): 
    """ Given a map-string map (see examples in the top of this file) that can be solved by only going east, this will return
    a list of states Pacman will traverse. The list it returns should therefore be of the form:

    [s0, s1, s2, ..., sn]

    where each sk is a GameState object, the first element s0 is the start-configuration (corresponding to that in the Map),
    and the last configuration sn is a won GameState obtained by going east.

    Note this function should work independently of the number of required east-actions.

    Hints:
        * Use the GymPacmanEnvironment class. The report description will contain information about how to set it up, as will pacman_demo.py
        * Use this environment to get the first GameState, then use the recommended functions to go east
    """
    # TODO: 5 lines missing.
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset()
    states = [x]
    while not (x.is_won() or x.is_lost()): # Continue until we win or lose
        x = x.f("East")
        states.append(x)
        print(x.is_won(), x.is_lost())
    return states

def get_future_states(x, N): 
    # TODO: 4 lines missing.
    state_spaces = [[x]]
    for _ in range(N):
        state_spaces.append([])
        for x in state_spaces[-2]:
            for u in x.A():
                for xp, p in p_next(x, u).items():
                    if p > 0.0 and xp not in state_spaces[-1]:  # __eq__ is defined in GameState
                        state_spaces[-1].append(xp)
    return state_spaces

def win_probability(map, N=10): 
    """ Assuming you get a reward of -1 on winning (and otherwise zero), the win probability is -J_pi(x_0). """
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset()
    pac = PacmanDPModelWithGhosts(x, N)
    J, pi_ = DP_stochastic(pac)
    return -J[0][x]

def shortest_path(map, N=10): 
    """ If each move has a cost of 1, the shortest path is the path with the lowest cost.
    The actions should be the list of actions taken.
    The states should be a list of states the agent visit. The first should be the initial state and the last
    should be the won state. """
    # TODO: 4 lines missing.
    env = PacmanEnvironment(layout_str=map)
    x, _ = env.reset()
    pac_model = PacManDPModel(x, N)

    agent = DynamicalProgrammingAgent(env, pac_model)
    actions = []
    states = [x]

    for k in range(N):
        if x.is_won() or x.is_lost():
            break
        u = agent.pi(x, k)
        x = x.f(u)
        env.step(u)
        actions.append(u)
        states.append(x)

    env.close()
    return actions, states


def no_ghosts():
    # Check the pacman_demo.py file for help on the GameState class and how to get started.
    # This function contains examples of calling your functions. However, you should use unitgrade to verify correctness.

    ## Problem 1: Lets try to go East. Run this code to see if the states you return looks sensible.
    states = go_east(east)
    for s in states:
        print(str(s))

    ## Problem 3: try the p_next function for a few empty environments. Does the result look sensible?
    x, _ = PacmanEnvironment(layout_str=east).reset()
    action = x.A()[0]
    print(f"Transitions when taking action {action} in map: 'east'")
    print(x)
    print(p_next(x, action))  # use str(state) to get a nicer representation.

    print(f"Transitions when taking action {action} in map: 'east2'")
    x, _ = PacmanEnvironment(layout_str=east2).reset()
    print(x)
    print(p_next(x, action))

    ## Problem 4
    print(f"Checking states space S_1 for k=1 in SS0tiny:")
    x, _ = PacmanEnvironment(layout_str=SS0tiny).reset()
    states = get_future_states(x, N=10)
    for s in states[1]: # Print all elements in S_1.
        print(s)
    print("States at time k=10, |S_10| =", len(states[10]))

    ## Problem 6
    N = 20  # Planning horizon
    action, states = shortest_path(east, N)
    print("east: Optimal action sequence:", action)

    action, states = shortest_path(datadiscs, N)
    print("datadiscs: Optimal action sequence:", action)

    action, states = shortest_path(SS0tiny, N)
    print("SS0tiny: Optimal action sequence:", action)


def one_ghost():
    # Win probability when planning using a single ghost. Notice this tends to increase with planning depth
    wp = []
    for n in range(10):
        wp.append(win_probability(SS1tiny, N=n))
    print(wp)
    print("One ghost:", win_probability(SS1tiny, N=12))


def two_ghosts():
    # Win probability when planning using two ghosts
    print("Two ghosts:", win_probability(SS2tiny, N=12))

if __name__ == "__main__":
    # no_ghosts()
    one_ghost()
    two_ghosts()
