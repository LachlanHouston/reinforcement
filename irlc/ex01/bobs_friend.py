# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import gymnasium
import numpy as np
from gymnasium.spaces.discrete import Discrete
from irlc.ex01.agent import Agent, train

class BobFriendEnvironment(gymnasium.Env): 
    def __init__(self, x0=20):
        self.x0 = x0
        self.action_space = Discrete(2)     # Possible actions {0, 1} 

    def reset(self):
        # TODO: 1 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        return self.s, {}

    def step(self, a):
        # TODO: 9 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        return s_next, reward, terminated, False, {}

class AlwaysAction_u0(Agent):
    def pi(self, s, k, info=None):  
        """This agent should always take action u=0."""
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

class AlwaysAction_u1(Agent):
    def pi(self, s, k, info=None):  
        """This agent should always take action u=1."""
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

def run_bob():
    # Part A:
    env = BobFriendEnvironment()
    x0, _ = env.reset()
    print(f"Initial amount of money is x0 = {x0} (should be 20 kroner)")
    print("Lets put it in the bank, we should end up in state x1=22 and get a reward of 2 kroner")
    x1, reward, _, _, _ = env.step(0)
    print("we got", x1, reward)
    # Since we reset the environment, we should get the same result as before:
    env.reset()
    x1, reward, _, _, _ = env.step(0)
    print("(once more) we got", x1, reward, "(should be the same as before)")

    env.reset()  # We must call reset -- the environment has possibly been changed!
    print("Lets lend it to our friend -- what happens will now be random")
    x1, reward, _, _, _ = env.step(1)
    print("we got", x1, reward)

    # Part B:
    stats, _ = train(env, AlwaysAction_u0(env), num_episodes=1000)
    average_u0 = np.mean([stat['Accumulated Reward'] for stat in stats])

    stats, _ = train(env, AlwaysAction_u1(env), num_episodes=1000)
    average_u1 = np.mean([stat['Accumulated Reward'] for stat in stats])
    print(f"Average reward while taking action u=0 was {average_u0} (should be 2)")
    print(f"Average reward while taking action u=1 was {average_u1} (should be 4)")


if __name__ == "__main__":
    run_bob()
