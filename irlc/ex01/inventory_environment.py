# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np
from gymnasium.spaces.discrete import Discrete
from gymnasium import Env
from irlc.ex01.agent import Agent, train

class InventoryEnvironment(Env): 
    def __init__(self, N=2):
        self.N = N                               # planning horizon
        self.action_space      = Discrete(3)     # Possible actions {0, 1, 2}
        self.observation_space = Discrete(3)     # Possible observations {0, 1, 2}

    def reset(self):
        self.s = 0                               # reset initial state x0=0
        self.k = 0                               # reset time step k=0
        return self.s, {}                        # Return the state we reset to (and an empty dict)

    def step(self, a):
        w = np.random.choice(3, p=(.1, .7, .2))       # Generate random disturbance
        # TODO: 5 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        return s_next, reward, terminated, False, {}  # return transition information  

class RandomAgent(Agent): 
    def pi(self, s, k, info=None): 
        """ Return action to take in state s at time step k """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")


def simplified_train(env: Env, agent: Agent) -> float: 
    s, _ = env.reset()
    J = 0  # Accumulated reward for this rollout
    for k in range(1000):
        ## TODO: Oy veh, the following 7 lines below have been permuted. Uncomment, rearrange to the correct order and remove the error.
        #-------------------------------------------------------------------------------------------------------------------------------
        # if terminated or truncated:
        # sp, r, terminated, truncated, metadata = env.step(a)
        # a = agent.pi(s, k) 
        # s = sp
        # J += r
        # agent.train(s, a, sp, r, terminated)
        #     break 
        raise NotImplementedError("Remove this exception after the above lines have been uncommented and rearranged.")
    return J 

def run_inventory():
    env = InventoryEnvironment() 
    agent = RandomAgent(env)
    stats, _ = train(env,agent,num_episodes=1,verbose=False)  # Perform one rollout.
    print("Accumulated reward of first episode", stats[0]['Accumulated Reward']) 
    # I recommend inspecting 'stats' in a debugger; why do you think it is a list of length 1?

    stats, _ = train(env, agent, num_episodes=1000,verbose=False)  # do 1000 rollouts 
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[RandomAgent class] Average cost of random policy J_pi_random(0)=", -avg_reward) 
    # Try to inspect stats again in a debugger here. How long is the list now?

    stats, _ = train(env, Agent(env), num_episodes=1000,verbose=False)  # Perform 1000 rollouts using Agent class 
    avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("[Agent class] Average cost of random policy J_pi_random(0)=", -avg_reward)  


    """ Second part: Using the simplified training method. I.e. do not use train() below.
     You can find some pretty strong hints about what goes on in simplified_train in the lecture slides for today. """
    avg_reward_simplified_train = np.mean( [simplified_train(env, agent) for i in range(1000)]) 
    print("[simplified train] Average cost of random policy J_pi_random(0) =", -avg_reward_simplified_train)  



if __name__ == "__main__":
    run_inventory()
