# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex01.agent import Agent
from irlc.ex02.dp import DP_stochastic
from irlc import train
import numpy as np


class DynamicalProgrammingAgent(Agent):
    """
    This is an agent which plan using dynamical programming.
    """
    def __init__(self, env, model=None):
        super().__init__(env)
        self.J, self.pi_ = DP_stochastic(model)

    def pi(self, s, k, info=None):
        if k >= len(self.pi_):
            raise Exception("k >= N; I have not planned this far!")
        ## TODO: Half of each line of code in the following 1 lines have been replaced by garbage. Make it work and remove the error.
        #----------------------------------------------------------------------------------------------------------------------------
        # action = se????????????
        raise NotImplementedError("Get the action according to the DP policy.")
        return action

    def train(self, s, a, r, sp, done=False, info_s=None, info_sp=None):  # Do nothing; this is DP so no learning takes place.
        pass


def main():
    from irlc.ex01.inventory_environment import InventoryEnvironment
    from irlc.ex02.inventory import InventoryDPModel

    env = InventoryEnvironment(N=3) 
    inventory_model = InventoryDPModel(N=3)
    agent = DynamicalProgrammingAgent(env, model=inventory_model)
    stats, _ = train(env, agent, num_episodes=5000) 

    s, _ = env.reset() # Get initial state
    Er = np.mean([stat['Accumulated Reward'] for stat in stats])
    print("Estimated reward using trained policy and MC rollouts", Er)  
    print("Reward as computed using DP", -agent.J[0][s])  

if __name__ == "__main__":
    main()
