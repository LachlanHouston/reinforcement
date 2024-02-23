# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
import irlc
# from irlc.ex01.frozen_lake import FrozenAgentDownRight
import gymnasium as gym
from unitgrade import UTestCase
from irlc.ex01.inventory_environment import InventoryEnvironment, simplified_train, RandomAgent
from unitgrade import Capturing2
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import RIGHT, DOWN  # The down and right-actions; may be relevant.
from irlc.ex01.pacman_hardcoded import GoAroundAgent, layout
from irlc.pacman.pacman_environment import PacmanEnvironment
from irlc import Agent, train
from irlc.ex01.bobs_friend import BobFriendEnvironment, AlwaysAction_u1, AlwaysAction_u0


class Problem1BobsFriend(UTestCase):
    def test_a_env_basic(self):
        env = BobFriendEnvironment()
        s0, _ = env.reset()
        self.assertEqual(s0, 20, msg="Reset must return the initial state, i.e. the amount of money we start out with")

    def test_a_env_u0(self):
        env = BobFriendEnvironment()
        env.reset()
        s1, r, done, _, _ = env.step(0)
        self.assertEqual(r, 2, msg="When taking action u0, we must get a reward of 2.")
        self.assertEqual(s1, 22, msg="When taking action u0, we must end in state x1=22")
        self.assertEqual(done, True, msg="After taking an action, the environment must terminate")

class Problem2BobsPolicy(UTestCase):
    def test_a_env_u1(self):
        env = BobFriendEnvironment()
        env.reset()
        s1, r, done, _, _ = env.step(1)
        print(r)
        self.assertTrue(r == 12 or r == -20, msg="When taking action u1, we must get a reward of 0 or 12.")
        self.assertTrue(s1 == 0 or s1 == 32, msg="When taking action u1, we must end in state x1=0 or x1 = 34")
        self.assertEqual(done, True, msg="After taking an action, the environment must terminate")

    def test_b_always_action_u0(self):
        env = BobFriendEnvironment()
        stats, _ = train(env, AlwaysAction_u0(env), num_episodes=1000)
        avg = np.mean( [stat['Accumulated Reward'] for stat in stats] )
        self.assertL2(avg, 2, msg="Average reward when we always take action u=0 must be 2.")

    def test_b_always_action_u1(self):
        env = BobFriendEnvironment()
        stats, _ = train(env, AlwaysAction_u1(env), num_episodes=10000)
        avg = np.mean( [stat['Accumulated Reward'] for stat in stats] )
        self.assertL2(avg, 4, tol=0.5, msg="Average reward when we always take action u=0 must be about 4.")

    def test_b_always_action_u1_starting_200(self):
        env = BobFriendEnvironment(x0=200)
        stats, _ = train(env, AlwaysAction_u1(env), num_episodes=10000)
        avg = np.mean( [stat['Accumulated Reward'] for stat in stats] )
        self.assertL2(avg, -42, tol=4, msg="Average reward when we always take action u=0 must be about 4.")

    def test_b_always_action_u0_starting_200(self):
        env = BobFriendEnvironment(x0=200)
        stats, _ = train(env, AlwaysAction_u0(env), num_episodes=10000)
        avg = np.mean( [stat['Accumulated Reward'] for stat in stats] )
        self.assertL2(avg, 20, msg="Average reward when we always take action u=0 must be about 4.")



class Problem5PacmanHardcoded(UTestCase):
    """ Test the hardcoded pacman agent """
    def test_pacman(self):
        env = PacmanEnvironment(layout_str=layout)
        agent = GoAroundAgent(env)
        stats, _ = train(env, agent, num_episodes=1)
        self.assertEqual(stats[0]['Length'] < 100, True)


class Problem6ChessTournament(UTestCase):
    def test_chess(self):
        """ Test the correct result in the little chess-tournament """
        from irlc.ex01.chess import main
        with Capturing2() as c:
            main()
        # Extract the numbers from the console output.
        print("Numbers extracted from console output was")
        print(c.numbers)
        self.assertLinf(c.numbers[-2], 26/33, tol=0.05)

class Problem3InventoryInventoryEnvironment(UTestCase):
    def test_environment(self):
        env = InventoryEnvironment()
        # agent = RandomAgent(env)
        stats, _ = train(env, Agent(env), num_episodes=2000, verbose=False)
        avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
        self.assertLinf(avg_reward, tol=0.6)

    def test_random_agent(self):
        env = InventoryEnvironment()
        stats, _ = train(env, RandomAgent(env), num_episodes=2000, verbose=False)
        avg_reward = np.mean([stat['Accumulated Reward'] for stat in stats])
        self.assertLinf(avg_reward, tol=0.6)

class Problem4InventoryTrain(UTestCase):
    def test_simplified_train(self):
        env = InventoryEnvironment()
        agent = Agent(env)
        avg_reward_simplified_train = np.mean([simplified_train(env, agent) for i in range(1000)])
        self.assertLinf(avg_reward_simplified_train, tol=0.5)

# class FrozenLakeTest(UTestCase):
#     def test_frozen_lake(self):
#         env = gym.make("FrozenLake-v1")
#         agent = FrozenAgentDownRight(env)
#         s = env.reset()
#         for k in range(10):
#             self.assertEqual(agent.pi(s, k), DOWN if k % 2 == 0 else RIGHT)


class Week01Tests(Report): #240 total.
    title = "Tests for week 01"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (Problem1BobsFriend, 10),
        (Problem2BobsPolicy, 10),
        (Problem3InventoryInventoryEnvironment, 10),
        (Problem4InventoryTrain, 10),
        (Problem5PacmanHardcoded, 10),
        (Problem6ChessTournament, 10),      # Week 1: Everything
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week01Tests())
