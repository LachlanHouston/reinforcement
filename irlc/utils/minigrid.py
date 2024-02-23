# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import gymnasium as gym
from gymnasium.spaces.discrete import Discrete
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.wrappers import FullyObsWrapper
import numpy as np


class ProjectObservationSpaceWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """
    def __init__(self, env, dims):
        super().__init__(env)
        os = self.observation_space.spaces['image']
        # if dims is not None:
        os.high = os.high[:,:,dims]
        os.low = os.low[:,:,dims]

        self.observation_space.spaces['image'] = os
        self.dims = dims

    def observation(self, obs):
        obs['image'] = obs['image'][:, :, self.dims]
        return obs


class SaneBoundsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """
    def __init__(self, env):
        super().__init__(env)
        os = self.observation_space.spaces['image']
        os.high[:, :, 0] = max(OBJECT_TO_IDX.values())
        if os.high.shape[2] >= 2:
            os.high[:, :, 1] = max(COLOR_TO_IDX.values())
        if os.high.shape[2] >= 3:
            os.high[:, :, 2] = 3
        self.observation_space.spaces['image'] = os

    def observation(self, obs):
        return obs

class HashableImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env,dims=None):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        # ls = obs['image'].flat.tolist()
        return tuple( obs['image'].flat )
        # return obs['image']


class LinearSpaceWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """
    def __init__(self, env):
        super().__init__(env)
        sz = self.observation_space.spaces['image'].shape
        npo = np.zeros( sz, dtype=np.object)
        for i in range(sz[0]):
            for j in range(sz[1]):
                for k in range(sz[2]):
                    if k == 0:
                        n = max(OBJECT_TO_IDX.values())+1
                    elif k == 1:
                        n = max(COLOR_TO_IDX.values())+1
                    elif k == 2:
                        n = 4
                    else:
                        raise Exception("Bad k")

                    npo[i,j,k] = Discrete(n)
        ospace = tuple(npo.flat)

        sz = np.cumsum([o.n for o in ospace])
        sz = sz - sz[0]
        self.sz = sz
        # from gym.spaces.box import Box
        self.observation_space = ospace

    def observation(self, obs):
        s = obs['image'].reshape((obs['image'].size,))
        return s


if __name__ == "__main__":
    """ Example use: """
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FullyObsWrapper(env) # use this
    env = LinearSpaceWrapper(env)
    s = env.reset()
    print(s)
    # Use with for instance:
    # agent = LinearSemiGradSarsa(env, gamma=1, epsilon=0.1, alpha=0.5)
