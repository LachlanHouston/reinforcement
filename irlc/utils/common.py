# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from gymnasium import spaces
import collections
import inspect
import types
import numpy as np
import os, glob, csv
from irlc.utils.lazylog import LazyLog

class defaultdict2(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))

        if isinstance(self.default_factory, types.FunctionType):
            nargs = len(inspect.getfullargspec(self.default_factory).args)
            self[key] = value = self.default_factory(key) if nargs == 1 else self.default_factory()
            return value
        else:
            return super().__missing__(key)

## Helper functions for saving/loading a time series
def load_time_series(experiment_name, exclude_empty=True):
    """
    Load most recent non-empty time series (we load non-empty since lazylog creates a new dir immediately)
    """
    files = list(filter(os.path.isdir, glob.glob(experiment_name+"/*")))
    if exclude_empty:
        files = [f for f in files if os.path.exists(os.path.join(f, "log.txt")) and os.stat(os.path.join(f, "log.txt")).st_size > 0]

    if len(files) == 0:
        return [], None
    recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
    stats = []
    with open(recent + '/log.txt', 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csv_reader):
            if i == 0:
                head = row
            else:
                def tofloat(v):
                    try:
                        return float(v)
                    except Exception:
                        return v

                stats.append( {k:tofloat(v) for k, v in zip(head, row) } )
    return stats, recent

def average_trajectories(trajectories):
    if len(trajectories) == 0:
        return None
    from irlc.ex01.agent import Trajectory, fields
    t = trajectories[0]
    # t._asdict()
    # n = max( [len(t.time) for t in trajectories] )
    trajectories2 = sorted(trajectories, key=lambda t: len(t.time))
    tlong = trajectories2[-1]
    dd = dict(state=[], action=[],reward=[])
    # keys = list(dd.keys())

    for t in range(len(tlong.time)):
        for k in ['state', 'action', 'reward']:
            avg = []
            for traj in trajectories:
                z = traj.__getattribute__(k)
                if len(z) > t:
                    avg.append(z[t])
            if len(avg) > 0:
                # avg = np.stack(avg)
                avg = np.mean(avg, axis=0)
                dd[k].append(avg)

    dd = {k: np.stack(v) for k, v in dd.items()}
    tavg = Trajectory(**dd, time=tlong.time, env_info=[])
    return tavg

    # tlong.state *= 0
    # tlong.action *= 0

    # for i in range(n):


def experiment_load(experiment_name, exclude_empty=True):
    files = list(filter(os.path.isdir, glob.glob(experiment_name + "/*")))
    if exclude_empty:
        files = [f for f in files if
                 os.path.exists(os.path.join(f, "log.txt")) and os.stat(os.path.join(f, "log.txt")).st_size > 0]
    if len(files) == 0:
        return []
    values = []
    files = sorted(files, key=lambda file: os.path.basename(file))
    for recent in files:
        # recent = sorted(files, key=lambda file: os.path.basename(file))[-1]
        stats = []
        with open(recent + '/log.txt', 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(csv_reader):
                if i == 0:
                    head = row
                else:
                    def tofloat(v):
                        try:
                            return float(v)
                        except Exception:
                            return v

                    stats.append({k: tofloat(v) for k, v in zip(head, row)})

        from irlc import cache_read, cache_write, cache_exists
        tpath = recent + "/trajectories.pkl"
        if cache_exists(tpath):
            trajectories = cache_read(tpath)
        else:
            trajectories = None
        values.append( (stats, trajectories, recent) )
    return values

def log_time_series(experiment, list_obs, max_xticks_to_log=None, run_name=None):
    logdir = f"{experiment}/"

    if max_xticks_to_log is not None and len(list_obs) > max_xticks_to_log:
        I = np.round(np.linspace(0, len(list_obs) - 1, max_xticks_to_log))
        list_obs = [o for i, o in enumerate(list_obs) if i in I.astype(np.int).tolist()]

    akeys = list(list_obs[0].keys())
    akeys += [k for k in list_obs[-1].keys() if k not in akeys]
    with LazyLog(logdir) as logz:
        for n,l in enumerate(list_obs):
            for k in akeys:
                v = None
                if k not in l:
                    for ll in list_obs[n:]:
                        if k in ll:
                            v = ll[k]
                            break
                    if v is None:
                        v = np.nan
                else:
                    v = l.get(k)
                logz.log_tabular(k,v)
            if "Steps" not in l:
                logz.log_tabular("Steps", n)
            if "Episode" not in l:
                logz.log_tabular("Episode",n)
            logz.dump_tabular(verbose=False)
        experiment_name = logz.experiment_name
    return experiment_name


class DiscreteTextActionSpace(spaces.Space):
    def __init__(self, actions, seed=None):
        # self.env = env
        # self._actions = actions
        self.actions = actions
        self.ds = spaces.Discrete(seed=seed, n=len(actions))
        # self.start = 0
        # self.actions = actions
        # super().__init__(shape=(len(actions),))

    # @property
    # def actions(self):
    #     return self._actions
    # return self.env.A(self.env.state)

    def sample(self, mask=None):
        return self.actions[self.ds.sample(mask)]

    @property
    def n(self):
        return self.ds.n

    def _make_mask(self, actions):
        mask = np.zeros((self.n,), dtype=np.int8)
        for a in actions:
            mask[self.actions.index(a)] = 1
        return mask

    def __str__(self):
        return f"<ExplicitAction space with actions: {', '.join(self.actions)}>"

    # def __contains__(self, action):
    #     return


class ExplicitActionSpace(spaces.Discrete):
    # Hacky stuff I don't think I need anymore.

    def __init__(self, env):
        self.env = env
        self.start = 0
        raise Exception()
        # pass
        # self.actions = actions
        # super().__init__(len(actions))

    @property
    def actions(self):
        return self.env.A(self.env.state)

    @property
    def n(self):
        return len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)
