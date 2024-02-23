# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import os
import numpy as np

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python lmpc_plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python lmpc_plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, y="accumulated_reward", x="Episode", ci=95, estimator='mean', **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    if isinstance(data, list): # is this correct even?
        data = pd.concat(data, ignore_index=True,axis=0)
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid", font_scale=1.5)
    lp = sns.lineplot(data=data, x=x, y=y, hue="Condition", errorbar=('ci', 95), estimator=estimator, **kwargs)
    plt.legend(loc='best') #.set_draggable(True)

def existing_runs(experiment):
    nex = 0
    for root, dir, files in os.walk(experiment):
        if 'log.txt' in files:
            nex += 1
    return nex

def _get_most_recent_log_dir(fpath):
    files = [os.path.basename(root) for root, dir, files in os.walk(fpath) if 'log.txt' in files]
    return sorted(files, key=lambda file: os.path.basename(file))[-1] if len(files) > 0 else None

def get_datasets(fpath, x, condition=None, smoothing_window=None, resample_key=None, resample_ticks=None, only_most_recent=False):
    import pandas as pd
    unit = 0
    if condition is None:
        condition = fpath
    datasets = []

    if only_most_recent:
        most_recent = _get_most_recent_log_dir(fpath)

    for root, dir, files in os.walk(fpath):
        # print(files)
        if 'log.txt' in files:
            if only_most_recent and most_recent is not None and os.path.basename(root) != most_recent: # Skip this log.
                continue
            json = os.path.join(root, 'params.json')
            if os.path.exists(json):
                with open(json) as f:
                    param_path = open(json)
                    params = json.load(param_path)
                    # exp_name = params['exp_name']

            log_path = os.path.join(root, 'log.txt')
            if os.stat(log_path).st_size == 0:
                print("Bad plot file", log_path, "size is zero. Skipping")
                continue
            experiment_data = pd.read_table(log_path)

            if smoothing_window:
                ed_x = experiment_data[x]
                experiment_data = experiment_data.rolling(smoothing_window,min_periods=1).mean()
                experiment_data[x] = ed_x

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition)

            datasets.append(experiment_data)
            unit += 1

    nc = f"({unit}x)"+condition[condition.rfind("/")+1:]
    for i, d in enumerate(datasets):
        datasets[i] = d.assign(Condition=lambda x: nc)

    if resample_key is not None:
        nmax = 0
        vmax = -np.inf
        vmin = np.inf
        for d in datasets:
            nmax = max( d.shape[0], nmax)
            vmax = max(d[resample_key].max(), vmax)
            vmin = min(d[resample_key].min(), vmin)
        if resample_ticks is not None:
            nmax = min(resample_ticks, nmax)

        new_datasets = []
        tnew = np.linspace(vmin + 1e-6, vmax - 1e-6, nmax)
        for d in datasets:
            nd = {}
            cols = d.columns.tolist()
            for c in cols:
                if c == resample_key:
                    y = tnew
                elif d[c].dtype == 'O':
                    y = [ d[c][0] ] * len(tnew)
                else:
                    y = np.interp(tnew, d[resample_key].tolist(), d[c], left=np.nan, right=np.nan)
                    y = y.astype(d[c].dtype)
                nd[c] = y

            ndata = pd.DataFrame(nd)
            ndata = ndata.dropna()
            new_datasets.append(ndata)
        datasets = new_datasets
    return datasets


def _load_data(experiments, legends=None, smoothing_window=None, resample_ticks=None,
              x_key="Episode",
              only_most_recent=False):
    ensure_list = lambda x: x if isinstance(x, list) else [x]
    experiments = ensure_list(experiments)
    if legends is None:
        legends = experiments
    legends = ensure_list(legends)

    data = []
    for logdir, legend_title in zip(experiments, legends):
        resample_key = x_key if resample_ticks is not None else None
        data += get_datasets(logdir, x=x_key, condition=legend_title, smoothing_window=smoothing_window, resample_key=resample_key, resample_ticks=resample_ticks,
                             only_most_recent=only_most_recent)
    return data

def main_plot(experiments, legends=None, smoothing_window=None, resample_ticks=None,
              x_key="Episode",
              y_key='Accumulated Reward',
              no_shading=False,
              **kwargs):
    if no_shading:
        kwargs['units'] = 'Unit'
        kwargs['estimator'] = None

    ensure_list = lambda x: x if isinstance(x, list) else [x]
    experiments = ensure_list(experiments)

    if legends is None:
        legends = experiments
    legends = ensure_list(legends)

    data = []
    for logdir, legend_title in zip(experiments, legends):
        resample_key = x_key if resample_ticks is not None else None
        data += get_datasets(logdir, x=x_key, condition=legend_title, smoothing_window=smoothing_window, resample_key=resample_key, resample_ticks=resample_ticks)

    plot_data(data, y=y_key, x=x_key, **kwargs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='AverageReturn', nargs='*')
    parser.add_argument('--title', default="please specify title", help="The title to show")
    parser.add_argument('--pdf_name', default=None, help="Name of pdf")

    args = parser.parse_args()
    main_plot(args.logdir, args.legend, args.value, title=args.title)

if __name__ == "__main__":
    main()


#### TRAJECTORY PLOTTING HERE ####
def plot_trajectory(trajectory, env=None, xkeys=None, ukeys=None):
    """
    Used to visualize trajectories returned from the :func:`~irlc.ex01.agent.train`-function. An example:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from irlc import Agent, plot_trajectory, train
        from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        env = GymSinCosPendulumEnvironment()
        stats, trajectories = train(env, Agent(env), num_episodes=1, return_trajectory=True)
        plot_trajectory(trajectories[0], env)

    Labels will be derived from the ``env`` if supplied. The parameters ``xkeys`` and ``ukeys`` can be used to limit which
    coordinates are plotted. For instance, if you only want to plot the first two x-coordinates you can set ``xkeys=[0,1]``:


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from irlc import Agent, plot_trajectory, train
        from irlc.ex04.model_pendulum import GymSinCosPendulumEnvironment
        env = GymSinCosPendulumEnvironment()
        stats, trajectories = train(env, Agent(env), num_episodes=1, return_trajectory=True)
        plot_trajectory(trajectories[0], env, xkeys=[0,1], ukeys=[])

    :param trajectory: A single trajectory computed using ``train`` (see example above)
    :param env: A gym control environment (optional)
    :param xkeys: List of integers corresponding to the coordinates of :math:`x` we wish to plot
    :param ukeys: List of integers corresponding to the coordinates of :math:`u` we wish to plot

    .. tip::
        If the plot does not show, you might want to import matplotlib as ``import matplotlib.pyplot as plt`` and call ``plt.show()``
    """
    if xkeys is None:
        xkeys = [i for i in range(trajectory.state.shape[1])]
    if ukeys is None: # all
        ukeys = [i for i in range(trajectory.action.shape[-1])]
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid", font_scale=1.5)
    def fp(time, X, keys, labels):
        for i, k in enumerate(keys):
            label = labels[k] if labels is not None else None
            sns.lineplot(x=time, y=X[:,k], label=label)

    time = trajectory.time.squeeze()
    fp(time, trajectory.state, xkeys, labels=env.state_labels if env is not None else None)
    fp(time[:-1], trajectory.action, ukeys, labels=env.action_labels if env is not None else None)
    plt.xlabel("Time / seconds")
    if env is not None:
        plt.legend()
