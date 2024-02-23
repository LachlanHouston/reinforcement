# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
Inspired by logz from berkleys deep RL course but re-written as a context manager like God intended.

To load the learning curves, you can do, for yafcport

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""
import json
import os
import time
from datetime import datetime

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class LazyLog(object):
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}

    def __init__(self, experiment_name, run_name=None, data=None):
        if run_name is None:
            experiment_name += "/"+ datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
        else:
            experiment_name += "/" + run_name
        self.experiment_name = experiment_name
        configure_output_dir(self, experiment_name)
        if data is not None:
            self.save_params(data)

    def __enter__(self):
        return self

    def save_params(self, data):
        save_params(self, data)

    def dump_tabular(self, verbose=False):
        dump_tabular(self, verbose)

    def log_tabular(self, key, value):
        log_tabular(self, key, value)

    def __exit__(self, type, value, traceback):
        self.output_file.close()


def configure_output_dir(G, d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    # CDIR = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    G.first_row = True
    G.output_dir = d or "/tmp/experiments/%i" % int(time.time())
    assert not os.path.exists(
        G.output_dir), "Log dir %s already exists! Delete it first or use a different dir" % G.output_dir
    os.makedirs(G.output_dir)
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    print(colorize("Logging data to %s" % G.output_file.name, 'green', bold=True))

def log_tabular(G, key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
    assert key not in G.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
    G.log_current_row[key] = val


def save_params(G, params):
    with open(os.path.join(G.output_dir, "params.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))


# def pickle_tf_vars():
#     import tensorflow as tf
#     """
#     Saves tensorflow variables
#     Requires them to be initialized first, also a default session must exist
#     """
#     _dict = {v.name: v.eval() for v in tf.global_variables()}
#     with open(osp.join(G.output_dir, "vars.pkl"), 'wb') as f:
#         pickle.dump(_dict, f)


def dump_tabular(G, verbose=True):
    """
    Write all of the diagnostics from the current iteration
    """
    vals = []
    key_lens = [len(key) for key in G.log_headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes) if verbose else None
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.3g" % val
        else:
            valstr = val
        print(fmt % (key, valstr)) if verbose else None
        vals.append(val)
    print("-" * n_slashes) if verbose else None
    if G.output_file is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False
