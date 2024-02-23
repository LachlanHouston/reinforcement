# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.inventory import InventoryDPModel
from irlc.ex02.dp import DP_stochastic
import numpy as np

# TODO: Code has been removed from here.
raise NotImplementedError("Insert your solution and remove this error.")

def a_get_policy(N: int, c: float, x0 : int) -> int:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return u

def b_prob_one(N : int, x0 : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return pr_empty


if __name__ == "__main__":
    model = InventoryDPModel()
    pi = [{s: 0 for s in model.S(k)} for k in range(model.N)]
    x0 = 0
    c = 0.5
    N = 3
    print(f"a) The policy choice for {c=} is {a_get_policy(N, c,x0)} should be 1")
    print(f"b) The probability of ending up with a single element in the inventory is {b_prob_one(N, x0)} and should be 0.492")
