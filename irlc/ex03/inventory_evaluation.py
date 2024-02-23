# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex02.inventory import InventoryDPModel

def a_expected_items_next_day(x : int, u : int) -> float:
    model = InventoryDPModel()
    expected_number_of_items = None
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return expected_number_of_items


def b_evaluate_policy(pi : list, x0 : int) -> float:
    # TODO: Code has been removed from here.
    raise NotImplementedError("Insert your solution and remove this error.")
    return J_pi_x0

if __name__ == "__main__":
    model = InventoryDPModel()
    # Create a policy that always buy an item if the inventory is empty.
    pi = [{s: 1 if s == 0 else 0 for s in model.S(k)} for k in range(model.N)]
    x = 0
    u = 1
    x0 = 1
    a_expected_items_next_day(x=0, u=1)
    print(f"Given inventory is {x=} and we buy {u=}, the expected items on day k=1 is {a_expected_items_next_day(x, u)} and should be 0.1")
    print(f"Evaluation of policy is {b_evaluate_policy(pi, x0)} and should be 2.7")
