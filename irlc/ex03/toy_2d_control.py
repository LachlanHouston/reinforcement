# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
import numpy as np

class Toy2DControl(ControlModel):
    def get_cost(self):
        # You get the cost-function for free because it can be anything as far as this problem is concerned.
        return SymbolicQRCost(Q=np.eye(2), R=np.eye(1))

    # TODO: 2 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")

def toy_simulation(u0 : float, T : float) -> float:
    # TODO: 4 lines missing.
    raise NotImplementedError("Create a Toy2dControl instance and use model.simulate(..) to get the final state.")
    return wT

if __name__ == "__main__":
    x0 = np.asarray([np.pi/2, 0])
    wT = toy_simulation(u0=0.4, T=5)
    print(f"Starting in x0=[pi/2, 0], after T=5 seconds the system is an an angle {wT=} (should be 1.265)") 
