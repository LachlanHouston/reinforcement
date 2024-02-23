# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
from gymnasium.spaces import Box

class LinearQuadraticModel(ControlModel):
    """
    Implements a model with update equations

    dx/dt = Ax + Bx + d
    Cost = integral_0^{t_F} (1/2 x^T Q x + 1/2 u^T R u + q' x + qc) dt
    """
    def __init__(self, A, B, Q, R, q=None, qc=None, d=None):  
        self._cost = SymbolicQRCost(R=R, Q=Q, q=q, qc=qc)
        self.A, self.B, self.d = A, B, d
        super().__init__()

    def sym_f(self, x, u, t=None):  
        xp = sym.Matrix(self.A) * sym.Matrix(x) + sym.Matrix(self.B) * sym.Matrix(u)
        if self.d is not None:
            xp += sym.Matrix(self.d)
        return [x for xr in xp.tolist() for x in xr]  

    def x0_bound(self) -> Box:
        return Box(0, 0, shape=(self.state_size,))

    def get_cost(self):
        return self._cost
