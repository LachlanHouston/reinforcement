# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
import sympy as sym
from irlc.ex03.control_model import ControlModel
from irlc.ex03.control_cost import SymbolicQRCost
import numpy as np
from irlc import savepdf
from gymnasium.spaces import Box


class KuramotoModel(ControlModel): 
    r"""
    The Kuramoto model. It implements the following dynamics:

    .. math::

        \dot{x}(t) = u(t) +\cos(x(t))

    I.e. the state and control variables are both one-dimensional. The cost function is simply:

    .. math::

        c(t) = \frac{1}{2}x(t)^2 + \frac{1}{2}u(t)^2

    This is a QR cost with :math:`Q=R=1`.
    """
    def u_bound(self) -> Box:
        return Box(-2, 2, shape=(1,))

    def x0_bound(self) -> Box:
        return Box(0, 0, shape=(1,))

    def get_cost(self) -> SymbolicQRCost:
        """
        Create a cost-object. The code defines a quadratic cost (with the given matrices) and allows easy computation
        of derivatives, etc. There are automatic ways to discretize the cost so you don't have to bother with that.
        See the online documentation for further details.
        """
        return SymbolicQRCost(Q=np.zeros((1, 1)), R=np.ones((1,1)))

    def sym_f(self, x: list, u: list, t=None): 
        r""" Return a symbolic expression representing the Kuramoto model.
        The inputs x, u are themselves *lists* of symbolic variables (insert breakpoint and check their value).
        you have to use them to create a symbolic object representing f, and return it as a list. That is, you are going to return

        .. codeblock:: python

            return [f_val]

        where ``f_val`` is the symbolic expression corresponding to the dynamics, i.e. :math:`u(t) + \cos( x(t))`.
        Note you can use trigonometric functions like ``sym.cos``.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement symbolic expression as a singleton list here")
        # define the symbolic expression 
        return symbolic_f_list  


def f(x, u):
    """ Implement the kuramoto osscilator model's dynamics, i.e. f such that dx/dt = f(x,u).
    The answer should be returned as a singleton list. """
    cmodel = KuramotoModel()
    # TODO: 1 lines missing.
    raise NotImplementedError("Insert your solution and remove this error.")
    # Use the ContiniousKuramotoModel to compute f(x,u). If in doubt, insert a breakpoint and let pycharms autocomplete
    # guide you. See my video to Exercise 2 for how to use the debugger. Don't forget to specify t (for instance t=0).
    # Note that sympys error messages can be a bit unforgiving.
    return f_value

def rk4_simulate(x0, u, t0, tF, N=1000):
    """
    Implement the RK4 algorithm (Her24, Algorithm 18).
    In this function, x0 and u are constant numpy ndarrays. I.e. u is not a function, which simplify the RK4
    algorithm a bit.

    The function you want to integrate, f, is already defined above. You can likewise assume f is not a function of
    time. t0 and tF play the same role as in the algorithm.

    The function should return a numpy ndarray xs of dimension (N,) (containing all the x-values) and a numpy ndarray
    tt containing the corresponding time points.

    Hints:
        * Call f as in f(x, u). You defined f earlier in this exercise.
    """
    tt = np.linspace(t0, tF, N+1)   # Time grid t_k = tt[k] between t0 and tF.
    xs = [ x0 ]
    f(x0, u) # This is how you can call f.
    for k in range(N):
        x_next = None # Obtain x_next = x_{k+1} using a single RK4 step.
        # Remember to insert breakpoints and use the console to examine what the various variables are.
        # TODO: 7 lines missing.
        raise NotImplementedError("Insert your solution and remove this error.")
        xs.append(x_next)
    xs = np.stack(xs, axis=0)
    return xs, tt

if __name__ == "__main__":
    # Create a symbolic model corresponding to the Kuramoto model:
    # Evaluate the dynamics dx / dt = f(x, u).

    print("Value of f(x,u) in x=2, u=0.3", f([2], [0.3])) 
    print("Value of f(x,u) in x=0, u=1", f([0], [1])) 

    cmodel = KuramotoModel()
    print(cmodel)
    x0 = cmodel.x0_bound().low  # Get the starting state x0. We exploit that the bound on x0 is an equality constraint.
    u = 1.3
    xs, ts = rk4_simulate(x0, [u], t0=0, tF=20, N=100)
    xs_true, us_true, ts_true = cmodel.simulate(x0, u_fun=u, t0=0, tF=20, N_steps=100)
    """You should generally use cmodel.simulate(...) to simulate the environment. Note that u_fun in the simulate 
    function can be set to a constant. Use this compute numpy ndarrays corresponding to the time, x and u values.
    """
    # Plot the exact simulation of the environment
    import matplotlib.pyplot as plt
    plt.plot(ts_true, xs_true, 'k.-', label='RK4 state sequence x(t) (using model.simulate)')
    plt.plot(ts, xs, 'r-', label='RK4 state sequence x(t) (using your code)')
    plt.legend()
    savepdf('kuramoto_rk4')
    plt.show(block=False)
