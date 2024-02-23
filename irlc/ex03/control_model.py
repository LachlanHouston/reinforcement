# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
from collections import defaultdict
import tabulate
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
from irlc.ex03.control_cost import SymbolicQRCost


class ControlModel: 
    r"""Represents the continious time model of a control environment.


    See (Her24, Section 13.2) for a top-level description.

    The model represents the physical system we are simulating and can be considered a control-equivalent of the
    :class:`irlc.ex02.dp_model.DPModel`. The class must keep track of the following:

    .. math::
        \frac{dx}{dt} = f(x, u, t)

    And the cost-function which is defined as an integral

    .. math::
        c_F(t_0, t_F, x(t_0), x(t_F)) + \int_{t_0}^{t_F} c(t, x, u) dt

    as well as constraints and boundary conditions on :math:`x`, :math:`u` and the initial conditions state :math:`x(t_0)`.
    this course, the cost function will always be quadratic, and can be accessed as ``model.get_cost``.

    If you want to implement your own model, the best approach is to start with an existing model and modify it for
    your needs. The overall idea is that you implement the dynamics,``sym_f``, and the cost function ``get_cost``,
    and optionally define bounds as needed.
    """
    state_labels = None     # Labels (as lists) used for visualizations.
    action_labels = None    # Labels (as lists) used for visualizations.

    def __init__(self): 
        """
        The cost must be an instance of :class:`irlc.ex04.cost_continuous.SymbolicQRCost`.
        Bounds is a dictionary but otherwise optional; the model should give it a default value.

        :param cost: A quadratic cost function
        :param dict bounds: A dictionary of boundary constraints.
        """
        if self.state_labels is None:
            self.state_labels = [f'x{i}' for i in range(self.state_size)]
        if self.action_labels is None:
            self.action_labels = [f'u{i}' for i in range(self.action_size)]

        t = sym.symbols("t") 
        x = sym.symbols(f"x0:{self.state_size}")
        u = sym.symbols(f"u0:{self.action_size}")
        try:
            f = self.sym_f(x, u, t)
        except Exception as e:
            print("control_model.py> There is a problem with the way you have specified the dynamics. The function sym_f must accept lists as inputs")
            raise e
        if len(f) != len(x):
            print("control_model.py> Your function ControlModel.sym_f must output a list of symbolic expressions.")
            assert len(f) == len(x)

        self._f_np = sym.lambdify((x, u, t), self.sym_f(x, u, t))  

    def x0_bound(self) -> Box: 
        r"""The bound on the initial state :math:`\mathbf{x}_0`.

        The default bound is ``Box(0, 0, shape=(self.state_size,))``, i.e. :math:`\mathbf{x}_0 = 0`.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(0, 0, shape=(self.state_size,)) 

    def xF_bound(self) -> Box: 
        r"""The bound on the terminal state :math:`\mathbf{x}_F`.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(-np.inf, np.inf, shape=(self.state_size,)) 

    def x_bound(self) -> Box: 
        r"""The bound on all other states :math:`\mathbf{x}(t)`.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(-np.inf, np.inf, shape=(self.state_size,)) 

    def u_bound(self) -> Box: 
        r"""The bound on the terminal state :math:`\mathbf{u}(t)`.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(-np.inf, np.inf, shape=(self.action_size,)) 

    def t0_bound(self) -> Box: 
        r"""The bound on the initial time :math:`\mathbf{t}_0`.

        I have included this bound for completeness: In practice, there is no reason why you should change it
        from the default bound is ``Box(0, 0, shape=(1,))``, i.e. :math:`\mathbf{t}_0 = 0`.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(0, 0, shape=(1,)) 

    def tF_bound(self) -> Box:  
        r"""The bound on the final time :math:`\mathbf{t}_F`, i.e. when the environment terminates.

        :return: An appropriate gymnasium Box instance.
        """
        return Box(-np.inf, np.inf, shape=(1,)) 

    def get_cost(self) -> SymbolicQRCost:
        raise NotImplementedError("When you implement the model, you must implement the get_cost() function.\nfor instance, use return SymbolicQRCost(Q=np.eye(n), R=np.eye(d))")

    def sym_f(self, x, u, t=None): 
        """
        The symbolic (``sympy``) version of the dynamics :math:`f(x, u, t)`. This is the main place where you specify
        the dynamics when you build a new model. you should look at concrete implementations of models for specifics.

        :param x: A list of symbolic expressions ``['x0', 'x1', ..]`` corresponding to :math:`x`
        :param u: A list of symbolic expressions ``['u0', 'u1', ..]`` corresponding to :math:`u`
        :param t: A single symbolic expression corresponding to the time :math:`t` (seconds)
        :return: A list of symbolic expressions ``[f0, f1, ...]`` of the same length as ``x`` where each element is a coordinate of :math:`f`
        """
        raise NotImplementedError("Implement a function which return the environment dynamics f(x,u,t) as a sympy exression") 

    def f(self, x, u, t=0) -> np.ndarray:
        r"""Evaluate the dynamics.

        This function will evaluate the dynamics. In other words, it will evaluate :math:`\mathbf{f}` in the following expression:

        .. math::

            \dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t)

        :param x: A numpy ndarray corresponding to the state
        :param u: A numpy ndarray corresponding to the control
        :param t: A :python:`float` corresponding to the time.
        :return: The time derivative of the state, :math:`\mathbf{x}(t)`.
        """
        return np.asarray( self._f_np(x, u, t) )


    def simulate(self, x0, u_fun, t0, tF, N_steps=1000, method='rk4'):  
        """
        Used to simulate the effect of a policy on the model. By default, it uses
        Runge-Kutta 4 (RK4) with a fine discretization -- this is slow, but in nearly all cases exact. See (Her24, Algorithm 18) for more information.

        The input argument ``u_fun`` should be a function which returns a list or tuple with same dimension as
        ``model.action_space``, :math:`d`.

        :param x0: The initial state of the simulation. Must be a list of floats of same dimension as ``env.observation_space``, :math:`n`.
        :param u_fun: Can be either:
            - Either a policy function that can be called as ``u_fun(x, t)`` and returns an action ``u`` in the ``action_space``
            - A single action (i.e. a list of floats of same length as the action space). The model will be simulated with a constant action in this case.
        :param float t0: Starting time :math:`t_0`
        :param float tF: Stopping time :math:`t_F`; the model will be simulated for :math:`t_F - t_0` seconds
        :param int N_steps: Steps :math:`N` in the RK4 simulation
        :param str method: Simulation method. Either ``'rk4'`` (default) or ``'euler'``
        :return:
            - xs - A numpy ``ndarray`` of dimension :math:`(N+1)\\times n` containing the observations :math:`x`
            - us - A numpy ``ndarray`` of dimension :math:`(N+1)\\times d` containing the actions :math:`u`
            - ts - A numpy ``ndarray`` of dimension :math:`(N+1)` containing the corresponding times :math:`t` (seconds)
        """

        u_fun = ensure_policy(u_fun)
        tt = np.linspace(t0, tF, N_steps+1)   # Time grid t_k = tt[k] between t0 and tF.
        xs = [ np.asarray(x0) ]
        us = [ u_fun(x0, t0 )]
        for k in range(N_steps):
            Delta = tt[k+1] - tt[k]
            tn = tt[k]
            xn = xs[k]
            un = us[k]   # ensure the action u is a vector.
            unp = u_fun(xn, tn + Delta)
            if method == 'rk4':
                """ Implementation of RK4 here. See: (Her24, Algorithm 18) """
                k1 = np.asarray(self.f(xn, un, tn))
                k2 = np.asarray(self.f(xn + Delta * k1/2, u_fun(xn, tn+Delta/2), tn+Delta/2))
                k3 = np.asarray(self.f(xn + Delta * k2/2, u_fun(xn, tn+Delta/2), tn+Delta/2))
                k4 = np.asarray(self.f(xn + Delta * k3,   u_fun(xn, tn + Delta), tn+Delta))
                xnp = xn + 1/6 * Delta * (k1 + 2*k2 + 2*k3 + k4)
            elif method == 'euler':
                xnp = xn + Delta * np.asarray(self.f(xn, un, tn))
            else:
                raise Exception("Bad integration method", method)
            xs.append(xnp)
            us.append(unp)
        xs = np.stack(xs, axis=0)
        us = np.stack(us, axis=0)
        return xs, us, tt 

    @property
    def state_size(self):
        """
        This field represents the dimensionality of the state-vector :math:`n`. Use it as ``model.state_size``
        :return: Dimensionality of the state vector :math:`x`
        """
        return self.get_cost().state_size
        # return len(list(self.bounds['x_low']))

    @property
    def action_size(self):
        """
        This field represents the dimensionality of the action-vector :math:`d`. Use it as ``model.action_size``
        :return: Dimensionality of the action vector :math:`u`
        """
        return self.get_cost().action_size
        # return len(list(self.bounds['u_low']))

    def render(self, x, render_mode="human"):
        """
        Responsible for rendering the state. You don't have to worry about this function.

        :param x: State to render
        :param str render_mode: Rendering mode. Select ``"human"`` for a visualization.
        :return:  Either none or a ``ndarray`` for plotting.
        """
        raise NotImplementedError()

    def close(self):
        pass

    def phi_x(self, x : list) -> list:
        r"""Coordinate transformation of the state when the model is discretized.

        This function specifies the coordinate transformation :math:`x_k = \Phi_x(x(t_k))` which is applied to the environment when it is
        discretized. It should accept a list of symbols, corresponding to :math:`x`, and return a new list
        of symbols corresponding to the (discrete) coordinates.

        :param x: A list of symbols ``[x0, x1, ..., xn]`` corresponding to :math:`\mathbf{x}(t)`
        :return: A new list of symbols corresponding to the discrete coordinates :math:`\mathbf{x}_k`.
        """
        return x

    def phi_x_inv(self, x: list) -> list:
        r"""Inverse of coordinate transformation for the state.

        This function should specify the inverse of the coordinate transformation :math:`\Phi_x`, i.e. :math:`\Phi_x^{-1}`.
        In other words, it has to map from the discrete coordinates to the continuous-time coordinates: :math:`x(t) = \Phi_x^{-1}(x_k)`.

        :param x: A list of symbols ``[x0, x1, ..., xn]`` corresponding to :math:`\mathbf{x}_k`
        :return: A new list of symbols corresponding to the continuous-time coordinates :math:`\mathbf{x}(t)`.
        """
        return x

    def phi_u(self, u: list)  -> list:
        r"""Coordinate transformation of the action when the model is discretized.

        This function specifies the coordinate transformation :math:`x_k = \Phi_x(x(t_k))` which is applied to the environment when it is
        discretized. It should accept a list of symbols, corresponding to :math:`x`, and return a new list
        of symbols corresponding to the (discrete) coordinates.

        :param x: A list of symbols ``[x0, x1, ..., xn]`` corresponding to :math:`\mathbf{x}(t)`
        :return: A new list of symbols corresponding to the discrete coordinates :math:`\mathbf{x}_k`.
        """
        return u

    def phi_u_inv(self, u: list)  -> list:
        r"""Inverse of coordinate transformation for the action.

        This function should specify the inverse of the coordinate transformation :math:`\Phi_u`, i.e. :math:`\Phi_u^{-1}`.
        In other words, it has to map from the discrete coordinates to the continuous-time coordinates: :math:`u(t) = \Phi_u^{-1}(u_k)`.

        :param x: A list of symbols ``[u0, u1, ..., ud]`` corresponding to :math:`\mathbf{u}_k`
        :return: A new list of symbols corresponding to the continuous-time coordinates :math:`\mathbf{u}(t)`.
        """
        return u

    def __str__(self):
        """
        Return a string representation of the model. This is a potentially helpful way to summarize the content of the
        model. You can use it as:

        .. runblock:: pycon

            >>> from irlc.ex04.model_pendulum import SinCosPendulumModel
            >>> model = SinCosPendulumModel()
            >>> print(model)

        :return: A string containing the details of the model.
        """
        split = "-"*20
        s = [f"{self.__class__}"] + ['='*50]
        s += ["Dynamics:", split]
        t = sym.symbols("t")
        x = sym.symbols(f"x0:{self.state_size}")
        u = sym.symbols(f"u0:{self.action_size}")

        s += [typeset_eq(x, u, self.sym_f(x, u, t) )]

        s += ["Cost:", split, str(self.get_cost())]

        dd = defaultdict(list)
        bounds = [ ('x', self.x_bound()), ('x0', self.x0_bound()), ('xF', self.xF_bound()),
                   ('u', self.u_bound()),
                   ('t0', self.t0_bound()), ('tF', self.tF_bound())]

        for v, box in bounds:
            if (box.low == -np.inf).all() and (box.high == np.inf).all():
                continue
            dd['low'].append(box.low_repr)
            dd['variable'].append("<= " + v + " <=")
            dd['high'].append(box.high_repr)

        if len(dd) > 0:
            s += ["Bounds:", split]
            s += [tabulate.tabulate(dd, headers='keys')]
        else:
            s += ['No bounds are applied to the x and u-variables.']
        return "\n".join(s)


def symv(s, n):
    """
    Returns a vector of symbolic functions. For instance if s='x' and n=3 then it will return
    [x0,x1,x2]
    where x0,..,x2 are symbolic variables.
    """
    return sym.symbols(" ".join(["%s%i," % (s, i) for i in range(n)]))

def ensure_policy(u):
    """
    Ensure u corresponds to a policy function with input arguments u(x, t)
    """
    if callable(u):
        return lambda x, t: np.asarray(u(x,t)).reshape((-1,))
    else:
        return lambda x, t: np.asarray(u).reshape((-1,))

def plot_trajectory(x_res, tt, lt='k-', ax=None, labels=None, legend=None):
    M = x_res.shape[1]
    if labels is None:
        labels = [f"x_{i}" for i in range(M)]

    if ax is None:
        if M == 2:
            a = 234
        if M == 3:
            r = 1
            c = 3
        else:
            r = 2 if M > 1 else 1
            c = (M + 1) // 2

        H = 2*r if r > 1 else 3
        W = 6*c
        # if M == 2:
        #     W = 12
        f, ax = plt.subplots(r,c, figsize=(W,H))
        if M == 1:
            ax = np.asarray([ax])
        print(M,r,c)

    for i in range(M):
        if len(ax) <= i:
            print("issue!")

        a = ax.flat[i]
        a.plot(tt, x_res[:, i], lt, label=legend)

        a.set_xlabel("Time/seconds")
        a.set_ylabel(labels[i])
        # a.set_title(labels[i])
        a.grid(True)
        if legend is not None and i == 0:
            a.legend()
        # if i == M:
    plt.tight_layout()
    return ax

def make_space_above(axes, topmargin=1.0):
    """ increase figure size to make topmargin (in inches) space for
        titles, without changing the axes sizes"""
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def typeset_eq(x, u, f):
    def ascii_vector(ls):
        ml = max(map(len, ls))
        ls = [" " * (ml - len(s)) + s for s in ls]
        ls = ["[" + s + "]" for s in ls]
        return "\n".join(ls)

    v = [str(z) for z in f]

    def cstack(ls: list):
        # ls = [l.splitlines() for l in ls]
        height = max([len(l) for l in ls])
        widths = [len(l[0]) for l in ls]

        for k in range(len(ls)):
            missing2 = (height - len(ls[k])) // 2
            missing1 = (height - len(ls[k]) - missing2)
            tpad = [" " * widths[k]] * missing1
            bpad = [" " * widths[k]] * missing2
            ls[k] = tpad + ls[k] + bpad

        r = [""] * len(ls[0])
        for w in range(len(ls)):
            for h in range(len(ls[0])):
                r[h] += ls[w][h]

        return r

    xx = [str(x) for x in x]
    uu = [str(u) for u in u]
    xx = ascii_vector(xx).splitlines()
    uu = ascii_vector(uu).splitlines()
    cm = cstack([xx, [", "], uu])
    eq = cstack([["f("], cm, [")"]])
    eq = cstack([["  "], eq, [" = "], ascii_vector(v).splitlines()])
    return "\n".join(eq)
