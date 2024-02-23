# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
import sympy as sym
import numpy as np


def mat(x): # Helper function.
    return sym.Matrix(x) if x is not None else x


class SymbolicQRCost:
    """
    This class represents the cost function for a continuous-time model. In the simulations, we are going to assume
    that the cost function takes the form:

    .. math::
        \int_{t_0}^{t_F} c(x(t), u(t)) dt + c_F(x_F)

    And this class will specifically implement the two functions :math:`c` and :math:`c_F`. They will be assumed to have the quadratic form:

    .. math::
        c(x, u) & = \\frac{1}{2} x^T Q x + \\frac{1}{2} u^T R u + u^T H x + q^T x + r^T u + q_0, \\\\
        c_F(x_F) & = \\frac{1}{2} x_F^T Q_F x_F + q_F^T x_F + q_{0,F}.

    So what all of this boils down to is that the class just need to store a bunch of matrices and vectors.

    You can add and scale cost-functions
    **********************************************************

    A slightly smart thing about the cost functions are that you can add and scale them. The following provides an
    example:

    .. runblock:: pycon

        >>> from irlc.ex03.control_cost import SymbolicQRCost
        >>> import numpy as np
        >>> cost1 = SymbolicQRCost(np.eye(2), np.zeros(1) ) # Set Q = I, R = 0
        >>> cost2 = SymbolicQRCost(np.ones((2,2)), np.zeros(1) ) # Set Q = 2x2 matrices of 1's, R = 0
        >>> print(cost1.Q) # Will be the identity matrix.
        >>> cost = cost1 * 3 +  cost2 * 2
        >>> print(cost.Q) # Will be 3 x I + 2

    """

    def __init__(self, Q, R, q=None, qc=None, r=None, H=None, QN=None, qN=None, qcN=None):
        """
        The constructor can be used to manually create a cost function. You will rarely want to call the constructor
        directly but instead use the helper methods (see class documentation).
        What the class basically does is that it stores the input parameters as fields. In other words, you can access the quadratic
        term of the cost function, :math:`\\frac{1}{2}x^T Q x`, as ``cost.Q``.

        :param Q: The matrix :math:`Q`
        :param R: The matrix :math:`R`
        :param q: The vector :math:`q`
        :param qc: The constant :math:`q_0`
        :param r: The vector :math:`r`
        :param H: The matrix :math:`H`
        :param QN: The terminal cost matrix :math:`Q_N`
        :param qN: The terminal cost vector :math:`q_N`
        :param qcN: The terminal cost constant :math:`q_{0,N}`
        """

        n = Q.shape[0]
        d = R.shape[0]
        self.Q = Q
        self.R = R
        self.q = np.zeros( (n,)) if q is None else q
        self.qc = 0 if qc == None else qc
        self.r = np.zeros( (d,)) if r is None else r
        self.H = np.zeros((d,n)) if H is None else H
        self.QN = np.zeros((n,n)) if QN is None else QN
        self.qN = np.zeros((n,)) if qN is None else qN
        self.qcN = 0 if qcN == None else qcN
        self.flds = ('Q', 'R', 'q', 'qc', 'r', 'H', 'QN', 'qN', 'qcN')
        self.flds_term = ('QN', 'qN', 'qcN')

        self.c_numpy = None
        self.cF_numpy = None


    @classmethod
    def zero(cls, state_size, action_size):
        """
        Creates an all-zero cost function, i.e. all terms :math:`Q`, :math:`R` are set to zer0.

        .. runblock:: pycon

            >>> from irlc.ex03.control_cost import SymbolicQRCost
            >>> cost = SymbolicQRCost.zero(2, 1)
            >>> cost.Q # 2x2 zero matrix
            >>> cost.R # 1x1 zero matrix.

        :param state_size: Dimension of the state vector :math:`n`
        :param action_size: Dimension of the action vector :math:`d`
        :return: A ``SymbolicQRCost`` with all zero terms.
        """

        return cls(Q=np.zeros( (state_size,state_size)), R=np.zeros((action_size,action_size)) )


    def sym_c(self, x, u, t=None):
        """
        Evaluate the (instantaneous) part of the function :math:`c(x,u, t)`. An example:

        .. runblock:: pycon

            >>> from irlc.ex03.control_cost import SymbolicQRCost
            >>> import numpy as np
            >>> cost = SymbolicQRCost(np.eye(2), np.eye(1)) # Set Q = I, R = 0
            >>> cost.sym_c(x = np.asarray([1,2]), u=np.asarray([0])) # should return 0.5 * x^T Q x = 0.5 * (1 + 4)

        :param x: The state :math:`x(t)`
        :param u: The action :math:`u(t)`
        :param t: The current time step :math:`t` (this will be ignored)
        :return: A ``sympy`` symbolic expression corresponding to the instantaneous cost.
        """
        u = sym.Matrix(u)
        x = sym.Matrix(x)
        c =  1 / 2 * (x.transpose() @ self.Q @ x) + 1 / 2 * (u.transpose() @ self.R @ u) + u.transpose() @ self.H @ x + sym.Matrix(self.q).transpose() @ x + sym.Matrix(self.r).transpose() @ u + sym.Matrix([[self.qc]])
        assert c.shape == (1,1)
        return c[0,0]


    def sym_cf(self, t0, tF, x0, xF):
        """
        Evaluate the terminal (constant) term in the cost function :math:`c_F(t_0, t_F, x_0, x_F)`. An example:

        .. runblock:: pycon

            >>> from irlc.ex03.control_cost import SymbolicQRCost
            >>> import numpy as np
            >>> cost = SymbolicQRCost(np.eye(2), np.zeros(1), QN=np.eye(2)) # Set Q = I, R = 0
            >>> cost.sym_cf(0, 0, 0, xF=2*np.ones((2,))) # should return 0.5 * xF^T * xF = 0.5 * 8

        :param t0: Starting time :math:`t_0` (not used)
        :param tF: Stopping time :math:`t_F` (not used)
        :param x0: Initial state :math:`x_0` (not used)
        :param xF: Termi lanstate :math:`x_F` (**this one is used**)
        :return: A ``sympy`` symbolic expression corresponding to the terminal cost.
        """
        xF = sym.Matrix(xF)
        c = 0.5 * xF.transpose() @ self.QN @ xF + xF.transpose() @ sym.Matrix(self.qN) + sym.Matrix([[self.qcN]])
        assert c.shape == (1,1)
        return c[0,0]

    def discretize(self, dt):
        """
        Discretize the cost function so it is suitable for a discrete control problem. See (Her24, Subsection 13.1.5) for more information.

        :param dt: The discretization time step :math:`\Delta`
        :return: An :class:`~irlc.ex04.cost_discrete.DiscreteQRCost` instance corresponding to a discretized version of this cost function.
        """
        from irlc.ex04.discrete_control_cost import DiscreteQRCost
        return DiscreteQRCost(**{f: self.__getattribute__(f) * (1 if f in self.flds_term else dt) for f in self.flds} )


    def __add__(self, c):
        return SymbolicQRCost(**{k: self.__dict__[k] + c.__dict__[k] for k in self.flds})

    def __mul__(self, c):
        return SymbolicQRCost(**{k: self.__dict__[k] * c for k in self.flds})

    def __str__(self):
        title = "Continuous-time cost function"
        label1 = "Non-zero terms in c(x, u)"
        label2 = "Non-zero terms in c_F(x)"
        terms1 = [s for s in self.flds if s not in self.flds_term]
        terms2 = self.flds_term
        return _repr_cost(self, title, label1, label2, terms1, terms2)

    def goal_seeking_terminal_cost(self, xF_target, QF=None):
        """
        Create a cost function which is minimal when the terminal state :math:`x_F` is equal to a goal state :math:`x_F^*`.
        Concretely, it will return a cost function of the form

        .. math::
            c_F(x_F) = \\frac{1}{2} (x_F^* - x_F)^\\top Q_F  (x_F^* - x_F)

        .. runblock:: pycon

            >>> from irlc.ex03.control_cost import SymbolicQRCost
            >>> import numpy as np
            >>> cost = SymbolicQRCost.zero(2, 1)
            >>> cost += cost.goal_seeking_terminal_cost(xF_target=np.ones((2,)))
            >>> print(cost.qN)
            >>> print(cost)

        :param xF_target: Target state :math:`x_F^*`
        :param QF: Cost matrix :math:`Q_F`
        :return: A ``SymbolicQRCost`` object corresponding to the goal-seeking cost function
        """
        if QF is None:
            QF = np.eye(xF_target.size)
        QF, qN, qcN = targ2matrices(xF_target, Q=QF)
        return SymbolicQRCost(Q=self.Q*0, R=self.R*0, QN=QF, qN=qN, qcN=qcN)

    def goal_seeking_cost(self, x_target, Q=None):
        """
        Create a cost function which is minimal when the state :math:`x` is equal to a goal state :math:`x^*`.
        Concretely, it will return a cost function of the form

        .. math::
            c(x, u) = \\frac{1}{2} (x^* - x)^\\top Q  (x^* - x)

        .. runblock:: pycon

            >>> from irlc.ex03.control_cost import SymbolicQRCost
            >>> import numpy as np
            >>> cost = SymbolicQRCost.zero(2, 1)
            >>> cost += cost.goal_seeking_cost(x_target=np.ones((2,)))
            >>> print(cost.q)
            >>> print(cost)

        :param x_target: Target state :math:`x^*`
        :param Q: Cost matrix :math:`Q`
        :return: A ``SymbolicQRCost`` object corresponding to the goal-seeking cost function
        """
        if Q is None:
            Q = np.eye(x_target.size)
        Q, q, qc = targ2matrices(x_target, Q=Q)
        return SymbolicQRCost(Q=Q, R=self.R*0, q=q, qc=qc)

    def term(self, Q=None, R=None,r=None):
        dd = {}
        lc = locals()
        for f in self.flds:
            if f in lc and lc[f] is not None:
                dd[f] = lc[f]
            else:
                dd[f] = self.__getattribute__(f)*0
        return SymbolicQRCost(**dd)

    @property
    def state_size(self):
        return self.Q.shape[0]

    @property
    def action_size(self):
        return self.R.shape[0]



def _repr_cost(cost, title, label1, label2, terms1, terms2):
    self = cost
    def _get(flds, label):
        d = {s: self.__dict__[s] for s in flds if np.sum(np.sum(self.__dict__[s] != 0)) != 0}
        out = ""
        if len(d) > 0:
            # out = ""
            out += f"> {label}:\n"
            for s, m in d.items():
                mm = f"{m}"
                if len(mm.splitlines()) > 1:
                    mm = "\n" + mm
                out += f" * {s} = {mm}\n"

        return d, out

    nz_c, o1 = _get([s for s in terms1], label1)
    out = ""
    out += f"{title}:\n"
    out += o1
    nz_term, o2 = _get(terms2, label2)
    out += o2
    if len(nz_c) + len(nz_term) == 0:
        print("All terms in the cost-function are zero.")
    return out


def targ2matrices(t, Q=None): # Helper function
    """
    Given a target vector :math:`t` and a matrix :math:`Q` this function returns cost-matrices suitable for implementing:

    .. math::
        \\frac{1}{2} * (x - t)^Q (x - t) = \\frac{1}{2} * x^T Q x + 1/2 * t^T * t - x * t

    :param t:
    :param Q:
    :return:
    """
    n = t.size
    if Q is None:
        Q = np.eye(n)

    return Q, -1/2 * (Q @ t + t @ Q.T), 1/2 * t @ Q @ t
