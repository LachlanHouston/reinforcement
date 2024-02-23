# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
Quadratic cost functions
"""
import numpy as np
from irlc.ex03.control_cost import targ2matrices

def nz(X,a,b=None):
    return np.zeros((a,) if b is None else (a,b)) if X is None else X

class DiscreteQRCost: #(DiscreteCost):
    """
    This class represents the cost function for a discrete-time model. In the simulations, we are going to assume
    that the cost function takes the form:

    .. math::
        \sum_{k=0}^{N-1} c_k(x_k, u_k) + c_N(x_N)


    And this class will specifically implement the two functions :math:`c` and :math:`c_N`.
    They will be assumed to have the quadratic form:

    .. math::
        c_k(x_k, u_k) & = \\frac{1}{2} x_k^T Q x_k + \\frac{1}{2} u_k^T R u_k + u^T_k H x_k + q^T x_k + r^T u_k + q_0, \\\\
        c_N(x_N) & = \\frac{1}{2} x_N^T Q_N x_N + q_N^T x_N + q_{0,N}.

    So what all of this boils down to is that the class just need to store a bunch of matrices and vectors.

    You can add and scale cost-functions
    **********************************************************

    A slightly smart thing about the cost functions are that you can add and scale them. The following provides an
    example:

    .. runblock:: pycon

        >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
        >>> import numpy as np
        >>> cost1 = DiscreteQRCost(np.eye(2), np.zeros(1) ) # Set Q = I, R = 0
        >>> cost2 = DiscreteQRCost(np.ones((2,2)), np.zeros(1) ) # Set Q = 2x2 matrices of 1's, R = 0
        >>> print(cost1.Q) # Will be the identity matrix.
        >>> cost = cost1 * 3 +  cost2 * 2
        >>> print(cost.Q) # Will be 3 x I + 2

    """
    def __init__(self, Q, R, H=None,q=None,r=None,qc=0, QN=None, qN=None,qcN=0):
        n, d = Q.shape[0], R.shape[0]
        self.QN, self.qN = nz(QN,n,n), nz(qN,n)
        self.Q, self.q = nz(Q, n, n), nz(q, n)
        self.R, self.H, self.r = nz(R, d, d), nz(H, d, n), nz(r, d)
        self.qc, self.qcN = qc, qcN
        self.flds_term = ['QN', 'qN', 'qcN']
        self.flds = ['Q', 'q', 'R', 'H', 'r', 'qc'] + self.flds_term

    def c(self, x, u, k=None, compute_gradients=False):
        """
        Evaluate the (instantaneous) part of the function :math:`c_k(x_k,u_k)`. An example:

        .. runblock:: pycon

            >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
            >>> import numpy as np
            >>> cost = DiscreteQRCost(np.eye(2), np.eye(1)) # Set Q = I, R = 0
            >>> cost.c(x = np.asarray([1,2]), u=np.asarray([0]), compute_gradients=False) # should return 0.5 * x^T Q x = 0.5 * (1 + 4)

        The function can also return the derivates of the cost function if ``compute_derivates=True``

        :param x: The state :math:`x_k`
        :param u: The action :math:`u_k`
        :param k: The time step :math:`k` (this will be ignored)
        :param compute_gradients: if ``True`` the function will compute gradients and Hessians.
        :return:
            - ``c`` - The cost as a ``float``
            - ``c_x`` - The derivative with respect to :math:`x`
        """
        c = 1/2 * (x @ self.Q @ x) + 1/2 * (u @ self.R @ u) + u @ self.H @ x + self.q @ x + self.r @ u + self.qc
        c_x = 1/2 * (self.Q + self.Q.T) @ x + self.q
        c_u = 1 / 2 * (self.R + self.R.T) @ u + self.r
        c_ux = self.H
        c_xx = self.Q
        c_uu = self.R
        if compute_gradients:
            # this is useful for MPC when we apply an optimizer rather than LQR (iLQR)
            return c, c_x, c_u, c_xx, c_ux, c_uu
        else:
            return c

    def cN(self, x, compute_gradients=False):
        """
        Evaluate the terminal (constant) term in the cost function :math:`c_N(x_N)`. An example:

        .. runblock:: pycon

            >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
            >>> import numpy as np
            >>> cost = DiscreteQRCost(np.eye(2), np.zeros(1), QN=np.eye(2)) # Set Q = I, R = 0
            >>> c, Jx, Jxx = cost.cN(x=2*np.ones((2,)), compute_gradients=True)
            >>> c # should return 0.5 * x_N^T * x_N = 0.5 * 8

        :param x: Terminal state :math:`x_N`
        :param compute_gradients: if ``True`` the function will compute gradients and Hessians of the cost function.
        :return: The last (terminal) part of the cost-function :math:`c_N`
        """
        J = 1/2 * (x @ self.QN @ x) + self.qN @ x + self.qcN
        if compute_gradients:
            J_x = 1 / 2 * (self.QN + self.QN.T) @ x + self.qN
            return J, J_x, self.QN
        else:
            return J

    def __add__(self, c):
        return DiscreteQRCost(**{k: self.__dict__[k] + c.__dict__[k] for k in self.flds})

    def __mul__(self, c):
        return DiscreteQRCost(**{k: self.__dict__[k] * c for k in self.flds})

    def __str__(self):
        title = "Discrete-time cost function"
        label1 = "Non-zero terms in c_k(x_k, u_k)"
        label2 = "Non-zero terms in c_N(x_N)"
        terms1 = [s for s in self.flds if s not in self.flds_term]
        terms2 = self.flds_term
        from irlc.ex03.control_cost import _repr_cost
        return _repr_cost(self, title, label1, label2, terms1, terms2)

    @classmethod
    def zero(cls, state_size, action_size):
        """
        Creates an all-zero cost function, i.e. all terms :math:`Q`, :math:`R` are set to zero.

        .. runblock:: pycon

            >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
            >>> cost = DiscreteQRCost.zero(2, 1)
            >>> cost.Q # 2x2 zero matrix
            >>> cost.R # 1x1 zero matrix.

        :param state_size: Dimension of the state vector :math:`n`
        :param action_size: Dimension of the action vector :math:`d`
        :return: A ``DiscreteQRCost`` with all zero terms.
        """
        return cls(Q=np.zeros((state_size, state_size)), R=np.zeros((action_size, action_size)))

    def goal_seeking_terminal_cost(self, xN_target, QN=None):
        """
        Create a discrete cost function which is minimal when the final state :math:`x_N` is equal to a goal state :math:`x_N^*`.
        Concretely, it will return a cost function of the form

        .. math::
            c_N(x_N) = \\frac{1}{2} (x^*_N - x_N)^\\top Q  (x^*_N - x_N)

        .. runblock:: pycon

            >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
            >>> import numpy as np
            >>> cost = DiscreteQRCost.zero(2, 1)
            >>> cost += cost.goal_seeking_terminal_cost(xN_target=np.ones((2,)))
            >>> print(cost.qN)
            >>> print(cost)

        :param xN_target: Target state :math:`x_N^*`
        :param Q: Cost matrix :math:`Q`
        :return: A ``DiscreteQRCost`` object corresponding to the goal-seeking cost function
        """

        if QN is None:
            QN = np.eye(xN_target.size)
        QN, qN, qcN = targ2matrices(xN_target, Q=QN)
        return DiscreteQRCost(Q=QN*0, R=self.R*0, QN=QN, qN=qN, qcN=qcN)

    def goal_seeking_cost(self, x_target, Q=None):
        """
        Create a discrete cost function which is minimal when the state :math:`x_k` is equal to a goal state :math:`x_k^*`.
        Concretely, it will return a cost function of the form

        .. math::
            c_k(x_k, u_k) = \\frac{1}{2} (x^*_k - x_k)^\\top Q  (x^*_k - x_k)

        .. runblock:: pycon

            >>> from irlc.ex04.discrete_control_cost import DiscreteQRCost
            >>> import numpy as np
            >>> cost = DiscreteQRCost.zero(2, 1)
            >>> cost += cost.goal_seeking_cost(x_target=np.ones((2,)))
            >>> print(cost.q)
            >>> print(cost)

        :param x_target: Target state :math:`x_k^*`
        :param Q: Cost matrix :math:`Q`
        :return: A ``DiscreteQRCost`` object corresponding to the goal-seeking cost function
        """
        if Q is None:
            Q = np.eye(x_target.size)
        Q, q, qc = targ2matrices(x_target, Q=Q)
        return DiscreteQRCost(Q=Q, R=self.R*0, q=q, qc=qc)
