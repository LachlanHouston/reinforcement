# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import numpy as np

class DPModel: 
    r""" The Dynamical Programming model class

    The purpose of this class is to translate a dynamical programming problem, defined by the equations,

    .. math::

        x_{k+1}              & = f_k(x_k, u_k, w_k) \\
        \text{cost}          & = g_k(x_k, u_k, w_k) \\
        \text{terminal cost} & = g_N(x_N) \\
        \text{Noise disturbances:} \quad w_k & \sim P_W(w_k | x_k, u_k) \\
        \text{State/action spaces:} \quad & \mathcal A_k(x_k), \mathcal S_k

    into a single python object which we can then use for planning.

    .. Note::

        This is the first time many of you encounter a class. If so, you might wonder why you can't just implement
        the functions as usual, i.e. ``def f(x, k, ...):``, ``def g(x, k, ...):``,
        as regular python function and just let that be it?

        The reason is that we want to pass all these function (which taken together represents a planning problem)
        to planning methods such as the DP-algorithm (see the function :func:`~irlc.ex02.dp.DP_stochastic`)
        all at once.
        It is not very convenient to pass the functions one at a time -- instead we collect them into a class and simply call the function as

        >>> from irlc.ex02.inventory import InventoryDPModel
        >>> from irlc.ex02.dp import DP_stochastic
        >>> model = InventoryDPModel()      # Intialize the model
        >>> J, pi = DP_stochastic(model)    # All functions are passed to DP_stochastic

        This completes the note.

    To actually use the model, you need to extend it and implement the methods. The basic recipe for this is something like::

        class MyDPModel(DPModel):
            def f(self, x, u, w, k): # Note the `self`-variable. You can use it to access class variables such as`self.N`.
                return x + u - w     # Just an example
            def S(self, k):
                return [0, 1, 2]    # State space S_k = {0, 1, 2}
                # Implement the other functions A, g, gN and Pw here.


    You should take a look at :func:`~irlc.ex02.inventory.InventoryDPModel` for a concrete example.
    Once the functions have been implemented, you can call them as:

    .. runblock:: pycon

        >>> from irlc.ex02.inventory import InventoryDPModel
        >>> model = InventoryDPModel(N=5) # Plan on a horizon of 5
        >>> print("State space S_2", model.S(2))
        >>> model.f(x=1, u=2, w=1, k=0)   # Just an example. You don't have to use named arguments, although it helps on readability.
        >>> model.A(1, k=2) # Action space A_1(2), i.e. the actions available at time step k=1 in state 2.

    """
    def __init__(self, N):
        """
        Called when the DP Model is initialized. By default, it simply stores the planning horizon ``N``

        :param N: The planning horizon in the DP problem :math:`N`
        """
        self.N = N  # Store the planning horizon.

    def f(self, x, u, w, k: int):
        """
        Implements the transition function :math:`x_{k+1} = f_k(x, u, w)` and returns the next state :math:`x_{k+1}`

        :param x: The state :math:`x_k`
        :param u: The action taken :math:`u_k`
        :param w: The random noise disturbance :math:`w_k`
        :param k: The current time step :math:`k`
        :return: The state the environment (deterministically) transitions to, i.e. :math:`x_{k+1}`
        """
        raise NotImplementedError("Return f_k(x,u,w)")

    def g(self, x, u, w, k: int) -> float:
        """
        Implements the cost function :math:`c = g_k(x, u, w)` and returns the cost :math:`c`

        :param x: The state :math:`x_k`
        :param u: The action taken :math:`u_k`
        :param w: The random noise disturbance :math:`w_k`
        :param k: The current time step :math:`k`
        :return: The cost (as a ``float``) incurred by the environment, i.e. :math:`g_k(x, u, w)`
        """
        raise NotImplementedError("Return g_k(x,u,w)")

    def gN(self, x) -> float:
        """
        Implements the terminal cost function :math:`c = g_N(x)` and returns the terminal cost :math:`c`.

        :param x: A state seen at the last time step :math:`x_N`
        :return: The terminal cost (as a ``float``) incurred by the environment, i.e. :math:`g_N(x)`
        """
        raise NotImplementedError("Return g_N(x)")

    def S(self, k: int):
        """
        Computes the state space :math:`\mathcal S_k` at time step :math:`k`.
        In other words, this function returns a set of all states the system can possibly be in at time step :math:`k`.

        .. Note::
            I think the cleanest implementation is one where this function returns a python ``set``. However, it won't matter
            if the function returns a ``list`` or ``tuple`` instead.

        :param k: The current time step :math:`k`
        :return: The state space (as a ``list`` or ``set``) available at time step ``k``, i.e. :math:`\mathcal S_k`
        """
        raise NotImplementedError("Return state space as set S_k = {x_1, x_2, ...}")

    def A(self, x, k: int):
        """
        Computes the action space :math:`\mathcal A_k(x)` at time step :math:`k` in state `x`.

        In other words, this function returns a ``set`` of all actions the agent can take in time step :math:`k`.

        .. Note::
            An example where the actions depend on the state is chess (in this case, the state is board position, and the actions are the legal moves)

        :param k: The current time step :math:`k`
        :param x: The state we want to compute the actions in :math:`x_k`
        :return: The action space (as a ``list`` or ``set``) available at time step ``k``, i.e. :math:`\mathcal A_k(x_k)`
        """
        raise NotImplementedError("Return action space as set A(x_k) = {u_1, u_2, ...}")

    def Pw(self, x, u, k: int):
        """
        Returns the random noise disturbances and their probability. In other words, this function implements the distribution:

        .. math::

            P_k(w_k | x_k, u_k)

        To implement this distribution, we must keep track of both the possible values of the noise disturbances :math:`w_k`
        as well as the (numerical) value of their probability :math:`p(w_k| ...)`.

        To do this, the function returns a dictionary of the form ``P = {w1: p_w1, w2: p_w2, ...}`` where

          - The keys ``w`` represents random noise disturbances
          - the values ``P[w]`` represents their probability (i.e. a ``float``)

        This can hopefully be made more clear with the Inventory environment:

        .. runblock:: pycon

            >>> from irlc.ex02.inventory import InventoryDPModel
            >>> model = InventoryDPModel(N=5) # Plan on a horizon of 5
            >>> print("Random noise disturbances in state x=1 using action u=0 is:", model.Pw(x=1, u=0, k=0))
            >>> for w, pw in model.Pw(x=1, u=0, k=0).items(): # Iterate and print:
            ...     print(f"p_k({w}|x, u) =", pw)


        :param x: The state :math:`x_k`
        :param u: The action taken :math:`u_k`
        :param k: The current time step :math:`k`
        :return: A dictionary representing the distribution of random noise disturbances :math:`P_k(w |x_k, u_k)` of the form  ``{..., w_i: pw_i, ...}``  such that  ``pw_i = P_k(w_i | x, u)``
        """
        # Compute and return the random noise disturbances here.
        # As an example:
        return {'w_dummy': 1/3, 42: 2/3}  # P(w_k="w_dummy") = 1/3, P(w_k =42)=2/3. 

    def w_rnd(self, x, u, k): 
        """
        This helper function computes generates a random noise disturbance using the function :func:`ex02.dp_model.DPModel.Pw`, i.e. it returns a sample:

        .. math::
            w \sim P_k(x_k, u_k)

        This will be useful for simulating the model.

        .. Note::
            You don't have to implement or change this function.

        :param x: The state :math:`x_k`
        :param u: The action taken :math:`u_k`
        :param k: The current time step :math:`k`
        :return: A random noise disturbance :math:`w` distributed as :math:`P_k(x_k, u_k)`
        """
        pW = self.Pw(x, u, k)
        w, pw = zip(*pW.items())  # seperate w and p(w)
        return np.random.choice(a=w, p=pw) 
