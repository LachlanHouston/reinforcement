# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
from irlc.ex02.graph_traversal import SmallGraphDP
from irlc.ex02.dp_model import DPModel

def DP_stochastic(model: DPModel): 
    """
    Implement the stochastic DP algorithm. The implementation follows (Her24, Algorithm 1).
    Once you are done, you should be able to call the function as:

    .. runblock:: pycon

        >>> from irlc.ex02.graph_traversal import SmallGraphDP
        >>> model = SmallGraphDP(t=5)  # Instantiate the small graph with target node 5
        >>> J, pi = DP_stochastic(model)
        >>> print(pi[0][2]) # Action taken in state ``x=2`` at time step ``k=0``.

    :param model: An instance of :class:`irlc.ex01.dp_model.DPModel` class. This represents the problem we wish to solve.
    :return:
        - ``J`` - A list of of cost function so that ``J[k][x]`` represents :math:`J_k(x)`
        - ``pi`` - A list of dictionaries so that ``pi[k][x]`` represents :math:`\mu_k(x)`
    """

    """ 
    In case you run into problems, I recommend following the hints in (Her24, Subsection 6.2.1) and focus on the
    case without a noise term; once it works, you can add the w-terms. When you don't loop over noise terms, just specify
    them as w = None in env.f and env.g.
    """
    N = model.N
    J = [{} for _ in range(N + 1)]
    pi = [{} for _ in range(N)]
    J[N] = {x: model.gN(x) for x in model.S(model.N)}
    for k in range(N-1, -1, -1):
        for x in model.S(k):
            """
            Update pi[k][x] and Jstar[k][x] using the general DP algorithm given in (Her24, Algorithm 1).
            If you implement it using the pseudo-code, I recommend you define Q (from the algorithm) as a dictionary like the J-function such that
                        
            > Q[u] = Q_u (for all u in model.A(x,k))
            
            Then you find the u with the lowest value of Q_u, i.e. 
            
            > umin = arg_min_u Q[u]
            
            (for help, google: `python find key in dictionary with minimum value').
            Then you can use this to update J[k][x] = Q_umin and pi[k][x] = umin.
            """
            # TODO: 4 lines missing.
            raise NotImplementedError("Insert your solution and remove this error.")
            """
            After the above update it should be the case that:

            J[k][x] = J_k(x)
            pi[k][x] = pi_k(x)
            """
    return J, pi 


if __name__ == "__main__":  # Test dp on small graph given in (Her24, Subsection 6.2.1)
    print("Testing the deterministic DP algorithm on the small graph environment")
    model = SmallGraphDP(t=5)  # Instantiate the small graph with target node 5 
    J, pi = DP_stochastic(model)
    # Print all optimal cost functions J_k(x_k) 
    for k in range(len(J)):
        print(", ".join([f"J_{k}({i}) = {v:.1f}" for i, v in J[k].items()]))
    print(f"Cost of shortest path when starting in node 2 is: {J[0][2]=} (and should be 4.5)") 
