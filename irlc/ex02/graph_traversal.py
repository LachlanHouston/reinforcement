# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
import numpy as np
from irlc.ex02.dp_model import DPModel

"""
Graph of shortest path problem of (Her24, Subsection 5.1.1)
"""
G222 = {(1, 2): 6,  (1, 3): 5, (1, 4): 2, (1, 5): 2,  
        (2, 3): .5, (2, 4): 5, (2, 5): 7,
        (3, 4): 1,  (3, 5): 5, (4, 5): 3}  

def symG(G):
    """ make a graph symmetric. I.e. if it contains edge (a,b) with cost z add edge (b,a) with cost c """
    G.update({(b, a): l for (a, b), l in G.items()})
symG(G222)

class SmallGraphDP(DPModel):
    """ Implement the small-graph example in (Her24, Subsection 5.1.1). t is the terminal node. """
    def __init__(self, t, G=None):  
        self.G = G.copy() if G is not None else G222.copy()  
        self.G[(t,t)] = 0  # make target vertex absorbing  
        self.t = t         # target vertex in graph
        self.nodes = {node for edge in self.G for node in edge} # set of all nodes
        super(SmallGraphDP, self).__init__(N=len(self.nodes)-1)  

    def f(self, x, u, w, k):
        if (x,u) in self.G:  
            # TODO: 1 lines missing.
            raise NotImplementedError("Implement function body")
        else:
            raise Exception("Nodes are not connected")

    def g(self, x, u, w, k): 
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def gN(self, x):  
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

    def S(self, k):   
        return self.nodes

    def A(self, x, k):
        return {j for (i,j) in self.G if i == x} 

def main():
    t = 5  # target node
    model = SmallGraphDP(t=t)
    x0 = 1  # starting node
    k = 0
    w = 0 # irrelevant.
    u = 2 # as an example.
    print(f"{model.f(x0, u, w, k)=} (should be 2)")
    print(f"{model.g(x0, u, w, k)=} (should be 6)")
    print(f"{model.gN(x0)=} (should be np.inf)")
    print(f"{model.S(k)=}", "(should be {1, 2, 3, 4, 5})")
    print(f"{model.A(x0, k)=}", "(should be {2, 3, 4, 5})")
    print("Run the tests to check your implementation.")

if __name__ == '__main__':
    main()
