# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
This project resembles the Inventory-control problem discussed in (Her24, Subsection 5.1.2) but with more complicated rules.
If you are stuck, the inventory-control problem will be a good place to start.

I recommend to use the DP_stochastic function (as we did with the inventory-control example). This means
your main problem is to build appropriate DPModel-classes to represent the different problems.
References:
  [Her24] Tue Herlau. Sequential decision making. (Freely available online), 2024.
"""
from irlc.ex02.dp_model import DPModel
from irlc.ex02.dp import DP_stochastic
import matplotlib.pyplot as plt
from scipy.stats import binom
from irlc import savepdf
import numpy as np

def plot_policy(pi, title, pdf):
    """ Helper function to plot the policy functions pi, as generated by the DP_stochastic function. This function
    can be used to visualize which actions are taken in which state (y-axis) at which time step (x-axis). """
    N = len(pi)
    W = max(pi[0].keys())
    A = np.zeros((W, N))
    for i in range(W):
        for j in range(N):
            A[i, j] = pi[j][i]
    plt.imshow(A)
    plt.title(title)
    savepdf(pdf)
    plt.show()

# TODO: 51 lines missing.
class KioskDPModel1(DPModel):
    def __init__(self, N = 14):
        super().__init__(N=N)
        
    def A(self, x, k):
        return list(range(16))
    
    def S(self, k):
        return list(range(21))
    
    def g(self, x, u, w, k):
        return 1.5*u - min(x + u, w)*2.1
    
    def f(self, x, u, w, k):
        return max(0, min(20, x - w + u))
    
    def Pw(self, x, u, k):
        return {0: 0.3, 3: 0.6, 6: 0.1}
    
    def gN(self, x):
        return 0


class KioskDPModel2(KioskDPModel1):
    def Pw(self, x, u, k):
        return {w: binom.pmf(w, 20, 1/5) for w in range(21)}
    
    def g(self, x, u, w, k):
        dispose = 0
        if x > 20:
            dispose = 3*(x -20)
        else:
            dispose = 0
        return 1.5*u - min(x + u, w)*2.1 + dispose*3
    
    
def warmup_states():
    # TODO: 1 lines missing.
    return KioskDPModel1().S(0)
    
def warmup_actions():
    # TODO: 1 lines missing.
    return KioskDPModel1().A(None, 0)


def solve_kiosk_1(): 
    # TODO: 1 lines missing.
    return DP_stochastic(KioskDPModel1())

def solve_kiosk_2(): 
    # TODO: 1 lines missing.
    return DP_stochastic(KioskDPModel2())
    
    
def main():
    # Problem 15
    print("Available states S_0:", warmup_states())
    print("Available actions A_0(x_0):", warmup_actions())

    J, pi = solve_kiosk_1() # Problem 16
    print("Kiosk1: Expected profits: ", -J[0][0], " imperial credits")
    plot_policy(pi, "Kiosk1", "Latex/figures/kiosk1")
    plt.show()

    J, pi = solve_kiosk_2() # Problem 17
    print("Kiosk 2: Expected profits: ", -J[0][0], " imperial credits")
    plot_policy(pi, "Kiosk2", "Latex/figures/kiosk2")
    plt.show()


if __name__ == "__main__":
    main()