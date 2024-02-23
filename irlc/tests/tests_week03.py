# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
import irlc
from unitgrade import UTestCase
from irlc.ex03.kuramoto import KuramotoModel, f
import sympy as sym
import numpy as np

class Problem1Kuramoto(UTestCase):
    """ Test the Kuromoto Osscilator """
    def test_continious_model(self):
        cmodel = KuramotoModel()
        x, u = sym.symbols("x u")
        expr = cmodel.sym_f([x], [u])
        # Check the expression has the right type.
        self.assertIsInstance(expr, list)
        # Evaluate the expression and check the result in a given point.
        self.assertEqualC(expr[0].subs([(x, 0.2), (u, 0.93)]))

    def test_f(self):
        self.assertLinf(f([0.1], [0.4]), tol=1e-6)


    def test_RK4(self):
        from irlc.ex03.kuramoto import rk4_simulate

        cmodel = KuramotoModel()
        x0 = np.asarray(cmodel.x0_bound().low)  # Get the starting state x=0.
        u = 1.3
        xs, ts = rk4_simulate(x0, [u], t0=0, tF=20, N=100)

        # xs, us, ts = cmodel.simulate(x0, u_fun=u , t0=0, tF=20)
        self.assertLinf(ts, tol=1e-6)
        # self.assertLinf(us, tol=1e-6)
        self.assertLinf(xs, tol=1e-6)

        # Test the same with a varying function:
        xs, ts = rk4_simulate(x0, [u+1], t0=0, tF=10, N=50)
        # xs, us, ts = cmodel.simulate(x0, u_fun=lambda x,t: np.sin(x + u) , t0=0, tF=10)
        self.assertLinf(ts, tol=1e-6)
        # self.assertLinf(us, tol=1e-6)
        self.assertLinf(xs, tol=1e-6)

class Exam5InventoryEvaluation(UTestCase):
    def test_a_test_expected_items_next_day(self):
        from irlc.ex03.inventory_evaluation import a_expected_items_next_day
        self.assertAlmostEqual(a_expected_items_next_day(x=0, u=1), 0.1, places=5)

    def test_b_test_expected_items_next_day(self):
        from irlc.ex03.inventory_evaluation import b_evaluate_policy
        pi = self.get_pi()
        self.assertAlmostEqual(b_evaluate_policy(pi, 1), 2.7, places=5)

    def get_pi(self):
        from irlc.ex02.inventory import InventoryDPModel
        model = InventoryDPModel()
        pi = [{x: 1 if x == 0 else 0 for x in model.S(k)} for k in range(model.N)]
        return pi

class Exam6Toy2d(UTestCase):
    def test_rk4_a(self):
        from irlc.ex03.toy_2d_control import toy_simulation
        w = toy_simulation(u0=0.4, T=5)
        self.assertFalse(isinstance(w, np.ndarray), msg="Your toy_simulation function must return a float")
        self.assertEqual(type(float(w)), float, msg="Your toy_simulation function must return a float")
        self.assertLinf(w, tol=0.01, msg="Your simulation ended up at the wrong angle")

    def test_rk4_b(self):
        from irlc.ex03.toy_2d_control import toy_simulation
        w = toy_simulation(u0=-0.1, T=2)
        self.assertFalse( isinstance(w, np.ndarray), msg="Your toy_simulation function must return a float")
        self.assertEqual(type(float(w)), float, msg="Your toy_simulation function must return a float")
        self.assertLinf(w, tol=0.01, msg="Your simulation ended up at the wrong angle")


class Week03Tests(Report): #240 total.
    title = "Tests for week 03"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
        (Problem1Kuramoto, 10),
        (Exam5InventoryEvaluation, 10),
        (Exam6Toy2d, 10),
                 ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week03Tests())
