# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import Report
from unitgrade import UTestCase
import irlc
from irlc.car.car_model import CarEnvironment
from irlc.ex04.pid_car import PIDCarAgent
from irlc import train
from irlc.ex04.pid_locomotive_agent import LocomotiveEnvironment, PIDLocomotiveAgent
from irlc.ex03.kuramoto import KuramotoModel, f
from irlc.ex04.discrete_kuramoto import fk, dfk_dx
import sympy as sym
import numpy as np

class Problem1DiscreteKuromoto(UTestCase):
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

    def test_fk(self):
        self.assertLinf(fk([0.1], [0.4]), tol=1e-6)

    def test_dfk_dx(self):
        self.assertLinf(dfk_dx([0.1], [0.4]), tol=1e-6)

class Problem3PID(UTestCase):
    """ PID Control """

    def test_pid_class(self, Kp=40, Ki=0, Kd=0, target=0, x=0):
        dt = 0.08
        from irlc.ex04.pid import PID
        pid = PID(Kp=Kp, Kd=Kd, Ki=Ki, target=target, dt=0.8)
        u = pid.pi(x)
        self.assertL2(u, tol=1e-4)

    def test_pid_Kp(self):
        self.test_pid_class(40, 0, 0, 0, 1)
        self.test_pid_class(10, 0, 0, 0, 2)


    def test_pid_target(self):
        self.test_pid_class(40, 0, 0, 3, 1)
        self.test_pid_class(20, 0, 0, 0, 2)


    def test_pid_all(self):
        self.test_pid_class(4, 3, 8, 1, 1)
        self.test_pid_class(40, 10, 3, 0, 2)


class Problem4PIDAgent(UTestCase):
    """ PID Control """

    def pid_locomotive(self, Kp=40, Ki=0, Kd=0, slope=0, target=0):
        dt = 0.08
        env = LocomotiveEnvironment(m=10, slope=slope, dt=dt, Tmax=5)
        agent = PIDLocomotiveAgent(env, dt=dt, Kp=Kp, Ki=Ki, Kd=Kd, target=target)
        stats, traj = train(env, agent, return_trajectory=True, verbose=False)
        self.assertL2(traj[0].state, tol=1e-4)

    def test_locomotive_flat(self):
        self.pid_locomotive()

    def test_locomotive_Kd(self):
        """ Test the derivative term """
        self.pid_locomotive(Kd = 10)

    def test_locomotive_Ki(self):
        """ Test the integral term """
        self.pid_locomotive(Kd = 10, Ki=5, slope=5)


    def test_locomotive_all(self):
        """ Test all terms """
        self.pid_locomotive(Kp=35, Kd = 10, Ki=5, slope=5, target=1)




class Problem7PIDCar(UTestCase):
    lt = -1

    @classmethod
    def setUpClass(cls) -> None:
        env = CarEnvironment(noise_scale=0, Tmax=80, max_laps=2)
        agent = PIDCarAgent(env, v_target=1.0)
        stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
        d = trajectories[0].state[:, 4]
        lt = len(d) * env.dt / 2
        print("Lap time", lt)
        cls.lt = lt

    def test_below_60(self):
        """ Testing if lap time is < 60 """
        self.assertTrue(0 < self.__class__.lt < 60)

    def test_below_40(self):
        """ Testing if lap time is < 60 """
        self.assertTrue(0 < self.__class__.lt < 40)


    def test_below_30(self):
        """ Testing if lap time is < 60 """
        self.assertTrue(0 < self.__class__.lt < 30)

    def test_below_22(self):
        """ Testing if lap time is < 22 """
        self.assertTrue(0 < self.__class__.lt < 22)

class Week04Tests(Report):
    title = "Tests for week 04"
    pack_imports = [irlc]
    individual_imports = []
    questions = [
                (Problem1DiscreteKuromoto, 10),
                (Problem3PID, 10),
                (Problem4PIDAgent, 10),  # ok
                (Problem7PIDCar, 10),  # ok
                ]

if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Week04Tests())
