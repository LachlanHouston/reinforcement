# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex04.discrete_control_model import DiscreteControlModel
from irlc.ex04.control_environment import ControlEnvironment
from irlc.ex04.model_harmonic import HarmonicOscilatorModel
import numpy as np
from irlc.utils.graphics_util_pygame import UpgradedGraphicsUtil
from gymnasium.spaces import Box

class LocomotiveModel(HarmonicOscilatorModel):
    viewer = None
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, m=1., slope=0.0, target=0):
        """
        Slope is the uphill slope of the train (in degrees). E.g. slope=15 makes it harder for the engine.

        :param m:
        :param slope:
        """
        self.target = target
        self.slope = slope
        super().__init__(m=m, k=0., drag=-np.sin(slope/360*2*np.pi) * m * 9.82)

    def x0_bound(self) -> Box:
        return Box(np.asarray([-1, 0]), np.asarray([-1,0]))

    def u_bound(self) -> Box:
        return Box(np.asarray([-100]), np.asarray([100])) # Min and Max engine power.

    def render(self, x, render_mode="human"):
        """ Initialize a viewer and update the states. """
        if self.viewer is None:
            self.viewer = LocomotiveViewer(self)
        self.viewer.update(x, self.target)
        import time
        time.sleep(0.05)
        return self.viewer.blit(render_mode=render_mode)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

class DiscreteLocomotiveModel(DiscreteControlModel):
    def __init__(self, *args, dt=0.1, **kwargs):
        model = LocomotiveModel(*args, **kwargs)
        super().__init__(model=model, dt=dt)

class LocomotiveEnvironment(ControlEnvironment):
    def __init__(self, *args, dt=0.1, Tmax=5, render_mode=None, **kwargs):
        model = DiscreteLocomotiveModel(*args, dt=dt, **kwargs)
        # self.dt = model.dt
        super().__init__(discrete_model=model, Tmax=Tmax, render_mode=render_mode)


class LocomotiveViewer(UpgradedGraphicsUtil):
    def __init__(self, train):
        self.train = train
        width = 1100
        self.scale = width / 4
        self.dw = self.scale * 0.1
        super().__init__(screen_width=width, xmin=-width / 2, xmax=width / 2, ymin=-width / 5, ymax=width / 5, title='Locomotive environment')
        from irlc.utils.graphics_util_pygame import Object
        self.locomotive = Object("locomotive.png", image_width=90, graphics=self)

    def render(self):
        # fugly rendering code.
        dw = self.dw
        scale = self.scale
        train = self.train
        red = (200, 40, 40)
        from irlc.utils.graphics_util_pygame import rotate_around
        ptrack = [(-2 * scale, -dw / 2*0),
                  (-2 * scale, dw / 2),
                  (2 * scale, dw / 2),
                  (2 * scale, -dw / 2*0)]
        ptrack.append( ptrack[-1])
        ptrack = rotate_around(ptrack,(0,0), -self.train.slope)
        self.draw_background(background_color=(255, 255, 255))
        self.polygon("asdf", coords=ptrack, fillColor=(int(.7 * 255),) * 3, filled=True)
        self.locomotive.surf.get_height()
        self.locomotive.rotate(self.train.slope)
        p0 = (0,0)
        self.locomotive.move_center_to_xy( *rotate_around( (self.scale * self.x[0], -self.locomotive.surf.get_height()//2), p0, -self.train.slope))
        self.locomotive.blit(self.surf)
        xx = 0*self.scale * self.x[0]
        triangle = [(train.target * scale - dw / 2+ xx, dw/2), (train.target * scale + xx, -0*dw / 2),
                    (train.target * scale + dw / 2 + xx, dw/2)]
        triangle = rotate_around(triangle, p0, -self.train.slope)
        ddw = dw/2
        xx = self.scale * self.x[0]
        trainloc = [(xx- ddw / 2, -ddw / 2), ( xx, -0 * ddw / 2), (xx + ddw / 2, -ddw / 2)]
        trainloc = rotate_around(trainloc, p0, -self.train.slope)
        self.trg = self.polygon("", coords=trainloc, fillColor=red, filled=True)
        self.trg = self.polygon("", coords=triangle, fillColor=red, filled=True)

    def update(self, x, xstar):
        self.x = x #*self.scale
        self.xstar = xstar
