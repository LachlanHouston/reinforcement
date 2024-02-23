# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# from pyglet.shapes import Rectangle, Circle
# from irlc.utils.pyglet_rendering import PygletViewer, PolygonOutline, GroupedElement
import pygame
from irlc.utils.graphics_util_pygame import UpgradedGraphicsUtil
import numpy as np

track_outline = (0, 0, 0)
track_middle = (220, 25, 25)

class CarViewerPygame(UpgradedGraphicsUtil):
    def __init__(self, car):

        n = int(10 * (car.map.PointAndTangent[-1, 3] + car.map.PointAndTangent[-1, 4]))
        center = [car.map.getGlobalPosition(i * 0.1, 0) for i in range(n)]
        outer = [car.map.getGlobalPosition(i * 0.1, -car.map.width) for i in range(n)]
        inner = [car.map.getGlobalPosition(i * 0.1, car.map.width) for i in range(n)]
        fudge = 0.2
        xs, ys = zip(*outer)
        super().__init__(screen_width=1000, xmin=min(xs) - fudge, xmax=max(xs) + fudge,
                         ymax=min(ys) - fudge, ymin=max(ys) + fudge, title="Racecar environment")
        self.center = center
        self.outer = outer
        self.inner = inner
        # Load ze sprite.
        from irlc.utils.graphics_util_pygame import Object
        self.car = Object("car.png", image_width=90)


    def render(self):
        green = (126, 200, 80)
        track = (144,)*3
        self.draw_background(background_color=green)

        self.polygon("safd", self.outer, fillColor=track, outlineColor=track_outline, width=3)
        self.polygon("in", self.inner, fillColor=green, outlineColor=track_outline, width=3)
        self.polygon("in", self.center, fillColor=None, filled=False, outlineColor=(100, 100, 100), width=5)
        # Now draw the pretty car.
        x, y, psi = self.xglob[4], self.xglob[5], self.xglob[3]
        xy = self.fixxy((x,y))
        # self.car.rect.move()
        self.car.rect.center = xy
        # self.car.rect.center = xy[1]

        self.car.rotate(psi / (2*np.pi) * 360)
        # self.car.rotate(45)
        self.car.blit(self.surf)
        self.circle("in", (x,y), 4, fillColor=(255, 0, 0)) # drawn on the center of the car.

    def update(self, xglob):
        self.xglob = xglob
