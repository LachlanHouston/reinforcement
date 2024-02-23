# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import math
import numpy as np
import pygame
from PIL import ImageColor
# from pyglet.shapes import Circle, Rectangle, Polygon, Sector
# from irlc.utils.pyglet_rendering import GroupedElement
from irlc.pacman.pacman_graphics_display import GHOST_COLORS, GHOST_SHAPE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# class Eye(GroupedElement):
#     normal, cross = None, None
#
#     def render(self):
#         self.normal = [Circle(0, 0, .2, color=WHITE, batch=self.batch, group=self.group),
#                        Circle(0, 0, 0.1, color=BLACK, batch=self.batch, group=self.group)]  # radius was 0.08
#         ew = 0.6
#         rw = ew/6
#         self.cross = [Rectangle(x=-ew/2, y=-rw/2, width=ew, height=rw, color=BLACK, group=self.group, batch=self.batch),
#                       Rectangle(x=-rw/2, y=-ew/2, width=rw, height=ew, color=BLACK, group=self.group, batch=self.batch)]
#         self.set_eye_dir('stop')
#
#     def set_eye_dir(self, direction='stop'):
#         dead = direction.lower() == 'dead'
#         for n in self.normal:
#             n.visible = not dead
#             pp = (0, 0)
#             if direction.lower() == 'stop':
#                 pass
#             dd = 0.1
#             if direction.lower() == 'east':
#                 pp = (dd, 0)
#                 # self.group.translate(dd, 0)
#
#             if direction.lower() == 'west':
#                 pp = (-dd, 0)
#                 # self.group.translate(-dd, 0)
#             if direction.lower() == 'south':
#                 pp = (0, -dd)
#                 # self.group.translate(0, -dd)
#             if direction.lower() == 'north':
#                 # self.group.translate(0, dd)
#                 pp = (0, dd)
#             self.normal[1].x = pp[0]
#             self.normal[1].y = pp[1]
#
#         for e in self.cross:
#             e.visible = dead
#         self.group.rotate(np.pi/4 if dead else 0)

from irlc.utils.graphics_util_pygame import rotate_around

class Ghost:
    body_, eyes_ = None, None
    def __init__(self, graphics_adaptor, agent_index=1, order=1, scale=10.):
        self.agentIndex = agent_index

        # GS = [(x*scale, y*scale) for x,y in GHOST_SHAPE]
        # self.GS = GS
        # xx, yy = zip(*GS)
        # xmin, xmax = min(xx), max(xx)
        # ymin, ymax = min(yy), max(yy)
        # this creates a surface
        # self.GS = GS
        # self.surf = pygame.Surface( (int(xmax-xmin), int(ymax-ymin)) )
        # Write ghost to this surface, then turn it to make it lie down.
        self.ga = graphics_adaptor
        # self.xmin = xmin
        # self.ymin = ymin
        # self.rect = self.surf.get_rect()
        self.x = 0
        self.y = 0
        self.angle = 0
        self.scale = scale

        self.direction = 'stop'
        # super().__init__(order=order)


    def set_scared(self, scared):
        return
        from irlc.pacman.devel.pyglet_pacman_graphics import SCARED_COLOR, GHOST_COLORS
        self.body_.color = SCARED_COLOR if scared else GHOST_COLORS[self.agentIndex]

    def eyes(self, direction):
        return
        for e in self.eyes_:
            e.set_eye_dir(direction)

    def set_position(self, x, y):
        # print("setting position", x,y)
        # self.group.x = x
        # self.group.y = y
        # self.group.translate(x, y)
        self.x = x
        self.y = y
        pass

    def rand_eyes(self):
        return ['stop', 'east', 'west', 'north', 'south'][np.random.randint(0, 5)]


    def set_direction(self, direction):
        self.direction = direction

        return
        self.eyes(direction)

    def kill(self):
        self.set_direction('dead')
        return
        # return
        # self.eyes('dead')
        self.body_color = ImageColor.getcolor(GHOST_COLORS[3], "RGB")
        # self.group.rotate(-np.pi/2)

    def resurrect(self):
        self.set_direction(self.rand_eyes())
        # return
        # self.eyes('straight')
        return
        self.body_.color = ImageColor.getcolor(GHOST_COLORS[self.agentIndex], "RGB")
        self.group.rotate(0)

    def render(self):
        # ghost_shape = tuple((x, -y) for (x, y) in GHOST_SHAPE)
        dead = self.direction.lower() == 'dead'
        angle = 0
        if dead:
            angle = -90

        ghost_shape = tuple((x*self.scale+self.x, -y*self.scale+self.y) for (x, y) in GHOST_SHAPE)

        # self.ga.polygon()
        # print(ghost_shape)
        xy0 = (self.x, self.y)
        self.ga.polygon("asdfasf", [rotate_around(c, xy0, angle) for c in ghost_shape], GHOST_COLORS[self.agentIndex] if not dead else GHOST_COLORS[3], filled=1)
        dx = 0.3
        dy = 0.3

        # pdx = 0.2
        # pdy = 0.2

        for k in range(2):
            pos =  (self.x + (-1 if k == 0 else 1)*dx*self.scale, self.y + dy*self.scale)
            self.ga.circle("asdfsF", rotate_around(pos, xy0, angle), 0.15*self.scale, None, WHITE)
            # Eyes:
            # continue

            direction = self.direction


            # for n in self.normal:
            #     n.visible = not dead
            pp = (0, 0)
            if direction.lower() == 'stop':
                pass
            dd = 0.1
            if direction.lower() == 'east':
                pp = (dd, 0)
                # self.group.translate(dd, 0)
            if direction.lower() == 'west':
                pp = (-dd, 0)
                # self.group.translate(-dd, 0)
            if direction.lower() == 'south':
                pp = (0, -dd)
                # self.group.translate(0, -dd)
            if direction.lower() == 'north':
                # self.group.translate(0, dd)
                pp = (0, dd)
            # self.normal[1].x = pp[0]
            # self.normal[1].y = pp[1]
            if not dead:
                self.ga.circle("asdfsF", rotate_around( (pos[0] + pp[0]*self.scale, pos[1] + pp[1]*self.scale), xy0, self.angle),
                               0.05 * self.scale, None, BLACK)
            else:
                ew = 0.6
                rw = ew / 6
                for k in range(2):
                    cross = [(-rw/2, ew/2),
                             (rw / 2, ew / 2),
                             (rw / 2, -ew / 2),
                             (-rw / 2, -ew / 2),
                             ]
                    cross = cross + [cross[0]]
                    cross = [rotate_around(c, (0,0), 45 + 90*k) for c in cross]
                    cc = [rotate_around( (pos[0]+x *self.scale+ pp[0], pos[1]+y *self.scale+ pp[1]), xy0, angle) for (x,y) in cross]
                    self.ga.polygon("asdfasf", cc, None, filled=True, fillColor=BLACK)






                # self.cross = [
                #     Rectangle(x=-ew / 2, y=-rw / 2, width=ew, height=rw, color=BLACK, group=self.group, batch=self.batch),
                #     Rectangle(x=-rw / 2, y=-ew / 2, width=rw, height=ew, color=BLACK, group=self.group, batch=self.batch)]


                pass
            # Circle(0, 0, .2, color=WHITE, batch=self.batch, group=self.group)
            #
            # self.normal = [Circle(0, 0, .2, color=WHITE, batch=self.batch, group=self.group),
            #                Circle(0, 0, 0.1, color=BLACK, batch=self.batch, group=self.group)]  # radius was 0.08
            # ew = 0.6
            # rw = ew / 6
            # self.cross = [
            #     Rectangle(x=-ew / 2, y=-rw / 2, width=ew, height=rw, color=BLACK, group=self.group, batch=self.batch),
            #     Rectangle(x=-rw / 2, y=-ew / 2, width=rw, height=ew, color=BLACK, group=self.group, batch=self.batch)]
            #
            # for e in self.cross:
            #     e.visible = dead
        # return
        # self.ga.polygon()
        # colour = ImageColor.getcolor(GHOST_COLORS[self.agentIndex], "RGB")
        # self.body_ = Polygon(*ghost_shape, color=colour, batch=self.batch, group=self.group)
        # self.eyes_ = [Eye(order=self.group.order+1+k, pg=self.group, batch=self.batch) for k in range(2)]
        # for k, e in enumerate(self.eyes_):
        #     e.group.translate(-.3 if k == 0 else .3, .3)


PACMAN_COLOR = (255, 255, 61)


# class Pacman(GroupedElement):
#     body = None
#
#     def __init__(self, grid_size, batch, pg=None, parent=None, order=0):
#         self.delta = 0
#         self.GRID_SIZE = grid_size
#         super().__init__(batch, pg=pg, parent=parent, order=order)
#         self.set_animation(0, 4)
#
#     def set_animation(self, frame, frames):
#         pos = frame/frames
#         width = 30 + 80 * math.sin(math.pi * pos)
#         delta = width / 2
#         self.delta = delta * np.pi / 180
#         self.body._angle = 2*np.pi-2*self.delta
#         self.body._start_angle = self.delta
#         self.body._update_position()
#
#     def set_direction(self, direction):
#         if direction == 'Stop':
#             pass
#         else:
#             angle = 0
#             if direction == 'East':
#                 angle = 0
#             elif direction == 'North':
#                 angle = np.pi/2
#             elif direction == 'West':
#                 angle = np.pi
#             elif direction == 'South':
#                 angle = np.pi*1.5
#             self.group.rotate(angle)
#
#     def render(self):
#         width = 30
#         delta = width/2
#         delta = delta/180 * np.pi
#         self.body = Sector(0, 0, self.GRID_SIZE/2, angle=2*np.pi-2*delta, start_angle=delta,
#                            color=PACMAN_COLOR, batch=self.batch, group=self.group)
