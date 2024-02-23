# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# graphicsUtils.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import numpy as np
import os
import pygame
from pygame import gfxdraw
import threading
import time
import pygame
import platform
import sys

ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list

def formatColor(r, g, b):
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

def colorToVector(color):
    return list(map(lambda x: int(x, 16) / 256.0, [color[1:3], color[3:5], color[5:7]]))

def h2rgb(color):
    if color is None or isinstance(color, tuple):
        return color
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

def h2rgb255(color):
    if isinstance(color, tuple):
        return color
    # c =
    return tuple(int(cc*255) for cc in h2rgb(color))
    if color is None:
        return None
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

class GraphicsCache:
    break_cache = False
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        # self._items_in_viewer = {}
        # self._seen_things = set()
        self.clear()
        self.verbose = verbose

    def copy_all(self):
        self._seen_things.update( set( self._items_in_viewer.keys() ) )

    def clear(self):
        self._seen_things = set()
        self.viewer.geoms.clear()
        self._items_in_viewer = {}

    def prune_frame(self):
        s0 = len(self._items_in_viewer)
        self._items_in_viewer = {k: v for k, v in self._items_in_viewer.items() if k in self._seen_things }
        if self.verbose:
            print("removed", len(self._items_in_viewer) - s0,  "geom size", len(self._items_in_viewer))
        self.viewer.geoms = list( self._items_in_viewer.values() )
        self._seen_things = set()


    def add_geometry(self, name, geom):
        if self.break_cache:
            if self._items_in_viewer == None:
                self.viewer.geoms = []
                self._items_in_viewer = {}

        self._items_in_viewer[name] = geom
        self._seen_things.add(name)



class GraphicsUtilGym:
    viewer = None
    _canvas_xs = None      # Size of canvas object
    _canvas_ys = None
    _canvas_x = None      # Current position on canvas
    _canvas_y = None

    def begin_graphics(self, width=640, height=480, color=formatColor(0, 0, 0), title="02465 environment", local_xmin_xmax_ymin_ymax=None, verbose=False,
                       frames_per_second=None):
        """ Main interface for managing graphics.
            The local_xmin_xmax_ymin_ymax controls the (local) coordinate system which is mapped onto screen coordinates. I.e. specify this
            to work in a native x/y coordinate system. If not, it will default to screen coordinates familiar from Gridworld.
        """
        width = int(width)
        height = int(height)    # For width/height to be integers to avoid crashes on some systems.

        icon = os.path.dirname(__file__) + "/../utils/graphics/dtu_icon.png"
        pygame_icon = pygame.image.load(icon)
        pygame.display.set_icon(pygame_icon)
        screen_width = width
        screen_height = height
        pygame.init()
        pygame.display.init()
        self.frames_per_second = frames_per_second


        self.screen = pygame.display.set_mode(
            (screen_width, screen_height)
        )
        self.screen_width = width
        self.screen_height = height

        pygame.display.set_caption(title)

        if height % 2 == 1:
            height += 1 # Must be divisible by 2.
        self._bg_color = color
        # viewer = Viewer(width=int(width), height=int(height))
        # viewer.window.set_caption(title)
        # self.viewer = viewer
        # self.gc = GraphicsCache(viewer, verbose=verbose)
        self._canvas_xs, self._canvas_ys = width - 1, height - 1
        self._canvas_x, self._canvas_y = 0, self._canvas_ys
        if local_xmin_xmax_ymin_ymax is None:
            # local_coordinates = []
            # This will align the coordinate system so it begins in the top-left corner.
            # This is the default behavior of pygame.
            local_xmin_xmax_ymin_ymax = (0, width, 0, height)
        self._local_xmin_xmax_ymin_ymax = local_xmin_xmax_ymin_ymax

        self.demand_termination = threading.Event()
        self.pause_refresh = False
        self.ask_for_pause = False
        self.is_paused = False
        self.time_last_blit = -1


        def refresh_window(gutils):
            refresh_interval_seconds = 0.1 # Miliseconds
            t0 = time.time()
            while not gutils.demand_termination.is_set():
                t1 = time.time()
                if t1 - t0 > refresh_interval_seconds:
                    if not self.ask_for_pause:
                        self.is_paused = False
                        if not (sys.platform == 'darwin' and platform.processor() == 'i386'):
                            pass # Disable the thread startup. This causes problems on linux (segfaults). Must find better fix, perhaps win-only.
                            # pygame.display.update()
                    else:
                        self.is_paused = True
                    t0 = t1
                time.sleep(refresh_interval_seconds/100)

        self.refresh_thread = threading.Thread(target=refresh_window, args=(self, ))
        self.refresh_thread.start()

    def close(self):
        self.demand_termination.set()
        self.refresh_thread.join(timeout=1000)
        pygame.display.quit()
        pygame.quit()
        # TH 2023: These two lines are super important.
        #  pdraw cache the fonts. So when pygame is loaded/quites,
        #  the font cache is not flushed. This is not a problem
        #  when determining the width of strings the font has seen,
        #  but causes a segfault with NEW strings.
        from irlc.utils import ptext
        ptext._font_cache = {}
        self.isopen = False

    def render(self):
        pass

    def blit(self, render_mode=None):
        self.render()
        self.screen.blit(self.surf, (0, 0))
        if render_mode == "human":
            tc = time.time()

            if self.frames_per_second is not None:

                if tc - self.time_last_blit < 1/self.frames_per_second:
                    tw = 1/self.frames_per_second - (tc - self.time_last_blit )
                    time.sleep(tw)
                else:
                    tw = 0

                self.time_last_blit = tc

            pygame.event.pump()
            pygame.display.flip()
        elif render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def rectangle(self, color, x, y, width, height, border=0, fill_color=None):
        x2,y2 = self.fixxy((x+width, y+height))
        x, y = self.fixxy((x,y))

        c1 = min([x, x2])
        c2 = min([y, y2])

        w = abs(x-x2)
        h = abs(y - y2)

        pygame.draw.rect(self.surf, color, pygame.Rect( int(c1), int(c2), int(w), int(h)), border)


    def draw_background(self, background_color=None):
        if background_color is None:
            background_color = (0, 0, 0)
        self._bg_color = background_color
        x1, x2, y1, y2 = self._local_xmin_xmax_ymin_ymax
        corners = [ (x1, y1), (x2, y1), (x2, y2), (x1, y2)  ]
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.polygon(name="background", coords=corners, outlineColor=self._bg_color, fillColor=self._bg_color, filled=True, smoothed=False)

    def fixxy(self, xy):
        x,y = xy
        x = (x - self._local_xmin_xmax_ymin_ymax[0]) / (self._local_xmin_xmax_ymin_ymax[1] - self._local_xmin_xmax_ymin_ymax[0]) * self.screen.get_width()
        y = (y - self._local_xmin_xmax_ymin_ymax[2]) / (self._local_xmin_xmax_ymin_ymax[3] - self._local_xmin_xmax_ymin_ymax[2]) * self.screen.get_height()
        return int(x), int(y)


    def plot(self, name, x, y, color=None, width=1.0):
        coords = [(x_,y_) for (x_, y_) in zip(x,y)]
        if color is None:
            color = "#000000"
        return self.polygon(name, coords, outlineColor=color, filled=False, width=width)

    def polygon(self, name, coords, outlineColor=None, fillColor=None, filled=True, smoothed=1, behind=0, width=1.0, closed=False):
        c = []
        for coord in coords:
            c.append(coord[0])
            c.append(coord[1])

        coords = [self.fixxy(c) for c in coords]
        if fillColor == None: fillColor = outlineColor
        poly = None
        if not filled: fillColor = ""

        c = [self.fixxy(tuple(c[i:i+2])) for i in range(0, len(c), 2)]
        if not filled:
            gfxdraw.polygon(self.surf, coords, h2rgb255(outlineColor))
            pygame.draw.polygon(self.surf, h2rgb255(outlineColor), coords, width=int(width))

        else:
            gfxdraw.filled_polygon(self.surf, coords, h2rgb255(fillColor))

        if outlineColor is not None and len(outlineColor) > 0 and filled: # Not sure why this cannot be merged with the filled case...
            # gfxdraw.polygon(self.surf, coords, h2rgb255(outlineColor), width=int(width))
            pygame.draw.polygon(self.surf, h2rgb255(outlineColor), coords, width=int(width))

        return poly

    def square(self, name, pos, r, color, filled=1, behind=0):
        x, y = pos
        coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return self.polygon(name, coords, color, color, filled, 0, behind=behind)

    def centered_arc(self, color, pos, r, start_angle, stop_angle, width=1):
        # Draw a centered arc (pygame defaults to boxed arcs)
        x, y = pos
        tt = np.linspace(start_angle / 360 * 2 * np.pi,stop_angle / 360 * 2 * np.pi, int(r * 10))
        px = np.cos(tt) * r
        py = -np.sin(tt) * r
        pp = list(zip(px.tolist(), py.tolist()))

        pp = [((x + a, y + b)) for (a, b) in pp]
        # if style == 'arc':  # For pacman. I guess this one makes the rounded wall segments.
        pp = [self.fixxy(p_) for p_ in pp]

        pygame.draw.lines(self.surf, h2rgb255(color), False, pp, width)

    def circle(self, name, pos, r, outlineColor=None, fillColor=None, endpoints=None, style='pieslice', width=2):
        pos = self.fixxy(pos)
        x, y = pos
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        if endpoints is not None and len(endpoints) > 0:
            tt = np.linspace(e[0]/360 * 2*np.pi, e[-1]/360 * 2*np.pi, int(r*20) )
            px = np.cos(tt) * r
            py = -np.sin(tt) * r
            pp = list(zip(px.tolist(), py.tolist()))
            if style == 'pieslice':
                pp = [(0,0),] + pp + [(0,0),]
            pp = [( (x+a, y+b)) for (a,b) in pp  ]
            if style == 'arc': # For pacman. I guess this one makes the rounded wall segments.
                pp = [self.fixxy(p_) for p_ in pp]
                pygame.draw.lines(self.surf, outlineColor, False, pp, width)
            elif style == 'pieslice':
                self.polygon(name, pp, fillColor=fillColor, outlineColor=outlineColor, width=width)
            else:
                raise Exception("bad style", style)
        else:
            gfxdraw.filled_circle(self.surf, x, y, int(r), h2rgb255(fillColor))

    def text(self, name, pos, color, contents, font='Helvetica', size=12, style='normal', anchor="w", fontsize=24,
             bold=False):
        pos = self.fixxy(pos)
        ax = "center"
        ax = "left" if anchor == "w" else ax
        ay = "center"
        ay = "baseline" if anchor == "s" else ay

        from irlc.utils.ptext import draw
        if anchor == 'w':
            opts = dict(midleft=pos)
        elif anchor == 'e':
            opts = dict(midright=pos)
        elif anchor == 's':
            opts = dict(midbottom=pos)
        elif anchor == 'n':
            opts = dict(midtop=pos)
        elif anchor == 'c':
            opts = dict(center=pos)
        else:
            raise Exception("Unknown anchor", anchor)
        opts['fontsize'] = fontsize
        opts['bold'] = bold
        draw(contents, surf=self.surf, color=h2rgb255(color), pos=pos, **opts)
        return


    def line(self, name, here, there, color=formatColor(0, 0, 0), width=2):

        here, there = self.fixxy(here), self.fixxy(there)
        pygame.draw.line(self.surf, h2rgb255(color), here, there, width)


def rotate_around(pos, xy0, angle):
    if isinstance(pos, list) and isinstance(pos[0], tuple):
        return [rotate_around(p, xy0, angle) for p in pos]
    return ((pos[0] - xy0[0]) * np.cos(angle / 180 * np.pi) - (pos[1] - xy0[1]) * np.sin(angle / 180 * np.pi) + xy0[0],
            (pos[0] - xy0[0]) * np.sin(angle / 180 * np.pi) + (pos[1] - xy0[1]) * np.cos(angle / 180 * np.pi) + xy0[1])

class Object(pygame.sprite.Sprite):
    def __init__(self, file, image_width=None, graphics=None):
        super(Object, self).__init__()
        fpath = os.path.dirname(__file__) +"/graphics/"+file
        image = pygame.image.load(fpath).convert_alpha()
        if image_width is not None:
            image_height = int( image_width / image.get_width() * image.get_height() )
            self.og_surf = pygame.transform.smoothscale(image, (image_width, image_height))
            # raise Exception("Implement this")
        else:
            self.og_surf = image
        # self.og_surf = pygame.transform.smoothscale(image, (100, 100))
        self.surf = self.og_surf
        self.rect = self.surf.get_rect(center=(400, 400))
        self.ga = graphics

    def move_center_to_xy(self, x, y):
        # Note: These are in the local coordinate system coordinates.
        x,y = self.ga.fixxy((x,y))
        self.rect.center = (x,y)

    def rotate(self, angle):
        """ Rotate sprite around it's center. """
        self.angle = angle
        self.surf = pygame.transform.rotate(self.og_surf, self.angle)
        self.rect = self.surf.get_rect(center=self.rect.center)

    def blit(self, surf):
        surf.blit(self.surf, self.rect)


class UpgradedGraphicsUtil(GraphicsUtilGym):
    def __init__(self, screen_width=800, screen_height=None, xmin=0., xmax=800., ymin=0., ymax=600., title="Gym window"):
        if screen_height is None:
            screen_height = np.abs( int(screen_width / (xmax - xmin) * (ymax-ymin)) )
        elif xmin is None:
            xmin = 0
            xmax = screen_width
            ymin = 0
            ymax = screen_height
        else:
            raise Exception()
        self.begin_graphics(width=screen_width, height=screen_height, local_xmin_xmax_ymin_ymax=(xmin, xmax, ymin, ymax), title=title)

    def get_sprite(self, name):
        """ Load a sprite from the graphics directory. """
        pass
