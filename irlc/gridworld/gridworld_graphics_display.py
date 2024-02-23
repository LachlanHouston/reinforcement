# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# gridworld_graphics_display.py
# ---------------------------
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

from irlc.utils.graphics_util_pygame import GraphicsUtilGym, formatColor
from irlc.pacman.pacman_graphics_display import PACMAN_OUTLINE_WIDTH, PACMAN_SCALE
from irlc.gridworld.gridworld_mdp import GridworldMDP
from collections import defaultdict
import math
import numpy as np

# import sphinx_autorun

BACKGROUND_COLOR = formatColor(0, 0, 0)
EDGE_COLOR = formatColor(1, 1, 1)
OBSTACLE_COLOR = formatColor(0.5, 0.5, 0.5)
TEXT_COLOR = formatColor(1, 1, 1)
MUTED_TEXT_COLOR = formatColor(0.7, 0.7, 0.7)
LOCATION_COLOR = formatColor(0, 0, 1)
RED_TEXT_COLOR = formatColor(.68, .93, 0.93)
from irlc.pacman.pacman_graphics_display import PACMAN_COLOR

def getEndpoints(direction, position=(0, 0)):
    x, y = position
    pos = x - int(x) + y - int(y)
    width = 30 + 80 * math.sin(math.pi * pos)
    delta = width / 2
    if direction == 'West':
        endpoints = (180 + delta, 180 - delta)
    elif direction == 'North':
        endpoints = (90 + delta, 90 - delta)
    elif direction == 'South':
        endpoints = (270 + delta, 270 - delta)
    else:
        endpoints = (0 + delta, 0 - delta)
    return endpoints


class GraphicsGridworldDisplay:
    time_since_last_update = 0
    key_queue = []
    def __init__(self, mdp, size=120, frames_per_second=None):
        self.mdp = mdp
        self.ga = GraphicsUtilGym()
        self.Q_old = None
        self.v_old = None
        self.Null_old = None
        title = "Gridworld Display"
        self.GRID_SIZE = size
        self.MARGIN = self.GRID_SIZE * 0.75
        screen_width = (mdp.width - 1) * self.GRID_SIZE + self.MARGIN * 2
        screen_height = (mdp.height - 0.5) * self.GRID_SIZE + self.MARGIN * 2
        self.ga.begin_graphics(screen_width, screen_height, BACKGROUND_COLOR, title=title, frames_per_second=frames_per_second)
        self.annotations = []
        # function to refresh the window


    def draw_annotation(self):
        for a in self.annotations:
            if a['type'] == 'text':
                self.ga.text(f"adf", (a['x'], a['y']), a['color'], a['message'], "Courier", anchor='c', fontsize=a['fontsize'], bold=a['bold'])


    def annotate_text(self, state, symbol='o', color=(200,50, 50), dx=0, dy=0, action=None, fontsize=30, bold=False):
        x,y = self.to_screen(state)
        x += int(dx * self.GRID_SIZE)
        y += int(dy * self.GRID_SIZE)
        if action is not None:
            from irlc.gridworld.gridworld_mdp import GridworldMDP
            dd = 0.2
            if action == GridworldMDP.NORTH:   y -= int( dd * self.GRID_SIZE)
            elif action == GridworldMDP.SOUTH: y += int( dd * self.GRID_SIZE)
            elif action == GridworldMDP.EAST: x += int(dd * self.GRID_SIZE)
            elif action == GridworldMDP.WEST: x -= int( dd * self.GRID_SIZE)

        self.annotations.append({'type': 'text', 'x': x, 'y': y, 'message': symbol, 'color': color, 'fontsize': fontsize, 'bold': bold})


    def close(self):
        # Stop pygame and refresh thread
        self.ga.close()

    def blit(self, render_mode=None):
        return self.ga.blit(render_mode=render_mode)
        # if render_mode == 'rgb_array':
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     )
        #
        #     pass



    # def autorefresh(self, env, interval=0.1):
    #     raise Exception("What is this?")
    #     def task(env, interval):
    #         while True:
    #             env.render()
    #             time.sleep(interval)
    #
    #     from threading import Thread
    #     thread = Thread(target=task, args=(env, interval))
    #     thread.start()

    # def end_frame(self):
    #     self.ga.end_frame()

    def displayValues(self, mdp, v, preferred_actions=None, currentState=None, message='Agent Values', returns_count=None, returns_sum=None):
        # if self.v_old == None:
        #     self.ga.gc.clear()
        #     self.v_old = {}
        # else:
        #     pass
        self.ga.draw_background()
        m = [v[s] for s in mdp.nonterminal_states]
        self.Q_old = None
        grid = mdp.grid
        minValue = min(m)
        maxValue = max(m)

        for x in range(mdp.width):
            for y in range(mdp.height):
                name = f"V_{x}_{y}_"
                state = (x, y)
                gridType = grid[x, y]
                isExit = str(gridType) != gridType
                isCurrent = currentState == state
                if gridType == '#':
                    self.drawSquare(name, x, y, 0, 0, 0, None, None, True, False, isCurrent)
                else:
                    value = v[state]
                    value = np.round(value, 2)
                    valString = '%.2f' % value
                    if mdp.is_terminal(state):
                        all_actions = []
                    else:
                        all_actions = mdp.A(state)
                        if preferred_actions != None:
                            all_actions = preferred_actions[state]

                    returns_sum_ = returns_sum[state] if returns_sum is not None else None
                    returns_count_ = returns_count[state] if returns_count is not None else None
                    self.drawSquare(name, x, y, value, minValue, maxValue, valString, all_actions, False, isExit, isCurrent,
                                    returns_sum=returns_sum_, returns_count=returns_count_)

        # print("Drawing...")
        if isinstance(currentState, tuple):
            # print("found pacman")
            screen_x, screen_y = self.to_screen(currentState)
            self.draw_player((screen_x, screen_y), 0.12 * self.GRID_SIZE)
        # else:
        #     print("no instance found??")

        pos = self.to_screen(((mdp.width - 1.0) / 2.0, - 0.8))
        self.ga.text(f"v_text_", pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")
        self.draw_annotation()

    def displayNullValues(self, mdp, currentState=None, message=''):
        self.ga.draw_background()
        grid = mdp.grid
        # self.blank()
        for x in range(mdp.width):
            for y in range(mdp.height):
                state = (x, y)
                gridType = grid[x,y]
                isExit = str(gridType) != gridType
                isCurrent = currentState == state
                name = f"sq_{x}_{y}"
                if gridType == '#':
                    self.drawSquare(name, x, y, 0, 0, 0, None, None, True, False, isCurrent)
                else:
                    self.drawNullSquare(name, mdp.grid, x, y, False, isExit, isCurrent)
        pos = self.to_screen(((mdp.width - 1.0) / 2.0, - 0.8))

        if isinstance(currentState, tuple):
            screen_x, screen_y = self.to_screen(currentState)
            self.draw_player((screen_x, screen_y), 0.12 * self.GRID_SIZE)
        else:
            pass
            # print("No player!")
        # pos = self.to_screen(((mdp.width - 1.0) / 2.0, - 0.8))
        # self.ga.text("Q_values_text", pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")

        self.ga.text("bottom_text", pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")
        self.draw_annotation()


    def displayQValues(self, mdp, Q, currentState=None, message="Agent Q-Values", eligibility_trace=None):
        """ Eligibility trace is an optional dictionary-like object. """
        self.ga.draw_background()
        if self.Q_old == None:
            # self.ga.gc.clear()

            self.Q_old = {}
            self.e_old = {}
        else:
            pass
            # self.ga.gc.copy_all()

        self.v_old = None
        self.Null_old = None

        m = [max(Q.get_Qs(s)[1]) for s in mdp.nonterminal_states]
        mv = [min(Q.get_Qs(s)[1]) for s in mdp.nonterminal_states]

        minValue = min(mv)
        maxValue = max(m)
        for x in range(mdp.width):
            for y in range(mdp.height):
                state = (x, y)
                if state not in mdp.nonterminal_states:
                    actions = []
                    Qs = []
                else:
                    actions, Qs = Q.get_Qs((x, y))
                    Qs = list(np.round(Qs, decimals=2))

                # Q_same = False
                if self.Q_old != None and Qs == self.Q_old.get((x, y), 0):
                    Q_same = True
                else:
                    Q_same = False
                Q_same = False
                E_same = True
                if eligibility_trace is not None:
                    es = [eligibility_trace[state, a] for a in actions]
                    if state in self.e_old and self.e_old[state] == es:
                        E_same = True
                    else:
                        E_same = False

                if E_same and Q_same:
                    continue
                else:
                    self.Q_old[state] = Qs
                    if eligibility_trace is not None:
                        self.e_old[state] = es

                name = f"Qsqr_{x}_{y}"
                gridType = mdp.grid[x, y]
                isExit = (str(gridType) != gridType)
                isCurrent = (currentState == state)
                # actions = mdp.A(state)
                if actions == None or len(actions) == 0:
                    actions = [None]
                q = defaultdict(lambda: 0)
                valStrings = {}

                if gridType == '#':
                    self.drawSquare(name, x, y, 0, 0, 0, None, None, True, False, isCurrent)
                elif isExit:
                    action = actions[0]  # next(iter(q.keys()))
                    value = Qs[0]  # q[action]  # q[action]
                    valString = '%.2f' % value
                    self.drawSquare(name, x, y, value, minValue, maxValue, valString, [action], False, isExit,
                                    isCurrent)
                else:
                    actions, Qs = Q.get_Qs(state)

                    for k, action in enumerate(actions):
                        v = Qs[k] # Get the Q-value.
                        # v = Q[state, action]
                        q[action] += v
                        valStrings[action] = '%.2f' % v
                        # etrace = None if eligibility_trace is None else eligibility_trace[]
                        # print(state, action, eligibility_trace[state, action])
                        de = None if eligibility_trace is None else {a: eligibility_trace[state, a] for a in actions}
                    self.drawSquareQ(name, x, y, q, minValue, maxValue, valStrings, actions, isCurrent, eligibility_trace=de)
        pos = self.to_screen(((mdp.width - 1.0) / 2.0, - 0.8))
        self.ga.text("Q_values_text", pos, TEXT_COLOR, message, "Courier", -32, "bold", "c")

        if isinstance(currentState, tuple):

            screen_x, screen_y = self.to_screen(currentState)
            self.draw_player((screen_x, screen_y), 0.12 * self.GRID_SIZE)
        self.draw_annotation()


    def drawNullSquare(self, name, grid, x, y, isObstacle, isTerminal, isCurrent):
        square_color = getColor(0, -1, 1)
        if isObstacle:
            square_color = OBSTACLE_COLOR
        (screen_x, screen_y) = self.to_screen((x, y))
        self.square(name + "_s1", (screen_x, screen_y),
                    0.5 * self.GRID_SIZE,
                    color=square_color,
                    filled=1,
                    width=1)
        self.square(name + "_s2", (screen_x, screen_y),
                    0.5 * self.GRID_SIZE,
                    color=EDGE_COLOR,
                    filled=0,
                    width=3)
        if isTerminal and not isObstacle:
            self.square(name + "_s3", (screen_x, screen_y),
                        0.4 * self.GRID_SIZE,
                        color=EDGE_COLOR,
                        filled=0,
                        width=2)
            self.ga.text(name + "_text", (screen_x, screen_y),
                         TEXT_COLOR,
                         str(grid[x,y]),
                         "Courier", -24, "bold", "c")
        self.draw_annotation()


    def drawSquare(self, name, x, y, val, min, max, valStr, all_action, isObstacle, isTerminal, isCurrent,
                   returns_count=None, returns_sum=None):
        square_color = getColor(val, min, max)
        (screen_x, screen_y) = self.to_screen((x, y))
        if isObstacle:
            square_color = OBSTACLE_COLOR

        self.square(name + "_o1", (screen_x, screen_y), 0.5 * self.GRID_SIZE, color=square_color, filled=1, width=1)

        self.square(name + "_o2", (screen_x, screen_y), 0.5 * self.GRID_SIZE, color=EDGE_COLOR, filled=0, width=3)
        if isTerminal and not isObstacle:
            self.square(name + "_o3", (screen_x, screen_y), 0.4 * self.GRID_SIZE, color=EDGE_COLOR, filled=0, width=2)

        if all_action is None:
            all_action = []
        GRID_SIZE = self.GRID_SIZE
        for action in all_action:
            if action == GridworldMDP.NORTH:
                self.ga.polygon(name + "_p1", [(screen_x, screen_y - 0.45 * GRID_SIZE),
                                               (screen_x + 0.05 * GRID_SIZE, screen_y - 0.40 * GRID_SIZE),
                                               (screen_x - 0.05 * GRID_SIZE, screen_y - 0.40 * GRID_SIZE)], EDGE_COLOR,
                                filled=1, smoothed=False)
            if action == GridworldMDP.SOUTH:
                self.ga.polygon(name + "_p2", [(screen_x, screen_y + 0.45 * GRID_SIZE),
                                               (screen_x + 0.05 * GRID_SIZE, screen_y + 0.40 * GRID_SIZE),
                                               (screen_x - 0.05 * GRID_SIZE, screen_y + 0.40 * GRID_SIZE)], EDGE_COLOR,
                                filled=1, smoothed=False)
            if action == GridworldMDP.WEST:
                self.ga.polygon(name + "_p3", [(screen_x - 0.45 * GRID_SIZE, screen_y),
                                               (screen_x - 0.4 * GRID_SIZE, screen_y + 0.05 * GRID_SIZE),
                                               (screen_x - 0.4 * GRID_SIZE, screen_y - 0.05 * GRID_SIZE)], EDGE_COLOR,
                                filled=1, smoothed=False)
            if action == GridworldMDP.EAST:
                self.ga.polygon(name + "_p4", [(screen_x + 0.45 * GRID_SIZE, screen_y),
                                               (screen_x + 0.4 * GRID_SIZE, screen_y + 0.05 * GRID_SIZE),
                                               (screen_x + 0.4 * GRID_SIZE, screen_y - 0.05 * GRID_SIZE)], EDGE_COLOR,
                                filled=1, smoothed=False)

        text_color = TEXT_COLOR
        if not isObstacle:
            self.ga.text(name + "_txt", (screen_x, screen_y - (GRID_SIZE/6 if isCurrent else 0) ), text_color, valStr, "Courier", -30, "bold", "c")

        if returns_count is not None:
            self.ga.text(name + "_rc", (screen_x-GRID_SIZE/3, screen_y+GRID_SIZE/7), RED_TEXT_COLOR, f"N(s)={int(returns_count)}", "Courier", -20, "bold", "w")
        if returns_sum is not None:
            self.ga.text(name + "_rs", (screen_x-GRID_SIZE/3, screen_y+2*GRID_SIZE/7), RED_TEXT_COLOR, f"S(s)={returns_sum:.2f}", "Courier", -20, "bold", "w")

        # if returns_count is not None:
        #     self.ga.text(name + "_rs", (screen_x, screen_y), text_color, valStr, "Courier", -30, "bold", "c")


    def drawSquareQ(self, name, x, y, qVals, minVal, maxVal, valStrs, bestActions, isCurrent, eligibility_trace=None):

        GRID_SIZE = self.GRID_SIZE
        (screen_x, screen_y) = self.to_screen((x, y))
        center = (screen_x, screen_y)
        nw = (screen_x - 0.5 * GRID_SIZE, screen_y - 0.5 * GRID_SIZE)
        ne = (screen_x + 0.5 * GRID_SIZE, screen_y - 0.5 * GRID_SIZE)
        se = (screen_x + 0.5 * GRID_SIZE, screen_y + 0.5 * GRID_SIZE)
        sw = (screen_x - 0.5 * GRID_SIZE, screen_y + 0.5 * GRID_SIZE)

        n = (screen_x, screen_y - 0.5 * GRID_SIZE + 5)
        s = (screen_x, screen_y + 0.5 * GRID_SIZE - 5)
        w = (screen_x - 0.5 * GRID_SIZE + 5, screen_y)
        e = (screen_x + 0.5 * GRID_SIZE - 5, screen_y)

        actions = qVals.keys()
        for action in actions:
            wedge_color = getColor(qVals[action], minVal, maxVal)
            if action == GridworldMDP.NORTH:
                self.ga.polygon(name + "_s1", (center, nw, ne), wedge_color, filled=1, smoothed=False)
            if action == GridworldMDP.SOUTH:
                self.ga.polygon(name + "_s2", (center, sw, se), wedge_color, filled=1, smoothed=False)
            if action == GridworldMDP.EAST:
                self.ga.polygon(name + "_s3", (center, ne, se), wedge_color, filled=1, smoothed=False)
            if action == GridworldMDP.WEST:
                self.ga.polygon(name + "_s4", (center, nw, sw), wedge_color, filled=1, smoothed=False)

        self.square(name + "_base_square", (screen_x, screen_y),
                    0.5 * GRID_SIZE,
                    color=EDGE_COLOR,
                    filled=0,
                    width=3)

        self.ga.line(name + "_l1", ne, sw, color=EDGE_COLOR)
        self.ga.line(name + "_l2", nw, se, color=EDGE_COLOR)

        for action in actions:
            text_color = TEXT_COLOR
            if qVals[action] < max(qVals.values()): text_color = MUTED_TEXT_COLOR
            valStr = ""
            if action in valStrs:
                valStr = valStrs[action]
            h = -20 # Font size (for reasons).
            if eligibility_trace is not None:
                estr = f'{eligibility_trace[action]:.2f}'
                dh = 0.105 * GRID_SIZE
                ECOL = RED_TEXT_COLOR if  eligibility_trace[action] != 0 else getColor(qVals[action], minVal, maxVal)
                esize = -16
            # dw = 0.095 * GRID_SIZE

            if action == GridworldMDP.NORTH:
                self.ga.text(name + "_txt1", n, text_color, valStr, "Courier", h, "bold", "n")

                if eligibility_trace is not None:
                    self.ga.text(name + "_txt1e", (n[0], n[1]+dh), ECOL, estr, "Courier", esize, "bold", "n")
            if action == GridworldMDP.SOUTH:
                self.ga.text(name + "_txt2", s, text_color, valStr, "Courier", h, "bold", "s")
                if eligibility_trace is not None:
                    self.ga.text(name + "_txt2e", (s[0], s[1]-dh), ECOL, estr, "Courier", esize, "bold", "s")
            if action == GridworldMDP.EAST:
                self.ga.text(name + "_txt3", e, text_color, valStr, "Courier", h, "bold", "e")
                if eligibility_trace is not None:
                    self.ga.text(name + "_txt3e", (e[0], e[1]+dh), ECOL, estr, "Courier", esize, "bold", "e")
            if action == GridworldMDP.WEST:
                self.ga.text(name + "_txt4", w, text_color, valStr, "Courier", h, "bold", "w")
                if eligibility_trace is not None:
                    self.ga.text(name + "_txt4e", (w[0], w[1]+dh), ECOL, estr, "Courier", esize, "bold", "w")


    def square(self, name, pos, size, color, filled, width):
        x, y = pos
        dx, dy = size, size
        return self.ga.polygon(name, [(x - dx, y - dy), (x - dx, y + dy), (x + dx, y + dy), (x + dx, y - dy)],
                               outlineColor=color,
                               fillColor=color, filled=filled, width=width, smoothed=False, closed=True)

    def draw_player(self, position, grid_size):
        # PACMAN_COLOR

        self.ga.circle("pacman", position, PACMAN_SCALE * grid_size * 2,
                       fillColor=PACMAN_COLOR, outlineColor=PACMAN_COLOR,
                       endpoints=getEndpoints(0),
                       width=PACMAN_OUTLINE_WIDTH)

    def to_screen(self, point):
        (gamex, gamey) = point
        x = gamex * self.GRID_SIZE + self.MARGIN
        y = (self.mdp.height - gamey - 1) * self.GRID_SIZE + self.MARGIN
        return (x, y)


def getColor(val, min_value, max_value):
    r = val * 0.65 / min_value if val < 0 and min_value < 0 else 0
    g = val * 0.65 / max_value if val > 0 and max_value > 0 else 0
    return formatColor(r, g, 0)


if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import OpenGridEnvironment
    env = OpenGridEnvironment(render_mode='human')
    # env = BookGridEnvironment()

    from irlc.ex11.q_agent import QAgent
    from irlc import train


    agent = QAgent(env)
    # env = VideoMonitor(env, agent=agent, fps=2000)
    import time

    t = time.time()
    n = 200
    train(env, agent, max_steps=n, num_episodes=10000, verbose=False)
    env.close()

    print("time per step", (time.time() - t) / n)
    # 0.458
    # 0.63
    # 0.61
    # Benchmark over 100 steps: everything else: 0.04 (11 %), setup: 0.25 (72 %), viewer.render: 0.06 (16 %)

# 423, 390, 342 (cur)
