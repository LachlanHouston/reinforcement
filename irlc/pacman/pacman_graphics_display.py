# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
# graphicsDisplay.py
# ------------------
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

# Most code by Dan Klein and John Denero written or rewritten for cs188, UC Berkeley.
# Some code from a Pacman implementation by LiveWires, and used / modified with permission.

# from irlc.utils.gym_graphics_utils import formatColor, GraphicsUtilGym, colorToVector
# from irlc.utils.gym_graphics_utils import formatColor, GraphicsUtilGym, colorToVector
from irlc.utils.graphics_util_pygame import formatColor, GraphicsUtilGym, colorToVector
from irlc.pacman.pacman_utils import Directions
import math
import time

DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = formatColor(0,0,0)
WALL_COLOR = formatColor(0.0/255.0, 51.0/255.0, 255.0/255.0)
INFO_PANE_COLOR = formatColor(.4,.4,0)
SCORE_COLOR = formatColor(.9, .9, .9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = []
GHOST_COLORS.append(formatColor(.9,0,0)) # Red
GHOST_COLORS.append(formatColor(0,.3,.9)) # Blue
GHOST_COLORS.append(formatColor(.98,.41,.07)) # Orange
GHOST_COLORS.append(formatColor(.1,.75,.7)) # Green
GHOST_COLORS.append(formatColor(1.0,0.6,0.0)) # Yellow
GHOST_COLORS.append(formatColor(.4,0.13,0.91)) # Purple

TEAM_COLORS = GHOST_COLORS[:2]

GHOST_SHAPE = [
    ( 0,    0.3 ),
    ( 0.25, 0.75 ),
    ( 0.5,  0.3 ),
    ( 0.75, 0.75 ),
    ( 0.75, -0.5 ),
    ( 0.5,  -0.75 ),
    (-0.5,  -0.75 ),
    (-0.75, -0.5 ),
    (-0.75, 0.75 ),
    (-0.5,  0.3 ),
    (-0.25, 0.75 )
  ]
GHOST_SIZE = 0.65
SCARED_COLOR = formatColor(1,1,1)

GHOST_VEC_COLORS = [colorToVector(gc) for gc in GHOST_COLORS]

PACMAN_COLOR = formatColor(255.0/255.0, 255.0/255.0, 61.0/255)
PACMAN_SCALE = 0.5

# Food
FOOD_COLOR = formatColor(1,1,1)
FOOD_SIZE = 0.1

# Laser
LASER_COLOR = formatColor(1,0,0)
LASER_SIZE = 0.02

# Capsule graphics
CAPSULE_COLOR = formatColor(1,1,1)
CAPSULE_SIZE = 0.25
# Drawing walls
WALL_RADIUS = 0.15

class InfoPane:
    def __init__(self, ga, layout, gridSize):
        self.gridSize = gridSize
        self.width = (layout.width) * gridSize
        self.base = (layout.height + 1) * gridSize
        self.height = INFO_PANE_HEIGHT
        self.fontSize = 24
        self.textColor = PACMAN_COLOR
        self.drawPane()
        self.ga = ga


    def toScreen(self, pos, y = None):
        """
          Translates a point relative from the bottom left of the info pane.
        """
        if y == None:
            x,y = pos
        else:
            x = pos
        x = self.gridSize + x # Margin
        y = self.base + y
        return x,y

    def drawPane(self):
        self.scoreText = {'pos':self.toScreen(0, 0),
                          'color':self.textColor,
                          'contents': "SCORE:    0",
                          'font': "Times",
                          'size': self.fontSize,
                          'style': "bold"}

    def initializeGhostDistances(self, distances):
        self.ghostDistanceText = []
        size = 20
        if self.width < 240:
            size = 12
        if self.width < 160:
            size = 10

        for i, d in enumerate(distances):
            t = {'pos': self.toScreen(self.width/2 + self.width/8 * i, 0),
                 'color': GHOST_COLORS[i+1],
                 'contents': str(d),
                 'font': "Times",
                 'size':size,
                 'style': "bold"}
            self.ghostDistanceText.append(t)

    def updateScore(self, score, method=''):
        self.scoreText['contents'] = "SCORE: % 4d %s" %(score, method)

    def setTeam(self, isBlue):
        txt = "RED TEAM"
        if isBlue: txt = "BLUE TEAM"
        self.teamText = {'pos': self.toScreen(300, 0  ),
                         'color': self.textColor,
                         'contents': txt,
                         'font': "Times",
                         'size': self.fontSize,
                         'style': "bold"}

    def updateGhostDistances(self, distances):
        if len(distances) == 0: return
        self.initializeGhostDistances(distances)

    def master_render(self):
        self.ga.text("master_test", **self.scoreText)
        if hasattr(self, 'teamText'):
            self.ga.text("team_test", **self.teamText)
        if hasattr(self, 'ghostDistanceText'):
            for d in self.ghostDistanceText:
                self.ga.text(f"ghost_distance_text_{d}_", **d)

    def drawGhost(self):
        pass

    def drawPacman(self):
        pass

    def drawWarning(self):
        pass

    def clearIcon(self):
        pass

    def updateMessage(self, message):
        pass

    def clearMessage(self):
        pass


class PacmanGraphics:
    def __init__(self, state, zoom=1.0, frameTime=0.0, capture=False, isBlue=False, method=''):
        self.have_window = 0
        self.currentGhostImages = {}
        self.pacmanImage = None
        self.zoom = zoom
        self.gridSize = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frameTime = frameTime
        # self.visitedlist = None
        # self.ghostbeliefs = None # for the ghost distributions
        self.ga = GraphicsUtilGym()
        # Used to be initialize.
        self.isBlue = isBlue
        self.startGraphics(state)
        self.distributionImages = None  # Initialized lazily
        self.previousState = state
        self.method = method

    # def initialize(self, state, isBlue = False):

    def master_render(self, state, ghostbeliefs=None, visitedlist=None, path=None):
        # self.viewer.geoms = []
        # self.ga.gc.
        # assert  False
        # state = state.data
        # This is completely needless. Just update the things that need to be updated and let everything else be.

        # self.ga.gc.clear()
        self.ga.draw_background()
        if visitedlist is not None:
            self.drawExpandedCells(cells=visitedlist)

        if path is not None:
            # draw the given path.
            path = [self.to_screen(p) for p in path]
            x, y = zip(*path)
            # name = f"render_path"
            for k in range(len(x)-1):
                self.ga.line('asdfasdf', here=(x[k], y[k]), there=(x[k+1], y[k+1]), width=4, color= formatColor(0.5, 0.95, 0.5) )

            # if len(path) > 1:
            #     self.ga.plot(name, x, y, width=4, color=formatColor(0.5, 0.95, 0.5) )

        if ghostbeliefs is not None:
            self.drawDistributions(state.data, ghostbeliefs=ghostbeliefs)

        self.drawStaticObjects(state.data)
        self.drawAgentObjects(state.data)
        self.infoPane.updateScore(state.data.score, self.method)

        if 'ghostDistances' in dir(state.data):
            self.infoPane.updateGhostDistances(state.data.ghostDistances)
        self.infoPane.master_render()
        # self.ga.gc.prune_frame()
        # self.viewer.render()

    def blit(self, render_mode=None):
        return self.ga.blit(render_mode=render_mode)



    def close(self):
        self.ga.close()

    def startGraphics(self, state):
        self.layout = state.data.layout
        # layout = self.layout
        self.width = self.layout.width
        self.height = self.layout.height
        self.make_window(self.width, self.height)
        self.ga.draw_background()
        self.infoPane = InfoPane(ga=self.ga, layout=self.layout, gridSize=self.gridSize)
        self.currentState = self.layout # Unclear.

    def drawDistributions(self, state, ghostbeliefs=None):
        ghostbeliefs = [gb.copy() for gb in ghostbeliefs] # uses a default dict.
        if ghostbeliefs is None or len(ghostbeliefs) == 0:
            return
        walls = state.layout.walls
        for x in range(walls.width):
            for y in range(walls.height):
                weights = [gb[(x,y)] for gb in ghostbeliefs]
                color = [0.0, 0.0, 0.0]
                colors = list(GHOST_VEC_COLORS)[1:]  # With Pacman
                if self.capture: colors = GHOST_VEC_COLORS

                for weight, gcolor in zip(weights, colors):
                    color = [min(1.0, c + 0.95 * g * weight ** .3) for c, g in zip(color, gcolor)]
                color = formatColor(*color)
                ( screen_x, screen_y ) = self.to_screen( (x, y) )
                self.ga.square(f"_belif_{x}_{y}_", (screen_x, screen_y),
                            0.5 * self.gridSize,
                            color = color, # BACKGROUND_COLOR,
                            filled = 1, behind=2)

    def drawStaticObjects(self, state):
        layout = self.layout
        self.drawWalls(layout.walls)
        self.food = self.drawFood(state.food)
        self.capsules = self.drawCapsules(state.capsules)

    def drawAgentObjects(self, state):
        self.agentImages = [] # (agentState, image)
        for index, agent in enumerate(state.agentStates):
            if agent.isPacman:
                image = self.drawPacman(agent, index)
                self.agentImages.append( (agent, image) )
            else:
                image = self.drawGhost(agent, index)
                self.agentImages.append( (agent, image) )


    def update(self, newState, animate=False, ghostbeliefs=None, path=None, visitedlist=None):
        # newState = newState.data
        agentIndex = newState.data._agentMoved
        agentState = newState.data.agentStates[agentIndex]
        # assert False
        if self.agentImages[agentIndex][0].isPacman != agentState.isPacman: self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        if animate:
            if agentState.isPacman:
                self.animatePacman(agentState, prevState, prevImage, state=newState, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
            else:
                self.moveGhost(agentState, agentIndex, prevState, prevImage)

        self.agentImages[agentIndex] = (agentState, prevImage)

        if newState.data._foodEaten != None:
            self.removeFood(newState.data._foodEaten, self.food)
        if newState.data._capsuleEaten != None:
            self.removeCapsule(newState.data._capsuleEaten, self.capsules)

        if 'ghostDistances' in dir(newState):
            self.infoPane.updateGhostDistances(newState.data.ghostDistances)
        self.master_render(newState, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)

    def make_window(self, width, height):
        grid_width = (width-1) * self.gridSize
        grid_height = (height-1) * self.gridSize
        screen_width = 2*self.gridSize + grid_width
        screen_height = 2*self.gridSize + grid_height + INFO_PANE_HEIGHT
        self.viewer = self.ga.begin_graphics(screen_width, screen_height, BACKGROUND_COLOR, "Pacman")

    def drawPacman(self, pacman, index):
        position = self.getPosition(pacman)
        d = pacman.draw_extra['delta_xy']
        position = (position[0] + d[0], position[1]+d[1])
        screen_point = self.to_screen(position)

        if 'endpoints' in pacman.draw_extra:
            endpoints = pacman.draw_extra['endpoints']
        else:
            endpoints = self.getEndpoints(self.getDirection(pacman))

        width = PACMAN_OUTLINE_WIDTH
        outlineColor = PACMAN_COLOR
        fillColor = PACMAN_COLOR

        if self.capture:
            outlineColor = TEAM_COLORS[index % 2]
            fillColor = GHOST_COLORS[index]
            width = PACMAN_CAPTURE_OUTLINE_WIDTH

        return [self.ga.circle("pacman", screen_point, PACMAN_SCALE * self.gridSize,
                       fillColor = fillColor, outlineColor = outlineColor,
                       endpoints = endpoints,
                       width = width)]

    def getEndpoints(self, direction, position=(0,0)):
        x, y = position
        pos = x - int(x) + y - int(y)
        width = 30 + 80 * math.sin(math.pi* pos)

        delta = width / 2
        if (direction == 'West'):
            endpoints = (180+delta, 180-delta)
        elif (direction == 'North'):
            endpoints = (90+delta, 90-delta)
        elif (direction == 'South'):
            endpoints = (270+delta, 270-delta)
        else:
            endpoints = (0+delta, 0-delta)
        return endpoints

    def movePacman(self, position, direction, image,pacman):
        # screenPosition = self.to_screen(position)
        endpoints = self.getEndpoints( direction, position )
        # r = PACMAN_SCALE * self.gridSize
        pacman.draw_extra['endpoints'] = endpoints

    def animatePacman(self, pacman, prevPacman, image, nframe=1, frames=4, state=None, ghostbeliefs=None, path=None, visitedlist=None):
        if self.frameTime < 0:
            print('Press any key to step forward, "q" to play')
        if self.frameTime > 0.01 or self.frameTime < 0:
            fx, fy = self.getPosition(prevPacman)
            px, py = self.getPosition(pacman)
            for nframe in range(1,int(frames) + 1):
                pos = px*nframe/frames + fx*(frames-nframe)/frames, py*nframe/frames + fy*(frames-nframe)/frames
                self.movePacman(pos, self.getDirection(pacman), image, pacman=pacman)
                pacman.draw_extra['delta_xy'] = (pos[0]-px, pos[1]-py)
                time.sleep(self.frameTime/frames)
                self.master_render(state, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
                self.blit(render_mode='human')
        else:
            self.movePacman(self.getPosition(pacman), self.getDirection(pacman), image, pacman=pacman)


    def getGhostColor(self, ghost, ghostIndex):
        if ghost.scaredTimer > 0:
            return SCARED_COLOR
        else:
            return GHOST_COLORS[ghostIndex]

    def drawGhost(self, ghost, agentIndex):
        pos = self.getPosition(ghost)
        dir = self.getDirection(ghost)
        (screen_x, screen_y) = (self.to_screen(pos) )
        coords = []
        for (x, y) in GHOST_SHAPE:
            coords.append((x*self.gridSize*GHOST_SIZE + screen_x, y*self.gridSize*GHOST_SIZE + screen_y))

        colour = self.getGhostColor(ghost, agentIndex)
        name = f"ghost_{agentIndex}_"
        body = self.ga.polygon(name, coords, colour, filled = 1)
        WHITE = formatColor(1.0, 1.0, 1.0)
        BLACK = formatColor(0.0, 0.0, 0.0)

        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        leftEye = self.ga.circle(name +"_s1", (screen_x+self.gridSize*GHOST_SIZE*(-0.3+dx/1.5), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy/1.5)), self.gridSize*GHOST_SIZE*0.2, WHITE, WHITE)
        rightEye = self.ga.circle(name +"_s2",(screen_x+self.gridSize*GHOST_SIZE*(0.3+dx/1.5), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy/1.5)), self.gridSize*GHOST_SIZE*0.2, WHITE, WHITE)
        leftPupil = self.ga.circle(name +"_s3",(screen_x+self.gridSize*GHOST_SIZE*(-0.3+dx), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy)), self.gridSize*GHOST_SIZE*0.08, BLACK, BLACK)
        rightPupil = self.ga.circle(name +"_s4",(screen_x+self.gridSize*GHOST_SIZE*(0.3+dx), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy)), self.gridSize*GHOST_SIZE*0.08, BLACK, BLACK)
        ghostImageParts = []
        ghostImageParts.append(body)
        ghostImageParts.append(leftEye)
        ghostImageParts.append(rightEye)
        ghostImageParts.append(leftPupil)
        ghostImageParts.append(rightPupil)
        return ghostImageParts

    def moveEyes(self, pos, dir, eyes): # does this do anything?
        (screen_x, screen_y) = (self.to_screen(pos) )
        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        self.ga.moveCircle(eyes[0],(screen_x+self.gridSize*GHOST_SIZE*(-0.3+dx/1.5), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy/1.5)), self.gridSize*GHOST_SIZE*0.2)
        self.ga.moveCircle(eyes[1],(screen_x+self.gridSize*GHOST_SIZE*(0.3+dx/1.5), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy/1.5)), self.gridSize*GHOST_SIZE*0.2)
        self.ga.moveCircle(eyes[2],(screen_x+self.gridSize*GHOST_SIZE*(-0.3+dx), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy)), self.gridSize*GHOST_SIZE*0.08)
        self.ga.moveCircle(eyes[3],(screen_x+self.gridSize*GHOST_SIZE*(0.3+dx), screen_y-self.gridSize*GHOST_SIZE*(0.3-dy)), self.gridSize*GHOST_SIZE*0.08)

    def moveGhost(self, ghost, ghostIndex, prevGhost, ghostImageParts):
        old_x, old_y = self.to_screen(self.getPosition(prevGhost))
        new_x, new_y = self.to_screen(self.getPosition(ghost))
        delta = new_x - old_x, new_y - old_y

        if ghost.scaredTimer > 0:
            color = SCARED_COLOR
        else:
            color = GHOST_COLORS[ghostIndex]
        self.ga.edit(ghostImageParts[0], ('fill', color), ('outline', color))
        self.moveEyes(self.getPosition(ghost), self.getDirection(ghost), ghostImageParts[-4:])


    def getPosition(self, agentState):
        if agentState.configuration == None: return (-1000, -1000)
        return agentState.getPosition()

    def getDirection(self, agentState):
        if agentState.configuration == None: return Directions.STOP
        return agentState.configuration.getDirection()

    def to_screen(self, point):
        ( x, y ) = point
        x = (x + 1)*self.gridSize
        y = (self.height  - y)*self.gridSize
        return ( x, y )

    # Fixes some TK issue with off-center circles
    def to_screen2(self, point):
        ( x, y ) = point
        #y = self.height - y
        x = (x + 1)*self.gridSize
        y = (self.height  - y)*self.gridSize
        return ( x, y )

    def drawWalls(self, wallMatrix):
        wallColor = WALL_COLOR

        for xNum, x in enumerate(wallMatrix):
            if self.capture and (xNum * 2) < wallMatrix.width: wallColor = TEAM_COLORS[0]
            if self.capture and (xNum * 2) >= wallMatrix.width: wallColor = TEAM_COLORS[1]

            for yNum, cell in enumerate(x):
                name = f"{xNum}_{yNum}_"
                if cell: # There's a wall here
                    pos = (xNum, yNum)
                    screen = self.to_screen(pos)
                    screen2 = self.to_screen2(pos)

                    # draw each quadrant of the square based on adjacent walls
                    wIsWall = self.isWall(xNum-1, yNum, wallMatrix)
                    eIsWall = self.isWall(xNum+1, yNum, wallMatrix)
                    nIsWall = self.isWall(xNum, yNum+1, wallMatrix)
                    sIsWall = self.isWall(xNum, yNum-1, wallMatrix)
                    nwIsWall = self.isWall(xNum-1, yNum+1, wallMatrix)
                    swIsWall = self.isWall(xNum-1, yNum-1, wallMatrix)
                    neIsWall = self.isWall(xNum+1, yNum+1, wallMatrix)
                    seIsWall = self.isWall(xNum+1, yNum-1, wallMatrix)

                    # NE quadrant
                    if (not nIsWall) and (not eIsWall):
                        # inner circle
                        # self.ga.circle(name + "s1", screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (0,91), 'arc')
                        self.ga.centered_arc(wallColor, screen2, WALL_RADIUS * self.gridSize, 0,90, width=2)

                    if (nIsWall) and (not eIsWall):
                        # vertical line
                        self.ga.line(name + "s2", add(screen, (self.gridSize*WALL_RADIUS, 0)), add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(-0.5)-0)), wallColor)
                    if (not nIsWall) and (eIsWall):
                        # horizontal line
                        self.ga.line(name + "s3", add(screen, (0, self.gridSize*(-1)*WALL_RADIUS)), add(screen, (self.gridSize*0.5+0, self.gridSize*(-1)*WALL_RADIUS)), wallColor)
                    if (nIsWall) and (eIsWall) and (not neIsWall):
                        # outer circle
                        # self.ga.circle(name + "s4", add(screen2, (self.gridSize*2*WALL_RADIUS, self.gridSize*(-2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-1, wallColor, wallColor, (180,271), 'arc')
                        self.ga.centered_arc(wallColor, add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)), WALL_RADIUS * self.gridSize- 0, 180, 270, width=2)
                        # centered_arc(self, color, pos, r, start_angle, stop_angle, width=1)
                        self.ga.line(name + "s5", add(screen, (self.gridSize*2*WALL_RADIUS-0, self.gridSize*(-1)*WALL_RADIUS)), add(screen, (self.gridSize*0.5+0, self.gridSize*(-1)*WALL_RADIUS)), wallColor)
                        self.ga.line(name + "s6", add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(-2)*WALL_RADIUS+0)), add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(-0.5))), wallColor)

                    # NW quadrant
                    if (not nIsWall) and (not wIsWall):
                        # inner circle
                        # self.ga.circle(name + "s8", screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (90,181), 'arc')
                        self.ga.centered_arc(wallColor, screen2, WALL_RADIUS * self.gridSize, 90,180, width=2)

                    if (nIsWall) and (not wIsWall):
                        # vertical line
                        self.ga.line(name + "s10", add(screen, (self.gridSize*(-1)*WALL_RADIUS, 0)), add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(-0.5)-0)), wallColor)
                    if (not nIsWall) and (wIsWall):
                        # horizontal line
                        self.ga.line(name + "s11", add(screen, (0, self.gridSize*(-1)*WALL_RADIUS)), add(screen, (self.gridSize*(-0.5)-0, self.gridSize*(-1)*WALL_RADIUS)), wallColor)
                    if (nIsWall) and (wIsWall) and (not nwIsWall):
                        # outer circle
                        # self.ga.circle(name + "s12", add(screen2, (self.gridSize*(-2)*WALL_RADIUS, self.gridSize*(-2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-1, wallColor, wallColor, (270,361), 'arc')
                        self.ga.centered_arc(wallColor, add(screen2, (self.gridSize*(-2)*WALL_RADIUS, self.gridSize*(-2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize, 270,360, width=2)

                        self.ga.line(name + "s13", add(screen, (self.gridSize*(-2)*WALL_RADIUS+0, self.gridSize*(-1)*WALL_RADIUS)), add(screen, (self.gridSize*(-0.5), self.gridSize*(-1)*WALL_RADIUS)), wallColor)
                        self.ga.line(name + "s14", add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(-2)*WALL_RADIUS+1)), add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(-0.5))), wallColor)

                    # SE quadrant
                    if (not sIsWall) and (not eIsWall):
                        # inner circle
                        # self.ga.circle(name + "s18", screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (270,361), 'arc')
                        self.ga.centered_arc(wallColor, screen2, WALL_RADIUS * self.gridSize, 270,360, width=2)

                    if (sIsWall) and (not eIsWall):
                        # vertical line
                        self.ga.line(name + "s20", add(screen, (self.gridSize*WALL_RADIUS, 0)), add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(0.5)+0)), wallColor)
                    if (not sIsWall) and (eIsWall):
                        # horizontal line
                        self.ga.line(name + "s21", add(screen, (0, self.gridSize*(1)*WALL_RADIUS)), add(screen, (self.gridSize*0.5+1, self.gridSize*(1)*WALL_RADIUS)), wallColor)
                    if (sIsWall) and (eIsWall) and (not seIsWall):
                        # outer circle
                        # self.ga.circle(name + "s22", add(screen2, (self.gridSize*2*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-1, wallColor, wallColor, (90,181), 'arc')
                        self.ga.centered_arc(wallColor, add(screen2, (self.gridSize*2*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-0, 90,180, width=2)
                        self.ga.line(name + "s23", add(screen, (self.gridSize*2*WALL_RADIUS-0, self.gridSize*(1)*WALL_RADIUS)), add(screen, (self.gridSize*0.5, self.gridSize*(1)*WALL_RADIUS)), wallColor)
                        self.ga.line(name + "s24", add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS-0)), add(screen, (self.gridSize*WALL_RADIUS, self.gridSize*(0.5))), wallColor)

                    # SW quadrant
                    if (not sIsWall) and (not wIsWall):
                        # inner circle
                        # self.ga.circle(name + "s30", screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (180,271), 'arc')
                        self.ga.centered_arc(wallColor, screen2, WALL_RADIUS * self.gridSize, 180,270, width=2)
                    if (sIsWall) and (not wIsWall):
                        # vertical line
                        self.ga.line(name + "s31", add(screen, (self.gridSize*(-1)*WALL_RADIUS, 0)), add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(0.5)+1)), wallColor)
                    if (not sIsWall) and (wIsWall):
                        # horizontal line
                        self.ga.line(name + "s32", add(screen, (0, self.gridSize*(1)*WALL_RADIUS)), add(screen, (self.gridSize*(-0.5)-0, self.gridSize*(1)*WALL_RADIUS)), wallColor)
                    if (sIsWall) and (wIsWall) and (not swIsWall):
                        # outer circle
                        # self.ga.circle(name + "s33", add(screen2, (self.gridSize*(-2)*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-1, wallColor, wallColor, (0,91), 'arc')
                        self.ga.centered_arc(wallColor, add(screen2, (self.gridSize*(-2)*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS)), WALL_RADIUS * self.gridSize-0, 0, 90, width=2)
                        self.ga.line(name + "s34", add(screen, (self.gridSize*(-2)*WALL_RADIUS+0, self.gridSize*(1)*WALL_RADIUS)), add(screen, (self.gridSize*(-0.5), self.gridSize*(1)*WALL_RADIUS)), wallColor)
                        self.ga.line(name + "s35", add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(2)*WALL_RADIUS-0)), add(screen, (self.gridSize*(-1)*WALL_RADIUS, self.gridSize*(0.5))), wallColor)

    def isWall(self, x, y, walls):
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def drawFood(self, foodMatrix ):
        foodImages = []
        color = FOOD_COLOR
        for xNum, x in enumerate(foodMatrix):
            if self.capture and (xNum * 2) <= foodMatrix.width: color = TEAM_COLORS[0]
            if self.capture and (xNum * 2) > foodMatrix.width: color = TEAM_COLORS[1]
            imageRow = []
            foodImages.append(imageRow)
            for yNum, cell in enumerate(x):
                name = f"food_{xNum}_{yNum}_"
                if cell: # There's food here
                    screen = self.to_screen((xNum, yNum ))
                    dot = self.ga.circle(name, screen,
                                  FOOD_SIZE * self.gridSize,
                                  outlineColor = color, fillColor = color,
                                  width = 1)
                    imageRow.append(dot)
                else:
                    imageRow.append(None)
        return foodImages

    def drawCapsules(self, capsules ):
        capsuleImages = {}
        for capsule in capsules:
            ( screen_x, screen_y ) = self.to_screen(capsule)
            name = f"capsule_{screen_y}_{screen_x}_"
            dot = self.ga.circle(name, (screen_x, screen_y),
                              CAPSULE_SIZE * self.gridSize,
                              outlineColor = CAPSULE_COLOR,
                              fillColor = CAPSULE_COLOR,
                              width = 1)
            capsuleImages[capsule] = dot
        return capsuleImages

    def removeFood(self, cell, foodImages ):
        x, y = cell

        # remove_from_screen(foodImages[x][y])

    def removeCapsule(self, cell, capsuleImages ):
        x, y = cell
        # remove_from_screen(capsuleImages[(x, y)])

    def drawExpandedCells(self, cells):
        """
        Draws an overlay of expanded grid positions for search agents
        """
        n = float(len(cells))
        baseColor = [1.0, 0.0, 0.0]
        self.clearExpandedCells()
        self.expandedCells = []
        for k, cell in enumerate(cells):
            screenPos = self.to_screen( cell)
            cellColor = formatColor(*[(n-k) * c * .5 / n + .25 for c in baseColor])
            name = f"exp_cell_{screenPos}_"
            block = self.ga.square(name, screenPos,
                     0.5 * self.gridSize,
                     color = cellColor,
                     filled = 1, behind=2)
            self.expandedCells.append(block)
            # if self.frameTime < 0:
            #     refresh()

    def clearExpandedCells(self):
        if 'expandedCells' in dir(self) and len(self.expandedCells) > 0:
            for cell in self.expandedCells:
                pass

class FirstPersonPacmanGraphics(PacmanGraphics):
    def __init__(self, state, zoom = 1.0, showGhosts = True, capture = False, frameTime=0, ghostbeliefs=None):
        PacmanGraphics.__init__(self, state, zoom=zoom, frameTime=frameTime)
        self.showGhosts = showGhosts
        self.capture = capture
        self.ghostbeliefs = ghostbeliefs


    def initialize(self, state, isBlue = False):
        self.isBlue = isBlue
        PacmanGraphics.startGraphics(self, state)
        self.layout = state.layout
        self.previousState = state

    def lookAhead(self, config, state):
        if config.getDirection() == 'Stop':
            return
        else:
            pass
            # Draw relevant ghosts
            allGhosts = state.getGhostStates()
            visibleGhosts = state.getVisibleGhosts()
            for i, ghost in enumerate(allGhosts):
                if ghost in visibleGhosts:
                    self.drawGhost(ghost, i)
                else:
                    self.currentGhostImages[i] = None

    def getGhostColor(self, ghost, ghostIndex):
        return GHOST_COLORS[ghostIndex]

    def getPosition(self, ghostState):
        if not self.showGhosts and not ghostState.isPacman and ghostState.getPosition()[1] > 1:
            return (-1000, -1000)
        else:
            return PacmanGraphics.getPosition(self, ghostState)

def add(x, y):
    return x[0] + y[0], x[1] + y[1]

# 790

if __name__ == '__main__':
    from irlc.pacman.pacman_environment import GymPacmanEnvironment
    env = GymPacmanEnvironment(animate_movement=True, layout='mediumClassic', frame_time=0.0001)
    # env = GymPacmanEnvironment(animate_movement=True, layout='smallClassic')
    from irlc import VideoMonitor, train, Agent
    env = VideoMonitor(env)
    n = 100
    train(env, Agent(env), max_steps=n, num_episodes=1000)
    # everything else: 0.20 (61 %), set up graphics: 0.03 (10 %), rendering: 0.09 (27 %)
