# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import pdb
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import sympy as sym

"""
This is a bunch of pretty awful code to define a map and compute useful quantities like tangents, etc. 
Defining a map is pretty straight forward (it consist of circle archs and lines), but 
don't try to read on.
"""
class SymMap:
    def plot(self, show=False):
        PointAndTangent, TrackLength, extra = self.spec2PointAndTangent(self.spec)
        for i in range(PointAndTangent.shape[0]-1):
            extra_ = extra[i]
            if 'CenterX' in extra_:
                CenterX, CenterY = extra_['CenterX'], extra_['CenterY']
                angle, spanAng = extra_['angle'], extra_['spanAng']
                r = self.spec[i,1]
                direction = 1 if r >= 0 else -1

                # Plotting. Ignore this
                plt.plot(CenterX, CenterY, 'ro')
                tt = np.linspace(angle, angle + direction * spanAng)
                plt.plot(CenterX + np.cos(tt) * np.abs(r), CenterY + np.abs(r) * np.sin(tt), 'r-')

        x, y = PointAndTangent[:, 0], PointAndTangent[:, 1]
        plt.plot(x, y, '.-')
        print(np.sum(np.sum(np.abs(self.PointAndTangent - PointAndTangent))))

        if show:
            plt.show()
    '''
    Format:
        PointAndTangent = [x, 
        y, 
        psi: angle of tangent vector at the last point of segment, 
        total-distance-travelled, 
        segment-length, curvature]
    
    Also creates a symbolic expression to evaluate track position.    
    '''
    def spec2PointAndTangent(self, spec):
        # also create a symbolic piecewise expression to evaluate the curvature as a function of track length location.

        # spec = self.spec
        # PointAndTangent = self.PointAndTangent.copy()
        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        extra = []

        N = spec.shape[0]
        segment_s_cur = 0  # Distance travelled to start of segment (s-coordinate).
        angle_prev = 0  # Angle of the tangent vector at the starting point of the segment
        x_prev, y_prev = 0, 0  # x,y coordinate of last point of previous segment.
        for i in range(N):
            l, r = spec[i,0], spec[i,1]  # Length of segment and radius of curvature
            ang = angle_prev  # Angle of the tangent vector at the starting point of the segment

            if r == 0.0:              # If the current segment is a straight line
                x = x_prev + l * np.cos(ang)  # x coordinate of the last point of the segment
                y = y_prev + l * np.sin(ang)  # y coordinate of the last point of the segment
                psi = ang  # Angle of the tangent vector at the last point of the segment
                curvature = 0
                extra_ = {}
            else:
                direction = 1 if r >= 0 else -1
                CenterX = x_prev + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = y_prev + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
                spanAng = l / np.abs(r)  # Angle spanned by the circle
                psi = wrap_angle(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment
                angleNormal = wrap_angle((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(angle + direction * spanAng)  # x coordinate of the last point of the segment
                y = CenterY + np.abs(r) * np.sin(angle + direction * spanAng)  # y coordinate of the last point of the segment
                curvature = 1/r

                extra_ = {'CenterX': CenterX,
                          'CenterY': CenterY,
                          'angle': angle,
                          'direction': direction,
                          'spanAng': spanAng}

            extra.append(extra_)
            NewLine = np.array([x, y, psi, segment_s_cur, l, curvature])
            PointAndTangent[i, :] = NewLine  # Write the new info
            x_prev, y_prev, angle_prev = PointAndTangent[i, 0], PointAndTangent[i, 1], PointAndTangent[i, 2]
            segment_s_cur += l

        xs = PointAndTangent[-2, 0]
        ys = PointAndTangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

        NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
        PointAndTangent[-1, :] = NewLine
        TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

        return PointAndTangent, TrackLength, extra


    """map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
    """
    def __init__(self, width):
        """Initialization
        width: track width
        Modify the vector spec to change the geometry of the track
        """
        self.width = width
        self.halfWidth = 0.4
        self.slack = 0.45
        lengthCurve = 3.5  # 3.0
        straight = 1.0
        spec = np.array([[1.0, 0],
                         [lengthCurve, lengthCurve / np.pi],
                         # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                         [straight, 0],
                         [lengthCurve / 2, -lengthCurve / np.pi],
                         [straight, 0],
                         [lengthCurve, lengthCurve / np.pi],
                         [lengthCurve / np.pi * 2 + 1.0, 0],
                         [lengthCurve / 2, lengthCurve / np.pi]])


        PointAndTangent, TrackLength, extra = self.spec2PointAndTangent(spec)
        self.PointAndTangent = PointAndTangent
        self.TrackLength = TrackLength
        self.spec = spec


    '''
    Creates a symbolic expression for the curvature
    
def Curvature(s, PointAndTangent):
    """curvature computation
    s: curvilinear abscissa at which the curvature has to be evaluated
    PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
    """
    TrackLength = PointAndTangent[-1,3]+PointAndTangent[-1,4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)

    i = int(np.where(np.squeeze(index))[0])
    curvature = PointAndTangent[i, 5]

    return curvature
    
    '''
    def sym_curvature(self, s):
        s = s - self.TrackLength * sym.floor(s / self.TrackLength)
        n = self.PointAndTangent.shape[0]
        pw = []
        for i in range(n):
            pw.append( (self.PointAndTangent[i,5], s - (self.PointAndTangent[i, 3] + self.PointAndTangent[i, 4]) <= 0) )
        p = sym.Piecewise(*pw)
        return p

    def getGlobalPosition(self, s, ey, epsi=None, vangle_true=None):
        """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """
        # wrap s along the track
        # while (s > self.TrackLength):
        #     s = s - self.TrackLength
        s = np.mod(s, self.TrackLength)

        # Compute the segment in which system is evolving
        PointAndTangent = self.PointAndTangent

        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        dx = np.where(np.squeeze(index))
        if len(dx) < 1:
            a = 234
            raise Exception("bad")
        try:
            i = int(np.where(np.squeeze(index))[0])
        except Exception as e:
            print(e)


        if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = PointAndTangent[i, 0]
            yf = PointAndTangent[i, 1]
            xs = PointAndTangent[i - 1, 0]
            ys = PointAndTangent[i - 1, 1]
            psi = PointAndTangent[i, 2]

            # Compute the segment length
            deltaL = PointAndTangent[i, 4]
            reltaL = s - PointAndTangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
            if epsi is not None:
                vangle = psi + epsi
        else:
            r = 1 / PointAndTangent[i, 5]  # Extract curvature
            ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            direction = 1 if r >= 0 else -1
            # if r >= 0:
            #     direction = 1
            # else:
            #     direction = -1

            CenterX = PointAndTangent[i - 1, 0] + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
            CenterY = PointAndTangent[i - 1, 1] + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

            spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap_angle(direction * np.pi / 2 + ang)

            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(angle + direction * spanAng)  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(angle + direction * spanAng)  # y coordinate of the last point of the segment

            if epsi is not None:
                vangle = epsi + direction * spanAng + PointAndTangent[i - 1, 2]

        if epsi is None:
            return x,y
        else:
            vangle = wrap_angle(vangle)
            if vangle_true is not None:
                vangle_true = wrap_angle(vangle_true)
                # vangle, vangle_true = np.unwrap([vangle, vangle_true])
                if err(vangle - vangle_true, exception=False) > 1e-3:  # debug code
                    print([vangle_true, vangle])
                    print("Bad angle, delta: ", vangle - vangle_true)
                    raise Exception("bad angle")
            return x, y, vangle

    def getLocalPosition(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        PointAndTangent = self.PointAndTangent
        CompletedFlag = 0

        for i in range(0, PointAndTangent.shape[0]):
            if CompletedFlag == 1:
                break

            if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
                # Extract the first final and initial point of the segment
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - PointAndTangent[i - 1, 2]
                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s  = PointAndTangent[i, 3]
                    ey = 0
                    CompletedFlag = 1

                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if np.abs(computeAngle( [x,y] , [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle( [x,y] , [xf, yf], [xs, ys])) <= np.pi/2:
                        v1 = np.array([x,y]) - np.array([xs, ys])
                        angle = computeAngle( [xf,yf] , [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s       = s_local + PointAndTangent[i, 3]
                        ey      = la.norm(v1) * np.sin(angle)

                        if np.abs(ey)<= self.width:
                            CompletedFlag = 1

            else:
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                r = 1 / PointAndTangent[i, 5]  # Extract curvature
                direction = 1 if r >= 0 else -1
                # if r >= 0:
                #     direction = 1
                # else:
                #     direction = -1
                ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

                # Compute the center of the arc
                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = PointAndTangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
                    epsi = psi_unwrap - PointAndTangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2)*np.abs(r)
                        s    = s_local + PointAndTangent[i, 3]
                        ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.width:
                            CompletedFlag = 1

        if epsi>1.0:
            raise Exception("epsi very large; car in wrong direction")
            pdb.set_trace()

        if CompletedFlag == 0:
            s    = 10000
            ey   = 10000
            epsi = 10000

            print("Error!! POINT OUT OF THE TRACK!!!! <==================")
            raise Exception("car outside track")
            # pdb.set_trace()

        return s, ey, epsi, CompletedFlag


    def curvature_and_angle(self, s):
        """curvature computation
        s: curvilinear abscissa at which the curvature has to be evaluated
        PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
        """
        PointAndTangent = self.PointAndTangent
        TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

        # In case on a lap after the first one
        while (s > TrackLength):
            s = s - TrackLength

        # Given s \in [0, TrackLength] compute the curvature
        # Compute the segment in which system is evolving
        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        i = int(np.where(np.squeeze(index))[0])
        curvature = PointAndTangent[i, 5]
        angle = PointAndTangent[i, 4]  # tangent angle of path
        return curvature, angle, i



# ======================================================================================================================
# ======================================================================================================================
# ====================================== Internal utilities functions ==================================================
# ======================================================================================================================
# ======================================================================================================================
def computeAngle(point1, origin, point2):
    # The orientation of this angle matches that of the coordinate system. Tha is why a minus sign is needed
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)

    dot = v1[0] * v2[0] + v1[1] * v2[1]  # dot product between [x1, y1] and [x2, y2]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle

'''
This is used because np.sign(a) return 0 when a=0, which is pretty stupid.
'''
def sign(a):
    return 1 if a >= 0 else -1

def wrap_angle(angle):
    return np.mod(angle+np.pi, 2 * np.pi) - np.pi

'''
Compute difference of these two vectors taking into account the angular component wraps
'''
def xy_diff(x,y):
    dx = x-y
    if len(dx.shape) == 1:
        dx[3] = wrap_angle(dx[3])
    else:
        dx[:,3] = wrap_angle(dx[:,3])
    return dx


def unityTestChangeOfCoordinates(map, ClosedLoopData):
    """For each point in ClosedLoopData change (X, Y) into (s, ey) and back to (X, Y) to check accurancy
    """
    TestResult = 1
    for i in range(0, ClosedLoopData.x.shape[0]):
        xdat = ClosedLoopData.x
        xglobdat = ClosedLoopData.x_glob

        s, ey, epsi, _ = map.getLocalPosition(x=xglobdat[i, 4], y=xglobdat[i, 5], psi=xglobdat[i, 3])
        v1 = np.array([epsi, s, ey])
        v2 = np.array(xdat[i, 3:6])
        x,y,vangle = np.array(map.getGlobalPosition(s=v1[1], ey=v1[2],epsi=v1[0], vangle_true=xglobdat[i,3] ))
        v3 = np.array([ vangle, x, y])
        v4 = np.array( [wrap_angle( xglobdat[i, 3] )] + xglobdat[i, 4:6].tolist() )
        # print(i)
        if np.abs( wrap_angle( xglobdat[i, 3] ) - vangle ) > 0.1:
            print("BAD")
            raise Exception("bad angle test result")

        if np.dot(v3 - v4, v3 - v4) > 0.00000001:
            TestResult = 0
            print("ERROR", v1, v2, v3, v4)
            # pdb.set_trace()
            v1 = np.array(map.getLocalPosition(xglobdat[i, 4], xglobdat[i, 5]))
            v2 = np.array(xdat[i, 4:6])
            v3 = np.array(map.getGlobalPosition(v1[0], v1[1]))
            v4 = np.array([xglobdat[i, 4], xglobdat[i, 5]])
            print(np.dot(v3 - v4, v3 - v4))
            # pdb.set_trace()

    if TestResult == 1:
        print("Change of coordinates test passed!")


def err(x, exception=True, tol=1e-5, message="Error too large!"):
    er = np.mean(np.abs(x).flat)
    if er > tol:
        print(message)
        print(x)
        print(er)
        if exception:
            raise Exception(message)
    return er
