
from math import sqrt, pi, sin, cos, asin, atan2
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, LineString, LinearRing, Polygon

_rMin = 50
_dMin = 2 *_rMin
_lMin = sqrt(2) *_rMin
_phiMax = 60 *pi /180

_Crcl = np.array([(_rMin *sin(phi), _rMin *cos(phi)) for phi in np.arange(0, 2 *pi, pi /16)])

def extend_line(a, b, d):
    alpha = atan2(b.x -a.x, b.y -a.y)
    return Point(a.x + d *sin(alpha), a.y + d *cos(alpha))

def turn(pSet3):
    i, j, k = pSet3
    hij = atan2(j.x -i.x, j.y -i.y)
    hjk = atan2(k.x -j.x, k.y -j.y)
    phi = hjk - hij
    if phi > pi: phi -= 2 *pi
    elif phi < -pi: phi += 2 *pi
    return phi

def smthCrv(path, show = True):

    smthPth = [path[0]]

    for i in range(1, len(path)-2):
        smthSeg = segment([smthPth[-1]] +path[i: i+3], 1).sPth
        for loc in smthSeg: smthPth.append(loc)

    if show:

        fig, axs = plt.subplots()
        axs.set_aspect('equal')
        x, y = LineString(path).xy
        axs.plot(x, y, 'k--', linewidth = 0.3)
        x, y = LineString(smthPth).xy
        # axs.plot(loc.x, loc.y, 'o', color = 'b', markersize = 2)
        axs.plot(x, y, 'b--', linewidth = 0.3)

        plt.show()


class segment():

    def __init__(self, path, i, show = False):

        if i > len(path)-2:
            print('+++ Error')
            return

        self.id = i

        self.aloc = path[i -1]
        self.iloc = path[i]
        self.jloc = path[i +1]
        self.kloc = path[i +2]

        self.pSet = path[i -1: i +3]
        self.lin3 = LineString(path[i -1: i +2])
        self.line = LineString(path[i -1: i +3])

        self.iRing = LinearRing(_Crcl + np.array(self.iloc.coords))
        self.jRing = LinearRing(_Crcl + np.array(self.jloc.coords))

        self.c = Point()
        self.trn1 = turn(self.pSet[ :3])
        self.trn2 = turn(self.pSet[1: ])

        self.sPth = [self.aloc, self.iloc]

        # self.chkphiMax()

        if abs(self.trn1) > _phiMax or (self.iloc.distance(self.jloc) < _rMin and abs(self.trn2) > _phiMax):
            if not self.iRing.intersects(self.jRing): self.findc1()
            else: self.findc2()
            self.smooth()

        if show: self.plot()

    def chkphiMax(self, show = False):

        # find secant line at _rMin
        s = LineString(self.iRing.intersection(self.lin3))
        # find _rMin center
        c = extend_line(self.iloc, s.interpolate(s.length /2), _rMin)
        cRing = LinearRing(_Crcl + np.array(c.coords))

        I = list(self.iRing.intersection(cRing))

        phiMax = abs(turn([I[0], self.iloc, I[1]]))
        print('+++ phiMax: %5.4f, %5.2f' % (phiMax, phiMax *180/pi))

        if show:

            fig, axs = plt.subplots()
            axs.set_aspect('equal')

            for loc, c in zip(list(self.pSet), ['k', 'b', 'b', 'b']):
                axs.plot(loc.x, loc.y, 'o', color = c, markersize = 2)

            x, y = self.line.xy
            axs.plot(x, y, 'k--', linewidth = 0.3)
            x, y = self.iRing.xy
            axs.plot(x, y, 'k--', linewidth = 0.2)
            x, y = cRing.xy
            axs.plot(x, y, 'r--', linewidth = 0.2)

            for p in I:
                x, y = LineString([self.iloc, p]).xy
                axs.plot(x, y, 'k--', linewidth = 0.2)

            plt.show()

    def findc1(self):
        # find secant line at _rMin
        s = LineString(self.iRing.intersection(self.lin3))
        # find _rMin center
        self.c = extend_line(self.iloc, s.interpolate(s.length /2), _rMin)

    def findc2(self):

        ijInter = list(self.iRing.intersection(self.jRing))
        self.c = LineString(ijInter).interpolate(_rMin)

        aijPoly = Polygon([(p.x, p.y) for p in self.pSet[ :3]])
        if not aijPoly.contains(self.c):
            ijInter.reverse()
            self.c = LineString(ijInter).interpolate(_rMin)

    def smooth(self):

        self.cRing = LinearRing(_Crcl + np.array(self.c.coords))

        p0 = self.aloc
        p1 = extend_line(self.c, self.iloc, _rMin)
        p2 = list(self.cRing.intersection(self.iRing))[0]
        p3 = list(self.cRing.intersection(self.iRing))[1]

        sPthLst = []
        if not LineString([p0, p1, p2]).crosses(self.lin3):
            sPthLst.append([p0, p1, p2])
        if not LineString([p0, p1, p3]).crosses(self.lin3):
            sPthLst.append([p0, p1, p3])
        if not LineString([p0, p2, p1]).crosses(self.lin3):
            sPthLst.append([p0, p2, p1])
        if not LineString([p0, p3, p1]).crosses(self.lin3):
            sPthLst.append([p0, p3, p1])

        if len(sPthLst):
            sPthTrn = [abs(turn([a, b, c])) for a, b, c in sPthLst]
            # if min(sPthTrn) < _phiMax:
            self.sPth = sPthLst[sPthTrn.index(min(sPthTrn))]

    def plot(self):

        print('+++ seg.%03d' % self.id)

        self.fig, self.axs = plt.subplots()
        self.axs.set_aspect('equal')

        for loc, c in zip(list(self.pSet), ['k', 'b', 'b', 'b']):
            self.axs.plot(loc.x, loc.y, 'o', color = c, markersize = 2)

        x, y = self.line.xy
        self.axs.plot(x, y, 'k--', linewidth = 0.3)

        for x, y in [self.iRing.xy, self.jRing.xy]:
            self.axs.plot(x, y, 'k--', linewidth = 0.2)

        if len(self.c.coords):
            x, y = self.c.x, self.c.y
            self.axs.plot(x, y, 'o', color = 'r', markersize = 2.5)
            x, y = self.cRing.xy
            self.axs.plot(x, y, 'r--', linewidth = 0.2)

        x, y = LineString(self.sPth).xy
        self.axs.plot(x, y, 'g--', linewidth = 0.5)

        plt.show()
