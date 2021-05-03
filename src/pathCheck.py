from math import pi, atan2, sqrt

import numpy as np

# matplotlib-1.5.1
import matplotlib.pyplot as plt
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

from shapely.geometry import Point

from static import turn3, curv3

def pathCheck(self):

    # flights
    F = [u.distance(v) for u, v in zip(self.path[:-1], self.path[1:])]
    # turns
    T = [turn3(u, v, w) for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:])]
    # curvature
    # C = []
    # for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:]):
    #     for c in curv3(u, v, w): C.append(c)
    C = [curv3(u, v, w) for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:])]

    fig = plt.figure(figsize = (12, 6))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.hist(F, bins = 40, range = (0, np.max(F)))
    ax2.hist(T, bins = 36, range = (-np.pi, np.pi))
    ax3.plot(C)
    ax4.plot([min(1/abs(c), 200) if c > 0 else 200 for c in C])
    ax4.axhline(y = 50, xmin = 0, xmax = len(C), color = 'r')

    plt.show()


def pathShow(self):

    def pSet_plot(self, axs, pSet, color, size, lw):

        XY = np.array([self.mde.index(p.x, p.y) for p in pSet])
        ul = self.upperLeft()
        XY[:, 0] -= ul[0]
        XY[:, 1] -= ul[1]

        axs.plot(XY[:, 1], XY[:, 0], 'o', color = color, markersize = size)
        axs.plot(XY[:, 1], XY[:, 0], 'k--', linewidth = lw)
        if len(pSet) != len(self.pSet):
            axs.plot(XY[0, 1], XY[0, 0], 'o', color = 'r', markersize = 2 *size)

    def pSet_annotate(self, axs):

        XY = np.array([self.mde.index(p.x, p.y) for p in self.pSet[:-1]])
        ul = self.upperLeft()
        XY[:, 0] -= ul[0]
        XY[:, 1] -= ul[1]

        LL = self.dProject(self.pSet[:-1])  # longitude, latitude

        for (x, y), ll in zip(XY, LL):
            # axs transposed !!! xy => (y, x)
            axs.annotate(str(ll), xy = (y, x), textcoords = 'data', va = "bottom", ha = "center", size = 9)

    def axs_annotate(self, axs):

        ul = self.upperLeft()
        xlocs = axs.get_xticks() +ul[0]
        ylocs = axs.get_yticks() +ul[1]
        lonlat = self.dProject([Point(self.mde.transform *(x, y)) for x, y in zip(xlocs, ylocs)], round = 3)
        axs.set_xticklabels(lonlat[:, 0])
        axs.set_yticklabels(lonlat[:, 1])

    Z = self.getZ()
    self.get_gTrack()

    fig = plt.figure(figsize = (12, 9))
    axs = plt.subplot()
    axs.grid(False)

    axs.imshow(Z, cmap = 'terrain')
    pSet_plot(self, axs, self.pSet, 'blue', 4, 0.2)
    pSet_plot(self, axs, self.gTrack, 'gray', 2, 0.2)
    pSet_plot(self, axs, self.path, 'black', 3, 0.3)
    pSet_annotate(self, axs)
    axs_annotate(self, axs)

    plt.tight_layout()
    plt.show()
