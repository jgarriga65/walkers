import numpy as np

from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from shapely.geometry import Point

_dSpeed = 15

def get_clstLbls1(self, dSpeed = _dSpeed, eps = 23, min = 3):

    # spatial distance matrix
    XY = np.array([[p.x, p.y] for p in self.gTrack])
    D1 = squareform(pdist(XY))

    # temporal distance component
    L = np.array([u.distance(v) for u, v in zip(self.gTrack[:-1], self.gTrack[1:])])

    sqSpeed = dSpeed**2

    for i in range(D1.shape[0]-1):
        Lij = 0
        for j in range(i+1, D1.shape[1]):
            Lij += L[j-1]
            D1[i, j] = sqrt(D1[i, j]**2 + Lij**2 /sqSpeed)
            D1[j, i] = D1[i, j]

    K1 = DBSCAN(eps = eps, min_samples = min, metric = 'precomputed')

    return K1.fit_predict(D1)

def get_clstLbls2(self, dSpeed = _dSpeed, eps = 60, min = 3):

    D2 = np.array([[p.x, p.y, t*_dSpeed] for t, p in enumerate(self.gTrack)])
    K2 = DBSCAN(eps = eps, min_samples = min).fit(D2)

    return K2.labels_

def shw_clstLbls(self, L):

    def __pSet_annotate(self, axs):

        XY = np.array([self.mde.index(p.x, p.y) for p in self.pSet[:-1]])
        ul = self.upperLeft()
        XY[:, 0] -= ul[0]
        XY[:, 1] -= ul[1]

        LL = self.dProject(self.pSet[:-1])  # longitude, latitude

        for (x, y), ll in zip(XY, LL):
            # axs transposed !!! xy => (y, x)
            axs.annotate(str(ll), xy = (y, x), textcoords = 'data', va = "bottom", ha = "center", size = 9)

    def __axs_annotate(self, axs):

        ul = self.upperLeft()
        xlocs = axs.get_xticks() +ul[0]
        ylocs = axs.get_yticks() +ul[1]
        lonlat = self.dProject([Point(self.mde.transform *(x, y)) for x, y in zip(xlocs, ylocs)], round = 3)
        axs.set_xticklabels(lonlat[:, 0])
        axs.set_yticklabels(lonlat[:, 1])

    def __cLbl_plot(self, axs, pSet, size):

        XY = np.array([self.mde.index(p.x, p.y) for p in pSet])
        ul = self.upperLeft()
        XY[:, 0] -= ul[0]
        XY[:, 1] -= ul[1]

        norm = colors.Normalize(L.min(), L.max())
        lbls = cm.jet(norm(L))

        for (y, x), l in zip(XY, lbls):
            axs.plot(x, y, 'o', color = l, markersize = size)

        axs.plot(XY[:, 1], XY[:, 0], 'k--', linewidth = 0.2)

    fig, axs = plt.subplots()
    axs.grid(True)

    axs.imshow(self.getZ(), cmap = 'terrain')

    # __pSet_plot(self, axs, self.path, 2)
    __cLbl_plot(self, axs, self.gTrack, 4)

    __pSet_annotate(self, axs)
    __axs_annotate(self, axs)

    plt.tight_layout()
    plt.show()
