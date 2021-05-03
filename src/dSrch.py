
import sys, os
from math import ceil

import numpy as np

# matplotlib-1.5.1
import matplotlib.pyplot as plt
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

from scipy.spatial import ConvexHull
from shapely.geometry import Point, LineString, Polygon, MultiPoint, box
from shapely.geometry import mapping

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show

import geopandas as gpd
from fiona.crs import from_epsg
import json

from pyproj import Proj, transform

_mdePth = '/home/jgarriga/dSrch/mde/'
_mssPth = '/home/jgarriga/dSrch/mss/'
_kmlPth = '/home/jgarriga/dSrch/kml/'

class aMDE(dict):

    def __init__(self):

        for fName in os.listdir(_mdePth):
            mdeKey = int(fName.split('f')[1][:4])
            self[mdeKey] = MDE(fName)

        nrow, ncol = 0, -1
        mdeAnt = None
        for mdeKey in sorted(self.keys()):
            if mdeAnt != None and mdeKey != mdeAnt +1 :
                ncol = -1
                nrow += 1
            mdeAnt = mdeKey
            ncol += 1
            self[mdeKey].cell = (nrow, ncol)

        self.getLayout()

    def info(self):
        for mdeKey in sorted(self.keys()): self[mdeKey]

    def getLayout(self):

        minRow = min([mde.cell[0] for mde in self.values()])
        maxRow = max([mde.cell[0] for mde in self.values()])
        minCol = min([mde.cell[1] for mde in self.values()])
        maxCol = max([mde.cell[1] for mde in self.values()])

        nRows = maxRow -minRow +1
        nCols = maxCol -minCol +1
        self.layout = np.zeros(nRows *nCols).reshape(nRows, nCols)

        for mdeKey in self.keys():
            i = self[mdeKey].cell[0] - minRow
            j = self[mdeKey].cell[1] - minCol
            self.layout[i, j] = mdeKey


class MDE:

    '''
    Digital Elevation Model (MDE) Single sheet
    Institut Cartogr\'afic i Geol\`ogic de Catalunya (ICGC)
    http://www.icgc.cat/Administracio-i-empresa/Descarregues/Elevacions/Model-d-Elevacions-del-Terreny-de-Catalunya

    Sistema geodèsic de referència: ETRS89 (European Terrestrial Reference System 1989)
    el.lipsoide: GRS80 (Geodètic Reference System 1980)

    Sistema de representació plana: Universal Transversa Mercator (UTM)
    Ordre de coordenades: Easting(X), Northing(Y)
    tot Catalunya inclosa en el Fus: 31

    UTM to longlat conversion for ETRS89 UTM zone 31 N ==> EPSG 25831
    proj4 = '+proj=utm +zone=31 +ellps=GRS80 +units=m +no_defs'

    +++ Nom dels fitxers de dades:
    met15v20//<format-versió>//f//<id-full>//<codi-subconjunt>//<marc-referència>//r//<revisió-correcció>//.txt
     on:

    <format-versió> està format per 3 caràcters, els dos primers indiquen el format del conjunt de
    dades, as per ASCII GRID d’ESRI, mentre que el tercer és un dígit que indica canvis en la
    distribució del producte lligats al format d’implementació;

    <id-full> és l’identificador seqüencial MTN 1:50.000 expressat amb 4 dígits, emplenat amb
    zeros per l’esquerra, més un caràcter per al tipus de full (A normal, B bis, C tris);

    <codi-subconjunt> el constitueixen 2 caràcters que prenen diferents valors segons el format de
    les dades: en el cas del MET-15 pren el valor mr;

    <marc-referència> dígit que diferencia marcs de referència. Pren el valor 0 per les dades en
    ED50 UTM 31 N i 1 per aquelles que són en ETRS89 UTM 31 N;

    <revisió-correcció> els primers 2 dígits indiquen el número de vegades que s’han actualitzat les
    dades; el tercer dígit indica si les dades d’una revisió s’han modificat una vegada distribuïdes.
    '''

    def __init__(self, fName):

        self.fName = fName
        self.fNmbr = int(fName.split('f')[1][:4])


        self.getInfo()

    def __str__(self):

        return '%03i - %02i:%02i ... %12.1f %12.1f ... %6i %6i' % (self.fNmbr, self.cell[0], self.cell[1], self.lon, self.lat, self.cols, self.rows)

    def getInfo(self):
        with open(_mdePth + self.fName) as f:
            self.cols = int(f.readline().split()[1])
            self.rows = int(f.readline().split()[1])
            self.lon = float(f.readline().split()[1])  # lower left center
            self.lat = float(f.readline().split()[1])  # lower left center
            self.size = float(f.readline().split()[1])
            self.null = float(f.readline().split()[1])
            self.loc = Point(self.lon, self.lat)

    def getData(self):
        with open(_mdePth + self.fName) as f:
            for i in range(6): f.readline()
            z = np.array(f.read().split(), dtype='float')
            return z.reshape(self.rows, self.cols)

    def show(self, axi = None):
        '''
        MDE contour plot
        '''
        if axi:
            axs = axi
        else:
            fig = plt.figure()
            axs = fig.add_subplot(111)
        axs.grid(True)
        #axs.set_facecolor('#DDDDDD') # grey background for null values
        axs.set_xlim(0, self.cols)
        axs.set_ylim(0, self.rows)
        x, y = np.mgrid[0:self.cols:1, 0:self.rows:1]
        z = np.transpose(self.getData()[::-1])
        axs.contourf(x, y, z, np.linspace(0, z.max(), 16), alpha=0.8, cmap = 'terrain')
        if not axi:
            plt.show()

class xMDE(MDE):

    '''
    Digital Elevation Model (MDE) Multiple sheets
    Institut Cartogr\'afic i Geol\`ogic de Catalunya (ICGC)
    '''

    def __init__(self, a, b, mdeTbl):

        if a not in mdeTbl.keys():
            return '%i not in mdeTbl' % a
        if b not in mdeTbl.keys():
            return '%i not in mdeTbl' % b

        a = np.where(mdeTbl.layout == a)
        b = np.where(mdeTbl.layout == b)

        ai = min(int(a[0]), int(b[0]))
        bi = max(int(a[0]), int(b[0])) +1
        aj = min(int(a[1]), int(b[1]))
        bj = max(int(a[1]), int(b[1])) +1
        self.layout = mdeTbl.layout[ai:bi, aj:bj]

        self.mdek = [mdek for mdeRow in self.layout for mdek in mdeRow if mdek != 0.]

        mdeLst = [_mdePth + mdeTbl[k].fName for k in self.mdek if k!= 0.]
        Z, self.trans = merge([rasterio.open(mde) for mde in mdeLst])
        self.Z = Z[0, :, :]

        # self.locs = np.array([[loc.x, loc.y] for loc in [self.mdeCat[mdeKey].loc for mdeKey in self.mdek]])
        # hull = ConvexHull(self.locs)
        # self.poly = Polygon([tuple(self.locs[vrtx]) for vrtx in hull.vertices])
        #
        # self.path = []

    def intraPointDistance(self):
        P = MultiPoint(self.poly.boundary.coords)
        pqDist = [(i, p.distance(q)) for i, p in enumerate(P) for q in P[i:]]
        return pqDist

    def getBBox(self):
        return box(self.poly.bounds[0], self.poly.bounds[1], self.poly.bounds[2], self.poly.bounds[3])

    def getSize(self):

        ll = Point(self.poly.bounds[0], self.poly.bounds[1])
        ul = Point(self.poly.bounds[0], self.poly.bounds[3])
        lr = Point(self.poly.bounds[2], self.poly.bounds[1])
        ur = Point(self.poly.bounds[2], self.poly.bounds[3])

        xSize = max(ll.distance(lr), ul.distance(ur))
        ySize = max(ll.distance(ul), lr.distance(ur))

        return (xSize, ySize)

    def show(self):
        fig = plt.figure()
        axs = fig.add_subplot(111)
        axs.grid(True)
        #axs.set_facecolor('#DDDDDD') # grey background for null values
        axs.set_xlim(0, self.Z.shape[1])
        axs.set_ylim(0, self.Z.shape[0])
        x, y = np.mgrid[0:self.Z.shape[1]:1, 0:self.Z.shape[0]:1]
        z = np.transpose(self.Z[::-1])
        axs.contourf(x, y, z, np.linspace(0, z.max(), 16), alpha=0.8, cmap='terrain')
        plt.show()

    def plot(self):
        fig, axs = plt.subplots()
        axs.set_aspect('equal')
        # axs.scatter(self.locs[:, 0], self.locs[:, 1], c = 'k', s = 10.0, alpha = 0.5)
        # bBox = self.getBBox()
        # axs.plot([p[0] for p in bBox.boundary.coords], [p[1] for p in bBox.boundary.coords], 'k-', lw = 0.2)
        # rect = self.poly.minimum_rotated_rectangle
        # axs.plot([p[0] for p in rect.boundary.coords], [p[1] for p in rect.boundary.coords], 'r-', lw = 0.2)
        # vrtx = self.poly.exterior.coords.xy
        # axs.plot(vrtx[0], vrtx[1], 'o')
        # hull = self.poly.convex_hull.exterior.coords.xy
        # axs.plot(hull[0], hull[1], 'k-', lw = 0.5)
        # cntr = self.poly.centroid.coords.xy
        # axs.plot(cntr[0], cntr[1], 'ro')
        if len(self.path):
            axs.plot([loc.x for loc in self.path], [loc.y for loc in self.path], 'g-', lw = 0.4)
        plt.show()

    def sweep(self, width = 100, show = True):
        pass


    def elipse(self, show = True):

        loc = self.poly.centroid

        xStep = min((loc.x - self.poly.bounds[0]), (self.poly.bounds[2] - loc.x))
        yStep = min((loc.y - self.poly.bounds[1]), (self.poly.bounds[3] - loc.y))

        self.path = [loc]
        for i in range(180):
            t = i * 2 * np.pi / 180
            x = loc.x + xStep * np.cos(t)
            y = loc.y + yStep * np.sin(t)
            self.path.append(Point(x, y))
            i += 1
        self.path.append(self.poly.centroid)

        print(len(self.path))
        if show: self.plot()

    def show(self):
        '''
        xMDE contour plot
        '''
        if self.layout.size > 1:
            fig, axs = plt.subplots(self.layout.shape[0], self.layout.shape[1])
            for row in axs:
                for ax in row:
                    ax.set_xticks([])
                    ax.set_yticks([])
            for i, mdeRow in enumerate(self.layout):
                for j, mdekey in enumerate(mdeRow):
                    if mdekey: self.mdeTbl[mdekey].show(axi = axs[i, j])
        else:
            fig, axs = plt.subplots(111)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            self.mdeTbl[self.layout[0,0]].show(axi = axs[0])

        plt.show()

    def clip(self, poly = None):
        pass


# this is given as lat, long !!!!

_blnsLon = [2.812797, 2.823759, 2.825289, 2.816398]
_blnsLat = [41.687333, 41.692006, 41.690913, 41.685079]

_rossLon = [3.1810, 3.2445, 3.2186, 3.1713]
_rossLat = [42.2714, 42.2546, 42.2302, 42.2502]

_buixLon = [2.5193, 2.5884, 2.6130, 2.5189]
_buixLat = [41.7922, 41.8015, 41.7787, 41.7660]

_and1Lon = [1.6517, 1.6584, 1.7336, 1.7310]
_and1Lat = [42.4826, 42.4978, 42.4795, 42.4623]

class mss:

    '''
    Dron Search Mission
    '''

    def __init__(self, name = 'Buix', lon = _buixLon, lat = _buixLat, org = 'EPSG:4326', dst = 'EPSG:25831', extW = 500):


        if len(lon) != len(lat):
            return 'Error! len(lon) != len(lat)'

        self.name = name + '.mss'
        self.org = Proj(init=org)
        self.dst = Proj(init=dst)
        self.extW = extW

        self.poly = Polygon(self.project(lon, lat))
        self.pSet = MultiPoint(self.poly.boundary.coords)

        self.path = []
        self.getMDE()

    def project(self, X, Y):
        return [transform(self.org, self.dst, x, y) for x, y in zip(X, Y)]

    def dProject(self, P, round = 4):
        return np.round([transform(self.dst, self.org, p.x, p.y) for p in P], round)

    def getMDE(self):

        mdeTbl = aMDE()
        mdeLst = []
        for mde in mdeTbl.values():
            r = rasterio.open(_mdePth + mde.fName)
            bbx = box(r.bounds.left, r.bounds.bottom, r.bounds.right, r.bounds.top)
            if np.any([bbx.contains(p) for p in self.pSet]):
                mdeLst.append(mde.fNmbr)

        mssMDE = xMDE(min(mdeLst), max(mdeLst), mdeTbl)

        self.mde = rasterio.open('/tmp/xMDE.txt', 'w', driver = 'AAIGrid', height = mssMDE.Z.shape[0], width = mssMDE.Z.shape[1], count = 1, dtype = mssMDE.Z.dtype, crs = None, transform = mssMDE.trans)

        # self.mde.write(mssMDE.Z, 1)
        self.Z = mssMDE.Z
        self.s = 15

    def upperLeft(self):
        return self.mde.index(self.poly.bounds[0] - self.extW, self.poly.bounds[3] + self.extW)

    def upperRight(self):
        return self.mde.index(self.poly.bounds[2] + self.extW, self.poly.bounds[3] + self.extW)

    def lowerLeft(self):
        return self.mde.index(self.poly.bounds[0] - self.extW, self.poly.bounds[1] - self.extW)

    def getZ(self):

        ul = self.upperLeft()
        ur = self.upperRight()
        ll = self.lowerLeft()

        Z = self.Z[ul[0]:ll[0], ul[1]:ur[1]]
        Z[np.where(Z < 0)] = Z[np.where(Z>=0)].min() -1

        return Z

    def pathLength(self):

        D = [u.distance(v) for u, v in zip(self.path[1: ], self.path[ :-1])]
        return round(sum(D) /1000, 2)

    def getWayPoints(self):

        self.wayPoints = []
        for u, v in zip(self.path[ :-1], self.path[1: ]):
            steps = ceil(u.distance(v) /self.s)
            X = np.linspace(u.x, v.x, steps)
            Y = np.linspace(u.y, v.y, steps)
            for x, y in zip(X, Y):
                self.wayPoints.append(Point(x, y))
            # Z = self.Z[self.mde.index(X, Y)]
            # for x, y, z in zip(X, Y, Z):
            #     self.wayPoints.append(Point(x, y, z))


    def show(self):

        def pSet_plot(self, axs, pSet, color, size):

            XY = np.array([self.mde.index(p.x, p.y) for p in pSet])
            ul = self.upperLeft()
            XY[:, 0] -= ul[0]
            XY[:, 1] -= ul[1]

            axs.plot(XY[:, 1], XY[:, 0], 'o', color = color, markersize = size)
            axs.plot(XY[:, 1], XY[:, 0], 'k--', linewidth = 0.2)

        def hPrf_plot(self, axs, pSet, color):

            H = self.Z[self.mde.index([p.x for p in pSet], [p.y for p in pSet])]
            axs.plot(range(len(pSet)), H, color)

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

        if not len(self.path):

            figsize = (Z.shape[1]/100 +1, Z.shape[0]/100 +1)
            fig, axs = plt.subplots(figsize = figsize)
            axs.grid(True)

            axs.imshow(Z, cmap = 'terrain')
            pSet_plot(self, axs, self.pSet, 'blue', 4)
            pSet_annotate(self, axs)
            axs_annotate(self, axs)

        else:

            #figsize = (Z.shape[1]/100 +1, Z.shape[0]/100 +5)
            fig = plt.figure(figsize = (12, 9))
            ax1 = plt.subplot2grid((2, 2), (0, 0))
            ax2 = plt.subplot2grid((2, 2), (0, 1))
            ax3 = plt.subplot2grid((2, 2), (1, 0), colspan = 2)

            ax1.grid(True)
            ax2.grid(False)
            ax3.grid(True)

            ax1.imshow(Z, cmap = 'terrain')
            pSet_plot(self, ax1, self.pSet, 'blue', 4)
            pSet_annotate(self, ax1)
            axs_annotate(self, ax1)

            ax2.imshow(Z, cmap = 'terrain')
            pSet_plot(self, ax2, self.pSet, 'blue', 4)
            pSet_plot(self, ax2, self.path, 'gray', 2)
            ax2.set_xticks([])
            ax2.set_yticks([])

            hPrf_plot(self, ax3, self.path, 'b')
            hPrf_plot(self, ax3.twiny(), self.wayPoints, 'r')
            # ax4 = ax3.twiny()
            # ax4.plot(range(len(self.wayPoints)), [p.z for p in self.wayPoints], 'r-')
            plt.title('area(km2): %6.4f,  length(km): %4.2f,  wayPoints: %5d' % (round(self.poly.area /10**6, 4), self.pathLength(), len(self.path)))

        plt.tight_layout()
        plt.show()

    def spiral(self, arcStep = 45, xStep = 1, yStep = 1, show = True):

        xSize, ySize = self.getSize()
        xStep = xSize / 1000 * xStep
        yStep = ySize / 1000 * yStep

        loc = self.poly.centroid
        self.path = []
        i = 0
        while loc.within(self.poly):
            self.path.append(loc)
            t = i * 2 * np.pi / arcStep
            x = loc.x + xStep * (t * np.sin(t))
            y = loc.y + yStep * (t * np.cos(t))
            loc = Point(x, y)
            i += 1
        self.path.append(self.poly.centroid)

        print(len(self.path))
        if show: self.show()

    def getSize(self):

        ll = Point(self.poly.bounds[0], self.poly.bounds[1])
        ul = Point(self.poly.bounds[0], self.poly.bounds[3])
        lr = Point(self.poly.bounds[2], self.poly.bounds[1])
        ur = Point(self.poly.bounds[2], self.poly.bounds[3])

        xSize = max(ll.distance(lr), ul.distance(ur))
        ySize = max(ll.distance(ul), lr.distance(ur))

        return (xSize, ySize)

    def sweep(self, width = 100, show = True):

        c = self.poly.centroid

        V = [p for p in self.pSet[ :-1]]
        D = [p.distance(c) for p in V]

        nSteps = min(D) /width

        dX = [(p.x -c.x) /nSteps for p in V]
        dY = [(p.y -c.y) /nSteps for p in V]

        self.path = V
        for s in range(int(nSteps)):
            V = [Point(p.x-dx, p.y-dy) for p, dx, dy in zip(V, dX, dY)]
            for p in V: self.path.append(p)

        self.path.append(self.pSet[0])

        self.getWayPoints()

        print(len(self.path), len(self.wayPoints))
        if show: self.show()
