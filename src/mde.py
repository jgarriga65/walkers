
import sys, os
from math import sin, cos, pi
import numpy as np

# matplotlib-1.5.1
import matplotlib.pyplot as plt
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

from shapely.geometry import Point, LineString, Polygon, MultiPoint, box
from shapely.geometry import mapping

import rasterio
from rasterio.merge import merge
# from rasterio.mask import mask
from rasterio.plot import show

# import geopandas as gpd
# from fiona.crs import from_epsg
# import json

from static import _mdePth

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

	def show_old(self):
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
			self.mdeTbl[self.layout[0, 0]].show(axi = axs[0])

		plt.show()
