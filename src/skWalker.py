import sys, os
import operator
from math import ceil, sin, cos, pi, atan2
from datetime import datetime, timedelta
import numpy as np

# matplotlib-1.5.1
import matplotlib.pyplot as plt
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

# from scipy.spatial import convex_hull
from scipy.signal import savgol_filter

from shapely.geometry import Point, LineString, Polygon, MultiPoint, box
from shapely.geometry import mapping

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.plot import show

# import geopandas as gpd
# from fiona.crs import from_epsg
# import json

from pyproj import Proj, transform

import _pickle as cPickle

from static import _mdePth, _mssPth, _kmlPth
# dron cam picture width/height factor
from static import _xW, _xH
# dron mean speed (m/s)
from static import _dSpeed
# dron max. flight height gradient
from static import _maxG

from mde import aMDE, xMDE
import static as stt

from levyWalk import Levy
from drovyWalk import Drovy
from bezierWalk import Bezier

# this is given as lat, long !!!!

_blnsLon = [2.812797, 2.823759, 2.825289, 2.816398]
_blnsLat = [41.687333, 41.692006, 41.690913, 41.685079]

_rossLon = [3.1810, 3.2445, 3.2186, 3.1713]
_rossLat = [42.2714, 42.2546, 42.2302, 42.2502]

_buixLon = [2.5193, 2.5884, 2.6130, 2.5189]
_buixLat = [41.7922, 41.8015, 41.7787, 41.7660]

_and1Lon = [1.502343, 1.499691, 1.492348, 1.481126, 1.480323, 1.484901]
_and1Lat = [42.631522, 42.643499, 42.651589, 42.649643, 42.642389, 42.633368]

class savedMission():

	def __init__(self, m, mName = None):

		if mName == None:
			if m.name.find('.txt') != -1:
				self.name = m.name[ :m.name.find('.txt')] + '.pkl'
			else:
				self.name = m.name + '.pkl'
		else:
			self.name = mName + '.pkl'

		self.org = m.org.srs.split('+')[-1].split('=')[-1][:-1]
		self.dst = m.dst.srs.split('+')[-1].split('=')[-1][:-1]
		self.extW = m.extW
		self.safeHeight = m.safeHeight

		self.lon = list(m.dProject(m.pSet, round = 6)[ :-1, 0])
		self.lat = list(m.dProject(m.pSet, round = 6)[ :-1, 1])

		self.path = m.path

	def toMission(self):

		_mission = Mission(name = self.name[:-4], lon = self.lon, lat = self.lat, org = self.org, dst = self.dst, extW = self.extW, safeHeight = self.safeHeight)

		_mission.getMDE()
		_mission.path = self.path
		_mission.ghTracks()

		return _mission

def missionSave(_mission, mName = None):

	sm = savedMission(_mission, mName = mName)
	fName = _mssPth + sm.name

	try:
		f = open(fName, 'wb')
		cPickle.dump(sm, f)
		f.close()
		print('+++ saved to %s' % fName)
	except:
		print(fName)
		print('+++ Mission not saved!')

def missionLoad(fName):

	if fName in os.listdir(_mssPth):
		fName = _mssPth + fName
	elif fName + '.pkl' in os.listdir(_mssPth):
		fName = _mssPth + fName + '.pkl'

	try:
		f = open(fName, 'rb')
		_mission = cPickle.load(f).toMission()
		f.close()
		print('+++ mission load from %s' % fName)
		return _mission
	except:
		print(fName)
		print('+++ Mission not found!')

def debug_pickle(instance):
	"""
	:return: Which attribute from instance can't be pickled?
	"""
	attribute = None

	for k, v in instance.__dict__.items():
		try:
			cPickle.dumps(v)
		except:
			attribute = k
			break

	return attribute

class Mission:

	'''
	Dron Search Mission
	'''

	def __init__(self, name, lon = _and1Lon, lat = _and1Lat, org = 'EPSG:4326', dst = 'EPSG:25831', extW = 500, safeHeight = 100):


		if len(lon) != len(lat):
			return 'Error! len(lon) != len(lat)'

		self.name = name + '.txt'
		self.org = Proj(init=org)
		self.dst = Proj(init=dst)
		self.extW = extW
		self.safeHeight = safeHeight

		self.poly = Polygon(self.project(lon, lat))
		self.pSet = MultiPoint(self.poly.boundary.coords)

		self.path = []
		self.gTrack = []
		self.hTrack = []

		self.getMDE()

	def project(self, X, Y):
		return [transform(self.org, self.dst, x, y) for x, y in zip(X, Y)]

	def dProject(self, P, round = 6):
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

	def upperLeft(self, index = True):
		uL = (self.poly.bounds[0] - self.extW, self.poly.bounds[3] + self.extW)
		if index: uL = self.mde.index(uL[0], uL[1])
		return uL

	def upperRight(self, index = True):
		uR = (self.poly.bounds[2] + self.extW, self.poly.bounds[3] + self.extW)
		if index: uR = self.mde.index(uR[0], uR[1])
		return uR

	def lowerLeft(self, index = True):
		lL = (self.poly.bounds[0] - self.extW, self.poly.bounds[1] - self.extW)
		if index: lL = self.mde.index(lL[0], lL[1])
		return lL

	def getZ(self):

		ul = self.upperLeft()
		ur = self.upperRight()
		ll = self.lowerLeft()

		Z = self.Z[ul[0]:ll[0], ul[1]:ur[1]]
		Z[np.where(Z < 0)] = Z[np.where(Z>=0)].min() -1

		return Z

	def pathLength(self):

		D = [u.distance(v) for u, v in zip(self.path[: -2], self.path[1: -1])]
		return round(sum(D) /1000, 2)

	def backHome(self):
		D = self.path[-2].distance(self.path[-1])
		return round(D /1000, 2)

	def fTime(self):
		fTime = (self.pathLength() +self.backHome()) *1000 /_dSpeed
		return '%02d:%02d' % (fTime //60, fTime%60)

	def get_gTrack(self):

		''' ground track '''

		self.gTrack = []
		self.pSteps = [0]
		for u, v in zip(self.path[ :-1], self.path[1: ]):
			steps = ceil(u.distance(v) /self.s)
			if steps == 1:
				self.gTrack.append(u)
				self.pSteps.append(1 + self.pSteps[-1])
			else:
				X = np.linspace(u.x, v.x, steps)
				Y = np.linspace(u.y, v.y, steps)
				for x, y in zip(X[: -1], Y[: -1]):
					self.gTrack.append(Point(x, y))
				self.pSteps.append(steps -1 + self.pSteps[-1])

		# add last point of self.path
		self.gTrack.append(self.path[-1])
		# self.pSteps[-1] += 1

	def hPrf(self, pSet):
		Z = self.Z[self.mde.index([p.x for p in pSet], [p.y for p in pSet])]
		Z[np.where(Z < 0)[0]] = 0
		return Z

	def get_hTrack(self, pSet, maxG = _maxG, window = 49, order = 2):

		L = [u.distance(v) for u, v in zip(self.gTrack[ :-1], self.gTrack[1: ])]
		H = self.hPrf(pSet) + self.safeHeight
		I = np.argsort(H)

		Z = np.zeros(len(self.gTrack))
		for i in I[::-1]:
			# highest left
			l = max(i -1, 0)
			while l > 1 and Z[l] == 0: l -= 1
			# highest right
			r = min(i +1, len(H)-1)
			while r < len(H) -1 and Z[r] == 0: r += 1
			# assign highest minus slope
			Z[i] = max(H[i], (Z[l] -sum(L[l: i]) *maxG), (Z[r] -sum(L[i: r]) *maxG))

		''' height-track: smoothing-savitzky-golay-filter '''
		Z = savgol_filter(Z, window , order)

		return Z

	''' internal plotting functions '''

	def __pSet_plot(self, axs, pSet, color, size):

		XY = np.array([self.mde.index(p.x, p.y) for p in pSet])
		ul = self.upperLeft()
		XY[:, 0] -= ul[0]
		XY[:, 1] -= ul[1]

		axs.plot(XY[:, 1], XY[:, 0], 'o', color = color, markersize = size)
		axs.plot(XY[:, 1], XY[:, 0], 'k--', linewidth = 0.2)
		if len(pSet) != len(self.pSet):
			axs.plot(XY[0, 1], XY[0, 0], 'o', color = 'r', markersize = 2 *size)

	def __iPth_plot(self, axs, pSet, color, size):

		def imgPoly(p, h, b):
			w = h *_xW
			h = h *_xH
			R = np.array([[cos(b), sin(b)], [-sin(b), cos(b)]])
			ul = np.array([p.x, p.y]) + np.dot(R, np.array([-w, h]))
			ur = np.array([p.x, p.y]) + np.dot(R, np.array([w, h]))
			lr = np.array([p.x, p.y]) + np.dot(R, np.array([w, -h]))
			ll = np.array([p.x, p.y]) + np.dot(R, np.array([-w, -h]))
			return Polygon([ul, ur, lr, ll])

		ul = self.upperLeft()

		# path headings
		B = [np.arctan2((v.x -u.x), (v.y -u.y)) for u, v in zip(pSet[1: ], pSet[: -1])]

		# path image half-size (w/2, h/2)
		XY = np.array([self.mde.index(p.x, p.y) for p in pSet])
		XY[:, 0] -= ul[0]
		XY[:, 1] -= ul[1]
		# S = [(self.Z[x, y] *_xW, self.Z[x, y] *_xH) for x, y in XY]

		# path images' polygons
		# 1. with pSet = self.path
		# K = [imgPoly(p, (self.fZ[s] -self.gZ[s]), b) for p, s, b in zip(pSet, self.pSteps, B)]
		# 2. with pSet = self.gTrack
		K = [imgPoly(p, (f -g), b) for p, f, g, b in zip(pSet, self.fZ, self.gZ, B)]

		axs.plot(XY[:, 1], XY[:, 0], 'o', color = color, markersize = size)
		axs.plot(XY[:, 1], XY[:, 0], 'k--', linewidth = 0.2)
		for img in K:
			V = np.array([self.mde.index(x, y) for x, y in img.boundary.coords])
			V[:, 0] -= ul[0]
			V[:, 1] -= ul[1]
			#axs.plot(V[:, 1], V[:, 0], '--', color='m', linewidth = 0.2)

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

		# uR = self.upperRight(index = False)
		# uR = transform(self.dst, self.org, uR[0], uR[1])
		# lL = self.lowerLeft(index = False)
		# lL = transform(self.dst, self.org, lL[0], lL[1])
		#
		# xDlta = (uR[0] -lL[0]) /(len(axs.get_xticks()) -2)
		# xlocs = [0] +[round(lL[0] +(x *xDlta), 4) for x in range(len(axs.get_xticks()))]
		# print(xlocs)
		# yDlta = (lL[1] -uR[1]) /(len(axs.get_yticks()) -2)
		# ylocs = [0] +[round(uR[1] +(y *yDlta), 4) for y in range(len(axs.get_yticks()))]
		# print(ylocs)
		axs.set_xticklabels([])
		axs.set_yticklabels([])

	''' end internal '''

	def show(self, amplify = 1):

		Z = self.getZ()

		if not len(self.path):

			figsize = (Z.shape[1]/100 +amplify, Z.shape[0]/100 +amplify)
			fig, axs = plt.subplots(figsize = figsize)
			axs.grid(True)

			axs.imshow(Z, cmap = 'terrain')
			self.__pSet_plot(axs, self.pSet, 'blue', 4)
			self.__pSet_annotate(axs)
			self.__axs_annotate(axs)

		else:

			#figsize = (Z.shape[1]/100 +1, Z.shape[0]/100 +5)
			fig = plt.figure(figsize = (12, 9))
			ax1 = plt.subplot2grid((2, 2), (0, 0))
			ax2 = plt.subplot2grid((2, 2), (0, 1))
			ax3 = plt.subplot2grid((2, 2), (1, 0), colspan = 3)

			ax1.grid(False)
			ax2.grid(False)
			ax3.grid(True)

			ax1.imshow(Z, cmap = 'terrain')
			self.__pSet_plot(ax1, self.pSet, 'blue', 4)
			self.__pSet_plot(ax1, self.path, 'gray', 2)
			self.__pSet_annotate(ax1)
			self.__axs_annotate(ax1)

			ax2.imshow(Z, cmap = 'terrain')
			self.__pSet_plot(ax2, self.pSet, 'blue', 4)
			self.__iPth_plot(ax2, self.gTrack, 'gray', 2)
			self.__axs_annotate(ax2)

			ax3.plot(self.pSteps, self.hPrf(self.path), 'k--', lw = 0.3)
			ax3.plot(range(len(self.gZ)), self.gZ, 'r', lw = 0.6)
			ax3.plot(range(len(self.fZ)), self.fZ, 'b', lw = 0.6)

			for i in range(len(self.hG)):
				if self.hG[i] < 0.15:
					ax3.plot((i+i+1)/2, self.fZ[i], 'o', color = 'b', markersize = 0.3)
				else:
					ax3.plot((i+i+1)/2, self.fZ[i], 'o', color = 'r', markersize = 1.5)

			# self.get_wPoints_qs()
			# ax3.plot(self.wayP[:, 0], self.wayP[:, 1], 'g', lw = 0.8)

			plt.title('area(km2): %6.4f,  length(km): %4.2f (+ %4.2f),  time(mm:ss): %s,  pathPoints: %5d, wayPoints: %5d' % (round(self.poly.area /10**6, 4), self.pathLength(), self.backHome(), self.fTime(), len(self.path), len(self.fZ)))

		plt.tight_layout()
		plt.show()

	def getSize(self):

		ll = Point(self.poly.bounds[0], self.poly.bounds[1])
		ul = Point(self.poly.bounds[0], self.poly.bounds[3])
		lr = Point(self.poly.bounds[2], self.poly.bounds[1])
		ur = Point(self.poly.bounds[2], self.poly.bounds[3])

		xSize = max(ll.distance(lr), ul.distance(ur))
		ySize = max(ll.distance(ul), lr.distance(ur))

		return (xSize, ySize)

	def ghTracks(self, maxG = _maxG, window = 49, order = 2):

		self.get_gTrack()
		# ground track heights
		self.gZ = self.hPrf(self.gTrack)
		# flight track heights
		self.fZ = self.get_hTrack(self.gTrack, maxG = maxG, window = 49, order = 2)
		# check height gradient
		self.hG = np.abs(np.diff(self.fZ))/[u.distance(v) for u, v in zip(self.gTrack[ :-1], self.gTrack[1: ])]

	def chkCrv(self): stt.chkCrv(self)

	def drovy(self, startLoc, max_steps = 725, min_flight = 15, max_flight = 15000, mu_flight = 2.2, min_radius = 40, max_radius = 4000, mu_radius = 2.0, fflight = 200, sflight = False, show = True):

		_drovy = Drovy(maxs = maxs, minf = minf, maxf = maxf, mu = mu)
		self.path = _drovy.walk(self.poly, startLoc = startLoc, minR = minR, maxR = maxR, muR = muR, fflight = fflight, sflight = sflight)
		self.path.append(self.path[0])

		self.ghTracks()
		if show: self.show()

	def spiral(self, arcStep = 45, xStep = 1, yStep = 1, show = True):

		xSize, ySize = self.getSize()
		xStep = xSize / 1000 * xStep
		yStep = ySize / 1000 * yStep

		loc = self.poly.centroid
		self.path = []
		i = 0
		while loc.within(self.poly):
			t = i * 2 * np.pi / arcStep
			x = loc.x + xStep * (t * np.sin(t))
			y = loc.y + yStep * (t * np.cos(t))
			loc = Point(x, y)
			self.path.append(loc)
			i += 1
		self.path.append(self.poly.centroid)

		self.ghTracks()
		if show: self.show()

	def sweep(self, startLoc, width = 100, show = True):

		V = [p for p in self.pSet[ :-1]]

		# reorder V from closest vertex to the starting location
		startLoc = Point(transform(self.org, self.dst, startLoc[0], startLoc[1]))
		startVtx = np.argmin([p.distance(startLoc) for p in V])
		vtxOrder = [i for i in range(startVtx, len(V))] + [i for i in range(startVtx)]
		V = [self.pSet[v] for v in vtxOrder]

		c = self.poly.centroid
		D = [p.distance(c) for p in V]

		nSteps = min(D) /width

		dX = [(p.x -c.x) /nSteps for p in V]
		dY = [(p.y -c.y) /nSteps for p in V]

		self.path = [startLoc]
		V = [Point(p.x -dx /2, p.y -dy /2) for p, dx, dy in zip(V, dX, dY)]
		for p in V: self.path.append(p)
		for s in range(int(nSteps)):
			V = [Point(p.x-dx, p.y-dy) for p, dx, dy in zip(V, dX, dY)]
			for p in V: self.path.append(p)

		self.path.append(startLoc)

		self.ghTracks()
		if show: self.show()

	def levy(self, maxs = 450, minf = 30, maxf = 45000, mu = 2.0, startLoc = 0, show = True):

		_levy = Levy(maxs=maxs, minf=minf, maxf=maxf, mu=mu)
		self.path = _levy.walk(self.poly, startLoc = startLoc)
		self.path.append(self.path[0])

		self.ghTracks()
		if show: self.show()

	def bezier(self, maxs = 450, minf = 50, maxf = 45000, mu = 2.0, startLoc = 0, show = True):

		_bezier = Bezier(maxs = maxs, minf = minf, maxf = maxf, mu = mu)
		self.path = _bezier.walk(self.poly, startLoc = startLoc)
		self.path.append(self.path[0])

		self.ghTracks()
		if show: self.show()

	def write(self):

		LL = self.dProject(self.gTrack, round=6)  # longitud, latitude

		# get home-relative heights
		H = self.fZ - self.gZ[0]

		f = open(_mssPth + self.name, 'w')
		'''head'''
		f.write('QGC WPL 110'),
		f.write('\n')
		'''home'''
		# 0	1	0	16	0	0	0	0	42.367328	1.775515	1052.000000	1
		f.write('%5d ' % 0),
		f.write('%1d ' % 1),
		f.write('%1d ' % 0),
		f.write('%3d ' % 16),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (LL[0, 1], LL[0, 0], self.gZ[0]))
		f.write('%1d ' % 1),
		f.write('\n')
		'''take-off'''
		# 1	0	3	22	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	50.000000	1
		f.write('%5d ' % 1),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 22),fflight
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (.0, .0, 50.00))
		f.write('%1d ' % 1),
		f.write('\n')
		'''switch flight-mode: 223 airplane-mode'''
		# 2	0	3	223	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1
		f.write('%5d ' % 2),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 223),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (.0, .0, .0))
		f.write('%1d ' % 1),
		f.write('\n')
		'''start image capture'''
		# 3	0	3	206	0.000000	0.000000	0.000000	0.000000	42.367153	1.775097	120.000000	1
		f.write('%5d ' % 3),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 206),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (.0, .0, .0)) # lon, lat needed ???
		f.write('%1d ' % 1),
		f.write('\n')
		'''waypoints'''
		# 4	0	3	16	0.000000	0.000000	0.000000	0.000000	42.367026	1.774807	120.000000	1
		for i in range(1, len(self.gTrack[:-1])):
			f.write('%5d ' % (i+3)),
			f.write('%1d ' % 0),
			f.write('%1d ' % 3),
			f.write('%3d ' % 16),
			f.write('%8.6f ' % .0),
			f.write('%8.6f ' % .0),
			f.write('%8.6f ' % .0),
			f.write('%8.6f ' % .0),
			f.write('%9.6f %9.6f %9.2f ' % (LL[i, 1], LL[i, 0], H[i]))
			f.write('%1d ' % 1),
			f.write('\n')
		'''Loiter turns: first'''
		# 12	0	3	18	1.000000	0.000000	0.000000	0.000000	42.366376	1.776180	80.000000	1
		f.write('%5d ' % (i+4)),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 18),
		f.write('%8.6f ' % 1.0),    # number of turns
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (LL[-1, 1], LL[-1, 0], 80.0))
		f.write('%1d ' % 1),
		f.write('\n')
		'''Loiter turns: second'''
		# 13    0	3	18	3.000000	0.000000	0.000000	0.000000	42.366947	1.775537	40.000000	1
		f.write('%5d ' % (i+5)),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 18),
		f.write('%8.6f ' % 3.0),    # number of turn
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (LL[-1, 1], LL[-1, 0], 40.0))
		f.write('%1d ' % 1),
		f.write('\n')
		'''Return to Launch'''
		# 14	0	3	20	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1
		f.write('%5d ' % (i+6)),
		f.write('%1d ' % 0),
		f.write('%1d ' % 3),
		f.write('%3d ' % 20),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%8.6f ' % .0),
		f.write('%9.6f %9.6f %9.2f ' % (.0, .0, .0)) # lon, lat needed ???
		f.write('%1d ' % 1),
		f.write('\n')

		f.close()
