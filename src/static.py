from math import pi, atan2, sin, cos, ceil, sqrt
from datetime import datetime, timedelta
import numpy as np
from shapely.geometry import Point

# matplotlib-1.5.1
import matplotlib.pyplot as plt
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')
from matplotlib.collections import LineCollection

from scipy.signal import savgol_filter

_mdePth = '/home/jgarriga/walkers/mde/'
_mssPth = '/home/jgarriga/walkers/mss/'
_kmlPth = '/home/jgarriga/walkers/kml/'

# _mdePth = '/home/bernatc/walkers/mde/'
# _mssPth = '/home/bernatc/walkers/mss/'
# _kmlPth = '/home/bernatc/walkers/kml/'

# dron mean speed (m/s)
_dSpeed = 15.0
# dron max. flight height gradient
_maxG = 0.10
# dron min. curvature radius (m)
_rMin = 50
# dron min. arcStep for spiral definition (rad)
_arcStep = np.pi /4

# dron cam picture width factor (sinus of cam opening (half) angle)
_xW = sin(43 * pi/180)
# dron cam picture height factor (assume height is 0.75 width, i.e h/w = 3000/4000 pixels)
_xH = _xW * 0.75

# -----------------------------------------------------------------------------
# static functions
# -----------------------------------------------------------------------------

def turn2(phi1, phi2):
	t = phi2 - phi1
	if abs(t) > pi:
		if t < 0: t = -2 *pi -t
		else: t = 2 *pi -t
	return t

def turn3(u, v, w):
	t = atan2((w.x - v.x), (w.y - v.y)) - atan2((v.x - u.x), (v.y - u.y))
	if abs(t) > pi:
		if t < 0: t = -2 *pi -t
		else: t = 2 *pi -t
	return t

def curv3(u, v, w, res = 0.5):
	# define a Bezier curve
	Bezier = []
	steps = max(ceil((u.distance(v) + v.distance(w)) /res), 3)
	for t in np.linspace(0, 1.0, steps):
		x = (1- t)**2 *u.x +2 *(1 -t) *t *v.x +t**2 *w.x
		y = (1- t)**2 *u.y +2 *(1 -t) *t *v.y +t**2 *w.y
		Bezier.append(Point(x, y))
	# curvature
	c3 = np.mean([turn3(u, v, w) /v.distance(w) for u, v, w in zip(Bezier[:-2], Bezier[1:-1], Bezier[2:])])
	# It's not quite clear (??) but given _lMin = 50 and _arcStep = np.pi /4, the measured curvature is ~0.01 (half of the theoretical value 0.02)
	return c3 *2.0

def Rpath(self, dSpeed = _dSpeed):

	lonLat = self.dProject(self.path, round = 6)
	secnds = [u.distance(v)/dSpeed for u, v in zip(self.path[:-1], self.path[1:])]
	dTimes = [datetime.now() +timedelta(seconds = t) for t in np.cumsum(secnds)]

	return np.array([[lon, lat] for lon, lat in lonLat])

def Rdata(self):

	flights = [u.distance(v) for u, v in zip(self.path[:-1], self.path[1:])]
	turns = [turn3(u, v, w) for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:])]

	return np.array([[f, t] for f, t, in zip(flights, turns)])


# -----------------------------------------------------------------------------
# path checks
# -----------------------------------------------------------------------------

def pathCheck(self, res = 1.0):

	# flights
	F = [u.distance(v) for u, v in zip(self.path[:-1], self.path[1:])]
	# turns
	T = [turn3(u, v, w) for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:])]
	# curvature
	C = [curv3(u, v, w, res = res) for u, v, w in zip(self.path[:-2], self.path[1:-1], self.path[2:])]
	# curvature radius
	R = [min(1/abs(c), 200) if c > 0 else 200 for c in C]

	fig = plt.figure(figsize = (12, 6))
	ax1 = plt.subplot(221)
	ax2 = plt.subplot(222)
	ax3 = plt.subplot(223)
	ax4 = plt.subplot(224)

	ax1.hist(F, bins = 40, range = (0, np.max(F)))
	ax2.hist(T, bins = 36, range = (-np.pi, np.pi))
	ax3.plot(C)
	ax4.plot(R)
	ax4.axhline(y = 50, xmin = 0, xmax = len(C), color = 'r')

	plt.show()

def getCrv(path, res = 1.0):
	C = [curv3(u, v, w, res = res) for u, v, w in zip(path[:-2], path[1:-1], path[2:])]
	return np.abs([0] +C +[0])
	def chkgt(self, a = 0, b = 100):

		Z = self.getZ()
		self.get_gTrack()

		fig = plt.figure(figsize = (12, 9))
		axs = plt.subplot()
		axs.grid(False)

		axs.imshow(Z, cmap = 'terrain')
		self.__pSet_plot(axs, self.pSet, 'blue', 4)
		# self.__pSet_plot(axs, self.gTrack[a: b], 'gray', 2)
		self.__pSet_plot(axs, self.path[a: b], 'black', 3)
		self.__pSet_annotate(axs)
		self.__axs_annotate(axs)

		plt.tight_layout()
		plt.show()

def chkCrv(self, res = 1.0):

	fig = plt.figure(figsize=(13, 6))
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)

	# plot polygon
	ax1.plot([loc.x for loc in self.pSet], [loc.y for loc in self.pSet], 'k-', linewidth = 0.5)
	# plot path
	ax1.plot([loc.x for loc in self.path], [loc.y for loc in self.path], 'k--', linewidth = 0.5)

	# curvature
	C = getCrv(self.path)
	# colors
	colors = plt.get_cmap('jet')(C /np.max(C))
	# set color-alpha based on C
	colors[:, 3] = (C +0.1) /(np.max(C) +0.1)
	for i in np.argsort(C):
		loc = self.path[i]
		ax1.plot(loc.x, loc.y, 'o', color = colors[i], markersize = 4.0)

	# radius > 200 --> c = 1/200 = 0.005
	# radius < 10 --> c = 1/10 = 0.01
	ax2.hist([1/c for c in C if 0.005 < c < 0.1], bins = 40)
	ax2.axvline(50, color = 'r')

	plt.show()

def chkSlope(self, maxSlope = 0.15):

	self.get_gTrack()
	# ground track heights
	self.gZ = self.hPrf(self.gTrack)
	# flight track heights
	self.fZ = self.get_hTrack()

	D = [u.distance(v) for u, v in zip(self.gTrack[: -1], self.gTrack[1: ])]
	S = np.diff(self.fZ)

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

def chkSpiral(mu = 2.0, minf = 15, maxf = 15000, minR = 5.0, maxR = 2500, flights = 4, sflight = False, res = 1.0):

	# flight range
	frange = maxf**(1-mu) - minf**(1-mu)
	# flight power
	fpower = 1/(1-mu)

	# Drovy walk parameters: radius range
	Rrange = maxR**(1-mu) - minR**(1-mu)
	# radius power
	Rpower = 1/(1-mu)

	# spiral resolution
	arcStep = 30

	loc, t, radInf = Point(0, 0), 0, 0
	pth = [loc]
	for s in range(flights):

		# flight distance
		flight = (frange *np.random.uniform(0, 1) + minf**(1 -mu))**fpower
		# straight flight
		if sflight: radInf = 1 if radInf == 0 else 0
		# flight radius
		if radInf:
			radius = np.inf
		else:
			radius = (Rrange *np.random.uniform(0, 1) + minR**(1 -mu))**Rpower
		# turn sign
		crvSgn = -1 if np.random.uniform(0, 1) < 0.5 else 1

		# flight starting direction
		moved, steps, t = 0, 0, t %(2.0 *np.pi)
		while moved < flight:

			steps += 1
			if radInf:
				x = loc.x + minf * np.sin(t)
				y = loc.y + minf * np.cos(t)
			else:
				t += 2 *np.pi /arcStep *crvSgn
				x = loc.x + radius * (steps * np.sin(t))
				y = loc.y + radius * (steps * np.cos(t))

			loc = Point(x, y)
			moved += pth[-1].distance(loc)
			pth.append(loc)

		print('+++', s, flight, radius, crvSgn, steps, moved)

	# curvature
	C = getCrv(pth, res = res)

	fig, axs = plt.subplots(figsize=(5, 5))

	X = [loc.x for loc in pth]
	Y = [loc.y for loc in pth]
	axs.plot(X, Y, 'k--', linewidth = 0.5)
	axs.set_xlim(min(min(X), min(Y)), max(max(X), max(Y)))
	axs.set_ylim(min(min(X), min(Y)), max(max(X), max(Y)))

	# colors
	colors = plt.get_cmap('jet')(C /np.max(C))
	# set color-alpha based on C
	colors[:, 3] = (C +0.1) /(np.max(C) +0.1)
	for i in np.argsort(C):
		loc = pth[i]
		plt.plot(loc.x, loc.y, 'o', color = colors[i], markersize = 4.0)

	# radius < 50 -->  1/r = c > 0.02
	print([1/c for c in C if c > 0.02])

	plt.show()

def chkSlp(self, maxG = 0.50):

	L = [u.distance(v) for u, v in zip(self.gTrack[ :-1], self.gTrack[1: ])]
	Z = self.hPrf(self.gTrack)

	def grdG(i):
		# gGi = 0
		# if (Z[i] < Z[i -1]) and (Z[i] < Z[i +1]):
		# 	gGi = (Z[i +1] -Z[i]) /L[i] - (Z[i] -Z[i -1]) /L[i -1]
		gGi = (Z[i +1] -Z[i]) /L[i] - (Z[i] -Z[i -1]) /L[i -1]
		return gGi

	while True:

		G = np.abs(np.diff(Z)) /L
		if not len(np.where(np.abs(np.diff(Z))/L > maxG)[0]): break

		dG = np.array([grdG(i) for i in range(1, len(Z) -1)], dtype = 'float')
		i = np.argmax(dG) +1
		if dG[i -1] < maxG: break

		print('+++ %04d %8.2f %8.2f %8.2f ... %8.2f - %8.2f = %8.2f ... cost %8.2f ' % (i, Z[i-1], Z[i], Z[i +1], (Z[i +1] -Z[i]) /L[i], (Z[i] -Z[i -1]) /L[i -1], dG[i -1], np.sum(G)), end = '\r')

		j = i + np.argmin(Z[i-1:i+2]) -1
		Z[j] += 1
		# if j == i -1:
		# 	Z[j] = (Z[j +1] +Z[j]) /2
		# elif j == i:
		# 	Z[j] = (Z[j +1] +Z[j -1]) /2
		# elif j == i +1:
		# 	Z[j] = (Z[j -1] +Z[j]) /2

	print()

	plt.plot(range(len(Z)), self.hPrf(self.gTrack))
	plt.plot(range(len(Z)), Z)
	plt.show()

	return Z

def safe_hPrf(self, maxg = 0.12, window = 49, order = 2):

	L = [u.distance(v) for u, v in zip(self.gTrack[ :-1], self.gTrack[1: ])]
	H = self.hPrf(self.gTrack) + self.safeHeight
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
		Z[i] = max(H[i], (Z[l] -sum(L[l: i]) *maxg), (Z[r] -sum(L[i: r]) *maxg))

	Z = savgol_filter(Z, window , order)
	plt.plot(range(len(Z)), self.hPrf(self.gTrack))
	plt.plot(range(len(Z)), Z)
	plt.show()

	return Z

def safe_hPrf_2(self, maxg = 0.15, s = 1, window = 49, order = 2):

	L = [u.distance(v) for u, v in zip(self.gTrack[ :-1], self.gTrack[1: ])]
	H = self.hPrf(self.gTrack) + self.safeHeight
	I = np.argmax(H)

	Z = np.zeros(len(self.gTrack))
	Z[I] = H[I]

	for i in range(I -1, -1, -1):
		Z[i] = max(H[i], Z[i +s] -(sum(L[i:min(I+1, i +s +1)]) *maxg))

	for i in range(I +1, len(H)):
		Z[i] = max(H[i], Z[i -s] -(sum(L[max(I, i -s):i]) *maxg))

	Z = savgol_filter(Z, window , order)
	# plt.plot(range(len(Z)), self.hPrf(self.gTrack), 'r')
	# plt.plot(range(len(Z)), Z, 'b')
	# plt.plot(range(len(H)), savgol_filter(H, window, order), 'g--')
	#
	# plt.show()

	return Z
