from math import cos, sin, sqrt, pi, log10, asin, ceil, atan2
import numpy as np
from scipy.stats import beta

from shapely.geometry import Point, Polygon

from static import turn2
from levyWalk import Levy

_rMin = 50
_phiMax = np.pi /2

class Bezier(Levy):

	def walk(self, poly, startLoc = 0):

		# walk path initialization
		self.steps, self.flights, self.moved  = 0, 0, 0

		# search starting point
		if startLoc < 0:
			loc = poly.centroid
		else:
			loc = Point(poly.boundary.coords.xy[0][startLoc], poly.boundary.coords.xy[1][startLoc])

		wlkPth = [loc]
		maxdst = self.step *self.maxs

		newloc = 0
		while self.moved < maxdst:

			# current path location
			x0, y0 = wlkPth[-1].x, wlkPth[-1].y

			# if last move didn't work, resample f1, phi1
			# assert poly.contains(Point(x1, y1))
			if not newloc:
				while True:
					f1 = (self.frange *np.random.uniform(0, 1) + self.step**(1 -self.mu))**self.fpower
					phi1 = np.random.uniform(0, (2 *np.pi), 1)
					x1, y1 = x0 +f1 *cos(phi1), y0 +f1 *sin(phi1)
					if np: break

			# check curvatureRadius < 50
			curvRad = 0
			while curvRad < 50:
				f2 = (self.frange *np.random.uniform(0, 1) + self.step**(1 -self.mu))**self.fpower
				phi2 = np.random.uniform(0, (2 *np.pi), 1)
				curvRad = (f1 +f2) /turn2(phi1, phi2)

			# next path locations
			x1, y1 = x0 +f1 *cos(phi1), y0 +f1 *sin(phi1)
			x2, y2 = x1 +f2 *cos(phi2), y1 +f2 *sin(phi2)

			# flight move
			# steps = ceil((f1 +f2) /self.step) /2 ??? why /2
			steps = ceil((f1 +f2) /self.step)
			newloc = 0
			for t in np.linspace(0, 0.5, steps)[1: ]:
				x = (1- t)**2 *x0 +2 *(1 -t) *t *x1 +t**2 *x2
				y = (1- t)**2 *y0 +2 *(1 -t) *t *y1 +t**2 *y2
				if not poly.contains(Point(x, y)): break
				newloc = Point(x, y)
				self.moved += self.step
				self.steps += 1

			# update current position
			if newloc:
				wlkPth.append(newloc)
				f1 = sqrt((x2 -newloc.x)**2 +(y2 -newloc.y)**2)
				phi1 = atan2(x2 -newloc.x, y2 -newloc.y) -pi /2

			# check max distance
			if self.steps >= self.maxs: break
			# update moved distance
			self.flights += 1

		return wlkPth
