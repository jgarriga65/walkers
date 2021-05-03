from math import pi, sin, cos, atan2, ceil
import numpy as np
import random
from shapely.geometry import Point, Polygon

from levyWalk import Levy
from static import _rMin, _arcStep

# When running out of the polygon, force circular flight until back inside of the polygon
# with _rMin = 50 and _arcStep = np.pi /4
# Note!!! we are setting a coarser arcStep than static._arcStep to reduce the number of
# waypoints waisted in defining back-in-loops.
# The theoretical arclength of these circular steps is _rMin * _arcStep
_lMin = _rMin *np.pi /4
# This yields a maximum curvature of _cMax = _arcStep / (_rMin *_arcStep) = 1 /_rMin = 0.02,
# that is, _rMin = 50

class Drovy(Levy):

	def toCenter(self, wlkPth, poly, fflight):

		# current location
		loc = wlkPth[-1]
		# direction to the center of the polygon
		phi = atan2((poly.centroid.x -loc.x), (poly.centroid.y -loc.y))
		# move
		x = loc.x +fflight *np.sin(phi)
		y = loc.y +fflight *np.cos(phi)
		loc = Point(x, y)
		wlkPth.append(loc)

		self.flights += 1
		self.moved += fflight
		self.steps += ceil(fflight /self.step)

		return loc

	def walk(self, poly, startLoc = 0, minR = 40, maxR = 4000, muR = 2.0, fflight = 200, sflight = False):

		self.muR, self.minR, self.maxR = muR, minR, maxR

		# Drovy walk parameters: radius range
		self.Rpower = 1/(1-muR)
		self.Rrange = maxR**(1-muR) - minR**(1-muR)
		self.flights, self.moved, self.steps = 0, 0, 0

		# search starting point, spiral start direction
		if startLoc < 0:
			loc, phi = poly.centroid, 0
			wlkPth = [loc]
		else:
			loc = Point(poly.boundary.coords.xy[0][startLoc], poly.boundary.coords.xy[1][startLoc])
			phi = atan2((poly.centroid.x -loc.x), (poly.centroid.y -loc.y))
			wlkPth = [loc]
			loc = self.toCenter(wlkPth, poly, fflight)

		maxdst = self.step *self.maxs
		radInf = 0

		while self.moved < maxdst:

			# flight distance
			flight = (self.frange *np.random.rand(1) + self.step**(1 -self.mu))**self.fpower
			# straight flight
			if sflight: radInf = 1 if radInf == 0 else 0
			# flight radius
			if radInf:
				radius = np.inf
			else:
				radius = (self.Rrange *np.random.rand(1) + self.minR**(1 -self.muR))**self.Rpower
			# turn sign
			crvSgn = -1 if np.random.rand(1) < 0.5 else 1

			# flight move
			moved, inPoly = .0, True
			# flight radius factor, flight starting direction
			s, phi = 0, phi %(2.0 *np.pi)
			while moved < flight:

				if radInf:
					x = loc.x +self.step *np.sin(phi)
					y = loc.y +self.step *np.cos(phi)
				else:
					s += _arcStep
					phi += _arcStep *crvSgn
					x = loc.x +radius *s *np.sin(phi)
					y = loc.y +radius *s *np.cos(phi)

				inPoly = poly.contains(Point(x, y))
				if not inPoly: break

				loc = Point(x, y)
				moved += wlkPth[-1].distance(loc)
				wlkPth.append(loc)
				# check max distance
				self.steps += 1
				if self.steps >= self.maxs: break

			# check we didn't step out of the polygon
			if not inPoly:
				# trick 1. circular flight to get back in
				i = 0
				while not inPoly:

					x = loc.x +_lMin *np.sin(phi)
					y = loc.y +_lMin *np.cos(phi)

					inPoly = poly.contains(Point(x, y))
					if inPoly and i > 2:
						# trick 2. if back in and cumturn is greater than pi/2
						# make last step twice longer to "avoid" stepping out again
						x = loc.x +_lMin *np.sin(phi) *2.0
						y = loc.y +_lMin *np.cos(phi) *2.0

					i += 1
					phi += np.pi /4 *crvSgn

					loc = Point(x, y)
					moved += wlkPth[-1].distance(loc)
					wlkPth.append(loc)
					# check max distance
					self.steps += 1
					if self.steps >= self.maxs: break

			# update moved distance
			if moved > 0:
				self.flights += 1
				self.moved += moved

		return wlkPth


# Alternative to go back inside the polygon
# does NOT work!!

# polygon info
# polyVrtxs = poly.exterior.coords
# polyLines = MultiLineString([(a,b) for a, b in zip(polyVrtxs[: -1], polyVrtxs[1: ])])

# if not poly.contains(Point(x,y)):
# 	pathLine = Line([wlkPth[-1], Point(x, y)])
# 	for polyLine in polyLines:
# 		if pathLine.intersects(polyLine):
# 			turnA = turn3(wlkPth[-2], wlkPth[-1], polyLine.coords[0])
# 			turnB = turn3(wlkPth[-2], wlkPth[-1], polyLine.coords[1])
# 			if abs(turnA) < abs(turnB):
# 				x, y = polyLine.coords[0]
# 			else:
# 				x, y = polyLine.coords[1]
# 			t = atan2(x -wlkPth[-1].x, y -wlkPth[-1].y)
# 			break
