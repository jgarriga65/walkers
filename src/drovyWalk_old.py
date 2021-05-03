from math import cos, sin, sqrt, pi, log10, asin
import numpy as np
from scipy.stats import beta

from shapely.geometry import Point, Polygon

from levyWalk import Levy

_rMin = 50
_phiMax = np.pi /2

class Drovy(Levy):

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

		# init f1, phi1 to check turns
		f1, phi1 = self.maxf, 0

		while self.moved < maxdst:

			# flight distance
			f2 = (self.frange *np.random.uniform(0, 1) + self.step**(1 -self.mu))**self.fpower

			# flight direction: from north clockwise
			# delta: back step to achieve minimum turn radius
			while True:
				phi2 = np.random.uniform(0, (2 *np.pi), 1)
				turn = abs(phi2 -phi1)
				if turn >np.pi: turn = 2 *np.pi -turn
				if  turn < _phiMax:
					delta = 0
					break
				else:
					delta = abs(_rMin /sin((np.pi -turn) /2))
					if delta < f1: break

			# flight move
			steps, moved = 0, 0
			while moved < f2:
				x = loc.x + self.step *cos(phi2)
				y = loc.y + self.step *sin(phi2)
				if not poly.contains(Point(x, y)):
					moved = 0
					break
				moved += self.step
				steps += 1
				# update current position
				loc = Point(x, y)
				# check max distance
				if self.steps +steps >= self.maxs: break

			if moved:

				# check flight turn
				if len(wlkPth) >1 and delta:
					# Att!! headings are measured from North clockwise.
 					# North is direction of y coordinates. Thus:
					# - delta in x direction is given by sin(-phi)
					# - delta in y direction is given by cos(-phi)
					x1 = wlkPth[-1].x -delta *sin(-phi1)
					y1 = wlkPth[-1].y -delta *cos(-phi1)
					x2 = wlkPth[-1].x +delta *sin(-phi2)
					y2 = wlkPth[-1].y +delta *cos(-phi2)
					# replace last location
					wlkPth[-1] = Point(x1, y1)
					# add intermediate location
					wlkPth.append(Point(x2, y2))

				# add new point
				wlkPth.append(loc)

				# update moved distance
				self.flights += 1
				self.steps += steps
				self.moved += moved

				# print('+++ %4d %6d %7.1f %6.2f %6.2f, %6.2f, %6.2f' %(self.flights, steps, moved, phi1, phi2, turn, delta))

				# update last flight and direction
				f1, phi1 = moved, phi2
				# if last two points have changed f1 must be recomputed
				f1 = sqrt((wlkPth[-1].x -wlkPth[-2].x)**2 +(wlkPth[-1].y -wlkPth[-2].y)**2)


		return wlkPth
