from math import cos, sin, sqrt, pi, log10, asin
import numpy as np
from scipy.stats import beta

from shapely.geometry import Point, Polygon

class Levy:

	def __init__(self, maxs=200, minf=45, maxf=45000, mu=2.0):

		# Levy walk parameters
		self.step = minf				# flight min length
		self.maxf = maxf				# flight max length
		self.maxs = maxs				# max number of steps
		self.mu = mu					# difussion coefficient
		# flight range
		self.frange = self.maxf**(1-self.mu) - self.step**(1-self.mu)
		# flight power
		self.fpower = 1/(1-self.mu)

	def reslts(self):
		# search results
		print('distance ... %7.2f, flights ... %4.0f' % (self.moved, self.flights) )

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

		while self.moved < maxdst:

			# flight direction: from north clockwise
			phi = np.random.uniform(0, (2*np.pi), 1)

			# flight distance
			flight = (self.frange * np.random.uniform(0,1) + self.step**(1-self.mu))**self.fpower
			self.flights += 1

			# flight move
			moved, hitLst = 0, []
			while moved < flight:
				x = loc.x + self.step *cos(phi)
				y = loc.y + self.step *sin(phi)
				if not poly.contains(Point(x, y)): break
				loc = Point(x, y)
				moved += self.step
				self.steps += 1
				# check max distance
				if self.steps >= self.maxs: break

			if moved > 0: wlkPth.append(loc)

			# moved distance
			self.moved += moved

		return wlkPth
