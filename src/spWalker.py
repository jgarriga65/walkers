
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

import sys, os

from math import cos, sin, sqrt, pi, log10
import numpy as np
from scipy.stats import multivariate_normal, beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import scale
import matplotlib.animation as animation

# import cPickle as pickle
# changed to python3
import _pickle as pickle

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy as sp

#from threading import Thread
from multiprocessing import Pool

_dataPath = os.path.abspath('./walkers/pkl')

def chkBeta(a, b):
	x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)
	plt.plot(x,beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6)
	plt.show()

# +++ load spGrid instance +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def spGridLoad(fNmbr):
	return pickle.load(open(_dataPath +'/spGrid%s.pkl' %str(fNmbr).zfill(2), 'rb'))


# +++ load batch instance ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mSearchLoad(fNmbr):
	return pickle.load(open(_dataPath +'/mSearch%s.pkl' %str(fNmbr).zfill(2), 'rb'))


# +++ landscape grid +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class spGrid:

	def __init__(self, side = 5, resolution = 0.005, targets = 50, correlation_factor = 0.5, gaussian_components = 2, ma = 1.2, mb = 1.2, va = 2.0, vb = 60, p = True):

		L, d, t, cf, c = side, resolution, targets, correlation_factor, gaussian_components

		# space grid
		self.size, self.dlta = L, d		# grid size and resolution
		self.shape = (int(L /d), int(L /d))

		# landscape
		if c == 0:
			self.z = np.zeros(self.shape, dtype = 'float')
			self.gmod, self.gphi = self.z, self.z
		else:
			self.getTht(c, ma, mb, va, vb)	# gradient space parameters:
											# c: number of gaussian components
											# means distribution: beta(ma,mb)
											# covar distribution: beta(va,vb)
			self.getpdf()					# get rasterized pdf
			if p: print(self.entropy())		# landscape pdf entropy

		# targets
		self.trgt, self.cf = t, cf
		self.tgtLst = []
		# place initial targets
		if t: self.plctgt(t, cf)		# place t targets,
		 								# with correlation factor cf
		# plot it
		if p: self.plot()

	def getTht(self, c, ma=1.2, mb=1.2, va=2.0, vb=60):

		# space gradient parameters with c gaussian components
		# means are drawn from a beta(ma, mb) distribution
		# covariance matrix values artargetse drawn form a beta(va, vb) distribution

		self.means = np.array([np.random.beta(ma, mb, 2) for i in range(c)])

		# fix variance matrices
		# self.covar = [np.array([[0.01,0],[0,0.01]]) for i in range(c)]

		# no covariances version
		# self.covar = [np.array([[np.random.beta(va,vb),0],[0,np.random.beta(va,vb)]]) for i in range(c)]

		# random covariances version
		self.covar = []
		for i in range(c):
			var = np.random.beta(va,vb,2)
			cov = np.random.beta(va,vb,2)
			cov *= np.array([1 if u>0.5 else -1 for u in np.random.rand(2)])
			cMtx = np.array([[var[0],cov[0]],[cov[1],var[0]]])
			self.covar.append(np.dot(cMtx, cMtx.T))

		# scale to landscape size
		self.means *= self.size
		self.covar = [cMtx/self.dlta for cMtx in self.covar]

	def getpdf(self):

		# get wrapped rasterized pdf
		x,y = np.mgrid[-1*self.size:2*self.size:self.dlta, -1*self.size:2*self.size:self.dlta]
		Z = np.zeros(x.shape, dtype='float')
		for i,cMtx in enumerate(self.covar):
			mn = multivariate_normal(self.means[i], cMtx)
			z = mn.pdf(np.dstack((x, y)))
			Z += z/np.sum(z)
		Z /= self.means.shape[0]
		q = np.arange(0, Z.shape[0]+1, Z.shape[0]//3)
		self.z = sum([Z[q[i]:q[i+1], q[j]:q[j+1]] for i in range(3) for j in range(3)])
		self.getgrd()

	def getgrd(self):

		# gradient value and direction
		dx, dy = np.gradient(self.z)
		g = np.sqrt(dx**2+dy**2)
		self.gmod = (g-g.min())/(g.max()-g.min())
		self.gphi = np.arctan2(dy, dx)

	def gettSp(self):
		# get cutoff value corresponding to self.cf and
		# set of possible cells to place targets
		z = np.sort(self.z.flatten())
		self.cutOff = z[np.where(np.cumsum(z) > self.cf)[0][0]]
		x,y = np.mgrid[0:self.size:self.dlta, 0:self.size:self.dlta]
		self.tSpace = [(cx,cy) for cx,cy in zip(x.ravel(), y.ravel()) if self.z[self.getCell((cx, cy))]>self.cutOff]

	def entropy(self):
		z = self.z[np.where(self.z>0)]
		return -np.sum(z*np.log(z))

	def getCell(self, xy):
		return int(xy[0]/self.dlta), int(xy[1]/self.dlta)

	def newtgt(self):
		if np.random.rand() > self.cf:
			tgx, tgy = np.random.rand(2)
			return (tgx *self.size, tgy *self.size)
		else:
			zi = np.sum(self.z, axis=1)
			r, tgx = np.random.rand(), -1
			while np.sum(zi[0:tgx+1]) < r: tgx += 1
			zj = self.z[tgx,:]/np.sum(self.z[tgx,:])
			r, tgy = np.random.rand(), -1
			while np.sum(zj[0:tgy+1]) < r: tgy += 1
			tgx += np.random.rand()
			tgy += np.random.rand()
			return (tgx *self.dlta, tgy *self.dlta)

	def newtgt_b(self):
		while True:
			tgx, tgy = np.random.rand(2) *self.size
			if self.z[self.getCell((tgx, tgy))] > self.cutOff: break
		return (tgx, tgy)

	def newtgt_c(self):
		return self.tSpace[int(np.random.rand()*(len(self.tSpace)-1))] +np.random.rand(2)*self.dlta

	def plctgt(self, targets = 50, correlation_factor = 0.5):

		# place t targets with correlation factor cf;
		# fc = 0, random placement of targets
		# fc = 1, totally correlated with landscape pdf

		self.trgt, self.cf = targets, correlation_factor
		if self.cf:	self.gettSp()
		self.tgtLst = [self.newtgt() for t in range(self.trgt)]

	def resetTargets(self, targets = 50, correlation_factor = 0.5):

		# reset targets and plot
		self.plctgt(targets = targets, correlation_factor = correlation_factor)
		self.plot()

	def shwTht(self):

		# show landscape parameters

		print('means .....\n', self.means)
		print('covar .....\n',)
		for cMtx in self.covar: print(cMtx)
		print('entropy ...', self.entropy())
		print('targets ...', self.trgt)
		print('cf(l,t) ...', self.cf)

	def plot(self, show = True):

		fig = plt.figure()
		axs = fig.add_subplot(111)
		axs.grid(True)
		axs.set_xlim(0, self.size)
		axs.set_ylim(0, self.size)
		if np.sum(self.z):
#			axs.contourf(self.x, self.y, self.z, alpha=0.8)
			x, y = np.mgrid[0:self.size:self.dlta, 0:self.size:self.dlta]
			axs.contourf(x, y, self.z, alpha=0.8)
		else:
			axs.set_facecolor('blue')
		if show:
			if len(self.tgtLst):
				axs.scatter([x for (x,y) in self.tgtLst], [y for (x,y) in self.tgtLst], c='w', s=200, marker='*')
			plt.show()

	def plot3d(self, nice=False):
		# does not work !!
		fig = plt.figure()
		axs = fig.add_subplot(111, projection='3d')
		X, Y = np.mgrid[0:self.size:self.dlta, 0:self.size:self.dlta]
		if nice:
			axs.plot_surface(X, Y, self.z, cstride=1, rstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		else:
			axs.plot_surface(X, Y, self.z, linewidth=0.2, antialiased=True)
		plt.show()

	def save(self, fName=''):
		if not len(fName):
			fNmbr = 1
			while os.path.exists(_dataPath +'/spGrid%s.pkl' %str(fNmbr).zfill(2)):
				fNmbr += 1
			fName = _dataPath + '/spGrid%s.pkl' %str(fNmbr).zfill(2)
		pickle.dump(self, open(fName, 'wb'))
		print('saved %s'%fName)

class walker():

	def dst2tgt(self, tgt, pos):
		# distance to target
		return sqrt((tgt[0] -pos[0])**2 +(tgt[1] -pos[1])**2)

	def tgthit(self, tgt, pos, hit_distance, target_detection_fail):
		# target hit
		# Att!! hitRange has a strong impact in target detection
		return (self.dst2tgt(tgt, pos) < hit_distance and np.random.rand() > target_detection_fail)

class levyWalker(walker):

	def __init__(self, min_flight = 0.001, max_flight = 10, mu = 2.0, show = True):

		# Levy walk parameters
		self.step = min_flight * sqrt(2)		# walker step
		self.min_flight = min_flight			# flight min length
		self.max_flight = max_flight			# flight max length
		self.mu = mu							# difussion coefficient
		# flight range
		self.frange = self.max_flight**(1-self.mu)-self.step**(1-self.mu)
		# flight power
		self.fpower = 1/(1-self.mu)
		# print output
		if show:
			print('+++ Levy-walk:')
			print('    step ........... %8.4f' %self.step)
			print('    max. flight .... %8.4f' %self.max_flight)
			print('    diffusion ...... %8.2f' %self.mu)

	def search(self, _spGrid, max_steps, min_gradient, gradient_fail, hit_range, target_detection_fail):

		# walk path initialization
		self.steps, self.flights, self.moved, self.hitLst  = 0, 0, 0, []
		# initial target list
		self.tgtLst = _spGrid.tgtLst[:]
		# search starting point
		self.wlkPth = [(0.5 *_spGrid.size, 0.5 *_spGrid.size)]
		# maximum distance for target hit
		hit_distance = hit_range *self.step

		maxdst = self.step *max_steps
		while self.moved < maxdst:

			# flight direction: from north clockwise
			# if gradient at the current cell > min_gradient
			# and do not fail at detect it (rand > gradient_fail): follow gradient
			# else: random

			if _spGrid.gmod[_spGrid.getCell(self.wlkPth[-1])] > min_gradient \
				and np.random.rand() > gradient_fail:
					phi = _spGrid.gphi[_spGrid.getCell(self.wlkPth[-1])]
			else:
				phi = np.random.uniform(0, (2 *np.pi), 1)

			# flight distance
			flight = (self.frange *np.random.uniform(0, 1) +self.step**(1-self.mu)) **self.fpower
			self.flights += 1

			# flight move
			moved, flghtHit = 0, []
			while moved < flight:
				x = self.wlkPth[-1][0] +self.step *cos(phi)
				y = self.wlkPth[-1][1] +self.step *sin(phi)
				if x >= _spGrid.size: x = x -_spGrid.size
				if y >= _spGrid.size: y = y -_spGrid.size
				if x < 0: x = x +_spGrid.size
				if y < 0: y = y +_spGrid.size
				self.wlkPth.append((x,y))
				moved += self.step
				self.steps += 1

				# check target hit
				flghtHit = [i for i, tgt in enumerate(self.tgtLst) if self.tgthit(tgt, self.wlkPth[-1], hit_distance, target_detection_fail)]
				if len(flghtHit): break

				# check gradient
				if _spGrid.gmod[_spGrid.getCell(self.wlkPth[-1])] > min_gradient \
					and np.random.rand() > gradient_fail:
						break

				# check max distance
				if self.steps >= max_steps: break

			# check hitLst
			for i in flghtHit:
				moved += self.dst2tgt(self.tgtLst[i], self.wlkPth[-1])
				self.wlkPth.append(self.tgtLst[i])
	#			self.steps += 1	# fede ?
				self.hitLst.append((self.tgtLst[i], self.steps))
				# target replacement to keep target density uniform (Fede !!??)
				self.tgtLst[i] = _spGrid.newtgt()

			# moved distance
			self.moved += moved


# +++ single search run ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class sSearch:

	def __init__(self, _spGrid, _spWalk, max_steps, gradient_detection_level, gradient_detection_fail, hit_range, target_detection_fail):

		self.spGrid, self.spWalk = _spGrid, _spWalk

		self.max_steps = max_steps
		self.gradient_detection_level = gradient_detection_level
		self.gradient_detection_fail = gradient_detection_fail
		self.hit_range = hit_range
		self.target_detection_fail = target_detection_fail

	def plot(self):
		# plot
		self.spGrid.plot(show = False)
		axs = plt.gca()
		if len(self.spGrid.tgtLst):
			axs.scatter([x for (x,y) in self.spGrid.tgtLst], [y for (x,y) in self.spGrid.tgtLst], c = 'w', s = 100, marker = '*')
		if len(self.spWalk.hitLst):
			axs.scatter([x for ((x, y), s) in self.spWalk.hitLst], [y for ((x, y), s) in self.spWalk.hitLst], c = 'm', s = 200, marker = '*')
		if len(self.spWalk.wlkPth):
			axs.scatter([x for (x, y) in self.spWalk.wlkPth], [y for (x, y) in self.spWalk.wlkPth], c = 'b', s = 0.5, marker = 'o')
		# search results
		print('distance %7.2f, flights %4.0f, hits %2.0f' %(self.spWalk.moved, self.spWalk.flights, len(self.spWalk.hitLst)))
		plt.show()

def singleSearch(_spGrid, _spWalk, max_steps = 10000, gradient_detection_level = 0.5, gradient_detection_fail = 0.5, hit_range = 10.0, target_detection_fail = 0.5):
	# perform search
	_spWalk.search(_spGrid, max_steps, gradient_detection_level, gradient_detection_fail, hit_range, target_detection_fail)
	# define search instance
	_sSearch = sSearch(_spGrid, _spWalk, max_steps, gradient_detection_level, gradient_detection_fail, hit_range, target_detection_fail)
	# output
	_sSearch.plot()
	return _sSearch


# +++ multi-process search run ++++++++++++++++++++++++++++++++++++++++++++++++++

class mSearch:

	def __init__(self, _spGrid, _spWalk, max_steps, gradient_detection_level = .5, gradient_detection_fail = .5, hit_range = 10.0, target_detection_fail = 0.5, dff = np.arange(1.1, 3.0, 0.2), runs = 100):

		self.spGrid, self.spWalk = _spGrid, _spWalk

		self.max_steps = max_steps
		self.gradient_detection_level = gradient_detection_level
		self.gradient_detection_fail = gradient_detection_fail
		self.hit_range = hit_range
		self.target_detection_fail = target_detection_fail

		# difussion coeficients
		self.dff = dff		# from 1.1 to 3.0 by 0.1
		 					# 0:1 subdifussive
							# 1:2 superdifussive
							# 2:3 balistic
		# number of runs
		self.runs = runs
		self.log = {}

	def bRun(self):
		pool = Pool(processes = len(self.dff))
		outQ = [pool.apply_async(wrkr, args = (self, i, mu)) for i, mu in enumerate(self.dff)]
		for _wlog in [q.get() for q in outQ]: self.log[_wlog.i] = _wlog
		pool.close()
		del pool

	def smmry(self):
		for _wLog in self.log.values(): _wLog.smmry()

	def meanEff(self, maxDst = 1.0):
		# plot mean efficiency at distance maxDst
		fig = plt.figure()
		axs = fig.add_subplot(111)
		axs.grid(True)
		# axs.set_xlim(1.1,3.0)
		# axs.set_ylim(0,0.1)
		dff = [log.mu for log in self.log.values()]
		axs.plot(dff, [log.runEff(maxDst = maxDst).mean() for log in self.log.values()])
		plt.show()

	def hitsBoxp(self, maxDst = 1.0):
		# efficiency box-plot at distance maxDst
		fig = plt.figure()
		axs = fig.add_subplot(111)
		axs.grid(True)
		# axs.set_xlim(1.1,3.0)
		# axs.set_ylim(0,0.1)
		axs.boxplot(np.array([log.hits(maxDst = maxDst) for log in self.log.values()]).T)
		axs.set_xticklabels(np.round([log.mu for log in self.log.values()], 2))
		plt.show()

	def save(self, fName=''):
		if not len(fName):
			fNmbr = 1
			while os.path.exists(_dataPath +'mSearch%s.pkl' %str(fNmbr).zfill(2)):
				fNmbr += 1
			fName = _dataPath + '/mSearch%s.pkl' %str(fNmbr).zfill(2)
		pickle.dump(self, open(fName, 'wb'))
		print('saved %s'%fName)


# multiprocessing worker
def wrkr(_mpSrch, i, mu):
	# new walker instance
	_spWalk = levyWalker(min_flight = _mpSrch.spWalk.min_flight, max_flight = _mpSrch.spWalk.max_flight, mu = mu, show = False)
	_wLog = wLog(i, _spWalk.mu, _spWalk.step)
	for run in range(_mpSrch.runs):
		# perform search
		_spWalk.search(_mpSrch.spGrid, _mpSrch.max_steps, _mpSrch.gradient_detection_level, _mpSrch.gradient_detection_fail, _mpSrch.hit_range, _mpSrch.target_detection_fail)
		_wLog.hitLst.append([htStep for (x, y), htStep in _spWalk.hitLst])
		_wLog.dstLst.append(_spWalk.moved)
	return _wLog


def multipleSearch(_spGrid, _spWalk, max_steps = 10000, gradient_detection_level = 0.5, gradient_detection_fail = 0.5, hit_range = 10.0, target_detection_fail = 0.5, dff = np.arange(1.1, 3.0, 0.2), runs = 1000):
	_mSearch = mSearch(_spGrid, _spWalk, max_steps, gradient_detection_level = gradient_detection_level, gradient_detection_fail = gradient_detection_fail, hit_range = hit_range, dff = dff, runs = runs)
	_mSearch.bRun()
	return _mSearch


class wLog():	# worker output

	def __init__(self, i, mu, step):

		self.i = i
		self.mu = mu
		self.step = step
		self.hitLst = []
		self.dstLst = []

	def dist(self):
		return np.mean(self.dstLst)

	def hits(self, maxDst = 1.0):
		if maxDst < 1.0:
			maxStep = self.dist() *maxDst / self.step
			return [sum([hitStep <= maxStep for hitStep in hitLst]) for hitLst in self.hitLst]
		else:
			return [len(hitLst) for hitLst in self.hitLst]

	def runEff(self, maxDst = 1.0):
		return np.array([hits /(dst *maxDst) for hits, dst in zip(self.hits(maxDst), self.dstLst)])

	def smmry(self):
		# print summary
		print('mu:%4.1f, hits:%5d, eff(0.5): %8.6f, eff(1.0): %8.6f' %(self.mu, sum(self.hits()), round(np.mean(self.runEff(maxDst = 0.5)), 6), round(np.mean(self.runEff()), 6)))
