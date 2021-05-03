
#force matplotlib to use no xwindows backend
#import matplotlib
#matplotlib.use('Agg')

import sys, os

from math import cos, sin, sqrt, pi, log10
import numpy as np
from scipy.stats import multivariate_normal, beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import matplotlib.animation as animation

# import cPickle as pickle
# changed to python3
import _pickle as pickle

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy as sp

#from threading import Thread
from multiprocessing import Pool

_dataPath = os.path.abspath('./')+'/walkers/pkl/'

def chkBeta(a, b):
	x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)
	plt.plot(x,beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6)
	plt.show()

# +++ load sgrd instance +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def sgLoad(fNmbr):
	return pickle.load(open(_dataPath+'sgrd%s.pkl'%str(fNmbr).zfill(2)))


# +++ load batch instance ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def mpLoad(fNmbr):
	return pickle.load(open(_dataPath+'mpsw%s.pkl'%str(fNmbr).zfill(2)))


# +++ single search ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class srch:

	def __init__(self, maxs=10000, minf=0.001, maxf=10, mu=2.0, dlvl=1, fail=0, hrng=1.0):

		# Levy walk parameters
		self.step = minf * sqrt(2)		# flight min length
		self.maxf = maxf				# flight max length
		self.maxs = maxs				# max number of steps
		self.mu = mu					# difussion coefficient

		# flight range
		self.frange = self.maxf**(1-self.mu)-self.step**(1-self.mu)
		# flight power
		self.fpower = 1/(1-self.mu)

		# follow gradient parameters
		self.dlvl = dlvl				# sensory defective level
		self.fail = fail				# probability of gradient detection fail

		# initial target list
		self.tgtLst, self.hitLst = [], []
		self.hrng = hrng				# hit range

		# walk path initialization
		self.steps, self.flights, self.moved  = 0, 0, 0

	def reslts(self):
		# search results
		print('distance ... %7.2f, flights ... %4.0f, hits ... %2.0f'%(self.moved,self.flights,len(self.hitLst)))
		return

	def plot2d(self):
		axs = plt.gca()
		if len(self.tgtLst):
			axs.scatter([x for (x,y) in self.tgtLst], [y for (x,y) in self.tgtLst], c='w', s=200, marker='*')
		if len(self.hitLst):
			axs.scatter([x for ((x,y),s) in self.hitLst], [y for ((x,y),s) in self.hitLst], c='r', s=200, marker='*')
		if len(self.wlkPth):
			axs.scatter([x for (x,y) in self.wlkPth], [y for (x,y) in self.wlkPth], c='r', s=1, marker='o')
		plt.show()

# +++ landscape grid +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class sgrd:

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
		if p: self.plot2d()

	def getTht(self, c, ma=1.2, mb=1.2, va=2.0, vb=60):

		# space gradient parameters with c gaussian components
		# means are drawn from a beta(ma, mb) distribution
		# covariance matrix values are drawn form a beta(va, vb) distribution

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

	def plctgt(self, t, cf = 0):

		# place t targets with correlation factor cf;
		# fc = 0, random placement of targets
		# fc = 1, totally correlated with landscape pdf

		self.trgt, self.cf = t, cf
		if self.cf:	self.gettSp()
		self.tgtLst = [self.newtgt() for t in range(self.trgt)]


	def rsttgt(self, t, cf=0):

		# reset targets and plot

		self.plctgt(t, cf=cf)
		self.plot2d()

	def shwTht(self):

		# show landscape parameters

		print('means .....\n', self.means)
		print('covar .....\n',)
		for cMtx in self.covar: print(cMtx)
		print('entropy ...', self.entropy())
		print('targets ...', self.trgt)
		print('cf(l,t) ...', self.cf)

	def plot2d(self, showIt=True):

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
		if showIt:
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

	def saveIt(self, fName=''):
		if not len(fName):
			fNmbr = 1
			while os.path.exists(_dataPath+'sgrd%s.pkl'%str(fNmbr).zfill(2)):
				fNmbr += 1
			fName = _dataPath + 'sgrd%s.pkl'%str(fNmbr).zfill(2)
		pickle.dump(self, open(fName, 'wb'))
		print('saved %s'%fName)

	def nwsrch(self, maxs=10000, minf=0.001, maxf=10, mu=2.0, dlvl=1, fail=0, hrng=1.0):
		# perform new search and show it
		_srch = srch(maxs=maxs, minf=minf, maxf=maxf, mu=mu, dlvl=dlvl, fail=fail, hrng=hrng)
		doSrch(self, _srch)
		_srch.reslts()
		self.plot2d(showIt=False)
		_srch.plot2d()


# +++ run search function ++++++++++++++++++++++++++++++++++++++++++++++++++++++

def doSrch(_sgrd, _srch):

	def dst2tgt(tgt, pos):
		# distance to target
		return sqrt((tgt[0]-pos[0])**2+(tgt[1]-pos[1])**2)

	def tgthit(tgt, pos):
		# target hit
		# Att!! hitRange has a strong impact in target detection
		return (dst2tgt(tgt, pos)<_srch.hrng*_srch.step)

	# initial target list
	_srch.tgtLst = _sgrd.tgtLst[:]

	# search starting point
	_srch.wlkPth = [(0.5*_sgrd.size, 0.5*_sgrd.size)]

	maxdst = _srch.step*_srch.maxs
	while _srch.moved < maxdst:

		# flight direction: from north clockwise
		# if dlvl < gradient at the current cell
		# and do not fail at detect it (flvl > rand): follow gradient
		# else: random
		if _srch.dlvl < _sgrd.gmod[_sgrd.getCell(_srch.wlkPth[-1])] and np.random.rand() >= _srch.fail:
			phi = _sgrd.gphi[_sgrd.getCell(_srch.wlkPth[-1])]
		else:
			phi = np.random.uniform(0, (2*np.pi), 1)

		# flight distance
		flight = (_srch.frange*np.random.uniform(0,1)+_srch.step**(1-_srch.mu)) **_srch.fpower
		_srch.flights += 1

		# flight move
		moved, hitLst = 0, []
		while moved < flight:
			x = _srch.wlkPth[-1][0] + _srch.step*cos(phi)
			y = _srch.wlkPth[-1][1] + _srch.step*sin(phi)
			if x >= _sgrd.size: x = x - _sgrd.size
			if y >= _sgrd.size: y = y - _sgrd.size
			if x < 0: x = x + _sgrd.size
			if y < 0: y = y + _sgrd.size
			_srch.wlkPth.append((x,y))
			moved += _srch.step
			_srch.steps += 1

			# check target hit
			hitLst = [i for i,tgt in enumerate(_srch.tgtLst) if tgthit(tgt, _srch.wlkPth[-1])]
			if len(hitLst):	break

			# check gradient
			if _srch.dlvl < _sgrd.gmod[_sgrd.getCell(_srch.wlkPth[-1])]: break

			# check max distance
			if _srch.steps >= _srch.maxs: break

		# check hitLst
		for i in hitLst:
			moved += dst2tgt(_srch.tgtLst[i], _srch.wlkPth[-1])
			_srch.wlkPth.append(_srch.tgtLst[i])
#			self.steps += 1	# fede ?
			_srch.hitLst.append((_srch.tgtLst[i], _srch.steps))
			# target replacement to keep target density uniform (Fede !!??)
			_srch.tgtLst[i] = _sgrd.newtgt()

		# moved distance
		_srch.moved += moved


# +++ multiprocessing version ++++++++++++++++++++++++++++++++++++++++++++++++++

class wLog():	# worker output

	def __init__(self, i, mu):

		self.i, self.mu = i, mu
		self.hitLst = []
		self.dstLst = []

	def hits(self):
		return sum([len(hitLst) for hitLst in self.hitLst])

	def dist(self):
		return sum(self.dstLst)

	def rEff(self, maxDst = 0, minf = 0):
		if maxDst:
			# compute run efficiencies at distance maxDst
			maxStp = maxDst /(minf *sqrt(2))
			return np.array([sum([hitStp <= maxStp for hitStp in hitLst]) /maxDst for hitLst in self.hitLst])
		else:
			# compute run efficiencies at total moved distance
			return np.array([len(hitLst) /dst for hitLst, dst in zip(self.hitLst, self.dstLst)])

	def smmr(self):
		# print summary
		print('mu:%4.1f, hits:%3.0f eff:%10.6f'%(self.mu, self.hits(), round(np.mean(self.rEff()),6)))

class mpsw:	# multiprocess search run

	def __init__(self, _sgrd, minf = 0.001, maxf = 10, maxs = 10000, dlvl = 1, fail = 0, hrng = 20.0, dff = np.arange(1.1, 3.0, 0.2), runs=1000):

		self.dff = dff		# difussion coeficients
							# from 1.1 to 3.0 by 0.1
		 					# 0:1 subdifussive
							# 1:2 superdifussive
							# 2:3 balistic
		self.log = {}

		# landscape grid
		self.sgrd = _sgrd
		# Levy walk parameters
		self.minf, self.maxf, self.maxs = minf, maxf, maxs
		# gradient follow parameters
		self.dlvl, self.fail = dlvl, fail
		# detection range (hit range)
		self.hrng = hrng
		# number of runs
		self.runs = runs

	def bRun(self):
		pool = Pool(processes=len(self.dff))
		outQ = [pool.apply_async(wrkr, args=(self, i, mu)) for i, mu in enumerate(self.dff)]
		for _wlog in [q.get() for q in outQ]: self.log[_wlog.i] = _wlog

	def smmr(self):
		for _wLog in self.log.values(): _wLog.smmr()

	def mEff(self, maxDst=0):
		# plot mean efficiency at distance maxDst
		fig = plt.figure()
		axs = fig.add_subplot(111)
		axs.grid(True)
		# axs.set_xlim(1.1,3.0)
		# axs.set_ylim(0,0.1)
		axs.plot(self.dff,[log.rEff(maxDst=maxDst, minf=self.minf).mean() for log in self.log.values()])
		plt.show()

	def bEff(self, maxDst=0):
		# efficiency box-plot at distance maxDst
		fig = plt.figure()
		axs = fig.add_subplot(111)
		axs.grid(True)
		# axs.set_xlim(1.1,3.0)
		# axs.set_ylim(0,0.1)
		axs.boxplot(np.array([log.rEff(maxDst = maxDst, minf = self.minf) for log in self.log.values()]).T)
		axs.set_xticklabels(np.round(self.dff, 2))
		plt.show()

	def saveIt(self, fName=''):
		if not len(fName):
			fNmbr = 1
			while os.path.exists(_dataPath+'mpsw%s.pkl'%str(fNmbr).zfill(2)):
				fNmbr += 1
			fName = _dataPath + 'mpsw%s.pkl'%str(fNmbr).zfill(2)
		pickle.dump(self,open(fName,'wb'))
		print('saved %s'%fName)

def wrkr(_mpsw, i, mu):	# multiprocessing worker
	_wLog = wLog(i, mu)
	for run in range(_mpsw.runs):
		_srch = srch(mu = mu, maxs = _mpsw.maxs, minf = _mpsw.minf, maxf = _mpsw.maxf, dlvl = _mpsw.dlvl, fail = _mpsw.fail, hrng = _mpsw.hrng)
		doSrch(_mpsw.sgrd, _srch)
		_wLog.hitLst.append([htStep for (x, y), htStep in _srch.hitLst])
		_wLog.dstLst.append(_srch.moved)
	return _wLog
