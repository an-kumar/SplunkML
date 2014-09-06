"""
	SplunkML PCA Preprocessor
	=========================

	Jay Hack
	jhack@stanford.edu
	Summer 2014
"""
import itertools
import scipy as sp
import numpy as np
from Cmd import Cmd, ArgCmd
from sklearn.decomposition import PCA
import splunklib.client as client
import splunklib.results as results

#==========[ Normalization	]==========
GetAverageStd = ArgCmd (	'eventstats [avg({0}) as {0}_avg, stdev({0}) as {0}_std]')
ToZeroMean = ArgCmd (		'[eval {0}_zm={0}-{0}_avg]', repeat_symbol=' | ')
ToUnitVar = ArgCmd (		'[eval {0}={0}_zm/{0}_std]', repeat_symbol=' | ')

#==========[ Common ArgCmds	]==========
GetProducts = ArgCmd (	'[eval {0}_{1}_product={0}*{1} ] ', repeat_symbol=' | ')
AvgProducts = ArgCmd (	'stats [avg({0}_{1}_product) as {0}_{1}_cov]')


class SplunkPCA ():

	"""
		Params:
		-------
		- X_fieldnames: field names of dependent variables to apply dimensionality
						reduction to
		- n_components: number of dimensions to reduce to
		- normalize_features: will convert every feature to zero-mean, unit variance 
	"""

	def __init__ (self, X_fieldnames, n_components=1, normalize_features=True):
		self.X_fieldnames = X_fieldnames
		self.n_components = n_components
		self.X_fieldpairs = [x for x in itertools.combinations(self.X_fieldnames, 2)]


	def normalize_X (self):
		"""
			converts all features to zero mean, unit variance
		"""
		return GetAverageStd(self.X_fieldnames) + ToZeroMean(self.X_fieldnames) + ToUnitVar (self.X_fieldnames)


	def compute_covariance (self):
		"""
			computes covariance in splunk
		"""
		return GetProducts (self.X_fieldpairs) + AvgProducts (self.X_fieldpairs)	


	def compute_principal_components (self, cov_mat):
		"""
			returns principal components
		"""
		pca = PCA (n_components=self.n_components)
		pca.fit (cov_mat)



if __name__ == '__main__':

	pca = SplunkPCA (['field%s' % n for n in range(1,10)])
	print pca.normalize_X () + pca.compute_covariance ()


