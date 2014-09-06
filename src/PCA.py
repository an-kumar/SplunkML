'''
SplunkML PCA

Ankit Kumar
ankitk@stanford.edu

'''
import splunklib.client as client
import splunklib.results as results
from collections import defaultdict
# from scipy.stats import multivariate_normal
import numpy as np
from base_classes import SplunkProjectorBase
import sys
import splunkmath as sm
from splunkmath.utils.strings import *
from splunkmath.classes import SplunkArray



reaction_features = ['field%s' % i for i in range(1,45)]
reaction_search = 'source="/users/ankitkumar/documents/code/splunkml/data/splunk_second_cont.txt"'


class SplunkPCA(SplunkProjectorBase):
	'''
	towrite (description)
	'''

	def __init__(self, host, port, username, password):
		super(SplunkPCA, self).__init__(host, port, username, password)

	def project_splunk_search(self, search_string, X_fields, output_field):
		'''
		uses splunkmath to project every event in the splunk search.
		'''
		# initialize search
		splunk_search = 'search %s'
		# instantiate arrays for x and components
		x = sm.array(X_fields)
		components = sm.array(self.components)
		# do the dot product
		projection = sm.dot(x, components)
		# rename the vector to output_field; projected field names will be output_field_i_j
		projection.rename(output_field)
		splunk_search_string = splunk_concat(splunk_search, projection.string)
		
		return splunk_search_string


	def project_single_event(self, event_to_project, X_fields):
		'''
		uses numpy to predict a single event
		'''
		# this is easy to do with pdfs, but just to make sure we have everything correct, we'll do everything out
		# turn the event into a numpy array
		x, y = sm.event_to_numpy_reps_continuous_continuous(event_to_project, self.feature_mapping, X_fields[0])	#x_fields[0] to throw away

		# projection defined as x dot self.components (1x45 * 45x4 -> 1x4)

		return np.dot(x, self.components)

	


	def train(self, search_string, X_fields, n):
		'''
		trains the model using PCA definition:

		Y = XV where the columns are V are the first n eigenvectors of the covariance matrix 1/m (X^T X)

		X is mxd, V is dxn -> Y is mxn, each row an example now with dimension n
		'''
		# doesn't seem like finding these values in splunk itself is the most efficient (it's done in Splunk for gda) if we just pull out X
		# we can find everything using numpy. So let's do that
		# make a feature mapping
		feature_mapping = {X_fields[i]:i for i in range(len(X_fields))}
		self.feature_mapping =feature_mapping
		# make the search job
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create('search ' +search_string + '| table %s' % ' '.join(X_fields),  **search_kwargs)
		# note that we are just disregarding the y here because of how job_to_numpy_reps was coded. will change later.
		X, y = sm.job_to_numpy_reps(job, feature_mapping, X_fields[0], ('continuous','continuous')) 

		# now we find the principal n eigenvectors
		# first subtract the means; save it to user later
		self.means = np.average(X, axis=0)
		X = X - self.means

		# now compute covariance matrix
		cov = np.dot(X.T, X)

		# now compute eigenvectors
		eigval, eigvec = np.linalg.eig(cov)

		# sort the eigvals
		sorted_indices = np.argsort(eigval)[::-1][:n]

		# find the right eigvecs; now the principal n eigvecs are the columns of principal_n_eigvecs
		principal_n_eigvecs = eigvec[:,sorted_indices]

		# store it in self.components
		self.components = principal_n_eigvecs
		print self.components.shape

if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	PCA = SplunkPCA(host="localhost", port=8089, username=username, password=password)
	PCA.train(reaction_search, reaction_features, 4)
	





