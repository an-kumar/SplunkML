'''
SplunkML Gaussian Discriminant Analysis

Ankit Kumar
ankitk@stanford.edu

'''
import splunklib.client as client
import splunklib.results as results
from collections import defaultdict
# from scipy.stats import multivariate_normal
import numpy as np
from base_classes import SplunkClassifierBase
import sys
import splunkmath as sm
from splunkmath.utils.strings import *
from splunkmath.classes import SplunkArray



reaction_features = ['field%s' % i for i in range(1,45)]
reaction_search = 'source="/users/ankitkumar/documents/code/splunkml/data/splunk_second_cont.txt"'
reaction_class = 'success'


class SplunkGaussianDiscriminantAnalysis(SplunkClassifierBase):
	def __init__(self, host, port, username, password, alpha=0.0):
		super(SplunkGaussianDiscriminantAnalysis, self).__init__(host, port, username, password)
		self.alpha = alpha
		self.feature_mapping = {}
		self.class_mapping = {}
		self.mnvs = []
		self.cov_matrix = []
		self.sufficient_statistics = []
		self.class_curr = 0
		self.feature_curr = 0
		self.num_classes = 0
		self.num_features = 0
		

	def sufficient_statistics_splunk_search(self, search_string, feature_fields, class_field):
		splunk_search = 'search %s | stats avg, count by %s' % (search_string, class_field)
		
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(splunk_search, **search_kwargs)
		return job


	def populate_sufficient_statistics_from_search(self, job, feature_fields, class_field):
		'''
		populates sufficient statistics from splunk search
		'''
		result_count = int(job["resultCount"])
		self.num_classes = result_count # result count is the number of classes
		# 1: initialize numpy arrays
		self.sufficient_statistics=np.zeros((result_count, len(feature_fields)))
		self.cov_matrix = np.zeros((len(feature_fields), len(feature_fields)))
		self.priors=np.zeros(result_count)
		offset = 0
		count = 50
		class_curr = 0
		feature_curr = 0
		# 2: iterate through search
		while (offset < int(result_count)):
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):
				# set mapping
				# self.class_mapping[result[class_field]] = class_curr
				self.class_mapping[class_curr] = result[class_field]
				# update prior
				self.priors[class_curr] = result['count']
				for field in feature_fields:
					if field in self.feature_mapping:
						self.sufficient_statistics[class_curr][self.feature_mapping[field]] = result['avg(%s)' % field]
					else:
						self.feature_mapping[field] = feature_curr
						self.feature_mapping[feature_curr] = field
					# update sufficient statistics
						self.sufficient_statistics[class_curr][feature_curr] = result['avg(%s)' % field]
						feature_curr += 1

				class_curr += 1
			offset += count


		# 3: normalize priors, send to log space
		self.priors = self.priors / self.priors.sum()
		self.log_prob_priors = np.log(self.priors)

		return


	def make_covariance_matrix(self, search_string, feature_fields, class_field):
		'''
		once we have the averages, we can go through once more and create the covariance matrix
		'''
		# 1: make eval strings to store averages depending on class
		eval_string = ''
		for i in range(len(feature_fields)):
			eval_string += 'eval averg%st=case(' % i
			for j in range(self.num_classes-1):
				eval_string += '%s=%s, %s, ' % (class_field,self.class_mapping[j],self.sufficient_statistics[j][i])
			eval_string += '%s=%s,%s) | ' % (class_field, self.class_mapping[self.num_classes-1], self.sufficient_statistics[self.num_classes-1][i])
			
		# 2: add logic to get the matrix entry for i,j
		for i in range(len(feature_fields)):
			for j in range(i,len(feature_fields)):
				eval_string += 'eval matrix_%s%s=(%s - %s)*(%s - %s) | ' % (i,j,self.feature_mapping[i], 'averg%st' % i, self.feature_mapping[j], 'averg%st' % j)
		eval_string = eval_string[:-2]

		# 3: add stats call
		eval_string += ' | stats avg(matrix*)'

		splunk_search = 'search %s | %s' % (search_string, eval_string)
		
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(splunk_search, **search_kwargs)

		# 4: populate covariance matrix
		search_results = job.results()
		for result in results.ResultsReader(search_results):
			for i in range(len(feature_fields)):
				for j in range(i,len(feature_fields)):
					num = float(result['avg(matrix_%s%s)'%(i,j)])
					self.cov_matrix[i][j] = num
					self.cov_matrix[j][i] = num


		# 5: find determinant, store the root of it, and store inverse covariance matrix as well
		cov_det = np.linalg.det(self.cov_matrix)
		self.cov_det_root = cov_det**(.5)
		self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)

		return
		

	def fill_mnvs(self):
		# make num_classes different mnvs
		self.mnvs = [multivariate_normal(mean=self.sufficient_statistics[i],cov=self.cov_matrix) for i in range(self.num_classes)]




	def train(self, search_string, feature_fields, class_field):
		'''
		finds sufficient statistics for the GDA model

		sufficient statistics are: 
		1) Class Priors (bernoulli)
		2) Multivariate mean vectors per-class
		3) Covariance matrix (NOT per-class)
		'''
		

		#1: create the job that searches for sufficient statistics
		suff_stat_search = self.sufficient_statistics_splunk_search(search_string, feature_fields, class_field)

		#2: populate the sufficient statistics (priors are mnv mean vectors)
		# priors stored in log form in 'self.log_prob_priors'
		# mnv mean vectors stored in self.sufficient_statistics (num_classes x num_variables matrix)
		self.populate_sufficient_statistics_from_search(suff_stat_search, feature_fields, class_field)

		#3: make covariance matrix
		self.make_covariance_matrix(search_string, feature_fields, class_field)
		# self.fill_mnvs()
		self.trained=True



	def to_numpy_rep(self, event_to_predict, feature_fields):
		#1: initialize
		np_rep = np.zeros(len(feature_fields))

		#2: fill
		for feature in feature_fields:
			np_rep[self.feature_mapping[feature]] = event_to_predict[feature]

		#3: return
		return np_rep


	def predict_splunk_search(self, search_string, feature_fields, class_field, output_field):
		'''
		makes a search string that populates events with a new prediction field
		'''
		# 1: search the right events
		splunk_search = 'search %s | ' % search_string 

		# 2: initialize feature fector and get the meandiff vector
		'''
		TODO COMMENT: if can pass in self.sufficient_statistics as a vector rather than first going to splunk string, would be better
		'''
		features = sm.array(feature_fields)
		suffstats = sm.array(self.sufficient_statistics)
		meandiff = sm.sub(features,suffstats)

		# 3: now we're making the exp term in the multivariate gaussian pdf. it's meandiff dot cov dot meandiff.T
		# first we get dot(meandiff, inv_cov_matrix)
		icm = sm.array(self.inv_cov_matrix)
		temp = sm.dot(meandiff,icm)
		# cxn * nxn -> cxn
		# now cxn * nxc => cxc
		final = sm.dot(temp, meandiff.T())
		# finally we only want the elemnts on the diagonals
		final_expterms = sm.diag(final)
		# and we scale by -.5
		multiplied_expterms = sm.mul(final_expterms,-.5)
		# multiplied_expterms.rename('expterm')
		# make the pi term and ln it
		pi_term = np.pi**(len(feature_fields)/float(2))
		multterm = sm.ln(sm.array((1/(self.cov_det_root*pi_term))))
		prob_vec = sm.array(self.log_prob_priors)
		# splunk vector broadcasting takes care of the rest
		new_prob_vec = sm.add(sm.sub(prob_vec,multterm),multiplied_expterms)
		new_prob_vec.rename('prob')
		# make argmax splunkarray
		argmax_sa = sm.argmax(new_prob_vec)
		argmax_sa.rename('argmax_prob')
		# now the field argmax_prob_0_0 is the index of new_prob_vec's maximum argument
		case_mapping_string = sm.case_mapping(self.class_mapping, 'argmax_prob_0_0', output_field)
		splunk_search += splunk_concat(argmax_sa.string, case_mapping_string)
		
		# eval string needs to change, but all math is done
		# splunk_search += 'eval %s=if(prob_0_0>prob_0_1,"%s","%s")' % (output_field, self.class_mapping[0], self.class_mapping[1]) ## NEED TO CHANGE THE STRINGS 0, 1!!!


		
		return splunk_search


		



if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	gda = SplunkGaussianDiscriminantAnalysis(host="localhost", port=8089, username=username, password=password)
	gda.test_accuracy_splunk_search(reaction_search, reaction_search, reaction_features, reaction_class)
	# snb.train(reaction_search, reaction_features, reaction_class)
	# snb.make_splunk_prediction_search_string(reaction_search, reaction_features, reaction_class)
	# snb.evaluate_accuracy(reaction_search, reaction_features, reaction_class)
	# snb.compare_sklearn()
