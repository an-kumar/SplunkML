'''
SplunkML Gaussian Naive Bayes

Ankit Kumar
ankitk@stanford.edu
'''
import splunkmath as sm
from base_classes import SplunkClassifierBase

import splunklib.client as client
import splunklib.results as results

import numpy as np


reaction_features = ['field%s' % i for i in range(1,45)]
reaction_search = 'source="/users/ankitkumar/documents/code/splunkml/data/splunk_second_cont.txt"'
reaction_class = 'success'

class SplunkGaussianNaiveBayes(SplunkClassifierBase):
	'''
	The Gaussian Naive Bayes classifier works with the assumption that the X fields (features) are normally distributed and independant.

	It returns simply the argmax_y P(X_fields|y), where because each field is assumed to be independant, we aren't using a multivariate normal distribution (the covariance matrix would just be eye(num_features)).

	Thus, the sufficient statistics are: 
		- the class priors P(y)
		- the feature means per class
		- the feature variances per class
	'''

	def __init__(self, host, port, username, password):
		super(SplunkGaussianNaiveBayes, self).__init__(host, port, username, password)

	def predict_splunk_search(self, X_fields, Y_field, search_string, output_field):
		pass

	def predict_single_event(self, event_to_predict, X_fields, Y_field):
		pass

	def train(self, search_string, X_fields, Y_field):
		'''
		We need to extract the sufficient statistics listed under the class definition. To do this, we'll run a splunk search, initialize some splunk vectors, do some vector math, and be done with it.

		Strategy:
			- get a (numpy) vector of averages per feature per class from the first pass
			- go through the data again and get variances of features per class
		'''
		# get averages per feature, priors, and set mappings.
		# sets self.feature_averages (num_classes, num_features), self.priors(1, num_classes), and self.log_prob_priors
		# also sets self.class_mapping and self.feature_mapping
		self.set_mapping_get_feature_averages_and_priors(search_string, X_fields, Y_field)

		# go through data again and set self.feature_variances
		self.set_feature_variances(search_string, X_fields, Y_field)




	def set_feature_variances(self, search_string, X_fields, Y_field):
		# initialize search
		splunk_search = 'search %s' % search_string

		# initialize splunk vector from the feature averages
		feature_averages=  sm.array(self.feature_averages)












	def set_mapping_get_feature_averages_and_priors(self, search_string, X_fields, Y_field):
		job = self.feature_averages_splunk_search(search_string, X_fields, Y_field)
		# priors has shape (1, num_classes), feature_averages has shape (num_classes, num_features)
		priors, feature_averages = self.populate_feature_averages_from_search(job, X_fields, Y_field)
		self.priors = priors
		self.log_prob_priors = np.log(priors)
		self.feature_averages = feature_averages

	def feature_averages_splunk_search(self, search_string, feature_fields, class_field):
		splunk_search = 'search %s | stats avg, count by %s' % (search_string, class_field)
		print splunk_search
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(splunk_search, **search_kwargs)
		return job


	def populate_feature_averages_from_search(self, job, feature_fields, class_field):
		'''
		populates sufficient statistics from splunk search
		'''
		result_count = int(job["resultCount"])
		self.num_classes = result_count # result count is the number of classes
		# 1: initialize numpy arrays
		feature_averages=np.zeros((result_count, len(feature_fields)))
		
		priors=np.zeros(result_count)
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
				self.class_mapping[result[class_field]] = class_curr
				self.class_mapping[class_curr] = result[class_field]
				# update prior
				priors[class_curr] = result['count']
				for field in feature_fields:
					if field in self.feature_mapping:
						feature_averages[class_curr][self.feature_mapping[field]] = result['avg(%s)' % field]
					else:
						self.feature_mapping[field] = feature_curr
						self.feature_mapping[feature_curr] = field
					# update sufficient statistics
						feature_averages[class_curr][feature_curr] = result['avg(%s)' % field]
						feature_curr += 1

				class_curr += 1
			offset += count

		priors = priors / priors.sum()
		return priors, feature_averages



if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	snb = SplunkGaussianNaiveBayes(host="localhost", port=8089, username=username, password=password)
	snb.train(reaction_search, reaction_features, reaction_class)

		