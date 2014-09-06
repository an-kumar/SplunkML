'''
SplunkML Gaussian Naive Bayes

Ankit Kumar
ankitk@stanford.edu
'''
import splunkmath as sm
from splunkmath.utils.strings import *
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

	def predict_splunk_search(self, search_string, X_fields, Y_field, output_field):
		'''
		uses splunkmath to predict every event in the splunk search.
		'''
		splunk_search = 'search %s' % search_string

		x = sm.array(X_fields)
		feature_averages = sm.array(self.feature_averages)
		feature_variances = sm.array(self.feature_variances)
		priors = sm.array(self.log_prob_priors)

		expterms = sm.div(sm.pow(sm.sub(x,feature_averages), 2), feature_variances)
		expterms = sm.mul(expterms, -.5)

		pi_terms = sm.mul(sm.ln(sm.mul(sm.mul(feature_variances,2),np.pi)), -.5)

		feature_probs = sm.add(pi_terms, expterms)

		class_probs = sm.sum(feature_probs, axis=1)

		final_probs = sm.add(priors, class_probs)
		final_probs.rename('final_probabilities')
		argmax_string = sm.argmax(final_probs) 

		strings_concatenated = splunk_concat(splunk_search, final_probs.string)
		strings_concatenated = splunk_concat(strings_concatenated, argmax_string)
		# now in the splunksearch, the field final_probabilities_maxval contains either 0,1,2... etc. so we add a final thing
		# this should be decomposed later
		final_string = 'eval %s = case(%s)' % (output_field, ','.join(['%s=%s, %s' % ('argmax_final_probabilities', i, i) for i in range(self.num_classes)]))
		return splunk_concat(strings_concatenated, final_string)


	def predict_single_event(self, event_to_predict, X_fields, Y_field):
		'''
		uses numpy to predict a single event
		'''
		# this is easy to do with pdfs, but just to make sure we have everything correct, we'll do everything out
		# turn the event into a numpy array
		x = np.zeros(len(X_fields))
		for feature in X_fields:
			x[self.feature_mapping[feature]] = event_to_predict[feature]

		# find the exp terms of the gaussian pdf (x - mu) ^2 / 2 (var)^2 (make it negative at the end)
		expterms = (x - self.feature_averages)**2 / self.feature_variances
		expterms = -.5*expterms

		# find the pi term
		pi_terms = -.5*(np.log(self.feature_variances*2*np.pi))

		# add them together since it's log space
		feature_probs = pi_terms + expterms
		# sum by rows (since in naive bayes P(x|y) = P(x1|y)P(x2|y)...)
		class_probs = np.sum(feature_probs, axis=1)
		# add back to original probabilities
		probabilities = self.log_prob_priors + class_probs
		return self.class_mapping[np.argmax(probabilities)]

	


	def train(self, search_string, X_fields, Y_field):
		'''
		We need to extract the sufficient statistics listed under the class definition. To do this, we'll run a splunk search with stats

		'''
		# get averages per feature, priors, and set mappings.
		# sets self.feature_variances, self.feature_averages (num_classes, num_features), self.priors(1, num_classes), and self.log_prob_priors
		# also sets self.class_mapping and self.feature_mapping
		job = self.feature_averages_variances_splunk_search(search_string, X_fields, Y_field)
		# priors has shape (1, num_classes), feature_averages has shape (num_classes, num_features)
		priors, feature_averages, feature_variances = self.populate_feature_averages_from_search(job, X_fields, Y_field)
		self.priors = priors
		self.log_prob_priors = np.log(priors)
		self.feature_averages = feature_averages
		self.feature_variances = feature_variances
		# that's all the sufficient statistics we need!






# --------------------------helper functions----------------------------------------





	def set_mapping_get_feature_averages_variances_and_priors(self, search_string, X_fields, Y_field):
		job = self.feature_averages_variances_splunk_search(search_string, X_fields, Y_field)
		# priors has shape (1, num_classes), feature_averages has shape (num_classes, num_features)
		priors, feature_averages, feature_variances = self.populate_feature_averages_from_search(job, X_fields, Y_field)
		self.priors = priors
		self.log_prob_priors = np.log(priors)
		self.feature_averages = feature_averages
		self.feature_variances = feature_variances

	def feature_averages_variances_splunk_search(self, search_string, feature_fields, class_field):
		splunk_search = 'search %s | stats avg, var, count by %s' % (search_string, class_field)
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
		feature_variances = np.zeros((result_count, len(feature_fields)))
		
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
						feature_variances[class_curr][self.feature_mapping[field]] = result['var(%s)' % field]

					else:
						self.feature_mapping[field] = feature_curr
						self.feature_mapping[feature_curr] = field
					# update sufficient statistics
						feature_averages[class_curr][feature_curr] = result['avg(%s)' % field]
						feature_variances[class_curr][feature_curr] = result['var(%s)' % field]
						feature_curr += 1

				class_curr += 1
			offset += count

		priors = priors / priors.sum()
		return priors, feature_averages, feature_variances



if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	gnb = SplunkGaussianNaiveBayes(host="localhost", port=8089, username=username, password=password)
	gnb.test_accuracy_splunk_search(reaction_search, reaction_search, reaction_features, reaction_class)
	# searchstring = snb.predict_splunk_search(reaction_features, reaction_class,reaction_search, 'test_output')
	# event = {x:60 for x in reaction_features}
	output= snb.test_accuracy_single_event(reaction_search, reaction_search, reaction_features, reaction_class)

		