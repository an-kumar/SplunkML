'''
SplunkML naive bayes classifier

Ankit Kumar
ankitk@stanford.edu


Todo: 	(1) get better with numpy arrays...
		(2) add laplace smoothing - it's not the same as sklearn right now!
'''
import splunklib.client as client
import splunklib.results as results
#from sklearn.naive_bayes import BernoulliNB
#from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import numpy as np
from base_classes import SplunkClassifierBase
import sys
import splunkmath as sm
from splunkmath.utils.strings import *

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/Code/SplunkML/data/splunk_votes_correct.txt"'
vote_class = 'party'


reaction_features = ['field%s' % i for i in range(1,11)]
reaction_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_continuous_classification.txt"'
reaction_class = 'success'


class SplunkNaiveBayes(SplunkClassifierBase):

	def __init__(self, host, port, username, password, alpha=0.0):
		super(SplunkNaiveBayes, self).__init__(host, port, username, password)
		self.alpha = alpha
		self.mapping = {}
		self.sufficient_statistics = []
		self.class_curr = 0
		self.feature_curr = 0
		self.num_classes = 0
		self.num_features = 0
		
	def update_sufficient_statistics(self,class_val, num_hits, field, value):
		self.sufficient_statistics[self.mapping[class_val]][self.mapping['%s_%s' % (field,value)]] = num_hits
		return




	def sufficient_statistics_splunk_search(self, search_string, feature_fields, class_field):
		csl = self.make_csl(feature_fields + [class_field])
		search_string = 'search %s | table %s | untable %s field value |stats count by %s field value' % (search_string, csl, class_field, class_field)
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	
		job = self.jobs.create(search_string, **search_kwargs)
		return job


	def populate_sufficient_statistics_from_search(self, job, class_field):
		result_count = job["resultCount"]
		offset = 0
		count = 50

		while (offset < int(result_count)):
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):
					class_val = result['%s' % class_field]
					num_hits = int(result['count'])
					field = result['field']
					value = result['value']
					self.update_sufficient_statistics(class_val, num_hits, field, value)
			offset += count



	def make_csl(self, fields):
		# Makes a comma-seperated list of the fields
		string = ''
		for field in fields[:-1]:
			string += '%s, ' % field
		string += fields[-1]
		return string


	def initialize_sufficient_statistics(self,search_string, feature_fields, class_field):
		'''
			intializes sufficient statistics array by finding out size, and creates the mapping
		'''
		#1: search for all values of all fields in splunk
		search_string = 'search %s | stats values' % (search_string)
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)
		
		search_results = job.results()
		for result in results.ResultsReader(search_results):
			self.mapping = {result['values(%s)' % class_field][i]:i for i in range(len(result['values(%s)' % class_field]))}
			self.num_classes = len(self.mapping)
			self.class_mapping = {}
			for elem in self.mapping.items():
				self.mapping[elem[1]] = elem[0]
				self.class_mapping[elem[1]] = elem[0]
			curr_index = 0
			self.feature_onehot_mapping= {}
			self.onehot_ordering = []
			for field in feature_fields:
				self.feature_onehot_mapping[field] = []
				self.onehot_ordering.append(field)
				for value in result['values(%s)' % field]:
					self.feature_onehot_mapping[field].append(value)
					
					self.mapping['%s_%s' % (field,value)] = curr_index

					curr_index += 1

		self.num_features = curr_index

		#2: make the numpy array for the sufficient statistics:
		self.sufficient_statistics = np.zeros((self.num_classes,self.num_features))

		return



	def counts_to_logprobs(self):
		#1: compute priors before smoothing
		priors = self.sufficient_statistics.sum(axis=1)
		priors = priors / priors.sum()
		self.log_prob_priors = np.log(priors)

		#2: add smoothing
		self.sufficient_statistics = self.sufficient_statistics + self.alpha

		#3: turn sufficient stats into log probabilities
		probabs = self.sufficient_statistics / self.sufficient_statistics.sum(axis=1)[:,np.newaxis]
		self.log_prob_suff_stats = np.log(probabs)

		


	def train(self, search_string, feature_fields, class_field):
		'''
			train_classifier(search_string, feature_fields, class_field)

			search_string: string to search splunk with
			feature_fields: fields to use as features
			class_field: field to predict

			returns: nothing, but sufficient statistics are populated

			notes: sufficient statistics are priors (P(c=x)) for each class c, and P(x_i=true) for each x_i, where an x_i exists for each field-value pair in the feature fields
		'''
		#1: find out how big the sufficient statistic array needs to be, and create the string->index mapping
		self.initialize_sufficient_statistics(search_string, feature_fields, class_field)
		

		#2: create the job that searches for sufficient statistics
		suff_stat_search = self.sufficient_statistics_splunk_search(search_string, feature_fields, class_field)

		#3: populate the sufficient statistics
		self.populate_sufficient_statistics_from_search(suff_stat_search, class_field)

		#4: turn counts into empirical log-probabilities
		self.counts_to_logprobs()
		


	def to_numpy_rep(self, event_to_predict, feature_fields):
		#1: initialize
		np_rep = np.zeros((self.num_features,1))

		#2: add features that the event has; if we've never seen one before, ignore it
		for field in feature_fields:
			if field not in event_to_predict:
				continue
			val = event_to_predict[field]
			if '%s_%s' % (field, val) in self.mapping:
				np_rep[self.mapping['%s_%s' % (field,val)]] = 1

		#3: return
		return np_rep




	def predict_single_event(self, event_to_predict, X_fields, Y_field):
		'''

			notes: uses naive bayes assumption: P(c=x) is proportional P(x_i's|c)P(c). P(c) is the prior, P(x_i's|c) decomposes to 
			P(x_1|c)P(x_2|c)...P(x_n|c); these are all calculated in log space using dot product.
		'''
		# 1: get numpy representation
		numpy_rep = self.to_numpy_rep(event_to_predict, X_fields)

		# 2: find P(x_'s|c) using naive bayes assumption
		class_log_prob = np.dot(self.log_prob_suff_stats, numpy_rep)[:,0]

		print class_log_prob
		
		# 3: multiply in (add in log space) priors
		class_log_prob += self.log_prob_priors


		
		return self.class_mapping[np.argmax(class_log_prob)]

	def predict_splunk_search(self, search_string, X_fields, Y_field, output_field):
		'''
		returns a string that contains the correct field
		'''
		splunk_search = 'search %s | ' % search_string 
		x = sm.to_one_hot(X_fields, onehot_mapping=self.feature_onehot_mapping, ordering=self.onehot_ordering)
		class_log_probs = sm.array(self.log_prob_suff_stats)

		# dot the two
		x_dot_logprobs = sm.dot(x, class_log_probs.T())
		priors = sm.array(self.log_prob_priors)

		# add the priors
		final_probs = sm.add(x_dot_logprobs, priors)

		argmax_sa = sm.argmax(final_probs)
		argmax_sa.rename('argmax_prob')
		# now the field argmax_prob_0_0 is the index of new_prob_vec's maximum argument
		case_mapping_string = sm.case_mapping(self.class_mapping, 'argmax_prob_0_0', output_field)
		splunk_search += splunk_concat(argmax_sa.string, case_mapping_string)
		print splunk_search
		return splunk_search





if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	snb = SplunkNaiveBayes(host="localhost", port=8089, username=username, password=password)
	snb.test_accuracy_splunk_search(vote_search, vote_search, vote_features, vote_class)
	# snb.compare_sklearn()
