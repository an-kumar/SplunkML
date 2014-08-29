'''
SplunkML Linear Regression

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
from splunkmath.classes import SplunkArray

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/Code/SplunkML/naivebayes/splunk_votes_correct.txt"'
vote_class = 'party'


reaction_features = ['field%s' % i for i in range(1,44)]
reaction_search = 'source="/users/ankitkumar/documents/code/splunkml/data/splunk_second_cont.txt"'
reaction_class = 'field44' # this can actually be any field, there are 44 of them; but make sure it's removed from reaction features
assert reaction_class not in reaction_features

class SplunkGaussianNaiveBayes(SplunkRegressorBase):
	'''
	towrite (description)
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
		trains the model using normal equations

		theta = (X^T X)^-1 X^T y
		'''
		# need to find X^T and X...

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






