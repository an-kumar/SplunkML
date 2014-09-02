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
from base_classes import SplunkRegressorBase
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

class SplunkLinearRegression(SplunkRegressorBase):
	'''
	towrite (description)
	'''

	def __init__(self, host, port, username, password):
		super(SplunkLinearRegression, self).__init__(host, port, username, password)

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
		x, y = sm.event_to_numpy_reps_continuous_continuous(event_to_predict, self.feature_mapping, Y_field, bias=True)	

		# h(x) defined as theta^T x
		h_x = np.dot(self.theta.T, x)
		
		return h_x

	


	def train(self, search_string, X_fields, Y_field):
		'''
		trains the model using normal equations

		theta = (X^T X)^-1 X^T y
		'''
		# doesn't seem like finding these values in splunk itself is the most efficient; if we just pull out X and y, 
		# we can find everything using numpy. So let's do that
		# make a feature mapping
		feature_mapping = {X_fields[i]:i for i in range(len(X_fields))}
		self.feature_mapping =feature_mapping
		# make the search job
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create('search ' +search_string + '| table %s' % ' '.join(X_fields) + ' ' + Y_field, **search_kwargs)
		X, y = sm.job_to_numpy_reps(job, feature_mapping, Y_field, ('continuous','continuous'), bias=True)

		# now we solve the normal equations to find theta
		self.theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
		print self.theta.shape

if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	slr = SplunkLinearRegression(host="localhost", port=8089, username=username, password=password)
	slr.test_accuracy_single_event(reaction_search, reaction_search,reaction_features, reaction_class)
	





