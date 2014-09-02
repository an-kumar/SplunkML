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
from splunkmath.utils.strings import *
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
		# make a bias term
		splunk_search = 'search %s | eval bias=1' % search_string
		# instantiate arrays for x and theta
		x = sm.array(X_fields + ['bias']) # this is sorta hacky as well. maybe make it easier to do this with splunkmath?
		theta = sm.array(self.theta)
		# do the dot product
		h_x = sm.dot(theta, sm.transpose(x)) # by convention we need to tranpose x... maybe this is the wrong convention.
		h_x.rename_elem(0,0,output_field)
		splunk_search_string = splunk_concat(splunk_search, h_x.string)
		
		return splunk_search_string


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
	slr.test_accuracy_splunk_search(reaction_search, reaction_search,reaction_features, reaction_class)
	





