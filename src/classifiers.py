'''
SplunkML classifiers

Ankit Kumar
ankitk@stanford.edu


Todo: 	(1) get better with numpy arrays...
		(2) add laplace smoothing - it's not the same as sklearn right now!
'''

import splunklib.client as client
import splunklib.results as results
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import numpy as np
import sys

vote_features = ['handicapped_infants', 'water_project_cost_sharing', 'adoption_of_the_budget_resolution','physician_fee_freeze', 'el_salvador_aid', 'religious_groups_in_schools', 'anti_satellite_test_ban','aid_to_nicaraguan_contras','mx_missile','immigration','synfuels_corporation_cutback','education_spending','superfund_right_to_sue','crime','duty_free_exports']
vote_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes_correct.txt"'
vote_class = 'party'


reaction_features = ['field%s' % i for i in range(1,11)]
reaction_search = 'source="/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_continuous_classification.txt"'
reaction_class = 'success'

# use host="localhost",port=8089,username="admin",password="flower00"
# use data_file = "/Users/ankitkumar/Documents/coding/205Consulting/OpenSource/SplunkML/naivebayes/splunk_votes.txt"

class SplunkClassifierBase(object):
	''' SplunkClassifierBase

	Base class of splunk classifiers. Functionality includes evaluate_accuracy(feature_fields, class_field)

	'''


	def __init__(self, host, port, username, password):
		self.service = client.connect(host=host, port=port, username=username, password=password)
		self.jobs = self.service.jobs
		self.trained = False
		self.feature_fields = None
		self.accuracy_tested = False


	def predict(self, feature_fields, class_field, event_to_predict):
		'''
			to overwrite
		'''
		pass


	def train_classifier(self, search_string, feature_fields, class_field):
		'''
			to overwrite
		'''
		pass

	def train(self, search_string, feature_fields, class_field):
		self.train_classifier(search_string, feature_fields, class_field)
		self.trained=True
		

	def compare_sklearn(self, np_reps, gold):
		'''
			to overwrite
		'''
		pass


	def check_accuracy(self, search_string, feature_fields, class_field):
		'''
			check_accuracy(search_string, feature_fields, class_field)

			search_string: string to use in the splunk search to narrow events
			feature_fields: which fields to use to predict
			class_field: field to predict

			returns: accuracy of prediction

			notes: assumes that classifier is already trained. calls predict on each event.
		'''
		# 1: check that classifier is trained:
		if not self.trained:
			raise 'classifier is not trained'

		# 2: search for the events
		print "searching"
		search_string = 'search %s | table *' % search_string
		print search_string
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(search_string, **search_kwargs)
		result_count = int(job["resultCount"])
		print "iterating"
		# 3: iterate through events, calling predict on each one. record results.
		correct = 0
		offset = 0
		count = 50
		np_reps = []
		gold = []
		while (offset < result_count):
			print offset
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):
				predicted_class, np_rep, actual_class = self.predict(feature_fields, class_field,result, return_numpy_rep=True)
				np_reps.append(np_rep)
				gold.append(actual_class)
				if predicted_class == result[class_field]:
					correct += 1
			offset += count
		self.np_reps = np_reps
		self.gold = gold


		# 4: calculate percentage
		perc_correct = float(correct)/result_count
		self.accuracy = perc_correct
		self.accuracy_tested = True

		# 5: return
		return perc_correct


	def evaluate_accuracy(self, search_string, feature_fields, class_field):
		'''
			evaluate_accuracy()

			trains the classifier, then predicts each of the events it trains on and records how many were correct
		'''
		print "Now evaluating %s test set accuracy." % self.__class__.__name__

		#1 : train the classifier
		print "--> Training the classifier..."
		self.train(search_string, feature_fields, class_field)
		print "... done."

		#2 : check accuracy
		print "--> Iterating through test set and checking accuracy..."
		accuracy = self.check_accuracy(search_string, feature_fields, class_field)
		print "done."

		#3 : print result
		print "Accuracy was %f." % accuracy







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
			for elem in self.mapping.items():
				self.mapping[elem[1]] = elem[0]
			curr_index = 0
			for field in feature_fields:
				for value in result['values(%s)' % field]:
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

		


	def train_classifier(self, search_string, feature_fields, class_field):
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



	def predict(self, feature_fields, class_field, event_to_predict, return_numpy_rep=False):
		'''
			predict(*)

			notes: uses naive bayes assumption: P(c=x) is proportional P(x_i's|c)P(c). P(c) is the prior, P(x_i's|c) decomposes to 
			P(x_1|c)P(x_2|c)...P(x_n|c); these are all calculated in log space using dot product.
		'''
		# 1: get numpy representation
		numpy_rep = self.to_numpy_rep(event_to_predict, feature_fields)

		# 2: find P(x_'s|c) using naive bayes assumption
		class_log_prob = np.dot(self.log_prob_suff_stats, numpy_rep)[:,0]

		# 3: multiply in (add in log space) priors
		class_log_prob += self.log_prob_priors
		if return_numpy_rep:
			actual_class = self.mapping[event_to_predict[class_field]]
			return self.mapping[np.argmax(class_log_prob)], numpy_rep.T[0], actual_class
		else:
			return self.mapping[np.argmax(class_log_prob)]


	def compare_sklearn(self):
		'''
			compares our implementation to sklearn's implementation. 

			assumes that evaluate_accuracy has been called.
		'''
		if not self.accuracy_tested:
			raise 'you must test the accuracy of the classifier before comparing to sklearn'
		print "--> Checking sklearn's accuracy..."
		X = np.array(self.np_reps)
		nb = BernoulliNB(alpha=0)
		y = np.array(self.gold)
		nb.fit(X,y)
		print "...done."
		print "sklearn accuracy is %f. Our accuracy was %f. " % (nb.score(X,y), self.accuracy)





#------------------------------------------------------------------------------------------------------------------------------------#

class SplunkLogisticRegression(SplunkClassifierBase):
	def __init__(self, host, port, username, password, training='batch', regularization=False):
		super(SplunkLogisticRegression, self).__init__(host, port, username, password)
		self.training = training
		self.mapping = {1:'1', 0:'0'} #change




	def initialization(self, feature_count):
		#1: initialize size of features
		self.feature_count = feature_count
		#1: set theta to be 0s
		self.theta = np.zeros(feature_count + 1) # +1 for the theta_0 term (see andrew ng's notes)



	def make_batch_gradient_descent_search(self, search_string, feature_fields, class_field):
		'''
		'''
		# 1: make the different strings to compute sigmoid
		z_string = 'eval z=(-1)*'
		eval_string = ''
		stats_sum_string = 'stats sum(sum_*)'
		for i in range(len(feature_fields)):
			z_string += '(%s*%s)+' %(feature_fields[i],self.theta[i])
			eval_string += 'eval sum_%s=result*%s | ' % (i, feature_fields[i])
		z_string += '%s' % self.theta[-1]
		eval_string += 'eval sum_end=result'
		
		# 2: turn into a splunk search
		splunk_search = 'search %s | %s | eval sigmoiddenom = 1 + exp(z) | eval sigmoid = if(sigmoiddenom=="inf",0,1/sigmoiddenom) | eval result = %s - sigmoid | %s | %s' % (search_string, z_string, class_field, eval_string, stats_sum_string)

		# 3: return
		print splunk_search
		return splunk_search




	def splunk_batch_gradient_descent(self, search_string, feature_fields, class_field, alphas=[2.0,1.5,1.0,.5,.3,.2,.1,.01,.001,.00001,.0000001], maxIter=1000, convergence=.01):
		'''
		'''
		# current_diff = np.ones((1,self.feature_count+1))*100 #initialize
		for iternum in range(maxIter):
			print 'iter: %s' % iternum

			#1: make the new splunk search
			splunk_search = self.make_batch_gradient_descent_search(search_string, feature_fields, class_field)
			search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
			job = self.jobs.create(splunk_search, **search_kwargs)

			search_results = job.results()
			old_theta = np.copy(self.theta)
			#2: iterate and update theta
			for result in results.ResultsReader(search_results):
				for i in range(self.feature_count):
					# update theta_i
					self.theta[i] += alphas[iternum]*float(result['sum(sum_%s)' % i])
				self.theta[-1] += alphas[iternum]*float(result['sum(sum_end)'])

			#3: check convergence
			diff = np.linalg.norm(old_theta - self.theta)
			print diff
			print old_theta
			print self.theta
			if diff < convergence:
				break
			else:
				print "difference: %f" % diff




	def sigmoid_function(self, z):
		return (1 / (1 + np.exp(z)))

		
	def find_h_x(self, feature_fields, event_to_predict):
		'''
		'''
		#1: find z = theta . x
		z = 0
		for i in range(len(feature_fields)):
			z += float(event_to_predict[feature_fields[i]])*self.theta[i]
		z += self.theta[-1] # add intercept
		z *= -1 #make negative

		#2: do sigmoid function
		sigmoid = self.sigmoid_function(z)

		#3: return
		return sigmoid



	def evaluate_accuracy(self, search_string, feature_fields, class_field):
		self.train(search_string, feature_fields, class_field)
		corr = 0
		total = 0
		job = self.predict(search_string, feature_fields, class_field, False)
		offset = 0
		count = 1000
		result_count = int(job["resultCount"])
		while (offset < result_count):
			print offset
			kwargs_paginate = {'count': count, 'offset':offset}
			search_results = job.results(**kwargs_paginate)
			for result in results.ResultsReader(search_results):

		
				if result[class_field] == result['predicted_splunkML']:
					corr += 1
					total += 1
				else:
					total += 1

			offset += count

		print "acc: "
		print float(corr)/(total)


	def predict(self,search_string, feature_fields,class_field,event_to_predict, return_numpy_rep=False):
		'''
		predict(*):
		takes in a string representing a search; returns a splunk job where each event in the search has a new field,
		'predicted_splunkML', which is the predicted value for that event.
		'''
		#1: make z string (z = theta transpose x)
		z_string = 'eval z=(-1)*'
		for i in range(len(feature_fields)):
			z_string += '(%s*%s)+' %(feature_fields[i],self.theta[i])

		z_string += '%s' % self.theta[-1]

		
		# 2: add logic to turn into sigmoid
		splunk_search = 'search %s | %s | eval sigmoiddenom = 1 + exp(z) | eval sigmoid = if(sigmoiddenom=="inf",0,1/sigmoiddenom) | eval predicted_splunkML=if(sigmoid<.5, %s, %s) | table %s, predicted_splunkML' % (search_string,z_string, self.mapping[0], self.mapping[1], class_field)
		print splunk_search

		# 3: search and return
		search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
		job = self.jobs.create(splunk_search, **search_kwargs)

		return job

		# #1: find h(x) for this event x
		# h_of_x = self.find_h_x(feature_fields, event_to_predict)

		# #2: return the closer value
		# if h_of_x > .5:
		# 	if return_numpy_rep:
		# 		return 1, False, False
		# 	else:
		# 		return '1'
		# else:
		# 	if return_numpy_rep:

		# 		return 0,False,False
		# 	else:
		# 		return '0'







	def train_classifier_batch(self, search_string, feature_fields, class_field):
		'''
		'''
		#1: initalize theta parameter
		self.initialization(len(feature_fields))

		#2: train theta using batch gradient descent
		self.splunk_batch_gradient_descent(search_string, feature_fields, class_field)


	def train_classifier(self, search_string, feature_fields, class_field):
		'''
			train_classifier

			trains the classifier given the feature fields and class field

			feature_fields: list of strings corresponding to features
			class_field: string corresponding to class field
		'''
		if self.training=='batch':
			self.train_classifier_batch(search_string, feature_fields, class_field)
		else:
			pass



	def compare_sklearn(self):
		'''
			compares our implementation to sklearn's implementation. 

			assumes that evaluate_accuracy has been called.
		'''
		if not self.accuracy_tested:
			raise 'you must test the accuracy of the classifier before comparing to sklearn'
		print "--> Checking sklearn's accuracy..."
		X = np.array(self.np_reps)
		LR = LogisticRegression(alpha=0)
		y = np.array(self.gold)
		LR.fit(X,y)
		print "...done."
		print "sklearn accuracy is %f. Our accuracy was %f. " % (LR.score(X,y), self.accuracy)






if __name__ == '__main__':
	username = raw_input("What is your username? ")
	password = raw_input("What is your password? ")
	# snb = SplunkNaiveBayes(host="localhost", port=8089, username=username, password=password)
	# snb.evaluate_accuracy(vote_search, vote_features, vote_class)
	# # snb.compare_sklearn()


	logit = SplunkLogisticRegression(host="localhost", port=8089, username=username, password=password)
	logit.evaluate_accuracy(reaction_search, reaction_features, reaction_class)






