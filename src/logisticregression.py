
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
