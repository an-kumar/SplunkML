'''
splunkvector.py
Ankit Kumar
ankitk@stanford.edu

This is an abstraction to quickly do vector operations in splunk.
'''
import numpy as np

'''
TODOS:
1. pass in a vector to initialize, can find length automatically
'''


class splunkmathbase(object):
	'''
	splunkmathbase:
	---------------

	base class for all splunk math classes. has name, curr_string, and elems.
	name: name of the object
	curr_string: a string that will end up with the object being correct
	elems: the name of the elements of the object
	'''
	def __init__(self, name):
		self.name = name
		self.curr_string = ''
		self.elems = []
		self.initialized = False
		self.shape = ()


	def get_string(self):
		return self.curr_string

	def __sub__(self, other):
		return sm.subtract(self, other)

class splunkscalar(splunkmathbase):
	'''
	splunkscalar:
	-------------

	simple scalar class for splunk math.
	the elements is simply the name (i.e if the scalar's name is "gamma", the field containing the value will be eval gamma =x)
	'''
	def __init__(self, name, number=None):
		self.name = name
		self.type = 'scalar'
		self.elems = [name]
		self.shape = (1,1)
		if number:
			self.initialize_from_number(number)

	def initialize_from_number(self, number):
		self.curr_string += 'eval %s = %s' % (self.name, str(number))
		self.initialized =True

class splunkvector(splunkmathbase):
	'''
	splunkvector:
	-------------

	vector class for splunk math. guaranteed to be 1d vector (else use splunkmatrix).
	elements are the name_i for i in range(length)
	'''

	def __init__(self, name, length, initialize_from=None):
		splunkmathbase.__init__(self, name)
		self.type = 'vector'
		# store the shape of the vector at all times
		self.length = length
		self.shape = (1,length)
		# not initialized yet (call initialize, initialize_random, or initialize_from_fields)
		if initialize_from is not None:
			self.initialize_from_given(initialize_from)
			

	def initialize_from_given(self, initialize_from):
		''' initializes from the string/array given '''
		if type(initialize_from) == str:
			if initialize_from == 'zeros':
				self.initialize()
			elif initialize_from == 'random':
				self.initialize_random()
		else:
			self.initialize_from_array(initialize_from)

	def initialize(self):
		''' initializes the vector at 0 '''
		for i in range(self.length):
			self.curr_string += 'eval %s_%s = 0 | ' % (self.name, i)
		self.curr_string = self.curr_string[:-2]
		self.initialized =True

	def initialize_random(self):
		''' initializes the vector randomly '''
		for i in range(self.length):
			self.curr_string += 'eval %s_%s = %f | ' % (self.name, i, np.random.rand())
		self.curr_string = self.curr_string[:-2]
		self.initialized =True


	def initialize_from_array(self, array):
		''' initializes the vector from the fields given '''
		for i in range(self.length):
			self.curr_string += 'eval %s_%s = %s | ' % (self.name, i, array[i])
		self.curr_string = self.curr_string[:-2]
		self.initialized =True



'''
OVERALL CLASS: SplunkArray
Implements scalar, vector, matrix all in one. i.e a scalar is a 1x1 SplunkArray
'''

class SplunkArray(object):

	'''
	only persistent features of a splunkarray (totally defined by the following):
		-shape 
		-name 
		-string
		-elems
	'''
	def __init__(self, name, shape):
		# shape: shape of the array
		self.shape = self.shape_from_passed_in(shape)
		# name: what to call it in the string
		self.name = name
		# string: how to create this object in a splunk search
		self.string = ''
		# elems: the names of the elements of this object
		self.elems = np.array([],dtype=object) #stored as np arrays


	def shape_from_passed_in(self, shape):
		# get shape/length
		if type(shape) == int:
			# make it a row vector
			return (1, shape)
		elif type(shape) == tuple:
			return shape
		else:
			raise Exception ("Shape passed in wasn't an int or a tuple. was a %s" % type(shape))

	def initialize_from_scalar(self, scalar):
		raise NotImplementedError



	def initialize_from_vector(self, vector):
		'''
		initializes the splunk array from a vector
		'''
		# todo: shape checking
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				# note that the shape of a SplunkArray vector is (1, length) i.e a row vector
				self.string += 'eval %s_%s_%s = %s | ' % (self.name, i, j, vector[j]) 
		self.string = self.string[:-2]
		self.find_elements()

	def find_elements(self):
		self.elems = np.array([['%s_%s_%s' % (self.name, i, j) for j in range(self.shape[1])] for i in range(self.shape[0])])
		

	def T(self):
		'''
		this function is used primarily for dot products as it has less overhead than splunkmath.tranpose(). 
		usage: dotprod = sm.dot(a, b.T())
		'''
		# new sa is a temporary one used for dot
		new_sa = SplunkArray('temp_T', (self.shape[1], self.shape[0]))
		new_sa.string = self.string
		new_sa.elems = self.elems.T
		return new_sa



	# ========================== [ MAGIC METHODS ] ========================== #

	def __sub__(self, other):
		# implements a - b
		









if __name__ == '__main__':
	print "===[ Simple Tests: splunkvector ]==="


	print "> test: initialize"
	zeros = splunkvector('testzeros',10, 'zeros')
	# zeros.initialize()
	print zeros.get_string()

	print "> test: initialize_random"
	random = splunkvector('testrandom',10, 'random')
	# random.initialize_random()
	print random.get_string()


	print "> test: initialize_from_fields"
	fields= splunkvector('testrandom',10, ['field_%s' % i for i in range(10)])
	# fields.initialize_from_fields(['field_%s' % i for i in range(10)])
	print fields.get_string()

	print "===[ Simple tests: splunkmath ]==="
	print "> test: vectordot"
	output = sm.dot(zeros, random)
	print output.get_string()


