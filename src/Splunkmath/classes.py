
'''
ankit kumar
ankit@205consulting.com
'''
'''
OVERALL CLASS: SplunkArray
Implements scalar, vector, matrix all in one. i.e a scalar is a 1x1 SplunkArray
'''

class SplunkArray(object):

	'''
	only persistent features of a splunkarray (totally defined by the following):
		-shape: shape of the array. Currently, vectors are (1, n) rather than (n,) as in numpy.
		-name: name of the array, and defines what the fields look like in splunk
		-string: a splunk string that defines how to "get to this point" with the vector
	'''
	def __init__(self, name, shape):
		'''
		INTERNAL FUNCTION: __init()
		usage: used internally to create a SplunkArray. To create a splunk array to use in code, etc, use one of the splunkmath APIs, such as .array(), .from_*, etc.
		'''
		# shape: shape of the array
		self.shape = self.shape_from_passed_in(shape)
		# name: what to call it in the string
		self.name = name
		# string: how to create this object in a splunk search
		self.string = ''
		# elems: the names of the elements of this object
		# self.elems = np.array([],dtype=object) #stored as np arrays
		self.find_elements()


	def shape_from_passed_in(self, shape):
		# get shape/length
		if type(shape) == int:
			# make it a row vector
			return (1, shape)
		elif type(shape) == tuple:
			return shape
		else:
			raise Exception ("Shape passed in wasn't an int or a tuple. was a %s" % type(shape))

# 	def initialize_from_scalar(self, scalar):
# 		raise NotImplementedError

# # -------- TO REWRITE: find_elements() should also set the string perhaps? pass in array to __init__ numpy style?

# 	def initialize_from_vector(self, vector):
# 		'''
# 		initializes the splunk array from a vector
# 		'''
# 		# todo: shape checking
# 		for i in range(self.shape[0]):
# 			for j in range(self.shape[1]):
# 				# note that the shape of a SplunkArray vector is (1, length) i.e a row vector
# 				self.string += 'eval %s_%s_%s = %s | ' % (self.name, i, j, vector[j]) 
# 		self.string = self.string[:-2]
# 		self.find_elements()

# 	def initialize_from_matrix(self, matrix):
# 		'''
# 		intializes the splunk array from a matrix (i.e M x N where M, N != 1)
# 		'''
# 		# todo: shape checking
# 		for i in range(self.shape[0]):
# 			for j in range(self.shape[1]):
# 				# note that the shape of a SplunkArray vector is (1, length) i.e a row vector
# 				self.string += 'eval %s_%s_%s = %s | ' % (self.name, i, j, matrix[i][j]) 
# 		self.string = self.string[:-2]
# 		self.find_elements()


# 	def find_elements(self, name=None):
# 		self.elems = find_elements_from_name_shape(self.name, self.shape)

		

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

	def rename(self, new_name):
		'''
		renames the splunkarray to the given name using regexp
		'''
		new_elems = find_elements_from_name_shape(new_name, self.shape)
		for i,j in self.iterable():
			self.string = self.string.replace(self.elems[i][j], new_elems[i][j])
		self.name = new_name
		self.elems = new_elems
		return self


	def iterable(self):
		'''
		returns a generator going through i,j in self.elems
		'''
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				yield i,j

	def set_element(self, i, j, val):
		'''
		sets element i,j to val
		'''
		# see if there's a trailing pipe
		if self.string.split()[-1] != '|':
			self.string += ' | '
		# set the eval string
		self.string += 'eval %s=%s' % (self.elems[i][j], val)




	# ========================== [ MAGIC METHODS ] ========================== #

	def __sub__(self, other):
		# implements a - b
		raise NotImplementedError
		# return sub(self, other)

	def __add__(self, other):
		#implements a + b
		raise NotImplementedError
		# return add(self,other)

	def __mul__(self, other):
		raise NotImplementedError

		# return mul(self, other)

	def __div__(self, other):
		raise NotImplementedError
		

if __name__ == '__main__':
	print find_elements_from_name_shape

