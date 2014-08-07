'''
splunkvector.py
Ankit Kumar
ankitk@stanford.edu

This is an abstraction to quickly do vector operations in splunk.
'''

'''
TODO:
******implement .dot, -, +, *, etc passing in a NP VECTOR as second argument (rather than making it a splunkarray first. see comment in gda.py)*****
refactor with the new "set element" i.e .set_element(elem), will figure out if there is a pipe before/after, etc
maybe find_elements is too important to not always call? sucks to always have to do it.

'''

import numpy as np

def find_elements_from_name_shape(name, shape):
	return np.array([['%s_%s_%s' % (name, i, j) for j in range(shape[1])] for i in range(shape[0])], dtype=object)

def diag(sa):
	'''
	implements np.diag-like operations: NxN matrix -> 1XN matrix of the diagonals
	'''
	#check for square shape
	if not sa.shape[0] == sa.shape[1]:
		raise Exception ("That shape is not square! %s" % sa.shape)
	# make new splunk vector
	output = SplunkArray(sa.name + '_diag', (1, sa.shape[0]))
	# init string
	output.string = sa.string + ' | '
	output.find_elements()
	# add new logic
	for i, j in output.iterable():
		output.set_element(i, j, sa.elems[j][j])
		# output.string += 'eval %s = %s | ' % (output.elems[i][j], sa.elems[j][j])
	# output.string = output.string[:-2]
	return output


def from_scalar(name, scalar):
	sa = SplunkArray(name, (1,1))
	sa.string = 'eval %s_0_0 = %s' % (sa.name, str(scalar))
	sa.find_elements()
	return sa

def from_matrix(name, matrix):
	sa = SplunkArray(name, matrix.shape)
	sa.initialize_from_matrix(matrix)
	return sa

def from_vector(name, vector):
	sa = SplunkArray(name, len(vector))
	sa.initialize_from_vector(vector)
	return sa


def shape_from_passed_in(shape):
	# get shape/length
	if type(shape) == int:
		# make it a row vector
		return (1, shape)
	elif type(shape) == tuple:
		return shape
	else:
		raise Exception ("Shape passed in wasn't an int or a tuple. was a %s" % type(shape))

# ======= [ MATHEMATIC OPERATIONS ] ======= #
def dot(one, two):
	# check shapes
	if one.shape[1] != two.shape[0]:
		raise Exception ("Those shapes don't dot with each other! shapes were %s, %s" % (one.shape, two.shape))

	# initialize the output array
	output_sa = SplunkArray(one.name + '_dot_' + two.name, (one.shape[0], two.shape[1]))
	# set the output array's string
	output_sa.string = splunk_concat(one.string, two.string)
	# now calculate the dot product
	for i in range(output_sa.shape[0]):
		for j in range(output_sa.shape[1]):
			# A_i,j = the i'th row of "one" dotted with the j'th column of "two":
			output_sa.set_element(i, j, vector_dot_string(one.elems[i], two.elems[:,j]))
			# output_sa.string += 'eval %s_%s_%s = %s | ' % (output_sa.name, i, j, vector_dot_string(one.elems[i], two.elems[:,j]))
	# output_sa.string = output_sa.string[:-2]
	output_sa.find_elements()
	return output_sa

def elementwise_func(sa, func):
	'''
	elementwise func "func" on the elements of sa. func expected to be the name of a func in splunk i.e "ln"
	'''
	output = SplunkArray(func+'_d'+sa.name, sa.shape)
	output.string = sa.string 
	output.find_elements()
	for i,j in sa.iterable():
		output.set_element(i,j, func+'(%s)' % sa.elems[i][j])
	return output

def ln(sa):
	return elementwise_func(sa, 'ln')


def mul_scalar(one, two):
	# assumes two is a scalar
	output = SplunkArray(one.name+'_mul_', one.shape)
	output.string = one.string + ' | '
	output.find_elements()
	for i,j in output.iterable():
		output.set_element(i, j, '%s*%s' % (one.elems[i][j], str(two)))
		# output.string += 'eval %s = %s*%s | ' % (output.elems[i][j], one.elems[i][j], str(two))
	# output.string = output.string [:-2]
	return output


'''
def elementwise_multiplication(output_name, elems_one, elems_two):
	
	elementwise mult elems_one * elems_two
	
	# check they're the same shape
	if elems_one.shape != elems_two.shape:
		raise ("elementwise subtraction error: shapes not the same. %s, %s" % (elems_one.shape, elems_two.shape))
	string = ''
	for i in range(elems_one.shape[0]):
		for j in range(elems_one.shape[1]):
			string += 'eval %s_%s_%s = %s - %s | ' % (output_name, i, j, elems_one[i][j], elems_two[i][j])
	return string[:-2]
'''


def add(one, two):
	'''
	implements one - two using numpy-like broadcasting
	note that because splunkarrays are only 2-dimensional, this is significantly easier than numpy stuff
	'''
	#check shape
	new_shape = check_broadcasting(one.shape, two.shape)
	output = SplunkArray(one.name + '_add_' + two.name, new_shape)
	# broadcast to temporary SAs
	temp_elems_one = broadcast_sa_to_shape(one, new_shape)
	temp_elems_two = broadcast_sa_to_shape(two, new_shape)
	# fill string
	output.string = splunk_concat(one.string, two.string)
	#since now the two temps are the same size, we can do simple element wise subtraction
	output = elementwise_addition(output, temp_elems_one, temp_elems_two)
	# output.find_elements()
	return output

def elementwise_addition(output, elems_one, elems_two):
	'''
	elementwise subtraction elems_one - elems_two
	'''
	# check they're the same shape
	if elems_one.shape != elems_two.shape:
		raise ("elementwise subtraction error: shapes not the same. %s, %s" % (elems_one.shape, elems_two.shape))
	for i,j in output.iterable():
		output.set_element(i,j, '%s + %s' % (elems_one[i][j], elems_two[i][j]))
	# # string = ''
	# # for i in range(elems_one.shape[0]):
	# # 	for j in range(elems_one.shape[1]):
	# # 		string += 'eval %s_%s_%s = %s + %s | ' % (output_name, i, j, elems_one[i][j], elems_two[i][j])
	# return string[:-2]
	return output


# ======== [ MATHEMATIC OPERATIONS END ] ========== #

def vector_dot_string(fields_one, fields_two):
	'''
	assuming fields_one and fields_two are iterables with names in them and the two have the same size, outputs a string representing their dot product in splunk
	'''
	assert len(fields_one) == len(fields_two)
	string = ''
	for i in range(len(fields_one)):
		string += '(%s*%s) + ' % (fields_one[i], fields_two[i])

	return string[:-2]

def transpose(sa):
	'''
	returns the transpose of the splunk array. if just using for a dot product, use sa.T() instead.
	'''
	new_sa = SplunkArray(sa.name + '_tranpose', (sa.shape[1], sa.shape[0]))
	new_sa.string = sa.string + '| '
	for i in range(new_sa.shape[0]):
		for j in range(new_sa.shape[1]):
			new_sa.string += 'eval %s_%s_%s = %s_%s_%s | ' % (new_sa.name, i, j, sa.name, j, i)
	new_sa.string = new_sa.string[:-2]
	new_sa.find_elements()
	return new_sa



def check_broadcasting(shape_one, shape_two):
	'''
	implements numpy style broadcasting and returns the output shape, or raises an error if the shapes are not broadcastable
	'''
	if (shape_one[1] == shape_two[1] or shape_one[1] == 1 or shape_two[1] == 1):
		if (shape_one[0] == shape_two[0] or shape_one[0] == 1 or shape_two[0] == 1):
			return (max(shape_one[0], shape_two[0]), max(shape_one[1], shape_two[1]))
		else:
			raise Exception ("Those aren't broadcastable (%s, %s)" % (shape_one, shape_two))
	else:
		raise Exception ("Those aren't broadcastable (%s, %s)" % (shape_one, shape_two))



def broadcast_sa_to_shape(sa, shape):
	'''
	returns a temporary SA whos .elems field has the correct broadcasting rules
	'''
	# if the shape is already correct, just return it
	if sa.shape == shape:
		return sa.elems
	# else, make a temp one that simulates what we need
	
	temp_elems = np.zeros(shape, dtype=object)
	for i in range(shape[0]):
		for j in range(shape[1]):
			# there's a better way to do this probably, but this is simple. refactor later, maybe.
			# the new element i,j is sa.elem[i][0] if sa.shape[0] == shape[0], else sa.elem[0][j]
			# if both match, we already returned the old sa.
			if sa.shape[0] == shape[0]:
				temp_elems[i][j] = sa.elems[i][0]
			elif sa.shape[1] == shape[1]:
				temp_elems[i][j] = sa.elems[0][j]
			else:
				raise Exception ("something went wrong with broadcasting. Check deeper.")
	return temp_elems

def splunk_concat(one, two):
	'''
	concats the two strings with a pipe in between
	'''
	if one.split()[-1] == '|':
		return one + two
	else:
		return one + ' | ' + two

def sub(one, two):
	'''
	implements one - two using numpy-like broadcasting
	note that because splunkarrays are only 2-dimensional, this is significantly easier than numpy stuff
	'''
	#check shape
	new_shape = check_broadcasting(one.shape, two.shape)
	output = SplunkArray(one.name + '_sub_' + two.name, new_shape)
	# broadcast to temporary SAs
	temp_elems_one = broadcast_sa_to_shape(one, new_shape)
	temp_elems_two = broadcast_sa_to_shape(two, new_shape)
	# fill string
	output.string = splunk_concat(one.string, two.string)# + '| ' + two.string + '| '
	#since now the two temps are the same size, we can do simple element wise subtraction
	output = elementwise_subtraction(output, temp_elems_one, temp_elems_two)
	# output.find_elements()
	return output

def elementwise_subtraction(output, elems_one, elems_two):
	'''
	elementwise subtraction elems_one - elems_two
	'''
	# check they're the same shape
	if elems_one.shape != elems_two.shape:
		raise ("elementwise subtraction error: shapes not the same. %s, %s" % (elems_one.shape, elems_two.shape))
	for i,j in output.iterable():
		output.set_element(i, j, '%s - %s' % (elems_one[i][j], elems_two[i][j]))
	return output
	# string = ''
	# for i in range(elems_one.shape[0]):
	# 	for j in range(elems_one.shape[1]):
	# 		string += 'eval %s_%s_%s = %s - %s | ' % (output_name, i, j, elems_one[i][j], elems_two[i][j])
	# return string[:-2]





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

	def initialize_from_matrix(self, matrix):
		'''
		intializes the splunk array from a matrix (i.e M x N where M, N != 1)
		'''
		# todo: shape checking
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				# note that the shape of a SplunkArray vector is (1, length) i.e a row vector
				self.string += 'eval %s_%s_%s = %s | ' % (self.name, i, j, matrix[i][j]) 
		self.string = self.string[:-2]
		self.find_elements()


	def find_elements(self, name=None):
		self.elems = find_elements_from_name_shape(self.name, self.shape)

		

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
		renames the splunkarray to the given name
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
		return sub(self, other)

	def __add__(self, other):
		#implements a + b
		return add(self,other)

	def __mul__(self, other):
		# CURRENTLY ONLY IMPLEMENTS SCALAR OTHERS (ACTUAL SCALARS)
		if (type(other) == int or type(other) == float):
			return mul_scalar(self, other)





if __name__ == '__main__':
	print "> testing from vector"
	a = from_vector('test', [1,2,3,4,5,6,7,8])
	print a.string
	print a.elems
	print "\n\n\n"

	print "> testing from matrix"
	a = from_matrix('test', np.zeros((5,3)))
	print a.string
	print a.elems
	print "\n\n\n"

	print "> testing vector dot with .T()"
	a = from_vector('test_one', [1,2,3,4,5])
	b = from_vector('test_two', [1,4,7,8,1])
	c = dot(a, b.T())
	print c.string
	print c.elems
	assert c.shape == (1,1)
	print "\n\n\n"

	print "> testing vector dot with transpose()"
	a = from_vector('test_one', [1,2,3,4,5])
	b = from_vector('test_two', [1,4,7,8,1])
	c = dot(a, transpose(b))
	print c.string
	print c.elems
	assert c.shape == (1,1)
	print "\n\n"

	print "> testing broadcasting"
	output = check_broadcasting((1,1), (5,1))
	print output
	try:
		output = check_broadcasting((5,1),(4,1))
	except Exception as e:
		print e
	print output
	print "\n\n"

	print "> testing subtraction"
	c = a - b
	print c.shape
	print c.string
	print "\n\n"

	print "> testing subtraction with broadcasting"
	d = from_scalar('asdf', 1)
	e = d - b
	print e.shape
	print e.string

	print "> testing rename"
	print e.string
	print e.name
	print "TO:"
	e.rename('e')
	print e.string
	print e.name

	print "> testing dot product vector to matrix"
	a = from_matrix('test', np.zeros((5,3)))
	b = from_vector('test_two', [1,4,7,8,1])
	# 1x5 X 5x3 -> 1x3
	c = dot(b,a)
	print c.shape
	print c.string






