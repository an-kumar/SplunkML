from utils.strings import *
import time
import numpy as np
from classes import *
'''
numpyfuncs: functions that copy numpy funtions, but written for splunkmath
also functions that aren't exactly numpy.
'''

def to_one_hot(fields, onehot_mapping=None, ordering=None):
	'''
	returns a splunkarray that has elements the onehot mapping of fields.

	onehot_mapping is assumed to be a dict that maps field -> list of values that field can take on.
	ordering is assumed to be a list of exactly the elements in onehot_mapping

	ONLY RETURNS A VECTOR
	'''
	if onehot_mapping is None or ordering is None:
		
		raise NotImplementedError #to implement; perhaps this is needed.
	# first, we make a list of tuples (field, field_value) in the correct order
	tuples = []
	for elem in ordering:
		for value in onehot_mapping[elem]:
			tuples.append((elem,value))
	# THIS MIGHT NOT WORK IF THE FIELD VALUES ARE INTS!!!
	splunk_string = ' | '.join(['eval onehot_0_{j}=if({correct_field} == "{field_value}",1,0)'.format(j=j, correct_field=tuples[j][0], field_value=tuples[j][1]) for j in range(len(tuples))])
	fields = ['onehot_0_%s' % i for i in range(len(tuples))]
	
	sa = array(fields)
	sa.string = splunk_concat(splunk_string, sa.string)
	return sa
	



def argmax(sa):
	'''
	implements splunkmath equivalent of np.argmax()

	params:
		-sa: a splunk array
		-mapping: (class_mapping, output_field) tuple for classifiers
	returns:
		-a string with an new eval field, 'argmax_sa.name', that contains in it 
	notes;
		-CURRENTLY ONLY SUPPORTS VECTORS!!!!
		- THIS NEEDS TO HEAVILY CHANGE ( return a splunkarray maybe )
	'''
	string = 'eval maxval = max(%s)' % ','.join([str(elem) for elem in sa.elems[0]])
	nextstring = 'eval argmax_%s = case(%s)' % (sa.name,','.join(['%s == maxval, %s' % (str(sa.elems[0][i]), str(i)) for i in range(len(sa.elems[0]))]))
	full = splunk_concat(string, nextstring)
	new_sa = array(['argmax_%s' % sa.name])
	new_sa.string = splunk_concat(sa.string, full)
	return new_sa

def array(argument):
	'''
	implements a splunkarray equivalent of np.array()

	params:
		-argument: either a scalar, a list, a list of lists, or a numpy array
	returns:
		- a splunkarray from the input argument
	'''
	# try a bunch of different types:
	if type(argument) in set([float, np.float64, int]):
		shape = (1,1)
		elems = np.array([[argument]])
	elif type(argument) == list:
		if type(argument[0]) == list:
			shape = (len(argument), len(argument[0]))
			elems = np.array(argument)
		else:
			shape = (1, len(argument))
			elems = np.array([argument])
	elif type(argument) == np.ndarray:
		# numpy uses the (n,) convention for n length arrays - so far, splunkmath uses (1,n). so we need to check for that.
		if len(argument.shape) == 1:
			shape = (1, argument.shape[0])
			elems = np.array([argument])
		else:
			shape = argument.shape
			elems = argument

	else:
		print argument
		raise Exception("You didn't pass in a float, int, list, or numpy array. You passed in a %s" % type(argument))

	# now initialize an empty SplunkArray, name doesn't matter
	sa = SplunkArray('field' + sha_hash(str(time.time())), shape)
	# set the elements to the argument itself
	sa.elems = elems
	# make sure the string is the empty string
	sa.string = ''
	return sa

def sum(sa, axis=1):
	if axis==0:
		sa = transpose(sa) # to test

	new_shape = (1, sa.shape[(axis+1) % 2])
	new_sa = SplunkArray('field' + time_hash(), new_shape)
	new_sa.string = sa.string
	for i,j in new_sa.iterable():
		# the add string for column j is the sum of all elements from the j'th row of sa
		add_string = '+'.join([str(elem) for elem in sa.elems[j]])
		new_sa.set_element(i, j, add_string)
	return new_sa




# def from_scalar(name, scalar):
# 	sa = SplunkArray(name, (1,1))
# 	sa.string = 'eval %s_0_0 = %s' % (sa.name, str(scalar))
# 	sa.find_elements()
# 	return sa

# def from_matrix(name, matrix):
# 	sa = SplunkArray(name, matrix.shape)
# 	sa.initialize_from_matrix(matrix)
# 	return sa

# def from_vector(name, vector):
# 	sa = SplunkArray(name, len(vector))
# 	sa.initialize_from_vector(vector)
# 	return sa

def zeros(shape):
	'''
	implements np.zeros(shape) operation
	'''
	sa = SplunkArray('field' +time_hash(), shape)
	sa.elems = np.zeros(shape)
	sa.string = ''
	return sa

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

def transpose(sa):
	'''
	returns the transpose of the splunk array. if just using for a dot product, use sa.T() instead.
	'''
	new_sa = SplunkArray(sa.name + '_tranpose', (sa.shape[1], sa.shape[0]))
	new_sa.string = sa.string + '| '
	for i in range(new_sa.shape[0]):
		for j in range(new_sa.shape[1]):
			new_sa.string += 'eval %s_%s_%s = %s | ' % (new_sa.name, i, j, sa.elems[j][i])
	new_sa.string = new_sa.string[:-2]
	new_sa.find_elements()
	return new_sa

	