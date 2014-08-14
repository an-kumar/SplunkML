from utils.strings import *
import time
import numpy as np
from classes import *

def array(argument):
	'''
	implements a splunkarray equivalent of np.array()

	params:
		-argument: either a scalar, a list, a list of lists, or a numpy array
	returns:
		- a splunkarray from the input argument
	'''
	# try a bunch of different types:
	if type(argument) == float or type(argument) == int:
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
		# numpy uses the (n,) convention for n length arrays - so far, splunkmath uses (1,n). so we need to check for htat.
		if len(argument.shape) == 1:
			shape = (1, argument.shape[0])
			elems = np.array([argument])
		else:
			shape = argument.shape
			elems = argument

	else:
		raise Exception("You didn't pass in a float, int, list, or numpy array. You passed in a %s" % type(argument))

	# now initialize an empty SplunkArray, name doesn't matter
	sa = SplunkArray(sha_hash(str(time.time())), shape)
	# set the elements to the argument itself
	sa.elems = elems
	# make sure the string is the empty string
	sa.string = ''
	return sa


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

def zeros(shape):
	'''
	implements np.zeros(shape) operation
	'''
	sa = SplunkArray(time_hash(), shape)
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
			new_sa.string += 'eval %s_%s_%s = %s_%s_%s | ' % (new_sa.name, i, j, sa.name, j, i)
	new_sa.string = new_sa.string[:-2]
	new_sa.find_elements()
	return new_sa

	