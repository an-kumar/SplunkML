from strings import *
from ..classes import SplunkArray
import numpy as np
def broadcast(one,two):
	'''
	implements broadcasting when given two splunk arrays. called during elementwise arithment operations.

	params:
		- one, two: SplunkArrays
	returns:
		- output: a splunk array of the correct, broadcasted shape ready to be filled in
		- temp_elems_(one, two): a temporary array of elements that correctly broadcasts the original values to the new shape
		- ^^^ should potentially allow to be changed if "elements" arrays aren't the correct thing to go with (could probably come up with a clever way)
	'''
	new_shape = check_broadcasting(one.shape, two.shape)
	output = SplunkArray(one.name + '_broadcast_arithmetic_' + two.name, new_shape) # TO CHANGE!#@$
	# broadcast to temporary SAs
	temp_elems_one = broadcast_sa_to_shape(one, new_shape)
	temp_elems_two = broadcast_sa_to_shape(two, new_shape)
	# fill string
	output.string = splunk_concat(one.string, two.string)
	return output, temp_elems_one, temp_elems_two



def find_elements_from_name_shape(name, shape):
	'''
	returns an array of shape "shape" that contains, in the i,j'th position, name_i_j.

	note that this function(or a similar one) could be written to return an interable over these names so that the array is never explicitly created
	'''
	return np.array([['%s_%s_%s' % (name, i, j) for j in range(shape[1])] for i in range(shape[0])], dtype=object)


def check_broadcasting(shape_one, shape_two):
	'''
	implements numpy style broadcasting and returns the output shape, or raises an error if the shapes are not broadcastable

	to learn more about broadcasting, google "numpy broadcasting"
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
	returns an elements array that contains the correct, broadcasted values from the original SplunkArray sa.
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
			elif sa.shape[0] == 1 and sa.shape[1] == 1:
				temp_elems[i][j] = sa.elems[0][0]
			else:
				raise Exception ("something went wrong with broadcasting. Check deeper.")
	return temp_elems



def shape_from_passed_in(shape):
	'''
	If shape was an int, interpret it as a row vector and return (1, shape). Else, shape must be a tuple, and return the same.
	'''
	# get shape/length
	if type(shape) == int:
		# make it a row vector
		return (1, shape)
	elif type(shape) == tuple:
		return shape
	else:
		raise Exception ("Shape passed in wasn't an int or a tuple. was a %s" % type(shape))
