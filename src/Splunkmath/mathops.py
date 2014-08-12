from utils.broadcasting import *
from utils.strings import *
from classes import SplunkArray


def sub(one, two):
	'''
	implements one - two using numpy-like broadcasting
	'''
	return broadcast_apply_elementwise(one, two, '-')


def mul(one, two):
	'''
	implements one * two using numpy-like broadcasting
	'''
	return broadcast_apply_elementwise(one, two, '*')


def add(one, two):
	'''
	implements one + two using numpy-like broadcasting
	'''
	return broadcast_apply_elementwise(one, two, '+')


def div(one, two):
	'''
	implements one / two using numpy-like broadcasting
	'''
	return broadcast_apply_elementwise(one, two, '/')

def broadcast_apply_elementwise(one, two, operation):
	'''
	overall function that implements +, -, *, / (all elementwise operations).
	
	params:
		- one, two: splunk arrays to perform the operation on
		- operation: either '+', '-', '*', '/'. These must be strings.
	returns:
		- a splunk array that represents the correct elementwise operation applied.

	notes:
		- the strategy here is to broadcast both arrays to the same shape, and then perform the operation given element-wise and dump it into a new SplunkArray
	'''
	# if two is not an array, try to make a temporary splunkarray to house it
	if type(two) != SplunkArray:
		two = make_temp_splunk_array(two)

	# initialize output SA and broadcast elements
	output, temp_elems_one, temp_elems_two = broadcast(one, two)
	
	# return the elementwise operation
	x= elementwise_arithmetic_operation(output,temp_elems_one, temp_elems_two, operation)
	return x


def elementwise_arithmetic_operation(output, elems_one, elems_two, operation):
	'''
	implements a (+|-|*|/) b, as all of these operations have the same syntax in splunk
	this function is called in all of the following: add, sub, mul, div. In all of them, the arrays are broadcasted to the same size, and then this function is called to do elementwise manipulation.

	params: 
		- output: splunk array to dump results in
		- elems_(one, two): sets of elements to do elementwise operation on. must be the same shape. elems_one contains the first operand (i.e elems_one[i][j] - elems_two[i][j]).
		- operation: operation to apply.

	returns: the same splunk array, now with each element set to the correct arithmetic operation
	'''
	if elems_one.shape != elems_two.shape:
		raise ("elementwise operation error: shapes not the same. Check broadcasting. Shapes were %s, %s" % (elems_one.shape, elems_two.shape))
	for i,j in output.iterable():
		output.set_element(i, j, '%s %s %s' % (elems_one[i][j], operation, elems_two[i][j]))
	return output


def dot(one, two):
	'''
	implements one dotprod two. 

	params:
		- one,two: splunkarrays to dot product
	returns:
		- a new splunk array that correctly represents the dot product of the two passed in arrays
	notes:
		- the strategy is to iterate through and set the new array with elements using vector dot products (vector_dot_string). i.e each element of the new array is the vector dot product of rows/columns of the old arrays
	'''
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


def vector_dot_string(fields_one, fields_two):
	'''
	assuming fields_one and fields_two are iterables with names in them and the two have the same size, outputs a string representing their dot product in splunk
	'''
	assert len(fields_one) == len(fields_two)
	string = ''
	for i in range(len(fields_one)):
		string += '(%s*%s) + ' % (fields_one[i], fields_two[i])

	return string[:-2]

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