
'''
SHAPE, BROADCASTING & SPLUNK UTILS
'''

def find_elements_from_name_shape(name, shape):
	return np.array([['%s_%s_%s' % (name, i, j) for j in range(shape[1])] for i in range(shape[0])], dtype=object)


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


def shape_from_passed_in(shape):
	# get shape/length
	if type(shape) == int:
		# make it a row vector
		return (1, shape)
	elif type(shape) == tuple:
		return shape
	else:
		raise Exception ("Shape passed in wasn't an int or a tuple. was a %s" % type(shape))


def make_temp_splunk_array(argument):
	'''
	usage: a = make_temp_splunk_array(1) or make_temp_splunk_array([1,2,3]) or make_temp_splunk_array(np.array([[1,2,3],[4,5,6]]))

	makes a temp splunk array with no string and with elems being the actual numbers given
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
	sa = SplunkArray('temp_UNIQUEHASHTOCHANGE', shape)
	# set the elements to the argument itself
	sa.elems = elems
	# make sure the string is the empty string
	sa.string = ''
	return sa
