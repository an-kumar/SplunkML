
def splunk_concat(one, two):
	'''
	checks to see if the end of string "one" has a pipe; if it does, simply concatenates it with two. Else, concantenates it with a pipe and two.
	'''
	if one.split()[-1] == '|':
		return one + two
	else:
		return one + ' | ' + two


def hash(string):
	'''
	hashes a string to another string
	'''
	return hashlib.sha1(string).hexdigest()