import hashlib
import time
def splunk_concat(one, two):
	'''
	checks to see if the end of string "one" has a pipe; if it does, simply concatenates it with two. Else, concantenates it with a pipe and two.
	'''
	if len(one) == 0 or len(two) == 0 or one.split()[-1] == '|' or two.split()[0] == '|':
		return one + two
	else:
		return one + ' | ' + two


def sha_hash(string):
	'''
	hashes a string to another string
	'''
	return hashlib.sha1(string).hexdigest()

def time_hash():
	'''
	hashes the current time
	'''
	return sha_hash(str(time.time()))