"""
	SplunkML Cmd Classes
	====================
	Maintain and manipulate splunk commands with Cmd and ArgCmd. They are intended 
	to provide an abstration above pure strings representing splunk commands such 
	that you can add them, multiply (repeat) them, and insert arbitrary arguments 
	into them. These come in handy for ML algorithms where certain operations are 
	used frequently, such as normalizing features to zero-mean and unit variance, 
	computing covariance matrices, etc.

	Cmd Examples:

		In [1]: c1 = Cmd('search sourcetype=my_dataset')
		In [2]: c2 = Cmd('stats 1_sum=sum(feature_1), 1_count=count(feature_1)')
		In [3]: c3 = Cmd('eval 1_avg=1_sum/1_count')
		In [4]: get_avg = c1 + c2 + c3
		In [5]: print get_avg
			search sourcetype=my_dataset | stats 1_sum=sum(feature_1), 1_count=count(feature_1) | eval 1_avg=1_sum/1_count


	ArgCmd Examples: 
			
		In [6]: c1 = ArgCmd ('search sourcetype={0}')
		In [7]: print c1('my_dataset')
			search sourcetype=my_dataset

		In [8]: c2 = ArgCmd ('stats sum=sum({0})')
		In [9]: print c1('my_dataset') + c2('feature_1')
			search sourcetype=my_dataset | stats sum=sum(feature_1)

		In [10]: c3 = ArgCmd ('search [count({0}) as {0}_count]', repeat_symbol=',')
		In [11]: print c3([feat1, feat2])
			count(feat1) as feat1_count, count(feat2) as feat2_count


	Ex.) Normalizing + Computing Covariance Matrix:
		
		GetAverageStd = ArgCmd ('eventstats [avg({0}) as {0}_avg, stdev({0}) as {0}_std]')
		ToZeroMean = 	ArgCmd ('[eval {0}_zm={0}-{0}_avg]', repeat_symbol=' | ')
		ToUnitVar = 	ArgCmd ('[eval {0}={0}_zm/{0}_std]', repeat_symbol=' | ')

		GetProducts = ArgCmd (	'[eval {0}_{1}_product={0}*{1} ] ', repeat_symbol=' | ')
		AvgProducts = ArgCmd (	'stats [avg({0}_{1}_product) as {0}_{1}_cov]')

		fieldnames = ['field1', 'field2', 'field3', 'field4']

		normalize_cmd = GetAverageStd(fieldnames) + ToZeroMean(fieldnames) + ToUnitVar (fieldnames)
		cov_mat_cmd = GetProducts (fieldnames) + AvgProduces (fieldnames)

	Jay Hack
	jhack@stanford.edu
	Summer 2014
"""
import re
import itertools
import scipy as sp
import numpy as np


class Cmd:
	"""
		Class: Cmd
		----------
		manages a static splunk search 


		Ideal Usage:
			c1 = Cmd('search sourcetype=my_dataset')
			c2 = Cmd('stats 1_sum=sum(feature_1), 1_count=count(feature_1)')
			c3 = Cmd('eval 1_avg=1_sum/1_count')
			get_avg = c1 + c2 + c3
	"""

	def __init__ (self, search_str):
		"""
			search_str: command to be executed
		"""
		assert type(search_str) == str
		self.search_str = ' '.join(search_str.split()).strip ()


	def endswith_pipe (self, s):
		"""
			returns true if the string ends with a pipe 
			(includes whitespace)
		"""
		return s.rstrip()[-1] == '|'

	
	def startswith_pipe (self, s):
		"""
			returns true if the string starts with a pipe
			(includes whitespace)
		"""
		return s.lstrip()[0] == '|'


	def concat_search_str (self, s1, s2):
		"""
			returns the splunk-esque concatenation of s1 and s2
		"""
		if self.endswith_pipe(s1) and self.startswith_pipe(s2):
			return s1 + s2.lstrip()[1:]
		elif not self.endswith_pipe(s1) and not self.startswith_pipe(s2):
			return s1 + ' | ' + s2


	def extract_search_str (self, other):
		"""
			given 'other', (a string or a Cmd), this returns 
			it's internal search string 
		"""
		if type(other) == str:
			return other
		if type(other) == type(self):
			return other.search_str


	def __add__ (self, other):
		"""
			returns a new SplunkCommand that is the concatenation
			of these two 
		"""
		add_str = self.extract_search_str(other)
		new_search_str = self.concat_search_str(self.search_str, add_str)
		return Cmd (new_search_str)


	def __iadd__ (self, other):
		"""
			appends to this SplunkCommand's search_str
		"""
		add_str = self.extract_search_str (other)
		self.search_str = self.concat_search_str (self.search_str, add_str)
		return Cmd (self.search_str)


	def __str__ (self):
		"""
			prints out the splunk string in a readable format 
		"""
		return self.search_str



class ArgCmd:
	"""
		Class: ArgCmd
		-------------
		Allows for the creation of an abstract command that takes 
		an arbitrary number of arguments and returns a Cmd with 
		them inserted.

		Ideal Usage:
			ac1 = ArgCmd ('search sourcetype={0}')
			ac2 = ArgCmd ('stats sum=sum({0})')
			c = ac1('my_datatype') + ac2('feature_1')

			rac = ArgCmd ('search [count({0}) as {0}_count, sum({1}) as {1}_sum]')
			~$: print rac([	(feat1x, feat1y), (feat2x, feat2y)])
				> search count(feat1x) as feat1x_count ...
	"""

	def __init__ (self, search_str, repeat_symbol=', '):
		"""
			search_str: a splunk search string with arguments specified inside
		"""
		assert self.str_is_valid (search_str)
		raw_search_str = search_str
		self.repeat_symbol = repeat_symbol

		if not self.has_repeat_str(raw_search_str):
			self.search_str = raw_search_str
			self.repeats = False
		else:
			self.start, self.repeat, self.end = self.split_raw_search_str (raw_search_str)
			self.repeats = True


	def has_repeat_str (self, s):
		"""
			returns true if there's a repeating portion
		"""
		return '[' in s


	def str_is_valid (self, s):
		"""
			performs syntax checking on s
		"""
		return s.count ('[') == s.count (']') and s.count('[') in [0, 1]


	def split_raw_search_str (self, s):
		"""
			returns the portion of s to be repeated 
		"""
		return re.split ('[\]\[]', s)


	def __call__ (self, *args):
		"""
			- args = elements to be inserted into search string 
		"""
		search_string = None
		if self.repeats == False:
			search_str = self.search_str.format(*args)
		else:
			if type(args[0][0]) == tuple:
				search_str = self.start + self.repeat_symbol.join([self.repeat.format(*a) for a in args[0]]) + self.end
			else:
				search_str = self.start + self.repeat_symbol.join([self.repeat.format(a) for a in args[0]]) + self.end

		return Cmd(search_str)





if __name__ == '__main__':

	print '==========[ Conducting Cmd/ArgCmd Unit Tests	]=========='

	#=====[ Test 1: Cmd __add__	]=====
	print "\n#[ Cmd - __add__ ]#"
	c1 = Cmd("""
				search sourcetype="my_dataset" |

				this is a test 
			""")
	c2 = Cmd('stats 1_sum=sum(feature_1), 1_count=count(feature_1)')
	c3 = Cmd('eval 1_avg=1_sum/1_count')
	print c1 + c2 + c3

	#=====[ Test 2: Cmd __iadd__	]=====
	print "\n#[ Cmd - __iadd__ ]#"
	c1 += c2
	print c1

	#=====[ Test 3: ArgCmd without repeating	]=====
	print "\n#[ ArgCmd - __str__ without repeating ]#"
	sc_search = ArgCmd('search sourcetype={0}')
	print sc_search('my_dataset')

	#=====[ Test4: ArgCmd with repeating	]=====
	print "\n#[ ArgCmd - __str__ with repeating ]#"
	sum_count = ArgCmd('stats [{0}_sum=sum({0}), {0}_count=count({0})]')
	print sum_count(['field1', 'field2'])

	#=====[ Test 5: ArgCmd with repeating, adding	]=====
	print "\n#[ ArgCmd - __add__ with repeating ]#"
	fields = ['field1', 'field2']
	get_avg = ArgCmd (	"""
							eval [
									{0}_avg={0}_sum/{0}_count
								]
						""")
	average_search = sc_search ('my_dataset') + sum_count(fields) + get_avg(fields)
	print average_search, '\n'
	GetAverageStd = ArgCmd ('stats [avg({0}) as {0}_avg, stdev({0}) as {0}_std]')
	print GetAverageStd (['field1', 'field2'])

	print '==========[ COVARIANCE ]=========='
	GetProducts = ArgCmd (	"""
								[eval {0}_{1}_product={0}*{1} ]
							""", repeat_symbol=' | ')
	AvgProducts = ArgCmd (	"""
								stats [avg({0}_{1}_product) as {0}_{1}_cov]
							""")
	fields = ['field1', 'field2', 'field3']
	field_pairs = [a for a in itertools.combinations (fields, 2)]
	print '\n', (GetProducts(field_pairs) + AvgProducts(field_pairs))




