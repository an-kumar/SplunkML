'''
mathinsplunk module: functions to pull math out of splunk (i.e, after one of these functions are called, no more splunkvector operations can be done)

uses things like stats avg,count,etc. Meant to be numpy-like.
'''
import splunklib.client as client
import splunklib.results as results

import numpy as np
# from ..classes import SplunkArray
from utils.strings import *
from numpyfuncs import *








#WIP#


def pull_sa_out(search_string, sa, jobs):
	'''
	pulls the given splunk array out of splunk and into a numpy array

	params:
		- search_string: search string that returns correct events in splunk
		- sa: splunk array to pull into numpy array
		- jobs: a splunk 'jobs' object to run the job
	returns:
		- numpy array corresponding to the splunk array contents after the search is return
	notes:
		- splunk array is assumed to have finished with some form of a stats command, so that 'events' no longer exist in the search
	'''
	# initialize the splunk search
	splunk_search = 'search %s' % search_string
	# add the splunk array's string
	splunk_search = splunk_concat(splunk_search, sa.string)
	# run the search
	search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	job = self.jobs.create(splunk_search, **search_kwargs)
	# read the results




#WIP#

def avg_over_events_by_field(sa, field):
	'''
	finds the average over events of the elements of sa, by field 'field'.

	params:
		- sa: splunk array to find average of (average is done elementwise, across events)
		- field: what to average by
	returns:
		- splunk vector ready to be pulled out of splunk (elements are averages over events)
	notes:
		- WARNING: this function uses stats, so no more mathinsplunk operations can be used
	'''
	avg_string = 'stats '
	new_elems = zeros(sa.shape)
	for i,j in sa.iterable():
		field = sa.elems[i][j]
		avg_string += 'avg(%s) as %s_avg, ' % (field, field)
		new_elems[i][j] = field + '_avg'
	new_sa = SplunkArray(time_hash(), sa.shape)
	new_sa.string = splunk_concat(sa.string, avg_string)
	new_sa.elems = new_elems
	return new_sa


def avg_over_events(sa):
	'''
	finds the average over events of the elements of sa.

	params:
		- sa: splunk array to find average of (average is done elementwise, across events)
	returns:
		- splunk vector ready to be pulled out of splunk (elements are averages over events)
	notes:
		- WARNING: this function uses stats, so no more mathinsplunk operations can be used
	'''
	avg_string = 'stats '
	new_elems = zeros(sa.shape)
	for i,j in sa.iterable():
		field = sa.elems[i][j]
		avg_string += 'avg(%s) as %s_avg ' % (field, field)
		new_elems[i][j] = field + '_avg'
	new_sa = SplunkArray(time_hash(), sa.shape)
	new_sa.string = splunk_concat(sa.string, avg_string)
	new_sa.elems = new_elems
	return new_sa




