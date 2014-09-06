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

def case_mapping(mapping, index_field, output_field):
	'''
	adds a string output_field which is equal to mapping[index_field]

	assumes index_field contains a single element that can be indexed by mapping
	'''
	string = 'eval %s = case(%s)' % (output_field, ','.join(['%s=%s,%s' % (index_field, elem, mapping[elem]) for elem in mapping]))
	return string

def search_to_numpy_reps(splunk_search, feature_mapping, class_field, type_tuple):
	'''
	turns a search that returns multiple events into an X, y numpy representation

	type_tuple must have, in both indices, 'continouous' or 'discrete'

	returns X, y 
	'''
	search_kwargs = {'timeout':1000, 'exec_mode':'blocking'}
	job = self.jobs.create(splunk_search, **search_kwargs)
	return events_to_numpy_reps(job, feature_mapping, class_field, type_tuple)

def job_to_numpy_reps(job, feature_mapping, class_field, type_tuple, bias=False):
	'''
	turns a job that returns multiple events into an X, y numpy representation. 

	type_tuple must have, in both indices, 'continouous' or 'discrete'

	returns X, y

	note: if only X is required (no y), for now just pass one of the X values into class field and disregard it
	'''
	# find correct numpy reps func
	to_numpy_reps_func = find_correct_to_numpy_reps_func(type_tuple)
	# iterate through events in the job, filling "X" and "y"
	X = []
	y = []
	offset = 0
	result_count = int(job["resultCount"])
	count = 50
	while (offset < int(result_count)):
		kwargs_paginate = {'count': count, 'offset':offset}
		search_results = job.results(**kwargs_paginate)
		for result in results.ResultsReader(search_results):
			try:
				x, curr_y = to_numpy_reps_func(result, feature_mapping, class_field, bias=bias)
			except:
				print "couldn't find something in"
				print result
			X.append(x)
			y.append(curr_y)
		offset += count
	X = np.array(X, dtype=np.float)
	y = np.array(y, dtype=np.float)
	return X, y


def find_correct_to_numpy_reps_func(type_tuple):
	if type_tuple == ('continuous', 'continuous'):
		return event_to_numpy_reps_continuous_continuous
	else:
		raise NotImplementedError

def event_to_numpy_reps_continuous_continuous(event, feature_mapping, class_field, bias=False):
	'''
	turns an event into a numpy rep X, y, where it is assumed that all teh values in both X and y are event_to_numpy_reps_continuous_continuous

	returns x, y
	'''
	if bias:
		# the last x term is the bias term, always 1
		x = np.zeros(len(feature_mapping)+1)
		x[len(feature_mapping)] = 1
	else:
		x = np.zeros(len(feature_mapping))
	for feature in feature_mapping:
		
		x[feature_mapping[feature]] = event[feature]
		
	y = event[class_field]
	
	return x, y



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




