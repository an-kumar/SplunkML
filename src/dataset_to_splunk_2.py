f = open('SPECTF.train')
field_names=['success']
field_names += ['field%s' % i for i in range(1,45)]
count = 0
for line in f:
	splits = line.rstrip('\n').split(',')
	dump_string = ''
	for i in range(len(splits)-1):
		dump_string += '%s=%s\t' % (field_names[i], splits[i])
	dump_string += '%s=%s' % (field_names[-1], splits[-1])
	print dump_string
	count += 1
	print "\n\n"
