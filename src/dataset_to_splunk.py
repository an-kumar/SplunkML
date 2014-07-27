f = open('voting.tab.txt','r')
for line in f:
	labels = line.split()
	break

for line in f:
	break

for line in f:
	break

count = 0
for line in f:

	splits = line.rstrip('\n').split('\t')
	
	# for i in range(len(splits)):
	# 	if splits[i] == '':
	# 		splits[i] = 'a'
	dump_string = ''
	for i in range(len(splits) - 1):
		if splits[i] == '':
			continue
		dump_string += '%s=%s\t' % (labels[i], splits[i])
		
	last = len(splits) -1
	dump_string += '%s=%s\n\n' % (labels[last],splits[last])
	print dump_string
	count += 1




