#!/bin/python

import csv

from sys import argv, exit
if len(argv) < 2:
	print('Usage: output_to_json.py [output log]')
	exit(1)

writer = None
f = open(argv[1], 'r')

while True:
	line = f.readline()
	if not line.startswith('epoch = '):
		continue

	tokens = line.split(',')
	epoch = int(tokens[0][len('epoch = '):])
	training_loss = float(tokens[1][tokens[1].rindex('='):].strip())

	while True:
		line = f.readline()
		if line.startswith('training accuracy: '):
			training_accuracy = float(line[len('training accuracy: '):line.rindex('±')])
		elif line.startswith('test accuracy = '):
			tokens = line.split(',')
			test_accuracy = float(tokens[0][len('test accuracy = '):tokens[0].rindex('±')])
			test_loss = float(tokens[1][tokens[1].rindex('='):].strip())
			if writer != None:
				break
		elif writer == None and line.startswith('saving to "'):
			ckpt_filepath = line[len('saving to "':line.rindex('"'))]
			csv_filepath = '/'.join(ckpt_filepath.split('/')[:-1]) + '/log.csv'
			import pdb; pdb.set_trace()
			f_out = open(csv_filepath, 'w')
			writer = csv.writer(f_out)
			writer.writerow(['epoch', 'training_loss', 'training_accuracy', 'test_accuracy', 'test_loss'])
			break

	# write the row to the CSV file
	writer.writerow([epoch, training_loss, training_accuracy, test_accuracy, test_loss])

f.close()
f_out.close()
