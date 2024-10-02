#!/bin/python

import csv

from sys import argv, exit
if len(argv) < 2:
	print('Usage: output_to_json.py [output log]')
	exit(1)

f_out = None
writer = None
f = open(argv[1], 'r')

while True:
	line = f.readline()
	if line == '':
		break
	elif not line.startswith('epoch = '):
		continue

	tokens = line.split(',')
	epoch = int(tokens[0][len('epoch = '):])
	training_loss = float(tokens[1][tokens[1].rindex('=')+1:].strip())

	# check if this log contains the beginning of another experiment; if so, write to a new CSV file
	if epoch == 0:
		if f_out != None:
			f_out.close()
		writer = None

	while True:
		line = f.readline()
		if line.startswith('training accuracy: '):
			training_accuracy = float(line[len('training accuracy: '):line.rindex('±')])
		elif line.startswith('test accuracy = '):
			tokens = line.split(',')
			test_accuracy = float(tokens[0][len('test accuracy = '):tokens[0].rindex('±')])
			test_loss = float(tokens[1][tokens[1].rindex('=')+1:].strip())
			if writer != None:
				break
		elif writer == None and line.startswith('saving to "'):
			ckpt_filepath = line[len('saving to "'):line.rindex('"')]
			csv_filepath = '/'.join(ckpt_filepath.split('/')[:-1]) + '/log.csv'
			try:
				f_out = open(csv_filepath, 'w')
				writer = csv.writer(f_out)
				writer.writerow(['epoch', 'training_loss', 'training_accuracy', 'test_accuracy', 'test_loss'])
			except FileNotFoundError:
				pass
			break

	# write the row to the CSV file
	if writer != None:
		writer.writerow([epoch, training_loss, training_accuracy, test_accuracy, test_loss])

f.close()
f_out.close()
