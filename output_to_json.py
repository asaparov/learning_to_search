#!/bin/python

import csv
from os.path import isfile

def read_csv(filename):
	rows = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			rows.append(row)
	return rows

from sys import argv, exit
if len(argv) < 2:
	print('Usage: output_to_json.py [output log]')
	exit(1)

writers = {}
existing_logs = {}
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

	while True:
		line = f.readline()
		if line.startswith('training accuracy: '):
			training_accuracy = float(line[len('training accuracy: '):line.rindex('±')])
		elif line.startswith('test accuracy = '):
			tokens = line.split(',')
			test_accuracy = float(tokens[0][len('test accuracy = '):tokens[0].rindex('±')])
			test_loss = float(tokens[1][tokens[1].rindex('=')+1:].strip())
		elif line.startswith('saving to "'):
			ckpt_filepath = line[len('saving to "'):line.rindex('"')]
			csv_filepath = '/'.join(ckpt_filepath.split('/')[:-1]) + '/log.csv'
			break

	if csv_filepath not in writers:
		try:
			# check to see if the file exists
			if csv_filepath not in existing_logs:
				try:
					rows = read_csv(csv_filepath)
					epoch_idx = rows[0].index('epoch')
					last_epoch = int(rows[-1][epoch_idx])
					existing_logs[csv_filepath] = last_epoch
				except Exception as e:
					if not isinstance(e, FileNotFoundError):
						print("WARNING: Unable to read csv file '{}'.".format(csv_filepath))
						print(e)
					existing_logs[csv_filepath] = None

			if existing_logs[csv_filepath] == None:
				f_out = open(csv_filepath, 'w')
				writer = csv.writer(f_out)
				writer.writerow(['epoch', 'training_loss', 'training_accuracy', 'test_accuracy', 'test_loss'])
			elif epoch == existing_logs[csv_filepath] + 1:
				f_out = open(csv_filepath, 'a')
				writer = csv.writer(f_out)
			else:
				continue
			writers[csv_filepath] = writer, f_out
		except FileNotFoundError:
			writers[csv_filepath] = None, None

	writer, _ = writers[csv_filepath]

	# write the row to the CSV file
	if writer != None:
		writer.writerow([epoch, training_loss, training_accuracy, test_accuracy, test_loss])

f.close()
for _, (_, f_out) in writers.items():
	f_out.close()
