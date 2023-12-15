from random import seed, shuffle, randrange, getstate, setstate
import numpy as np
import torch
from train import evaluate_model, generate_eval_data
from os.path import isfile
from os import listdir
from math import exp, log
from sys import stdout

def lighten_color(color, amount=0.5):
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

seed(2)
torch.manual_seed(2)
np.random.seed(2)

if not torch.cuda.is_available():
	print("ERROR: CUDA device is not available.")
	#from sys import exit
	#exit(-1)
	device = torch.device('cpu')
else:
	device = torch.device('cuda')

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.style.use('ggplot')
colors = []
for c in rcParams["axes.prop_cycle"]:
	colors.append(c['color'])

def plot_loss_vs_data(ckpt_directory, delta_log_examples, max_lookahead, streaming_block_size):
	suffix = ckpt_directory[ckpt_directory.index('inputsize')+len('inputsize'):]
	max_input_size = int(suffix[:suffix.index('_')])

	suffix = ckpt_directory[ckpt_directory.index('seed')+len('seed'):]
	training_seed = int(suffix[:suffix.index('_')])

	# generate test data
	seed(training_seed)
	torch.manual_seed(training_seed)
	np.random.seed(training_seed)
	random_state = getstate()
	np_random_state = np.random.get_state()
	torch_random_state = torch.get_rng_state()
	test_sets = {}
	for lookahead in list(range(1, max_lookahead + 1)) + [None]:
		setstate(random_state)
		np.random.set_state(np_random_state)
		torch.set_rng_state(torch_random_state)

		inputs,outputs = generate_eval_data(max_input_size, min_path_length=2, distance_from_start=None, distance_from_end=None, lookahead_steps=lookahead, num_paths_at_fork=None, num_samples=10000)
		test_sets[lookahead] = (inputs,outputs)

	filenames = [file for file in listdir(ckpt_directory) if file.startswith('epoch') and file.endswith('.pt')]
	max_epoch = None
	for file in filenames:
		epoch = int(file[len('epoch'):-len('.pt')])
		if max_epoch == None or epoch > max_epoch:
			max_epoch = epoch
	max_examples = max_epoch * streaming_block_size

	log_example_list = np.arange(log(streaming_block_size + 1), log(max_examples), delta_log_examples)
	if log_example_list[-1] != log(max_examples):
		log_example_list = np.append(log_example_list, [log(max_examples)])
	losses = np.empty((len(log_example_list),max_lookahead+1))
	index = 0
	for log_examples in log_example_list:
		# find the closest checkpoint
		target_epoch = max(1, int(exp(log_examples) // streaming_block_size))
		print('Searching for checkpoint with number of examples = {}, target "epoch" = {}'.format(exp(log_examples), target_epoch))
		stdout.flush()
		for j in range(5):
			filename = ckpt_directory + '/epoch' + str(target_epoch + j) + '.pt'
			if isfile(filename):
				actual_epoch = target_epoch + j
				break
			filename = ckpt_directory + '/epoch' + str(target_epoch - j) + '.pt'
			if isfile(filename):
				actual_epoch = target_epoch - j
				break

		if not isfile(filename):
			break
		print('Evaluating model with log examples = ' + str(actual_epoch * streaming_block_size))
		stdout.flush()
		loaded_obj = torch.load(filename, map_location=device)
		if type(loaded_obj) == tuple:
			model, _, _, _ = loaded_obj
		else:
			model = loaded_obj

		eval_inputs,eval_outputs = test_sets[None]
		_,loss = evaluate_model(model, eval_inputs, eval_outputs)
		losses[index,0] = loss
		for lookahead in range(1, max_lookahead + 1):
			print('lookahead = ' + str(lookahead))
			stdout.flush()
			eval_inputs,eval_outputs = test_sets[lookahead]
			_,loss = evaluate_model(model, eval_inputs, eval_outputs)
			losses[index,lookahead] = loss
		log_example_list[index] = log(actual_epoch * streaming_block_size)
		print('Test losses: ' + str(losses[index,:]))
		stdout.flush()
		index += 1

	dir_name = ckpt_directory
	if dir_name[-1] == '/':
		dir_name = dir_name[:-1]

	suffix = ckpt_directory[ckpt_directory.index('maxlookahead') + len('maxlookahead'):]
	max_train_lookahead = int(suffix[:suffix.index('_')])

	fig = plt.gcf()
	ax = plt.gca()
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	fig.set_size_inches(6, 2, forward=True)
	ax.plot(np.exp(log_example_list), losses[:,0], color=colors[5], label='OOD', linewidth=0.5)
	for lookahead in range(1, max_train_lookahead + 1):
		alpha = (lookahead - 1) / (max_train_lookahead - 1)
		ax.plot(np.exp(log_example_list), losses[:,lookahead], color=lighten_color(colors[0], 0.7*alpha + 0.4), label=str(lookahead), linewidth=0.5)
	for lookahead in range(max_train_lookahead + 1, max_lookahead + 1):
		alpha = (lookahead - max_train_lookahead - 1) / (max_lookahead - max_train_lookahead)
		ax.plot(np.exp(log_example_list), losses[:,lookahead], color=lighten_color(colors[1], 0.7*alpha + 0.4), label=str(lookahead), linewidth=0.5)
	ax.legend(loc='lower left', ncol=2)
	plt.xlim(60000, max_examples * 1.1)
	plt.ylim(0.0001, 10.0)
	plt.xlabel('examples seen')
	plt.ylabel('loss')
	plt.grid(True)
	fig.savefig(dir_name + '_loss.png', dpi=256)
	plt.clf()

from sys import argv
plot_loss_vs_data(argv[1], 0.3, int(argv[2]), int(argv[3]))
