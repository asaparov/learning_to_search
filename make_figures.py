from analyze import do_evaluate_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox
from os import listdir, makedirs
import csv

def find_ckpt(directory, max_epoch):
	existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(directory) if ckpt.startswith('epoch')]
	selected_epoch = max([epoch for epoch in existing_epochs if epoch <= max_epoch])
	return directory + '/epoch' + str(selected_epoch) + '.pt'

def draw_brace(ax, xspan, yy, text):
	"""Draws an annotated brace on the axes."""
	xmin, xmax = xspan
	xspan = xmax - xmin
	ax_xmin, ax_xmax = ax.get_xlim()
	xax_span = ax_xmax - ax_xmin

	ymin, ymax = ax.get_ylim()
	yspan = ymax - ymin
	resolution = int(abs(xspan)/xax_span*1000)*2+1 # guaranteed uneven
	beta = 300./xax_span # the higher this is, the smaller the radius

	x = np.linspace(xmin, xmax, resolution)
	x_half = x[:int(resolution/2)+1]
	y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
	                + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
	y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
	y = yy + (.2*y - .01)*yspan # adjust vertical position

	ax.autoscale(False)
	ax.plot(x, y, color='#555', lw=1, clip_on=False)

	text_y = yy
	if xmax > xmin:
		text_y += 0.08*yspan
	else:
		text_y += -0.08*yspan
	ax.text((xmax+xmin)/2., text_y, text, ha='center', va='bottom', color='#555')

def make_sensitivity_figures(max_epoch=100000):
	greedy_ckpt_dir  = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed1_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_OOD_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead12_seed2_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	greedy_ckpt = find_ckpt(greedy_ckpt_dir, max_epoch)
	crafted_ckpt = find_ckpt(crafted_ckpt_dir, max_epoch)
	crafted_OOD_ckpt = find_ckpt(crafted_OOD_ckpt_dir, max_epoch)
	greedy_accuracies = do_evaluate_model(greedy_ckpt)
	crafted_accuracies = do_evaluate_model(crafted_ckpt)
	crafted_OOD_accuracies = do_evaluate_model(crafted_OOD_ckpt)
	#greedy_star_accuracies = do_evaluate_model(greedy_ckpt, star_distribution=True)
	#crafted_star_accuracies = do_evaluate_model(crafted_ckpt, star_distribution=True)
	#crafted_OOD_star_accuracies = do_evaluate_model(crafted_OOD_ckpt, star_distribution=True)

	data = np.empty((3, 21))
	data[0,:21] = [acc for acc,_,_ in greedy_accuracies]
	data[1,:21] = [acc for acc,_,_ in crafted_accuracies]
	data[2,:21] = [acc for acc,_,_ in crafted_OOD_accuracies]

	fig = plt.gcf()
	fig.set_size_inches(8.0, 1.9, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([-0.2] + [i for i in range(1,21)], labels=(['Tested on\nnaïve distr.'] + [i+1 for i in range(20)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(3), labels=['Naïve distr.', 'Balanced distr.', 'Balanced distr.\n(lookahead $\\le 12$)', ])
	plt.text(-4.5, 2.3, '\\textbf{Trained on:}', color='#555', rotation='vertical')
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			c = "w" if data[i,j] > 0.6 else "k"
			plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)

	plt.xticks(np.arange(-.5, 21, 1), minor=True)
	plt.yticks(np.arange(-.5, 3, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)

	ax = plt.gca()
	draw_brace(ax, (20.4, 0.6), 4.0, 'Tested on balanced distribution with lookahead')

	plt.tight_layout()
	fig.savefig('figures/sensitivity.pdf', dpi=128)
	plt.clf()

def make_dfs_figures(max_epoch=100000):
	unpadded_ckpt_dirs = {
		#48 : 'dfs_results/checkpoints_v3_7layer_inputsize48_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#64 : 'dfs_results/checkpoints_v3_7layer_inputsize64_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#80 : 'dfs_results/checkpoints_v3_7layer_inputsize80_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#96 : 'dfs_results/checkpoints_v3_7layer_inputsize96_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed2_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed2_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed4_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed5_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed6_trainstreaming_nomask_unablated_dfs'
	}
	padded_ckpt_dirs = {
		#48 : 'dfs_results/checkpoints_v3_7layer_inputsize48_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#64 : 'dfs_results/checkpoints_v3_7layer_inputsize64_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#80 : 'dfs_results/checkpoints_v3_7layer_inputsize80_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#96 : 'dfs_results/checkpoints_v3_7layer_inputsize96_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed2_trainstreaming_nomask_unablated_padded_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded_dfs',
		128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed4_trainstreaming_nomask_unablated_padded_dfs'
	}

	unpadded_ckpts = {}
	padded_ckpts = {}
	for inputsize,ckpt_dir in unpadded_ckpt_dirs.items():
		unpadded_ckpts[inputsize] = find_ckpt(ckpt_dir, max_epoch)
	for inputsize,ckpt_dir in padded_ckpt_dirs.items():
		padded_ckpts[inputsize] = find_ckpt(ckpt_dir, max_epoch)

	unpadded_accuracies = do_evaluate_model(unpadded_ckpts[128], max_backtrack_distance=15)
	padded_accuracies = do_evaluate_model(padded_ckpts[128], max_backtrack_distance=15)

	data = np.empty((2, 16))
	data[0,:16] = [acc for acc,_,_ in unpadded_accuracies[1:]]
	data[1,:16] = [acc for acc,_,_ in padded_accuracies[1:]]

	fig = plt.gcf()
	fig.set_size_inches(8.0, 1.6, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([i for i in range(16)], labels=([i for i in range(16)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(2), labels=['Standard training', 'Trained with\nrandom padding'])
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			c = "w" if data[i,j] > 0.6 else "k"
			plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)

	plt.xticks(np.arange(-.5, 16, 1), minor=True)
	plt.yticks(np.arange(-.5, 2, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)
	plt.xlabel('Tested on examples with backtrack distance')

	#ax = plt.gca()
	#draw_brace(ax, (20.3, 0.7), 2.7, 'Tested on crafted distribution with lookahead')

	plt.tight_layout()
	fig.savefig('figures/dfs.pdf', dpi=128)
	plt.clf()

def read_csv(filename):
	rows = []
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			rows.append(row)
	return rows

def make_scaling_figures(epoch=1500, variable='input_sizes', keep_incomplete_seeds=False, train_loss_scaling_bug=True):
	if variable in ('dfs_padded','dfs_unpadded'):
		csv_dir = 'dfs_csvs/'
	else:
		csv_dir = 'scaling_experiments/csvs/'

	seeds = [int(d[len('seed_'):]) for d in listdir(csv_dir + 'scaling_' + variable) if d.startswith('seed_')]

	train_losses = {}
	test_losses = {}
	has_converged = {}
	first_seed = True
	incomplete_seeds = []
	TRAIN_LOSS_WINDOW = 0
	TEST_LOSS_WINDOW = 40
	IGNORE_HIDS = [2048,4096]
	IGNORE_LAYERS = [256,1024,2048]
	IGNORE_NL_INPUTSIZES = [96,112,128]
	for seed in sorted(seeds):
		if variable == 'input_sizes':
			var_name = 'Input size'
			prefix = '1e-5_seed' + str(seed) + '_'
		elif variable == 'layers':
			var_name = 'Num layers'
			prefix = 'nlayers_'
		elif variable == 'hidden_dim':
			var_name = 'Hidden dim'
			prefix = 'hid_'
		elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
			var_name = 'Input size'
			prefix = 'input_'
		elif variable in ('dfs_padded', 'dfs_unpadded'):
			var_name = 'Input size'
			prefix = 'input_'
		inputs = [f for f in listdir(csv_dir + 'scaling_' + variable + '/seed_' + str(seed)) if f.startswith(prefix) and f.endswith('.csv')]
		for f in inputs:
			inputsize = int(f[len(prefix):-len('.csv')])
			if variable == 'hidden_dim' and inputsize in IGNORE_HIDS:
				continue
			elif variable == 'layers' and inputsize in IGNORE_LAYERS:
				continue
			elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer') and inputsize in IGNORE_NL_INPUTSIZES:
				continue
			csv = read_csv(csv_dir + 'scaling_{}/seed_{}/{}'.format(variable, seed, f))
			csv[0] = [s.replace(' ','_').lower() for s in csv[0]]
			if 'epoch' not in csv[0]:
				print('WARNING: {} {}, seed {}, has no results.'.format(var_name, inputsize, seed))
				continue
			epoch_idx = csv[0].index('epoch')
			train_loss_idx = csv[0].index('training_loss')
			test_loss_idx = csv[0].index('test_loss')
			train_acc_idx = csv[0].index('training_accuracy')
			try:
				row_idx = next(i for i in range(1,len(csv)) if int(csv[i][epoch_idx]) == epoch)
			except StopIteration:
				print('WARNING: {} {}, seed {}, is missing epoch {}.'.format(var_name, inputsize, seed, epoch))
				continue
			row = csv[row_idx]
			train_loss_column = np.array([float(csv[i][train_loss_idx]) if i < len(csv) else float(csv[-1][train_loss_idx]) for i in range(1,row_idx+TRAIN_LOSS_WINDOW+2)])
			train_loss_column = np.concatenate((np.ones(TRAIN_LOSS_WINDOW)*float(csv[1][train_loss_idx]), train_loss_column))
			if train_loss_scaling_bug:
				# TODO: this is due to an earlier bug in the `train.py` code where the loss wasn't being correctly scaled by the batch size; this line should be removed if this script is being used on data produced from a more recent version of `train.py` where the bug is fixed
				if variable == 'layers' and inputsize >= 64:
					train_loss_column /= 2**12
				else:
					train_loss_column /= 2**10
			train_loss = np.convolve(train_loss_column, np.ones(TRAIN_LOSS_WINDOW*2 + 1)/(TRAIN_LOSS_WINDOW*2 + 1), mode='valid')

			test_loss_column = np.array([float(csv[i][test_loss_idx]) if i < len(csv) else float(csv[-1][test_loss_idx]) for i in range(1,row_idx+TEST_LOSS_WINDOW+2)])
			test_loss_column = np.concatenate((np.ones(TEST_LOSS_WINDOW)*float(csv[1][test_loss_idx]), test_loss_column))
			test_loss = np.convolve(test_loss_column, np.ones(TEST_LOSS_WINDOW*2 + 1)/(TEST_LOSS_WINDOW*2 + 1), mode='valid')
			def to_acc(s):
				if '±' in s:
					return float(s[:s.index('±')])
				else:
					return float(s)
			has_seed_converged = any([to_acc(r[train_acc_idx]) >= 0.995 for r in csv[1:(row_idx+1)]])

			if inputsize not in test_losses:
				if not first_seed:
					print('WARNING: {} {} is missing seed {}.'.format(var_name, inputsize, seed))
				train_losses[inputsize] = {}
				test_losses[inputsize] = {}
				has_converged[inputsize] = {}
			train_losses[inputsize][seed] = train_loss
			test_losses[inputsize][seed] = test_loss
			has_converged[inputsize][seed] = has_seed_converged
		first_seed = False

	if not keep_incomplete_seeds:
		# check if there are any seeds with missing inputsizes
		for seed in seeds:
			if seed in incomplete_seeds:
				continue
			if any([seed not in losses for losses in train_losses.values()]):
				incomplete_seeds.append(seed)

		# remove incomplete seeds from results
		if len(incomplete_seeds) != 0:
			print('WARNING: The following seeds have incomplete results {}; they will not be included in the plots.'.format(incomplete_seeds))
		for inputsize,losses in list(train_losses.items()):
			for seed in incomplete_seeds:
				if seed in losses:
					del losses[seed]
			if len(losses) == 0:
				del train_losses[inputsize]
		for inputsize,losses in list(test_losses.items()):
			for seed in incomplete_seeds:
				if seed in losses:
					del losses[seed]
			if len(losses) == 0:
				del test_losses[inputsize]
		for inputsize,losses in list(has_converged.items()):
			for seed in incomplete_seeds:
				if seed in losses:
					del losses[seed]
			if len(losses) == 0:
				del has_converged[inputsize]

	if variable == 'input_sizes':
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 8, 52
	elif variable == 'layers':
		xaxis = 'Number of transformer layers'
		legend_title = 'Number of\ntransformer layers'
		xmin, xmax = 4, 45
	elif variable == 'hidden_dim':
		xaxis = 'Non-embedding parameters'
		legend_title = 'Non-embedding\nparameters'
		nlayers = 8
		xmin, xmax = 100000, 1000000000
	elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 8, 34
	elif variable in ('dfs_padded', 'dfs_unpadded'):
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 8, 52

	inputsizes = np.empty(len(test_losses), dtype=np.uint64)
	counter = 0
	for inputsize,_ in test_losses.items():
		inputsizes[counter] = inputsize
		counter += 1
	sorted_idx = inputsizes.argsort()
	inputsizes = inputsizes[sorted_idx]

	if variable in ('input_sizes', 'NL_16hid_8layer', 'NL_32hid_16layer'):
		x = (inputsizes - 5) // 3
	elif variable == 'layers':
		x = inputsizes
	elif variable == 'hidden_dim':
		fixed_max_input_size = 98
		ntoken = (fixed_max_input_size-5) // 3 + 5
		dmodel = np.maximum(ntoken, inputsizes) + fixed_max_input_size
		x = 6*dmodel*dmodel*nlayers
	elif variable in ('dfs_padded', 'dfs_unpadded'):
		x = (inputsizes - 5) // 3

	converged = np.empty(len(has_converged))
	counter = 0
	for _,converged_list in has_converged.items():
		if len(converged_list) == 0:
			converged[counter] = 0.0
		else:
			converged[counter] = sum(converged_list.values()) / len(converged_list)
		counter += 1
	converged = converged[sorted_idx]

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	ax.plot(x, converged, '.')
	plt.xlim(xmin, xmax)
	plt.ylim(-0.05, 1.05)
	ax.set_xscale("log", base=10)
	if variable != 'hidden_dim':
		ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
		ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Fraction of converged seeds')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_convergence.pdf', dpi=256)
	plt.clf()

	mintestlosses = np.empty(len(test_losses))
	counter = 0
	for inputsize,loss_list in test_losses.items():
		losses = [v[-1] for k,v in loss_list.items() if has_converged[inputsize][k]]
		if len(losses) == 0:
			mintestlosses[counter] = float('inf')
		else:
			mintestlosses[counter] = np.min(losses) #min(loss_list.values())
		counter += 1
	mintestlosses = mintestlosses[sorted_idx]

	if variable == 'input_sizes':
		a,b = np.polyfit(x[converged > 0], np.log(mintestlosses[converged > 0] - np.min(mintestlosses) + 1.0e-9), 1, w=np.sqrt(mintestlosses[converged > 0]))

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	if variable == 'input_sizes':
		x_fit = np.arange(xmin,xmax,(xmax-xmin)/10000)
		ax.plot(x_fit, mintestlosses[0]-1.0e-9+np.exp(b+a*x_fit), '--', color='k', linewidth=0.5)
	ax.plot(x[converged > 0], mintestlosses[converged > 0], '.')
	plt.xlim(xmin, xmax)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	if variable != 'hidden_dim':
		ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
		ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Minimum test loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_test.pdf', dpi=256)
	plt.clf()

	mintrainlosses = np.empty(len(test_losses))
	counter = 0
	for _,loss_list in train_losses.items():
		losses = [v[-1] for k,v in loss_list.items()]
		mintrainlosses[counter] = np.mean(losses)
		counter += 1
	mintrainlosses = mintrainlosses[sorted_idx]

	if variable == 'input_sizes':
		m,b = np.polyfit(x, mintrainlosses, 1)

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	#if variable == 'input_sizes':
	#	x_fit = np.arange(xmin,xmax,(xmax-xmin)/10000)
	#	ax.plot(x_fit, m*x_fit + b, '--', color='k', linewidth=0.5)
	ax.plot(x, mintrainlosses, '.')
	plt.xlim(xmin, xmax)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	if variable != 'hidden_dim':
		ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
		ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Average training loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_train.pdf', dpi=256)
	plt.clf()

	cmap = plt.get_cmap('viridis')
	start_color_point = 0.0
	end_color_point = 0.85
	colors = {}
	counter = 0
	for inputsize in inputsizes:
		t = counter/(len(inputsizes) - 1)
		colors[inputsize] = cmap(t*start_color_point + (1-t)*end_color_point)
		counter += 1

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	max_x = 0.0
	handles = {}
	for i in range(len(inputsizes)):
		inputsize = inputsizes[i]
		for _,loss_vals in test_losses[inputsize].items():
			examples_seen = np.arange(0,len(loss_vals)) * (2**18)
			line, = ax.plot(examples_seen, loss_vals, c=colors[inputsize], alpha=0.8)
			max_x = max(examples_seen[-1], max_x)
			if inputsize not in handles:
				handles[inputsize] = (line, x[i])
	if variable == 'hidden_dim':
		labels = ['{0:.1f}M'.format(l/1000000) for _,(_,l) in handles.items()]
	else:
		labels = [str(l) for _,(_,l) in handles.items()]
	if len(labels) <= 4:
		num_cols = len(labels)
	else:
		num_cols = np.ceil(len(labels)/6)
	plt.legend([h for _,(h,_) in handles.items()], labels, title=legend_title, fontsize=6.0, title_fontsize=7.5, ncols=num_cols, handlelength=1.0)
	plt.xlim(1000000, max_x)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	plt.xlabel('Training examples')
	plt.ylabel('Test loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_test_loss.pdf', dpi=256)
	plt.clf()

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	max_x = 0.0
	handles = {}
	for i in range(len(inputsizes)):
		inputsize = inputsizes[i]
		for _,loss_vals in train_losses[inputsize].items():
			examples_seen = np.arange(0,len(loss_vals)) * (2**18)
			line, = ax.plot(examples_seen, loss_vals, c=colors[inputsize], alpha=0.8)
			max_x = max(examples_seen[-1], max_x)
			if inputsize not in handles:
				handles[inputsize] = (line, x[i])
	if variable == 'hidden_dim':
		labels = ['{0:.1f}M'.format(l/1000000) for _,(_,l) in handles.items()]
	else:
		labels = [str(l) for _,(_,l) in handles.items()]
	if len(labels) <= 4:
		num_cols = len(labels)
	else:
		num_cols = np.ceil(len(labels)/6)
	plt.legend([h for _,(h,_) in handles.items()], labels, title=legend_title, fontsize=6.0, title_fontsize=7.5, ncols=num_cols, handlelength=1.0)
	plt.xlim(100000, max_x)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	plt.xlabel('Training examples')
	plt.ylabel('Training loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_train_loss.pdf', dpi=256)
	plt.clf()

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	max_x = 0.0
	handles = {}
	for i in range(len(inputsizes)):
		inputsize = inputsizes[i]
		min_loss_len = min([len(loss_vals) for loss_vals in test_losses[inputsize].values()])
		all_losses = np.empty((len(test_losses[inputsize]), min_loss_len))
		counter = 0
		for _,loss_vals in test_losses[inputsize].items():
			all_losses[counter,:] = loss_vals[:min_loss_len]
			counter += 1
		examples_seen = np.arange(0,min_loss_len) * (2**18)
		line, = ax.plot(examples_seen, np.min(all_losses, axis=0), c=colors[inputsize], alpha=0.8)
		max_x = max(examples_seen[-1], max_x)
		handles[inputsize] = (line, x[i])
	if variable == 'hidden_dim':
		labels = ['{0:.1f}M'.format(l/1000000) for _,(_,l) in handles.items()]
	else:
		labels = [str(l) for _,(_,l) in handles.items()]
	if len(labels) <= 4:
		num_cols = len(labels)
	else:
		num_cols = np.ceil(len(labels)/6)
	plt.legend([h for _,(h,_) in handles.items()], labels, title=legend_title, fontsize=6.0, title_fontsize=7.5, ncols=num_cols, handlelength=1.0)
	plt.xlim(1000000, max_x)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	plt.xlabel('Training examples')
	plt.ylabel('Minimum test loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_test_minloss.pdf', dpi=256)
	plt.clf()

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	max_x = 0.0
	handles = {}
	for i in range(len(inputsizes)):
		inputsize = inputsizes[i]
		min_loss_len = min([len(loss_vals) for loss_vals in train_losses[inputsize].values()])
		all_losses = np.empty((len(train_losses[inputsize]), min_loss_len))
		counter = 0
		for _,loss_vals in train_losses[inputsize].items():
			all_losses[counter,:] = loss_vals[:min_loss_len]
			counter += 1
		examples_seen = np.arange(0,min_loss_len) * (2**18)
		line, = ax.plot(examples_seen, np.min(all_losses, axis=0), c=colors[inputsize], alpha=0.8)
		max_x = max(examples_seen[-1], max_x)
		handles[inputsize] = (line, x[i])
	if variable == 'hidden_dim':
		labels = ['{0:.1f}M'.format(l/1000000) for _,(_,l) in handles.items()]
	else:
		labels = [str(l) for _,(_,l) in handles.items()]
	if len(labels) <= 4:
		num_cols = len(labels)
	else:
		num_cols = np.ceil(len(labels)/6)
	plt.legend([h for _,(h,_) in handles.items()], labels, title=legend_title, fontsize=6.0, title_fontsize=7.5, ncols=num_cols, handlelength=1.0)
	plt.xlim(100000, max_x)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	plt.xlabel('Training examples')
	plt.ylabel('Minimum training loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_train_minloss.pdf', dpi=256)
	plt.clf()

def get_mi_results(ckpt_dir, epoch=3370):
	from trace_circuit import do_analysis, MissingSampleError
	results = np.empty((3,20))
	counter = 0
	for lookahead in [-1] + list(range(1,20)):
		ckpt_filepath = ckpt_dir + '/epoch{}.pt'.format(epoch)
		try:
			print('Reading trace_circuit analysis from {} for lookahead {}.'.format(ckpt_filepath, lookahead))
			explainable_example_proportion, average_merge_ops_per_example, suboptimal_merge_op_proportion = do_analysis(ckpt_filepath, None, lookahead, 100, quiet=True)
			results[0,counter] = explainable_example_proportion
			results[1,counter] = average_merge_ops_per_example
			results[2,counter] = 1.0 - suboptimal_merge_op_proportion
		except MissingSampleError:
			print('WARNING: Checkpoint for {} is missing tracing results for lookahead {}.'.format(ckpt_filepath, lookahead))
			results[:,counter] = np.nan
		counter += 1
	return results

def make_mi_figures(epoch=3370):
	greedy_ckpt_dir  = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed1_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_OOD_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead12_seed2_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	greedy_results = get_mi_results(greedy_ckpt_dir, epoch)
	crafted_results = get_mi_results(crafted_ckpt_dir, epoch)
	crafted_OOD_results = get_mi_results(crafted_OOD_ckpt_dir, epoch)
	untrained_results = get_mi_results(crafted_ckpt_dir, epoch=0)

	data = np.empty((4,20))
	data[0,:] = greedy_results[0,:]
	data[1,:] = crafted_results[0,:]
	data[2,:] = crafted_OOD_results[0,:]
	data[3,:] = untrained_results[0,:]

	fig = plt.gcf()
	fig.set_size_inches(8.0, 2.3, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	r_cmap.set_bad(color='white')
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([-0.2] + [i for i in range(1,20)], labels=(['Tested on\nnaïve distr.'] + [i for i in range(1,20)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(4), labels=['Naïve distr.', 'Balanced distr.', 'Balanced distr.\n(lookahead $\\le 12$)', 'Random model'])
	plt.text(-4.5, 2.3, '\\textbf{Trained on:}', color='#555', rotation='vertical')
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	plt.title('Proportion of examples explained by path-merging algorithm')
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if np.isnan(data)[i,j]:
				plt.text(j, i, 'N/A', ha="center", va="center", color="k", fontsize=7)
			else:
				c = "w" if data[i,j] > 0.6 else "k"
				plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)

	plt.xticks(np.arange(-.5, 19, 1), minor=True)
	plt.yticks(np.arange(-.5, 3, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)

	ax = plt.gca()
	draw_brace(ax, (19.4, 0.6), 5.2, 'Tested on balanced distribution with lookahead')

	plt.tight_layout()
	fig.savefig('figures/mi_explainable_proportion.pdf', dpi=128)
	plt.clf()

	data = np.empty((3,20))
	data[0,:] = greedy_results[2,:]
	data[1,:] = crafted_results[2,:]
	data[2,:] = crafted_OOD_results[2,:]

	fig = plt.gcf()
	fig.set_size_inches(8.0, 1.95, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	r_cmap.set_bad(color='white')
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([-0.2] + [i for i in range(1,20)], labels=(['Tested on\nnaïve distr.'] + [i for i in range(1,20)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(3), labels=['Naïve distr.', 'Balanced distr.', 'Balanced distr.\n(lookahead $\\le 12$)'])
	plt.text(-4.5, 2.3, '\\textbf{Trained on:}', color='#555', rotation='vertical')
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	plt.title('Proportion of ``maximal\'\' path-merge operations')
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if np.isnan(data)[i,j]:
				plt.text(j, i, 'N/A', ha="center", va="center", color="k", fontsize=7)
			else:
				c = "w" if data[i,j] > 0.6 else "k"
				plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)

	plt.xticks(np.arange(-.5, 19, 1), minor=True)
	plt.yticks(np.arange(-.5, 2, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)

	ax = plt.gca()
	draw_brace(ax, (19.4, 0.6), 4.0, 'Tested on balanced distribution with lookahead')

	plt.tight_layout()
	fig.savefig('figures/mi_optimal_merge_op_proportion.pdf', dpi=128)
	plt.clf()

def make_lookahead_histogram(num_samples=1000000):
	def build_module(name):
		from os import system
		if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I../ {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
			print(f"ERROR: Unable to compile `{name}.cpp`.")
			import sys
			sys.exit(1)
	try:
		from os.path import getmtime
		from importlib.util import find_spec
		generator_module = find_spec('generator')
		if generator_module == None:
			raise ModuleNotFoundError
		elif getmtime(generator_module.origin) < getmtime('generator.cpp'):
			print("C++ module `generator` is out-of-date. Compiling from source...")
			build_module("generator")
		import generator
	except ModuleNotFoundError:
		print("C++ module `generator` not found. Compiling from source...")
		build_module("generator")
		import generator
	print("C++ module `generator` loaded.")

	output = generator.lookahead_histogram(128, 10000000, 9999, 1, False)
	output = output / np.sum(output)

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	ax.bar(np.arange(0,len(output)), output)
	plt.xlim(-0.5, 20.5)
	#plt.ylim(1.0e-9, 0.6)
	ax.set_yscale("log", base=10)
	plt.xlabel('Lookahead')
	plt.ylabel('Example frequency')
	#plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/lookahead_histogram.pdf', dpi=256)
	plt.clf()

from sys import argv
do_all = (len(argv) == 1 or '--all' in argv)

plt.style.use('ggplot')
makedirs('figures', exist_ok=True)

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

if do_all or '--sensitivity' in argv:
	make_sensitivity_figures(max_epoch=3370)
if do_all or '--dfs' in argv:
	make_dfs_figures(max_epoch=2845)
if do_all or '--scaling-inputsize' in argv:
	make_scaling_figures(epoch=900, variable='input_sizes')
if do_all or '--scaling-layers' in argv:
	make_scaling_figures(epoch=290, variable='layers')
if do_all or '--scaling-hiddendim' in argv:
	make_scaling_figures(epoch=900, variable='hidden_dim')
if do_all or '--scaling-NL' in argv:
	make_scaling_figures(epoch=535, variable='NL_16hid_8layer')
if do_all or '--scaling-dfs' in argv:
	make_scaling_figures(epoch=1100, variable='dfs_padded', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--mi' in argv:
	make_mi_figures(epoch=3370)
if do_all or '--lookahead-histogram' in argv:
	make_lookahead_histogram()
