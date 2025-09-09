from analyze import do_evaluate_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox
from os import listdir, makedirs
import csv

def find_ckpt(directory, max_epoch):
	existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(directory) if ckpt.startswith('epoch') and ckpt.endswith('.pt')]
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
	star_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed2_trainstreaming_nomask_unablated_padded_star'
	greedy_ckpt = find_ckpt(greedy_ckpt_dir, max_epoch)
	crafted_ckpt = find_ckpt(crafted_ckpt_dir, max_epoch)
	crafted_OOD_ckpt = find_ckpt(crafted_OOD_ckpt_dir, max_epoch)
	star_ckpt = find_ckpt(star_ckpt_dir, max_epoch)
	greedy_accuracies = do_evaluate_model(greedy_ckpt)
	crafted_accuracies = do_evaluate_model(crafted_ckpt)
	crafted_OOD_accuracies = do_evaluate_model(crafted_OOD_ckpt)
	star_accuracies = do_evaluate_model(star_ckpt)
	greedy_star_accuracies = do_evaluate_model(greedy_ckpt, star_distribution=True)
	crafted_star_accuracies = do_evaluate_model(crafted_ckpt, star_distribution=True)
	crafted_OOD_star_accuracies = do_evaluate_model(crafted_OOD_ckpt, star_distribution=True)
	star_star_accuracies = do_evaluate_model(star_ckpt, star_distribution=True)

	data = np.empty((4, 22))
	data[0,0] = greedy_accuracies[0][0]
	data[1,0] = star_accuracies[0][0]
	data[2,0] = crafted_accuracies[0][0]
	data[3,0] = crafted_OOD_accuracies[0][0]
	data[0,1] = np.mean([acc for acc,_,_ in greedy_star_accuracies])
	data[1,1] = np.mean([acc for acc,_,_ in star_star_accuracies])
	data[2,1] = np.mean([acc for acc,_,_ in crafted_star_accuracies])
	data[3,1] = np.mean([acc for acc,_,_ in crafted_OOD_star_accuracies])
	data[0,2:] = [acc for acc,_,_ in greedy_accuracies[1:]]
	data[1,2:] = [acc for acc,_,_ in star_accuracies[1:]]
	data[2,2:] = [acc for acc,_,_ in crafted_accuracies[1:]]
	data[3,2:] = [acc for acc,_,_ in crafted_OOD_accuracies[1:]]

	fig = plt.gcf()
	fig.set_size_inches(8.0, 1.9, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([-0.2] + [i for i in range(1,22)], labels=(['Naïve distr.'] + ['Star distr.'] + [i+1 for i in range(20)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(4), labels=['Naïve distr.', 'Star distr.', 'Balanced distr.', 'Balanced distr.\n(lookahead $\\le 12$)'])
	plt.text(-4.5, 2.8, '\\textbf{Trained on:}', color='#555', rotation='vertical')
	plt.text(-4.1, 4.5, '\\textbf{Tested on:}', color='#555')
	plt.tick_params(axis='both', which='both', length=0, pad=0.8)
	plt.grid(False)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			c = "w" if data[i,j] > 0.6 else "k"
			plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)

	plt.xticks(np.arange(-.5, 22, 1), minor=True)
	plt.yticks(np.arange(-.5, 4, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)

	ax = plt.gca()
	draw_brace(ax, (21.4, 1.6), 5.0, 'Balanced distribution with lookahead')

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
	if variable in ('dfs_padded','dfs_unpadded', 'dfs_balanced', 'dfs_params'):
		csv_dir = 'dfs_csvs/'
	elif variable in ('si', 'si_params'):
		csv_dir = 'si_csvs/'
	elif variable in ('decoder_RoPE', 'decoder_RoPE_params'):
		csv_dir = 'decoder_csvs/'
	elif variable in ('direct_schedule', 'dfs_schedule'):
		csv_dir = 'schedule_csvs/'
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
	IGNORE_INPUTSIZES = [138]
	IGNORE_NL_INPUTSIZES = [112,128]
	for seed in sorted(seeds):
		if variable == 'input_sizes':
			var_name = 'input size'
			prefix = '1e-5_seed' + str(seed) + '_'
		elif variable == 'layers':
			var_name = 'num layers'
			prefix = 'nlayers_'
		elif variable == 'hidden_dim':
			var_name = 'hidden dim'
			prefix = 'hid_'
		elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
			var_name = 'input size'
			prefix = 'input_'
		elif variable == 'decoder_RoPE':
			var_name = 'input size'
			prefix = 'input'
		elif variable in ('dfs_padded', 'dfs_unpadded', 'dfs_balanced', 'si'):
			var_name = 'input size'
			prefix = 'input_'
		elif variable in ('dfs_params', 'si_params', 'decoder_RoPE_params'):
			var_name = 'non-embedding parameters'
			prefix = ''
		elif variable in ('direct_schedule', 'dfs_schedule'):
			var_name = 'warmup'
			prefix = 'warmup_'
		elif variable == 'causal_masking':
			var_name = 'input size'
			prefix = f'scaling_n_c_1e-5_seed_{seed}_'
		elif variable == 'causal_masking_hidden_dim':
			var_name = 'non-embedding parameters'
			prefix = f'seed{seed}_98_max_8_hid_'
		inputs = [f for f in listdir(csv_dir + 'scaling_' + variable + '/seed_' + str(seed)) if f.startswith(prefix) and f.endswith('.csv')]
		for f in inputs:
			if variable in ('dfs_params', 'si_params', 'decoder_RoPE_params'):
				tokens = f[:-len('.csv')].split('_')
				if len(tokens) != 3 or not tokens[0].endswith('layer') or not tokens[1].endswith('hid') or not tokens[2].endswith('head'):
					continue
				nlayers = int(tokens[0][:-len('layer')])
				hid = int(tokens[1][:-len('hid')])
				nheads = int(tokens[2][:-len('head')])
				if variable == 'dfs_params':
					fixed_max_input_size = 112
				elif variable == 'si_params':
					fixed_max_input_size = 272
				elif variable == 'decoder_RoPE_params':
					fixed_max_input_size = 98
				ntoken = (fixed_max_input_size-5) // 3 + 5
				dmodel = max(ntoken, hid) + fixed_max_input_size
				inputsize = 6*dmodel*dmodel*nlayers
			elif variable == 'causal_masking':
				substr = f[len(prefix):]
				inputsize = int(substr[:substr.index('_')])
			elif variable == 'causal_masking_hidden_dim':
				substr = f[len(prefix):]
				hid = int(substr[:substr.index('_')])
				nlayers = 8
				nheads = 1
				fixed_max_input_size = 98
				ntoken = (fixed_max_input_size-5) // 3 + 5
				dmodel = max(ntoken, hid) + fixed_max_input_size
				inputsize = 6*dmodel*dmodel*nlayers
			elif variable == 'decoder_RoPE':
				tokens = f[:-len('.csv')].split('_')
				if not tokens[0].startswith('input'):
					continue
				inputsize = int(tokens[0][len('input'):])
			else:
				inputsize = int(f[len(prefix):-len('.csv')])
			if variable == 'input_sizes' and inputsize in IGNORE_INPUTSIZES:
				continue
			elif variable == 'hidden_dim' and inputsize in IGNORE_HIDS:
				continue
			elif variable == 'layers' and inputsize in IGNORE_LAYERS:
				continue
			elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer') and inputsize in IGNORE_NL_INPUTSIZES:
				continue
			csv = read_csv(csv_dir + 'scaling_{}/seed_{}/{}'.format(variable, seed, f))
			csv[0] = [s.replace(' ','_').lower() for s in csv[0]]
			if 'epoch' not in csv[0]:
				print('WARNING: {} {}, seed {}, has no results.'.format(var_name.capitalize(), inputsize, seed))
				continue
			epoch_idx = csv[0].index('epoch')
			train_loss_idx = csv[0].index('training_loss')
			test_loss_idx = csv[0].index('test_loss')
			train_acc_idx = csv[0].index('training_accuracy')
			try:
				row_idx = next(i for i in range(1,len(csv)) if int(csv[i][epoch_idx]) == epoch)
			except StopIteration:
				print('WARNING: {} {}, seed {}, is missing epoch {}.'.format(var_name.capitalize(), inputsize, seed, epoch))
				if not keep_incomplete_seeds:
					continue
				row_idx = len(csv) - 1
			train_loss_column = np.array([float(csv[i][train_loss_idx]) if i < len(csv) else float(csv[-1][train_loss_idx]) for i in range(1,row_idx+TRAIN_LOSS_WINDOW+2)])
			train_loss_column = np.concatenate((np.ones(TRAIN_LOSS_WINDOW)*float(csv[1][train_loss_idx]), train_loss_column))
			if train_loss_scaling_bug:
				# TODO: this is due to an earlier bug in the `train.py` code where the loss wasn't being correctly scaled by the batch size; this line should be removed if this script is being used on data produced from a more recent version of `train.py` where the bug is fixed
				if variable == 'layers' and inputsize >= 64:
					train_loss_column /= 2**12
				elif variable == 'hidden_dim' and inputsize == 2048:
					pass
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
					print('WARNING: {} {} is missing seed {}.'.format(var_name.capitalize(), inputsize, seed))
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
			missing_inputsizes = [inputsize for inputsize,losses in train_losses.items() if seed not in losses]
			if len(missing_inputsizes) != 0:
				print('WARNING: Seed {} is missing {} {}.'.format(seed, var_name, inputsize))
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

	update_period = 2**18
	if variable in ('si', 'si_params'):
		update_period = 2**16

	if variable in ('input_sizes', 'causal_masking'):
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 8, 52
	elif variable == 'layers':
		xaxis = 'Number of transformer layers'
		legend_title = 'Number of\ntransformer layers'
		xmin, xmax = 4, 45
	elif variable in ('hidden_dim', 'dfs_params', 'si_params', 'causal_masking_hidden_dim', 'decoder_RoPE_params'):
		xaxis = 'Non-embedding parameters'
		legend_title = 'Non-embedding\nparameters'
		nlayers = 8
		xmin, xmax = update_period, 1000000000
	elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 8, 34
	elif variable in ('dfs_padded', 'dfs_unpadded', 'dfs_balanced', 'si', 'decoder_RoPE'):
		xaxis = 'Maximum input graph size'
		legend_title = 'Maximum input graph size'
		xmin, xmax = 10, 65
	elif variable in ('dfs_schedule', 'direct_schedule'):
		xaxis = 'Warmup steps'
		legend_title = 'Warmup steps'
		xmin, xmax = 0, 300000

	inputsizes = np.empty(len(test_losses), dtype=np.uint64)
	counter = 0
	for inputsize,_ in test_losses.items():
		inputsizes[counter] = inputsize
		counter += 1
	inputsizes = np.array(sorted(inputsizes))

	if variable in ('input_sizes', 'NL_16hid_8layer', 'NL_32hid_16layer', 'causal_masking', 'decoder_RoPE'):
		x = (inputsizes - 5) // 3
	elif variable in ('layers', 'dfs_params', 'si_params', 'causal_masking_hidden_dim', 'decoder_RoPE_params'):
		x = inputsizes
	elif variable in ('hidden_dim'):
		fixed_max_input_size = 98
		ntoken = (fixed_max_input_size-5) // 3 + 5
		dmodel = np.maximum(ntoken, inputsizes) + fixed_max_input_size
		x = 6*dmodel*dmodel*nlayers
	elif variable in ('dfs_padded', 'dfs_unpadded', 'dfs_balanced'):
		x = (inputsizes - 4) // 4
	elif variable == 'si':
		x = (inputsizes - 2) // 6
	elif variable in ('dfs_schedule', 'direct_schedule'):
		x = inputsizes / 2**8

	converged = np.zeros(len(has_converged))
	for inputsize,converged_list in has_converged.items():
		idx = np.where(inputsizes == inputsize)[0].item()
		if len(converged_list) != 0:
			converged[idx] = sum(converged_list.values()) / len(converged_list)

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	ax.plot(x, converged, '.')
	plt.xlim(xmin, xmax)
	plt.ylim(-0.05, 1.05)
	ax.set_xscale("log", base=10)
	if variable not in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'direct_schedule', 'dfs_schedule', 'causal_masking_hidden_dim'):
		ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
		ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Fraction of converged seeds')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_convergence.pdf', dpi=256)
	plt.clf()

	mintestlosses = np.empty(len(test_losses))
	mintestlosses.fill(float('inf'))
	for inputsize,loss_list in test_losses.items():
		idx = np.where(inputsizes == inputsize)[0].item()
		losses = [v[-1] for k,v in loss_list.items() if has_converged[inputsize][k]]
		if len(losses) != 0:
			mintestlosses[idx] = np.min(losses) #min(loss_list.values())

	if variable == 'input_sizes':
		a,b = np.polyfit(x[converged > 0], np.log(mintestlosses[converged > 0] - np.min(mintestlosses) + 1.0e-9), 1, w=np.sqrt(mintestlosses[converged > 0]))

	if np.sum(converged > 0) > 1:
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
		if variable not in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'direct_schedule', 'dfs_schedule', 'causal_masking_hidden_dim'):
			ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
			ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
		plt.xlabel(xaxis)
		plt.ylabel('Minimum test loss')
		plt.grid(True, axis='x', which='both')
		plt.tight_layout()
		fig.savefig('figures/scaling_' + variable + '_test.pdf', dpi=256)
		plt.clf()

	mintrainlosses = np.empty(len(test_losses))
	for inputsize,loss_list in train_losses.items():
		idx = np.where(inputsizes == inputsize)[0].item()
		losses = [v[-1] for k,v in loss_list.items()]
		mintrainlosses[idx] = np.mean(losses)

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
	if variable not in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'direct_schedule', 'dfs_schedule', 'causal_masking_hidden_dim'):
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
	if len(inputsizes) == 1:
		colors[inputsizes[0]] = cmap(0.0)
	else:
		for inputsize in inputsizes:
			t = counter/(len(inputsizes) - 1)
			colors[inputsize] = cmap(t*start_color_point + (1-t)*end_color_point)
			counter += 1

	def plot_loss(loss_mode, plot_min_only, vs_flops):
		fig = plt.gcf()
		ax = plt.gca()
		fig.set_size_inches(4, 2.5, forward=True)
		max_x = 0.0
		handles = {}
		losses = train_losses if loss_mode == 'train' else test_losses
		for i in range(len(inputsizes)):
			inputsize = inputsizes[i]
			if plot_min_only:
				min_loss_len = min([len(loss_vals) for loss_vals in losses[inputsize].values()])
				all_losses = np.empty((len(losses[inputsize]), min_loss_len))
				counter = 0
				for _,loss_vals in losses[inputsize].items():
					all_losses[counter,:] = loss_vals[:min_loss_len]
					counter += 1
				examples_seen = np.arange(0,min_loss_len) * update_period
				if vs_flops:
					if variable in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'causal_masking_hidden_dim'):
						nonembedding_params = x[i]
					elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
						vocab_size = 109
						dmodel = 5*inputsizes[i] + vocab_size
						nlayers = 8
						nonembedding_params = 6*dmodel*dmodel*nlayers
					else:
						raise Exception('Unimplemented')
					examples_seen = 6*examples_seen*nonembedding_params
				line, = ax.plot(examples_seen, np.min(all_losses, axis=0), c=colors[inputsize], alpha=0.8)
				max_x = max(examples_seen[-1], max_x)
				handles[inputsize] = (line, x[i])
			else:
				for _,loss_vals in losses[inputsize].items():
					examples_seen = np.arange(0,len(loss_vals)) * update_period
					if vs_flops:
						if variable in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'causal_masking_hidden_dim'):
							nonembedding_params = x[i]
						elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer'):
							vocab_size = 109
							dmodel = 5*inputsizes[i] + vocab_size
							nlayers = 8
							nonembedding_params = 6*dmodel*dmodel*nlayers
						else:
							raise Exception('Unimplemented')
						examples_seen = 6*examples_seen*nonembedding_params
					line, = ax.plot(examples_seen, loss_vals, c=colors[inputsize], alpha=0.8)
					max_x = max(examples_seen[-1], max_x)
					if inputsize not in handles:
						handles[inputsize] = (line, x[i])
		if variable in ('hidden_dim', 'dfs_params', 'si_params', 'decoder_RoPE_params', 'direct_schedule', 'dfs_schedule', 'causal_masking_hidden_dim'):
			labels = ['{0:.1f}M'.format(l/1000000) for _,(_,l) in handles.items()]
		else:
			labels = [str(l) for _,(_,l) in handles.items()]
		if variable == 'decoder_RoPE_params':
			num_cols = 2
		elif len(labels) <= 4 and variable not in ('hidden_dim', 'causal_masking_hidden_dim'):
			num_cols = len(labels)
		elif variable in ('NL_16hid_8layer', 'NL_32hid_16layer', 'si'):
			num_cols = 2
		else:
			num_cols = np.ceil(len(labels)/6)
		plt.legend([h for _,(h,_) in handles.items()], labels, title=legend_title, fontsize=6.0, title_fontsize=7.5, ncols=num_cols, handlelength=1.0, loc=('lower left' if variable in ('NL_16hid_8layer', 'NL_32hid_16layer') else 'best'))
		if loss_mode == 'train':
			if vs_flops:
				if variable == 'si_params':
					plt.xlim(235000000*update_period, 6*epoch*update_period*min(inputsizes))
				else:
					plt.xlim(10000000*update_period, max_x)
			else:
				plt.xlim(update_period, max_x)
		else:
			if vs_flops:
				plt.xlim(10000000*1000000, 6*epoch*update_period*min(inputsizes))
			else:
				plt.xlim(1000000, max_x)
		#plt.ylim(0.0001, 10.0)
		ax.set_xscale("log", base=10)
		ax.set_yscale("log", base=10)
		if vs_flops:
			plt.xlabel('Floating-point operations')
		else:
			plt.xlabel('Training examples')
		if loss_mode == 'train':
			if plot_min_only:
				plt.ylabel('Minimum train loss')
			else:
				plt.ylabel('Training loss')
		else:
			if plot_min_only:
				plt.ylabel('Minimum test loss')
			else:
				plt.ylabel('Test loss')
		plt.grid(True, axis='x', which='both')
		plt.tight_layout()
		fig.savefig('figures/scaling_' + variable + '_' + loss_mode + '_' + ('min' if plot_min_only else '') + 'loss' + ('_flops' if vs_flops else '') + '.pdf', dpi=256)
		plt.clf()

	plot_loss(loss_mode='test', plot_min_only=False, vs_flops=False)
	plot_loss(loss_mode='train', plot_min_only=False, vs_flops=False)
	plot_loss(loss_mode='test', plot_min_only=True, vs_flops=False)
	plot_loss(loss_mode='train', plot_min_only=True, vs_flops=False)
	if variable in ('hidden_dim', 'dfs_params', 'si_params', 'NL_16hid_8layer', 'NL_32hid_16layer', 'causal_masking_hidden_dim', 'decoder_RoPE_params'):
		plot_loss(loss_mode='test', plot_min_only=False, vs_flops=True)
		plot_loss(loss_mode='train', plot_min_only=False, vs_flops=True)
		plot_loss(loss_mode='test', plot_min_only=True, vs_flops=True)
		plot_loss(loss_mode='train', plot_min_only=True, vs_flops=True)

def get_mi_path_merge_stats(ckpt_dir, epoch, lookahead):
	from trace_circuit import do_analysis, MissingSampleError
	ckpt_filepath = ckpt_dir + '/epoch{}.pt'.format(epoch)
	try:
		print('Reading trace_circuit analysis from {} for lookahead {}.'.format(ckpt_filepath, lookahead))
		_, _, _, path_merge_stats, _, _, _ = do_analysis(ckpt_filepath, None, lookahead, 100, quiet=True)
		return path_merge_stats
	except MissingSampleError:
		print('WARNING: Checkpoint for {} is missing tracing results for lookahead {}.'.format(ckpt_filepath, lookahead))
		return None

def get_mi_path_merge_ops(ckpt_dir, epoch, example_input):
	import torch
	from trace_circuit import do_analysis_on_example, TransformerTracer
	ckpt_filepath = ckpt_dir + '/epoch{}.pt'.format(epoch)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	tfm_model, _, _, _ = torch.load(ckpt_filepath, map_location=device, weights_only=False)
	for transformer in tfm_model.transformers:
		if not hasattr(transformer, 'pre_ln'):
			transformer.pre_ln = True
	tfm_model = torch.compile(tfm_model)
	tracer = TransformerTracer(tfm_model)
	return do_analysis_on_example(tracer, example_input, device)

def get_mi_results(ckpt_dir, epoch=3370):
	from trace_circuit import do_analysis, MissingSampleError
	results = np.empty((3,20))
	counter = 0
	for lookahead in [-1] + list(range(1,20)):
		ckpt_filepath = ckpt_dir + '/epoch{}.pt'.format(epoch)
		try:
			print('Reading trace_circuit analysis from {} for lookahead {}.'.format(ckpt_filepath, lookahead))
			explainable_example_proportion, average_merge_ops_per_example, suboptimal_merge_op_proportion, _, _, _ = do_analysis(ckpt_filepath, None, lookahead, 100, quiet=True)
			results[0,counter] = explainable_example_proportion
			results[1,counter] = average_merge_ops_per_example
			results[2,counter] = 1.0 - suboptimal_merge_op_proportion
		except MissingSampleError:
			print('WARNING: Checkpoint for {} is missing tracing results for lookahead {}.'.format(ckpt_filepath, lookahead))
			results[:,counter] = np.nan
		counter += 1
	return results

def make_mi_path_merge_stats(epoch=3370, lookahead=9):
	greedy_ckpt_dir  = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed1_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_OOD_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead12_seed2_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	path_merge_stats = get_mi_path_merge_stats(crafted_ckpt_dir, epoch, lookahead=lookahead)

	edge_list = []
	for l in range(len(path_merge_stats)):
		data = np.zeros((lookahead+1, lookahead+1))
		for src, dst_histogram in path_merge_stats[l].items():
			average_freq = np.mean(list(dst_histogram.values()))
			max_freq_diff = np.max(np.abs(np.array(list(dst_histogram.values())) - average_freq))
			for dst, frequency in dst_histogram.items():
				if src != -1 and dst != -1 and src <= lookahead and dst <= lookahead:
					#edge_list.append((l, src, dst, (frequency - average_freq) / max_freq_diff))
					pass
				else:
					continue
				data[dst,src] += frequency
		data = data.T / (np.sum(data.T, axis=0)+1e-9)[np.newaxis,:]

		fig = plt.gcf()
		fig.set_size_inches(3.0, 3.0, forward=True)
		ax = plt.gca()
		ax.xaxis.get_major_locator().set_params(integer=True)
		ax.yaxis.get_major_locator().set_params(integer=True)
		r_cmap = plt.get_cmap('magma').reversed()
		plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
		#plt.xticks([-0.2] + [i for i in range(1,22)], labels=(['Naïve distr.'] + ['Star distr.'] + [i+1 for i in range(20)]), rotation=45, ha="right", rotation_mode="anchor")
		#plt.yticks(np.arange(4), labels=['Naïve distr.', 'Star distr.', 'Balanced distr.', 'Balanced distr.\n(lookahead $\\le 12$)'])
		plt.tick_params(axis='both', which='both', length=0, pad=0.8)
		plt.grid(False)
		'''for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				c = "w" if data[i,j] > 0.6 else "k"
				plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=8)'''

		#plt.xticks(np.arange(-.5, 22, 1), minor=True)
		#plt.yticks(np.arange(-.5, 4, 1), minor=True)
		plt.xlabel('Target vertex (distance from start)')
		plt.ylabel('Source vertex (distance from start)')
		plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
		plt.tick_params(which='minor', bottom=False, left=False)
		plt.tight_layout()
		fig.savefig('figures/mi_path_merge_stats_layer{}.pdf'.format(l), dpi=128)
		plt.clf()

	# save the standalone colorbar
	a = np.array([[0,1]])
	plt.figure(figsize=(3, 0.6))
	img = plt.imshow(a, cmap=r_cmap)
	plt.gca().set_visible(False)
	cax = plt.axes([0.1, 0.72, 0.8, 0.2])
	plt.colorbar(orientation="horizontal", cax=cax, label="frequency")
	plt.savefig("figures/colorbar.pdf", dpi=128)

	return

	template = r"""
\centering
\tikzset{vertex/.style={shape=circle,draw,node distance=1.5em,fill=white}}
\tikzset{edge/.style = {-{Latex[length=3.5pt,width=3.5pt]},color=black}}
\scalebox{0.82}{
\begin{tikzpicture}[shorten >=0.5pt,node distance=1.1cm]
	% vertices and edges
	\foreach \x in {0,...,12}
		\foreach \y in {0,...,5}
			{\node [vertex] (n\y_\x) at (1.3*\x-1.3,-4*\y) {\x};}

	\foreach \y in {0,...,5}
		\foreach \x in {1,...,12}
			{\draw [edge,thick] (n\y_\number\numexpr\x-1\relax) -- (n\y_\x);}

	\foreach \y in {0,...,5} {
		\node [label] [rotate=45,above left=-0.1em of n\y_0,anchor=south] {\color{red!80!black}\footnotesize\textsf{start}};
		\node [label] [rotate=-45,above right=-0.3em of n\y_12,anchor=south] {\color{red!80!black}\footnotesize\textsf{goal}};
	}
"""

	template += '\n'
	theta = np.pi/5
	node_radius = 0.3
	cmap = plt.get_cmap('seismic').reversed()
	for layer, src, dst, frequency in edge_list:
		node_distance = 1.3*(dst - src)
		if node_distance == 0:
			# TODO: should we depict this in a different way?
			continue
		psi = np.asin(node_radius*np.sin(theta)/np.abs(node_distance))
		(r,g,b,a) = cmap(frequency)
		template += '	\\definecolor{col}{RGB}{' + str(int(np.round(r*255))) + ',' + str(int(np.round(g*255))) + ',' + str(int(np.round(b*255))) + '}\n'
		template += f'	\\draw [color=col,opacity=0.8] (n{layer}_{src}) (n{layer}_{dst})++({-np.sin(np.pi/2 - psi - theta)*node_radius*np.sign(node_distance)},{np.cos(np.pi/2 - psi - theta)*node_radius*np.sign(node_distance)}) arc [start angle={(np.pi/2 - theta + 2*psi)*180/np.pi}, delta angle={(2*theta - 4*psi)*180/np.pi}, radius={node_distance/(2*np.sin(theta))}];\n'

	template += r"""
	\end{tikzpicture}
	} % scalebox
"""
	with open('figures/mi_path_merge_stats.tikz', 'w') as fout:
		fout.write(template)

def make_mi_path_merge_ops(epoch=3370, example_index=0):
	max_input_size = 128
	nlayers = 6
	greedy_ckpt_dir  = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed1_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_OOD_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead12_seed2_trainstreaming_nomask_unablated_padded' # available seeds: 1-8

	QUERY_PREFIX_TOKEN = (max_input_size-5) // 3 + 4
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	PATH_PREFIX_TOKEN = (max_input_size-5) // 3 + 1
	max_vertex_id = (max_input_size - 5) // 3

	input_tokens = []
	path_length = 10
	for i in range(path_length):
		input_tokens += [EDGE_PREFIX_TOKEN, i + 1, i + 2]
	input_tokens += [QUERY_PREFIX_TOKEN, 1, path_length + 1, PATH_PREFIX_TOKEN, 1]
	input_length = len(input_tokens)
	input_offset = max_input_size - len(input_tokens)
	input_tokens = [PADDING_TOKEN] * input_offset + input_tokens
	example_input = np.array(input_tokens)

	example_output, path_merge_ops = get_mi_path_merge_ops(crafted_OOD_ckpt_dir, epoch, example_input)
	print("Model output: ", example_output)

	template = r"""
		\tikzset{token/.style={shape=rectangle,draw,minimum height=1.4em,minimum width=1.3em,scale=0.9}}
		\tikzset{tfmedge/.style = {-{Latex[length=3.5pt,width=3.5pt]},color=black!30}}
		\begin{tikzpicture}[shorten >=0.5pt,node distance=1.1cm]
	"""

	for layer in range(nlayers + 1):
		for i in range(input_length):
			if i == 0 and layer != 0:
				start_node_pos = '[below=3em of t{}_{}]'.format(layer-1, 0)
			elif i != 0:
				start_node_pos = '[right=0em of t{}_{}]'.format(layer, i-1)
			else:
				start_node_pos = ''
			input_token = example_input[input_offset + i]
			if input_token == EDGE_PREFIX_TOKEN:
				token_text = '{\\color{RoyalBlue!80!black}\\textbf{\\texttt{E}}}'
			elif input_token == QUERY_PREFIX_TOKEN:
				token_text = '{\\color{red!80!black}\\textbf{\\texttt{Q}}}'
			elif input_token == PATH_PREFIX_TOKEN:
				token_text = '{\\color{OliveGreen}\\textbf{\\texttt{P}}}'
			else:
				token_text = '{\\texttt{' + str(input_token) + '}}'
			template += '\\node [token] (t{}_{}) {} {};\n'.format(layer, i, start_node_pos, token_text)
	weights_per_layer = []
	for layer in range(nlayers + 1):
		weights_per_layer.append([])
		for op in [op for op in path_merge_ops if op.layer == layer]:
			weights_per_layer[layer].extend([w for w in op.weights if w != None])
	for op in path_merge_ops:
		for parent in op.predecessors:
			index = parent.successors.index(op)
			causes = parent.op_explanations[index]
			if parent.weights[index] == None:
				weight = None
			else:
				weight = (parent.weights[index] - min(weights_per_layer[layer-1])) / (max(weights_per_layer[layer-1]) - min(weights_per_layer[layer-1]))
			is_position_op = False
			is_token_matching = False
			if causes != None:
				for cause in causes:
					if type(cause) == tuple:
						if type(cause[0]) == int and type(cause[1]) == int and cause[0] > max_vertex_id+5 and cause[1] > max_vertex_id+5:
							is_position_op = True
					elif type(cause) == int and cause <= max_vertex_id:
						is_token_matching = True
			if is_position_op and not is_token_matching:
				edge_color = 'red'
			elif is_token_matching and not is_position_op:
				edge_color = 'RoyalBlue'
			elif is_position_op and is_token_matching:
				edge_color = 'Orange'
			elif weight != None:
				edge_color = 'Orange'
				continue
			else:
				edge_color = 'OliveGreen'
			template += '\\draw [tfmedge,draw={}!85!black,opacity={}] (t{}_{}.south) -- (t{}_{}.north);\n'.format(edge_color, weight if weight != None else 1.0, parent.layer, parent.row_id-input_offset, op.layer, op.row_id-input_offset)
	template += '\\end{tikzpicture}\n'

	with open('figures/mi_path_merge_ops.tikz', 'w') as fout:
		fout.write(template)

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
	make_scaling_figures(epoch=900, variable='hidden_dim', keep_incomplete_seeds=False)
if do_all or '--scaling-NL' in argv:
	make_scaling_figures(epoch=1500, variable='NL_16hid_8layer', keep_incomplete_seeds=True)
if do_all or '--scaling-dfs' in argv:
	make_scaling_figures(epoch=2845, variable='dfs_balanced', train_loss_scaling_bug=False)
if do_all or '--scaling-dfs-params' in argv:
	make_scaling_figures(epoch=700, variable='dfs_params', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--scaling-si' in argv:
	make_scaling_figures(epoch=5125, variable='si', train_loss_scaling_bug=False)
if do_all or '--scaling-si-params' in argv:
	make_scaling_figures(epoch=8000, variable='si_params', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--scaling-schedule' in argv:
	make_scaling_figures(epoch=1550, variable='dfs_schedule', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--scaling-decoder' in argv:
	make_scaling_figures(epoch=950, variable='causal_masking', keep_incomplete_seeds=True)
if do_all or '--scaling-decoder-params' in argv:
	make_scaling_figures(epoch=15000, variable='causal_masking_hidden_dim', keep_incomplete_seeds=True)
if do_all or '--scaling-decoder-RoPE' in argv:
	make_scaling_figures(epoch=2000, variable='decoder_RoPE', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--scaling-decoder-RoPE-params' in argv:
	make_scaling_figures(epoch=20000, variable='decoder_RoPE_params', keep_incomplete_seeds=True, train_loss_scaling_bug=False)
if do_all or '--mi' in argv:
	make_mi_figures(epoch=3370)
if do_all or '--mi-path-merge-stats' in argv:
	make_mi_path_merge_stats(epoch=3370)
if do_all or '--mi-path-merge-ops' in argv:
	make_mi_path_merge_ops(epoch=3370)
if do_all or '--lookahead-histogram' in argv:
	make_lookahead_histogram()
