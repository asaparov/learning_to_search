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
		text_y += 0.2*yspan
	else:
		text_y += -0.2*yspan
	ax.text((xmax+xmin)/2., text_y, text, ha='center', va='bottom', color='#555')

def make_sensitivity_figures(max_epoch=100000):
	greedy_ckpt_dir  = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	crafted_ckpt_dir = 'useful_path_results/checkpoints_v3_6layer_inputsize128_maxlookahead20_seed1_trainstreaming_nomask_unablated_padded' # available seeds: 1-8
	greedy_ckpt = find_ckpt(greedy_ckpt_dir, max_epoch)
	crafted_ckpt = find_ckpt(crafted_ckpt_dir, max_epoch)
	greedy_accuracies = do_evaluate_model(greedy_ckpt)
	crafted_accuracies = do_evaluate_model(crafted_ckpt)
	#greedy_star_accuracies = do_evaluate_model(greedy_ckpt, star_distribution=True)
	#crafted_star_accuracies = do_evaluate_model(crafted_ckpt, star_distribution=True)

	data = np.empty((2, 21))
	data[0,:21] = [acc for acc,_,_ in greedy_accuracies]
	data[1,:21] = [acc for acc,_,_ in crafted_accuracies]

	fig = plt.gcf()
	fig.set_size_inches(10.0, 2.1, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([-0.2] + [i for i in range(1,21)], labels=(['Tested on\nsimple distr.'] + [i+1 for i in range(20)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(2), labels=['Trained on\nsimple distr.', 'Trained on\ncrafted distr.'])
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			c = "w" if data[i,j] > 0.6 else "k"
			plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=9)

	plt.xticks(np.arange(-.5, 21, 1), minor=True)
	plt.yticks(np.arange(-.5, 2, 1), minor=True)
	plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
	plt.tick_params(which='minor', bottom=False, left=False)

	ax = plt.gca()
	draw_brace(ax, (20.3, 0.7), 2.7, 'Tested on crafted distribution with lookahead')

	plt.tight_layout()
	fig.savefig('figures/sensitivity.pdf', dpi=128)
	plt.clf()

def make_dfs_figures(max_epoch=100000):
	unpadded_ckpt_dirs = {
		48 : 'dfs_results/checkpoints_v3_7layer_inputsize48_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		64 : 'dfs_results/checkpoints_v3_7layer_inputsize64_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		80 : 'dfs_results/checkpoints_v3_7layer_inputsize80_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		96 : 'dfs_results/checkpoints_v3_7layer_inputsize96_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		#112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed2_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed1_trainstreaming_nomask_unablated_dfs',
		128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed2_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed3_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed4_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed5_trainstreaming_nomask_unablated_dfs',
		#128 : 'dfs_results/checkpoints_v3_7layer_inputsize128_maxlookahead-1_seed6_trainstreaming_nomask_unablated_dfs'
	}
	padded_ckpt_dirs = {
		48 : 'dfs_results/checkpoints_v3_7layer_inputsize48_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		64 : 'dfs_results/checkpoints_v3_7layer_inputsize64_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		80 : 'dfs_results/checkpoints_v3_7layer_inputsize80_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		96 : 'dfs_results/checkpoints_v3_7layer_inputsize96_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
		112 : 'dfs_results/checkpoints_v3_7layer_inputsize112_maxlookahead-1_seed1_trainstreaming_nomask_unablated_padded_dfs',
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
	fig.set_size_inches(10.0, 2.1, forward=True)
	r_cmap = plt.get_cmap('plasma').reversed()
	plt.imshow(data, cmap=r_cmap, vmin=0.0, vmax=1.0)
	plt.xticks([i for i in range(16)], labels=([i for i in range(16)]), rotation=45, ha="right", rotation_mode="anchor")
	plt.yticks(np.arange(2), labels=['Standard training', 'Trained with\nrandom padding'])
	plt.tick_params(axis='both', which='both', length=0)
	plt.grid(False)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			c = "w" if data[i,j] > 0.6 else "k"
			plt.text(j, i, '%.2f' % data[i,j], ha="center", va="center", color=c, fontsize=9)

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

def make_scaling_figures(epoch=1500, variable='input_sizes'):
	seeds = [int(d[len('seed_'):]) for d in listdir('scaling_experiments/csvs/scaling_' + variable) if d.startswith('seed_')]

	train_losses = {}
	test_losses = {}
	has_converged = {}
	first_seed = True
	for seed in seeds:
		if variable == 'input_sizes':
			var_name = 'Input size'
			prefix = '1e-5_seed' + str(seed) + '_'
		elif variable == 'layers':
			var_name = 'Num layers'
			prefix = 'nlayers_'
		elif variable == 'hidden_dim':
			var_name = 'Hidden dim'
			prefix = 'hid_'
		inputs = [f for f in listdir('scaling_experiments/csvs/scaling_' + variable + '/seed_' + str(seed)) if f.startswith(prefix) and f.endswith('.csv')]
		for f in inputs:
			inputsize = int(f[len(prefix):-len('.csv')])
			csv = read_csv('scaling_experiments/csvs/scaling_{}/seed_{}/{}'.format(variable, seed, f))
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
			train_loss = float(row[train_loss_idx])
			test_loss = float(row[test_loss_idx])
			has_seed_converged = any([r[train_acc_idx] == '1.0' for r in csv[1:(row_idx+1)]])

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

	if variable == 'input_sizes':
		xaxis = 'Maximum input graph size'
		xmin, xmax = 8, 52
	elif variable == 'layers':
		xaxis = 'Number of transformer layers'
		xmin, xmax = 4, 45
	elif variable == 'hidden_dim':
		xaxis = 'Non-embedding parameters'
		nlayers = 8 # TODO: double-check the number of layers in the hiddendim experiments
		xmin, xmax = 100000, 1000000000

	inputsizes = np.empty(len(test_losses), dtype=np.uint64)
	counter = 0
	for inputsize,_ in test_losses.items():
		inputsizes[counter] = inputsize
		counter += 1
	sorted_idx = inputsizes.argsort()
	inputsizes = inputsizes[sorted_idx]

	if variable == 'input_sizes':
		x = (inputsize - 5) // 3
	elif variable == 'layers':
		x = inputsize
	elif variable == 'hidden_dim':
		fixed_max_input_size = 96 # TODO: double-check the max input size in the hiddendim experiments
		ntoken = (fixed_max_input_size-5) // 3 + 5
		dmodel = np.max(ntoken, inputsize) + fixed_max_input_size
		x = 6*dmodel*dmodel*nlayers

	converged = np.empty(len(has_converged))
	counter = 0
	for _,converged_list in has_converged.items():
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
	ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
	ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Convergence frequency')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_convergence.pdf', dpi=256)
	plt.clf()

	mintestlosses = np.empty(len(test_losses))
	counter = 0
	for inputsize,loss_list in test_losses.items():
		mintestlosses[counter] = np.mean([v for k,v in loss_list.items() if converged[inputsize][k]]) #min(loss_list.values())
		counter += 1
	mintestlosses = mintestlosses[sorted_idx]

	if variable == 'input_sizes':
		a,b = np.polyfit(x[converged > 0], np.log(mintestlosses[converged > 0] - 0.0003), 1, w=np.sqrt(mintestlosses[converged > 0]))

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	x = np.arange(xmin,xmax,(xmax-xmin)/10000)
	if variable == 'input_sizes':
		ax.plot(x, np.exp(a*x + b) + 0.0003, '--', color='k', linewidth=0.5)
	ax.plot(x[converged > 0], mintestlosses[converged > 0], '.')
	plt.xlim(xmin, xmax)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
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
		mintrainlosses[counter] = min(loss_list.values())
		counter += 1
	mintrainlosses = mintrainlosses[sorted_idx]

	if variable == 'input_sizes':
		m,b = np.polyfit(x[converged > 0], mintrainlosses[converged > 0], 1)

	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(4, 2.5, forward=True)
	x = np.arange(xmin,xmax,(xmax-xmin)/10000)
	if variable == 'input_sizes':
		ax.plot(x, m*x + b, '--', color='k', linewidth=0.5)
	ax.plot(x[converged > 0], mintrainlosses[converged > 0], '.')
	plt.xlim(xmin, xmax)
	#plt.ylim(0.0001, 10.0)
	ax.set_xscale("log", base=10)
	ax.set_yscale("log", base=10)
	ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
	ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
	plt.xlabel(xaxis)
	plt.ylabel('Minimum training loss')
	plt.grid(True, axis='x', which='both')
	plt.tight_layout()
	fig.savefig('figures/scaling_' + variable + '_train.pdf', dpi=256)
	plt.clf()

from sys import argv
all = (len(argv) == 1 or '--all' in argv)

plt.style.use('ggplot')
makedirs('figures', exist_ok=True)

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif"
})

if all or '--sensitivity' in argv:
	make_sensitivity_figures()
if all or '--dfs' in argv:
	make_dfs_figures()
if all or '--scaling-inputsize' in argv:
	make_scaling_figures(epoch=1500, variable='input_sizes')
if all or '--scaling-layers' in argv:
	make_scaling_figures(epoch=450, variable='layers')
if all or '--scaling-hiddendim' in argv:
	make_scaling_figures(epoch=120, variable='hidden_dim')
