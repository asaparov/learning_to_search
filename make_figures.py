from analyze import do_evaluate_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from os import listdir, makedirs

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


from sys import argv
all = (len(argv) == 1 or '--all' in argv)

plt.style.use('ggplot')
makedirs('figures', exist_ok=True)

#plt.rcParams.update({
#	"text.usetex": True,
#	"font.family": "serif",
#	"font.serif": ["Palatino"],
#})

if all or '--sensitivity' in argv:
	make_sensitivity_figures()
if all or '--dfs' in argv:
	make_dfs_figures()
