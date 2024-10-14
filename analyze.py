from random import seed, shuffle, randrange, Random
import numpy as np
import torch
from train import generate_eval_data, generate_star_graph_data, evaluate_model, binomial_confidence_int
import generator
from gpt2 import Transformer, ToeplitzMode

def perturb_vertex_ids(input, fix_index, num_examples, max_input_size):
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	max_vertex_id = (max_input_size-5) // 3

	# compute the correct next vertex
	graph = {}
	for i in range(len(input)):
		if input[i] == EDGE_PREFIX_TOKEN:
			if int(input[i+1]) not in graph:
				graph[int(input[i+1])] = [int(input[i+2])]
			else:
				graph[int(input[i+1])].append(int(input[i+2]))
	useful_steps = []
	for neighbor in graph[int(input[-1])]:
		stack = [neighbor]
		reachable = []
		while len(stack) != 0:
			current = stack.pop()
			reachable.append(current)
			if current not in graph:
				continue
			for child in graph[current]:
				if child not in reachable:
					stack.append(child)
		if int(input[-3]) in reachable:
			useful_steps.append(neighbor)
	if len(useful_steps) == 0:
		raise Exception('Given input has no path to goal vertex.')
	elif len(useful_steps) != 1:
		raise Exception('Given input has more than one next step to goal vertex.')

	out = torch.empty((num_examples, input.shape[0]), dtype=torch.int64)
	out_labels = torch.empty((num_examples), dtype=torch.int64)
	out[0,:] = input
	edge_indices = [i for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]
	edge_count = len(edge_indices)
	if fix_index != None:
		fixed_edge_index = next(i for i in range(len(edge_indices)) if fix_index >= edge_indices[i] and fix_index < edge_indices[i] + 3)
		fixed_edge = edge_indices[fixed_edge_index]
	padding_size = next(i for i in range(len(input)) if input[i] != PADDING_TOKEN)
	out[:,:padding_size] = PADDING_TOKEN
	out_labels[0] = useful_steps[0]
	for i in range(1, num_examples):
		id_map = list(range(1, max_vertex_id + 1))
		shuffle(id_map)
		id_map = [0] + id_map
		if fix_index != None:
			del edge_indices[fixed_edge_index]
		shuffle(edge_indices)
		if fix_index != None:
			edge_indices.insert(fixed_edge_index, fixed_edge)
		for j in range(len(edge_indices)):
			out[i,padding_size+(3*j):padding_size+(3*j)+3] = torch.LongTensor([EDGE_PREFIX_TOKEN, id_map[input[edge_indices[j]+1]], id_map[input[edge_indices[j]+2]]])
		out[i,padding_size+(3*edge_count):] = torch.LongTensor([(id_map[v] if v <= max_vertex_id else v) for v in input[padding_size+(3*edge_count):]])
		out_labels[i] = id_map[useful_steps[0]]
	return out, out_labels

def run_model(model, input, fix_index, max_input_size, num_perturbations=2**14):
	if len(input) > max_input_size:
		raise ValueError("Input length must be at most 'max_input_size'.")
	device = next(model.parameters()).device
	QUERY_PREFIX_TOKEN = (max_input_size-5) // 3 + 4
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	PATH_PREFIX_TOKEN = (max_input_size-5) // 3 + 1

	model.eval()
	padded_input = [PADDING_TOKEN] * (max_input_size - len(input)) + input
	padded_input = torch.LongTensor(padded_input).to(device)
	print("running model on input")
	print(padded_input)
	perturbed_input, perturbed_output = perturb_vertex_ids(padded_input, fix_index, 1+num_perturbations, max_input_size)
	predictions, _ = model(perturbed_input)
	if len(predictions.shape) == 3:
		predictions = predictions[:, -1, :]
		perturbed_output = perturbed_output.to(device)
		num_correct = torch.sum(torch.argmax(predictions,1) == perturbed_output)
		print("accuracy: {}".format(num_correct / predictions.shape[0]))
	else:
		predictions = predictions[-1, :]
		print("output:")
		print(predictions)
		print("prediction: {}\n".format(torch.argmax(predictions)))

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

def ideal_model(max_input_size, num_layers, hidden_dim, bidirectional, absolute_pos_emb, learnable_token_emb):
	from math import sqrt
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	max_vertex_id = (max_input_size-5) // 3

	ntoken = (max_input_size-5) // 3 + 5
	nhead = 1
	d_hid = ntoken + hidden_dim
	dropout = 0
	toeplitz = ToeplitzMode.NONE
	model = Transformer(
			layers=num_layers,
			pad_idx=PADDING_TOKEN,
			words=ntoken,
			seq_len=max_input_size,
			heads=nhead,
			dims=max(ntoken,d_hid),
			rate=1,
			dropout=dropout,
			bidirectional=bidirectional,
			absolute_pos_emb=absolute_pos_emb,
			learn_token_emb=learnable_token_emb,
			ablate=False,
			toeplitz=toeplitz,
			pre_ln=False)

	total_dims = d_hid + max_input_size
	if model.ln_head:
		model.ln_head.weight = torch.nn.Parameter(torch.ones(model.ln_head.weight.size(0)))
		model.ln_head.bias = torch.nn.Parameter(torch.zeros(model.ln_head.bias.size(0)))
	for i, transformer in enumerate(model.transformers):
		transformer.ln_attn.weight = torch.nn.Parameter(torch.ones(transformer.ln_attn.weight.size(0)))
		transformer.ln_attn.bias = torch.nn.Parameter(torch.zeros(transformer.ln_attn.bias.size(0)))
		if i == 0:
			if transformer.ff:
				transformer.ln_ff.weight = torch.nn.Parameter(torch.ones(transformer.ln_ff.weight.size(0)))
				transformer.ln_ff.bias = torch.nn.Parameter(torch.zeros(transformer.ln_ff.bias.size(0)))
				transformer.ff[0].weight = torch.nn.Parameter(torch.eye(transformer.ff[0].weight.size(0)))
				transformer.ff[0].bias = torch.nn.Parameter(-torch.ones(transformer.ff[0].bias.size(0)))
				transformer.ff[3].weight = torch.nn.Parameter(-torch.eye(transformer.ff[3].weight.size(0)))
				transformer.ff[3].bias = torch.nn.Parameter(torch.zeros(transformer.ff[3].bias.size(0)))
			proj_k = torch.zeros(transformer.attn.proj_k.weight.shape)
			for j in range(0, max_vertex_id + 1):
				proj_k[j, j] = 10000.0
			for j in range(d_hid + 1, d_hid + max_input_size - 1, 3):
				proj_k[d_hid:, j] = -10000.0
				proj_k[-1, j] = 100.0
			for j in range(d_hid, d_hid + max_input_size - 1, 3):
				proj_k[d_hid:, j] = -10000.0
			proj_k[-2,-2] = 10000.0
			transformer.attn.proj_k.weight = torch.nn.Parameter(proj_k)
			transformer.attn.proj_q.weight = torch.nn.Parameter(torch.eye(transformer.attn.proj_q.weight.size(0)))
			transformer.attn.proj_q.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_q.bias.size(0)))
			transformer.attn.proj_k.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_k.bias.size(0)))
			if transformer.attn.proj_v:
				transformer.attn.proj_v.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_v.bias.size(0)))
				proj_v = max_vertex_id*torch.eye(transformer.attn.proj_v.weight.size(0))
				for j in range(0, max_vertex_id + 1):
					proj_v[j, j] = -0.5
				transformer.attn.proj_v.weight = torch.nn.Parameter(proj_v)
			if transformer.attn.linear:
				transformer.attn.linear.bias = torch.nn.Parameter(torch.zeros(transformer.attn.linear.bias.size(0)))
				transformer.attn.linear.weight = torch.nn.Parameter(torch.eye(transformer.attn.linear.weight.size(0)))
		elif i < len(model.transformers) - 1:
			if transformer.ff:
				transformer.ln_ff.weight = torch.nn.Parameter(torch.ones(transformer.ln_ff.weight.size(0)))
				transformer.ln_ff.bias = torch.nn.Parameter(torch.zeros(transformer.ln_ff.bias.size(0)))
				transformer.ff[0].weight = torch.nn.Parameter(torch.eye(transformer.ff[0].weight.size(0)))
				transformer.ff[0].bias = torch.nn.Parameter(-torch.ones(transformer.ff[0].bias.size(0)))
				if i == 1:
					transformer.ff[3].weight = torch.nn.Parameter(-1.01*torch.eye(transformer.ff[3].weight.size(0)))
				else:
					transformer.ff[3].weight = torch.nn.Parameter(-0.99*torch.eye(transformer.ff[3].weight.size(0)))
				transformer.ff[3].bias = torch.nn.Parameter(torch.zeros(transformer.ff[3].bias.size(0)))
			proj_k = torch.zeros(transformer.attn.proj_k.weight.shape)
			for j in range(0, max_vertex_id + 1):
				proj_k[j, j] = -100.0
			for j in range(d_hid + 1, d_hid + max_input_size - 5):
				proj_k[j, j-1] = 100.0
			if i == 1:
				for j in range(d_hid + 1, d_hid + max_input_size - 5, 3):
					proj_k[j, j-1] = 0.0
					proj_k[j, j+1] = 20.0
					proj_k[j,-1] = 0.0
			proj_k[-2,-2] = 1.0
			proj_k[-1,-1] = 1.0
			transformer.attn.proj_k.weight = torch.nn.Parameter(proj_k)
			proj_q_bias = torch.zeros(transformer.attn.proj_q.bias.size(0))
			proj_q_bias[-2] = 100.0
			proj_q = torch.eye(transformer.attn.proj_q.weight.size(0))
			transformer.attn.proj_q.weight = torch.nn.Parameter(proj_q)
			transformer.attn.proj_q.bias = torch.nn.Parameter(proj_q_bias)
			transformer.attn.proj_k.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_k.bias.size(0)))
			if transformer.attn.proj_v:
				transformer.attn.proj_v.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_v.bias.size(0)))
				proj_v = torch.eye(transformer.attn.proj_v.weight.size(0))
				if i == 1:
					proj_v = 2 * proj_v
				transformer.attn.proj_v.weight = torch.nn.Parameter(proj_v)
			if transformer.attn.linear:
				transformer.attn.linear.bias = torch.nn.Parameter(torch.zeros(transformer.attn.linear.bias.size(0)))
				transformer.attn.linear.weight = torch.nn.Parameter(torch.eye(transformer.attn.linear.weight.size(0)))
		else:
			if transformer.ff:
				transformer.ln_ff.weight = torch.nn.Parameter(torch.ones(transformer.ln_ff.weight.size(0)))
				transformer.ln_ff.bias = torch.nn.Parameter(torch.zeros(transformer.ln_ff.bias.size(0)))
				transformer.ff[0].weight = torch.nn.Parameter(torch.eye(transformer.ff[0].weight.size(0)))
				transformer.ff[0].bias = torch.nn.Parameter(-3*torch.ones(transformer.ff[0].bias.size(0)))
				transformer.ff[3].weight = torch.nn.Parameter(-1.1*torch.eye(transformer.ff[3].weight.size(0)))
				transformer.ff[3].bias = torch.nn.Parameter(torch.zeros(transformer.ff[3].bias.size(0)))
			proj_k = torch.zeros(transformer.attn.proj_q.weight.shape)
			proj_k[-1,-3] = 100.0
			transformer.attn.proj_k.weight = torch.nn.Parameter(proj_k)
			transformer.attn.proj_q.weight = torch.nn.Parameter(torch.eye(transformer.attn.proj_q.weight.size(0)))
			transformer.attn.proj_q.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_q.bias.size(0)))
			transformer.attn.proj_k.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_q.bias.size(0)))
			if transformer.attn.proj_v:
				transformer.attn.proj_v.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_v.bias.size(0)))
				transformer.attn.proj_v.weight = torch.nn.Parameter(torch.eye(transformer.attn.proj_v.weight.size(0)))
			if transformer.attn.linear:
				transformer.attn.linear.bias = torch.nn.Parameter(torch.zeros(transformer.attn.linear.bias.size(0)))
				transformer.attn.linear.weight = torch.nn.Parameter(torch.eye(transformer.attn.linear.weight.size(0)))
	return model

def print_graph(input):
	n = len(input)
	QUERY_PREFIX_TOKEN = (n-5) // 3 + 4
	PADDING_TOKEN = (n-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (n-5) // 3 + 2
	PATH_PREFIX_TOKEN = (n-5) // 3 + 1

	forward_edges = []
	for i in range((n - 5) // 3 + 1):
		forward_edges.append([])
	edge_indices = np.nonzero(input == EDGE_PREFIX_TOKEN)[0] + 1
	for i in edge_indices:
		forward_edges[input[i].item()].append(input[i+1].item())

	# find the major paths from the start vertex in the graph
	query_index = np.nonzero(input == QUERY_PREFIX_TOKEN)[0][-1].item()
	start = input[query_index + 1]
	queue = [(start,None,0)]
	distances_from_start = {}
	while len(queue) != 0:
		current, parent, distance = queue.pop()
		if current not in distances_from_start or distance < distances_from_start[current][0]:
			distances_from_start[current] = distance, parent
		else:
			continue
		for child in forward_edges[current]:
			queue.append((child,current,distance+1))
	# find the vertices with distance at least that from the start to the goal vertex
	goal = input[np.nonzero(input == QUERY_PREFIX_TOKEN)[0]+2].item()
	if goal not in distances_from_start:
		import pdb; pdb.set_trace()
	goal_distance,_ = distances_from_start[goal]
	furthest_vertices = []
	for vertex,(distance,_) in distances_from_start.items():
		if distance == goal_distance:
			furthest_vertices.append(vertex)
	# compute the paths to each furthest vertex
	forks = {}
	for v in furthest_vertices:
		current = v
		path = [current]
		while True:
			_,parent = distances_from_start[current]
			if parent == None:
				break
			path.append(parent)
			current = parent
		forks[v] = list(reversed(path))

	printed_edges = []
	goal_fork = forks[goal]
	del forks[goal]
	for i in range(len(goal_fork) - 1):
		if i == 0:
			print(goal_fork[i], end='')
		print(' -> ' + str(goal_fork[i+1]), end='')
		printed_edges.append((goal_fork[i],goal_fork[i+1]))
	for _,fork in forks.items():
		first_vertex = True
		for i in range(len(fork) - 1):
			if (fork[i],fork[i+1]) in printed_edges:
				first_vertex = True
				continue
			if first_vertex:
				print('\n' + str(fork[i]), end='')
				first_vertex = False
			print(' -> ' + str(fork[i+1]), end='')
			printed_edges.append((fork[i],fork[i+1]))
	print('\n', end='')
	for src in range(len(forward_edges)):
		for dst in forward_edges[src]:
			if (src,dst) not in printed_edges:
				print(str(src) + ' -> ' + str(dst))
				printed_edges.append((src,dst))
	start = input[np.nonzero(input == QUERY_PREFIX_TOKEN)[0]+1].item()
	print('Start: ' + str(start) + ', Goal: ' + str(goal))

	path_prefix_index = np.nonzero(input == PATH_PREFIX_TOKEN)[0][-1].item()
	print('Path: ' + str(input[(path_prefix_index+1):]))


def do_evaluate_model(filepath, star_distribution=False, max_backtrack_distance=None):
	model, _, _, _ = torch.load(filepath, map_location=device)

	from os.path import sep
	dirname = filepath.split(sep)[-2]
	suffix = dirname[dirname.index('inputsize')+len('inputsize'):]
	max_input_size = int(suffix[:suffix.index('_')])

	suffix = dirname[dirname.index('seed')+len('seed'):]
	training_seed = int(suffix[:suffix.index('_')])

	suffix = dirname[dirname.index('maxlookahead')+len('maxlookahead'):]
	training_max_lookahead = int(suffix[:suffix.index('_')])
	max_lookahead = ((max_input_size - 5) // 3 - 1) // 2

	is_dfs = 'dfs' in dirname
	if max_backtrack_distance == None:
		max_backtrack_distance = (max_input_size - 4) // 4 - 1

	if not hasattr(model, 'looped'):
		model.looped = False
	for transformer in model.transformers:
		if not hasattr(transformer, 'pre_ln'):
			transformer.pre_ln = True

	seed_generator = Random(training_seed)
	seed_values = []

	def get_seed(index):
		if index < len(seed_values):
			return seed_values[index]
		while index >= len(seed_values):
			seed_values.append(seed_generator.randrange(2 ** 32))
		return seed_values[index]

	NUM_TEST_SAMPLES = 1000
	reserved_inputs = set()
	test_accuracies = []
	if is_dfs:
		for backtrack_distance in [-1] + list(range(0, max_backtrack_distance + 1)):
			generator.set_seed(get_seed(1))
			inputs,outputs,labels,_ = generator.generate_dfs_training_set(max_input_size, NUM_TEST_SAMPLES, reserved_inputs, backtrack_distance, False, False, True)
			test_acc,test_loss,predictions = evaluate_model(model, inputs, outputs)
			confidence_int = binomial_confidence_int(test_acc, NUM_TEST_SAMPLES)
			'''print("Mistaken inputs:")
			predictions = np.array(predictions.cpu())
			incorrect_indices,_ = np.nonzero(np.take_along_axis(outputs, predictions[:,None], axis=1) == 0)
			np.set_printoptions(threshold=10_000)
			for incorrect_index in incorrect_indices:
				print_graph(inputs[incorrect_index, :])
				print("Expected answer: {}, predicted answer: {} (label: {})\n".format(np.nonzero(outputs[incorrect_index])[0], predictions[incorrect_index], labels[incorrect_index]))'''
			import pdb; pdb.set_trace()
			print("Test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, confidence_int, test_loss))
			test_accuracies.append((test_acc, confidence_int, test_loss))
	elif star_distribution:
		for spoke_length in range(1, max_lookahead + 1):
			max_spoke_count = ((max_input_size - 5) // 3 - 1) // spoke_length
			spoke_accuracies = []
			for num_spokes in range(1, max_spoke_count + 1):
				print('spoke_length: {}, num_spokes: {}'.format(spoke_length, num_spokes))
				inputs,outputs = generate_star_graph_data(max_input_size, num_spokes, spoke_length, num_samples=NUM_TEST_SAMPLES)
				test_acc,test_loss,predictions = evaluate_model(model, inputs, outputs)
				confidence_int = binomial_confidence_int(test_acc, NUM_TEST_SAMPLES)
				print("Test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, confidence_int, test_loss))
				spoke_accuracies.append((test_acc, confidence_int, test_loss))
			test_accuracies.append(spoke_accuracies)
	else:
		for lookahead in [None] + list(range(1, max_lookahead + 1)):
			seed(training_seed)
			torch.manual_seed(training_seed)
			np.random.seed(training_seed)

			inputs,outputs = generate_eval_data(max_input_size, min_path_length=2, distance_from_start=1, distance_from_end=-1, lookahead_steps=lookahead, num_paths_at_fork=None, num_samples=NUM_TEST_SAMPLES)

			print("Evaluating model...")
			test_acc,test_loss,predictions = evaluate_model(model, inputs, outputs)
			'''print("Mistaken inputs:")
			predictions = np.array(predictions.cpu())
			incorrect_indices = np.nonzero(predictions != outputs)[0]
			np.set_printoptions(threshold=10_000)
			for incorrect_index in incorrect_indices:
				print_graph(inputs[incorrect_index, :])
				print("Expected answer: {}, predicted answer: {}\n".format(outputs[incorrect_index], predictions[incorrect_index]))
			import pdb; pdb.set_trace()'''
			confidence_int = binomial_confidence_int(test_acc, NUM_TEST_SAMPLES)
			print("Test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, confidence_int, test_loss))
			test_accuracies.append((test_acc, confidence_int, test_loss))

	return test_accuracies


if __name__ == "__main__":
	from sys import argv
	import os
	filepath = argv[1]
	if os.path.isfile(filepath):
		test_accuracies = do_evaluate_model(filepath)
		print("\nTest accuracies:")
		print(["%.2f±%.2f" % (acc,conf_int) for acc,conf_int,_ in test_accuracies])

	elif os.path.isdir(argv[1]):
		import matplotlib
		import matplotlib.pyplot as plt
		from matplotlib import rcParams
		plt.style.use('ggplot')
		colors = []
		for c in rcParams["axes.prop_cycle"]:
			colors.append(c['color'])

		# count how many seeds there are
		checkpoint_dirs = [argv[1]]
		import glob
		for name in glob.glob(argv[1] + '/seed*'):
			checkpoint_dirs.append(name)
		print(checkpoint_dirs)

		first_plot_data = []
		second_plot_data = []
		third_plot_data = []
		for checkpoint_dir in checkpoint_dirs:
			epoch = 0
			epochs = []
			first_columns = []
			second_columns = []
			third_columns = []
			while True:
				filename = checkpoint_dir + '/epoch' + str(epoch) + '.pt'
				if not os.path.isfile(filename):
					break

				model, _, _, _ = torch.load(filename, map_location=device)

				params = {k:v for k,v in model.named_parameters()}
				# compute the first A matrix
				P_k = params['transformers.0.attn.proj_k.weight']
				P_q = params['transformers.0.attn.proj_q.weight']
				U_k = torch.cat((P_k,params['transformers.0.attn.proj_k.bias'].unsqueeze(1)),1)
				U_q = torch.cat((P_q,params['transformers.0.attn.proj_k.bias'].unsqueeze(1)),1)
				A = torch.matmul(U_q.transpose(-2,-1),U_k)

				# compute how many columns of the top-left submatrix of A is learned
				first_columns_learned = 0
				max_input_size = 24
				for i in range(1, max_input_size-4):
					if all(A[1:i,i] > A[i,i]/5) and A[i,i] < -1.0 and all(A[(i+1):(max_input_size-4),i] > A[i,i]/5):
						first_columns_learned += 1

				# compute the second A matrix
				P_k = params['transformers.1.attn.proj_k.weight']
				P_q = params['transformers.1.attn.proj_q.weight']
				U_k = torch.cat((P_k,params['transformers.1.attn.proj_k.bias'].unsqueeze(1)),1)
				U_q = torch.cat((P_q,params['transformers.1.attn.proj_k.bias'].unsqueeze(1)),1)
				A = torch.matmul(U_q.transpose(-2,-1),U_k)

				# compute how many columns of the bottom-right submatrix of A is learned
				second_columns_learned = 0
				for i in range(max_input_size-3):
					if A[48+i,48+i+1] < -1.0:
						second_columns_learned += 1

				# compute how many rows of the top-right submatrix of the last linear layer is learned
				third_columns_learned = 0
				W = params['transformers.1.attn.linear.weight']
				for i in range(1,max_input_size-4):
					if W[i,i] > 1.0 and all(W[i,:i] < 1.0) and all(W[i,(i+1):] < 1.0):
						third_columns_learned += 1

				epochs.append(epoch)
				first_columns.append(first_columns_learned)
				second_columns.append(second_columns_learned)
				third_columns.append(third_columns_learned)
				epoch += 5

			first_plot_data.append((epochs, first_columns))
			second_plot_data.append((epochs, second_columns))
			third_plot_data.append((epochs, third_columns))

		dir_name = argv[1]
		if dir_name[-1] == '/':
			dir_name = dir_name[:-1]
		max_epochs = max([x[-1] for x, _ in first_plot_data])

		fig = plt.gcf()
		ax = plt.gca()
		fig.set_size_inches(6, 2, forward=True)
		for x, y in first_plot_data:
			ax.plot(x, y)
		plt.xlim(0, max_epochs)
		plt.ylim(0, 19)
		plt.xlabel('epochs')
		plt.ylabel('number of \n correct columns')
		plt.grid(True)
		fig.savefig(dir_name + '_A1.png', dpi=256)
		plt.clf()

		ax = plt.gca()
		fig.set_size_inches(6, 2, forward=True)
		for x, y in second_plot_data:
			ax.plot(x, y)
		plt.xlim(0, max_epochs)
		plt.ylim(0, 18)
		plt.xlabel('epochs')
		plt.ylabel('number of \n correct columns')
		plt.grid(True)
		fig.savefig(dir_name + '_A2.png', dpi=256)
		plt.clf()

		ax = plt.gca()
		fig.set_size_inches(6, 2, forward=True)
		for x, y in third_plot_data:
			ax.plot(x, y)
		plt.xlim(0, max_epochs)
		plt.ylim(0, 19)
		plt.xlabel('epochs')
		plt.ylabel('number of \n correct rows')
		plt.grid(True)
		fig.savefig(dir_name + '_W.png', dpi=256)
		plt.clf()
