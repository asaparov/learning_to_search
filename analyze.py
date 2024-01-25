from random import seed, shuffle, randrange, Random
import numpy as np
import torch
from train import generate_eval_data, evaluate_model, binomial_confidence_int
import generator
from gpt2 import Transformer, ToeplitzMode

def perturb_vertex_ids(input, fix_index, num_examples, max_input_size):
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	max_vertex_id = (max_input_size-5) // 3

	out = torch.empty((num_examples, input.shape[0]), dtype=torch.int64)
	out_labels = torch.empty((num_examples), dtype=torch.int64)
	out[0,:] = input
	edge_indices = [i for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]
	edge_count = len(edge_indices)
	fixed_edge_index = next(i for i in range(len(edge_indices)) if fix_index >= edge_indices[i] and fix_index < edge_indices[i] + 3)
	fixed_edge = edge_indices[fixed_edge_index]
	padding_size = next(i for i in range(len(input)) if input[i] != PADDING_TOKEN)
	out[:,:padding_size] = PADDING_TOKEN
	out_labels[0] = out[0,fix_index+1]
	for i in range(1, num_examples):
		id_map = list(range(1, max_vertex_id + 1))
		shuffle(id_map)
		id_map = [0] + id_map
		del edge_indices[fixed_edge_index]
		shuffle(edge_indices)
		edge_indices.insert(fixed_edge_index, fixed_edge)
		for j in range(len(edge_indices)):
			out[i,padding_size+(3*j):padding_size+(3*j)+3] = torch.LongTensor([EDGE_PREFIX_TOKEN, id_map[input[edge_indices[j]+1]], id_map[input[edge_indices[j]+2]]])
		out[i,padding_size+(3*edge_count):] = torch.LongTensor([(id_map[v] if v <= max_vertex_id else v) for v in input[padding_size+(3*edge_count):]])
		out_labels[i] = out[i,fix_index+1]
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
	import pdb; pdb.set_trace()
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
		transformer.attn.proj_q.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_q.bias.size(0)))
		transformer.attn.proj_k.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_k.bias.size(0)))
		transformer.attn.proj_q.weight = torch.nn.Parameter(torch.eye(transformer.attn.proj_q.weight.size(0)))
		if transformer.attn.linear:
			transformer.attn.linear.bias = torch.nn.Parameter(torch.zeros(transformer.attn.linear.bias.size(0)))
			transformer.attn.linear.weight = torch.nn.Parameter(torch.eye(transformer.attn.linear.weight.size(0)))
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
			transformer.attn.proj_k.weight = torch.nn.Parameter(proj_k)
			if transformer.attn.proj_v:
				transformer.attn.proj_v.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_v.bias.size(0)))
				transformer.attn.proj_v.weight = torch.nn.Parameter(max_vertex_id*torch.eye(transformer.attn.proj_v.weight.size(0)))
		else:
			if transformer.ff:
				transformer.ln_ff.weight = torch.nn.Parameter(torch.ones(transformer.ln_ff.weight.size(0)))
				transformer.ln_ff.bias = torch.nn.Parameter(torch.zeros(transformer.ln_ff.bias.size(0)))
				transformer.ff[0].weight = torch.nn.Parameter(torch.eye(transformer.ff[0].weight.size(0)))
				transformer.ff[0].bias = torch.nn.Parameter(-torch.ones(transformer.ff[0].bias.size(0)))
				transformer.ff[3].weight = torch.nn.Parameter(-torch.eye(transformer.ff[3].weight.size(0)))
				transformer.ff[3].bias = torch.nn.Parameter(torch.zeros(transformer.ff[3].bias.size(0)))
			proj_k = torch.zeros(transformer.attn.proj_k.weight.shape)
			for j in range(0, max_vertex_id + 1):
				proj_k[j, j] = -100.0
			for j in range(d_hid + 1, d_hid + max_input_size - 5):
				proj_k[j, j-1] = 100.0
			proj_k[-1,-1] = 1000.0
			transformer.attn.proj_k.weight = torch.nn.Parameter(proj_k)
			if transformer.attn.proj_v:
				transformer.attn.proj_v.bias = torch.nn.Parameter(torch.zeros(transformer.attn.proj_v.bias.size(0)))
				transformer.attn.proj_v.weight = torch.nn.Parameter(torch.eye(transformer.attn.proj_v.weight.size(0)))
	return model

from sys import argv
import os
filepath = argv[1]
if os.path.isfile(filepath):
	model, _, _, _ = torch.load(filepath, map_location=device)

	suffix = filepath[filepath.index('inputsize')+len('inputsize'):]
	max_input_size = int(suffix[:suffix.index('_')])

	suffix = filepath[filepath.index('seed')+len('seed'):]
	training_seed = int(suffix[:suffix.index('_')])

	suffix = filepath[filepath.index('maxlookahead')+len('maxlookahead'):]
	training_max_lookahead = int(suffix[:suffix.index('_')])

	model = ideal_model(max_input_size=max_input_size, num_layers=6, hidden_dim=16, bidirectional=True, absolute_pos_emb=True, learnable_token_emb=False)

	#run_model(model, [22, 21,  5, 19, 21, 11,  5, 21, 10,  3, 21,  4, 10, 21,  9,  4, 21,  9, 11, 23,  9,  3, 20,  9], max_input_size=24)
	#run_model(model, [22, 22, 22, 22, 22, 22, 22, 21, 1,  2, 21,  1,  4, 21,  2,  3, 21, 4,  5, 23,  1,  3, 20, 1], max_input_size=24)
	#run_model(model, [46, 45,  3, 19, 45, 18, 39, 45, 36, 15, 45, 24, 42, 45, 37,  3, 45, 37, 36, 45, 23, 32, 45,  8, 24, 45, 19, 30, 45, 15, 23, 45, 39, 40, 45, 40, 34, 45, 30, 18, 45, 32,  8, 47, 37, 34, 44, 37], max_input_size=48)
	run_model(model, [43, 15, 34, 43, 30,  9, 43, 14, 22, 43,  8, 13, 43,  8,  2, 43, 26,  1, 43,  1, 14, 43, 36,  7, 43, 22,  4, 43, 22,  2, 43, 34, 26, 43, 34, 25, 43, 28, 30, 43, 16, 3, 43, 16, 32, 43, 13, 33, 43, 12, 15, 43, 25, 21, 43,  9, 36, 43, 3, 12, 43, 32,  8, 43, 33, 28, 45, 16,  4, 42, 16], fix_index=98, max_input_size=max_input_size, num_perturbations=0)

	seed(training_seed)
	torch.manual_seed(training_seed)
	np.random.seed(training_seed)

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
	print("Generating eval data...")
	inputs,outputs = generate_eval_data(max_input_size, min_path_length=2, distance_from_start=1, distance_from_end=-1, lookahead_steps=9, num_paths_at_fork=None, num_samples=NUM_TEST_SAMPLES)
	#generator.set_seed(get_seed(1))
	#inputs, outputs, _, _ = generator.generate_training_set(max_input_size, NUM_TEST_SAMPLES, training_max_lookahead, reserved_inputs, 1, False)
	print("Evaluating model...")
	test_acc,test_loss,predictions = evaluate_model(model, inputs, outputs)
	import pdb; pdb.set_trace()
	print("Mistaken inputs:")
	predictions = np.array(predictions.cpu())
	incorrect_indices = np.nonzero(predictions != outputs)[0]
	np.set_printoptions(threshold=10_000)
	for incorrect_index in incorrect_indices:
		print(inputs[incorrect_index, :])
		print("Expected answer: {}, predicted answer: {}\n".format(outputs[incorrect_index], predictions[incorrect_index]))
	print("Test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, binomial_confidence_int(test_acc, 1000), test_loss))

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
