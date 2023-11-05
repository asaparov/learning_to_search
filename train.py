from random import sample, randrange, choice, shuffle, seed, uniform
from os import listdir, makedirs
from os.path import isfile, isdir
from sys import stdout
import pickle
import numpy as np
import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from gpt2 import Transformer
from Sophia import SophiaG

RESERVED_INDICES = (0,)

class Node(object):
	def __init__(self, id):
		self.id = id
		self.children = []
		self.parents = []

	def __eq__(self, other):
		return self.id == other.id

	def __hash__(self):
		return hash(self.id)

	def __str__(self):
		return 'n(' + str(self.id) + ')'

	def __repr__(self):
		return 'n(' + str(self.id) + ')'

# computes the number of lookahead steps to find the answer
def lookahead_depth(vertex, next_vertex, goal):
	frontier = [(v,v) for v in vertex.children]
	visited = {v for v in vertex.children}
	lookahead = 0
	while len(frontier) != 0:
		if all(branch == next_vertex for _, branch in frontier):
			return lookahead
		lookahead += 1
		new_frontier = []
		for v, branch in frontier:
			if v == goal:
				return lookahead
			for child in v.children:
				if child not in visited:
					new_frontier.append((child, branch))
					visited.add(child)
				elif branch == next_vertex:
					for i in range(len(new_frontier)):
						if new_frontier[i][0] == child:
							new_frontier[i] = child, branch
		frontier = new_frontier
	return lookahead

def generate_graph(num_vertices, max_num_parents, max_vertex_id):
	vertices = []
	for i in range(num_vertices):
		vertices.append(Node(i))

	# sample a random DAG
	num_sources = 1 #choice([1, 2])
	for i in range(num_sources, num_vertices):
		# sample the number of parent vertices
		if choice([True, False]):
			num_parents = 1
		else:
			num_parents = randrange(1, max_num_parents)
		num_parents = min(num_parents, i)

		for parent_id in sample(range(i), num_parents):
			vertices[parent_id].children.append(vertices[i])
			vertices[i].parents.append(vertices[parent_id])

	# remove any correlation between graph topology and vertex IDs by shuffling the vertices
	new_indices = list(range(max_vertex_id + 1))
	shuffle(new_indices)
	src_index = 0
	for i in range(len(vertices)):
		if new_indices[src_index] in RESERVED_INDICES:
			src_index += 1
		vertices[i].id = new_indices[src_index]
		src_index += 1
	shuffle(vertices)
	return vertices

def generate_graph_with_lookahead(num_vertices, max_num_parents, max_vertex_id, lookahead):
	num_vertices = max(num_vertices, 1 + 2 * lookahead)

	vertices = []
	for i in range(num_vertices):
		vertices.append(Node(i))

	vertices[1].parents.append(vertices[0])
	vertices[0].children.append(vertices[1])
	for i in range(1, lookahead):
		vertices[1 + i].parents.append(vertices[i])
		vertices[i].children.append(vertices[1 + i])
	index = 1 + lookahead
	if lookahead != 0:
		vertices[1 + lookahead].parents.append(vertices[0])
		vertices[0].children.append(vertices[1 + lookahead])
		other_branch_length = lookahead + randrange(min(2, num_vertices - index - lookahead + 1))
		for i in range(1, other_branch_length):
			vertices[1 + lookahead + i].parents.append(vertices[lookahead + i])
			vertices[lookahead + i].children.append(vertices[1 + lookahead + i])
		index += other_branch_length

	num_prefix_vertices = randrange(num_vertices - index + 1)
	prev_vertex = vertices[0]
	for i in range(num_prefix_vertices):
		vertices[index].children.append(prev_vertex)
		prev_vertex.parents.append(vertices[index])
		prev_vertex = vertices[index]
		index += 1

	if uniform(0, 1) < 0.75:
		start = vertices[index - 1]
	else:
		start = vertices[choice([0] + list(range(index - num_prefix_vertices, index)))]
	end = vertices[max(1, lookahead)]

	for i in range(index, num_vertices):
		# sample the number of parent vertices
		if choice([True, False]):
			num_parents = 1
		else:
			num_parents = randrange(1, max_num_parents)
		num_parents = min(num_parents, i)

		for parent_id in sample(range(i), num_parents):
			vertices[parent_id].children.append(vertices[i])
			vertices[i].parents.append(vertices[parent_id])

	# remove any correlation between graph topology and vertex IDs by shuffling the vertices
	new_indices = list(range(max_vertex_id + 1))
	shuffle(new_indices)
	src_index = 0
	for i in range(len(vertices)):
		if new_indices[src_index] in RESERVED_INDICES:
			src_index += 1
		vertices[i].id = new_indices[src_index]
		src_index += 1
	shuffle(vertices)
	return vertices, start, end

def generate_example(num_vertices, max_num_parents, max_vertex_id, get_shortest_paths=True, lookahead=None):
	if lookahead == None:
		graph = generate_graph(num_vertices, max_num_parents, max_vertex_id)

		# randomly select two vertices
		start = graph[randrange(len(graph) - 1)]
		while True:
			end = graph[randrange(len(graph) - 1)]
			if end != start:
				break
	else:
		graph, start, end = generate_graph_with_lookahead(num_vertices, max_num_parents, max_vertex_id, lookahead)
		if graph == None:
			return None, None, None, None

	# find the shortest paths from `start` to `end`
	queue = [(start, 0)]
	reverse_pointers = {}
	while len(queue) != 0:
		(current, distance) = queue.pop()

		for child in current.children:
			if child not in reverse_pointers:
				reverse_pointers[child] = {current:(distance+1)}
			elif current not in reverse_pointers[child]:
				reverse_pointers[child][current] = distance + 1
			elif reverse_pointers[child][current] > distance + 1:
				reverse_pointers[child][current] = distance + 1
			else:
				continue
			found_child = False
			for j in range(len(queue)):
				if queue[j][0] == child:
					queue[j][1] = min(queue[j][1], distance + 1)
					found_child = True
					break
			if not found_child:
				queue.append((child, distance + 1))

	if end not in reverse_pointers:
		return None, None, None, None

	forward_pointers = {}
	queue = [end]
	while len(queue) != 0:
		current = queue.pop()
		if current == start:
			continue
		if get_shortest_paths:
			min_distance = min(reverse_pointers[current].values())
			prev_nodes = [n for n, d in reverse_pointers[current].items() if d == min_distance]
		else:
			prev_nodes = [n for n, _ in reverse_pointers[current].items()]
		for prev in prev_nodes:
			if prev not in forward_pointers:
				forward_pointers[prev] = [current]
			else:
				forward_pointers[prev].append(current)
		queue.extend(prev_nodes)

	# construct the shortest paths from the forward pointers
	paths = []
	queue = [[start]]
	while len(queue) != 0:
		partial_path = queue.pop()
		if partial_path[-1] == end:
			paths.append(partial_path)
			if len(paths) > 64:
				return None, None, None, None
			continue
		for next in forward_pointers[partial_path[-1]]:
			queue.append(partial_path + [next])

	return (graph, start, end, paths)

def binomial_confidence_int(p, n):
	return 1.96 * np.sqrt(p * (1.0 - p) / n)

def evaluate_model(model, max_input_size, min_path_length=2, distance_from_start=None, distance_from_end=None, lookahead_steps=None, print_progress=False):
	device = next(model.parameters()).device
	QUERY_PREFIX_TOKEN = max_input_size - 1
	PADDING_TOKEN = max_input_size - 2
	EDGE_PREFIX_TOKEN = max_input_size - 3
	PATH_PREFIX_TOKEN = max_input_size - 4

	min_vertices = max(3, min_path_length)

	model.eval()
	total_predictions = 0
	best_predictions = 0
	useful_predictions = 0
	valid_predictions = 0
	best_edge_counts = []
	useful_edge_counts = []
	valid_edge_counts = []
	graph_size_counts = []
	num_samples = 1000
	#distances_from_end = [0] * max_input_size
	#MAX_FREQ_PER_BUCKET = 0.30
	while total_predictions < num_samples:
		while True:
			num_vertices = randrange(min_vertices, (max_input_size - 5) // 3)
			if lookahead_steps != None:
				num_vertices = max(num_vertices, min(lookahead_steps * 2 + 1 + randrange(0, 3), (max_input_size - 5) // 3))
			g, start, end, paths = generate_example(num_vertices, 4, (max_input_size - 5) // 3, get_shortest_paths=False, lookahead=lookahead_steps)
			if paths != None and min([len(path) for path in paths]) > min_path_length:
				break

		prefix = []
		for vertex in g:
			for child in vertex.children:
				prefix.extend([EDGE_PREFIX_TOKEN, vertex.id, child.id])
		prefix.extend([QUERY_PREFIX_TOKEN, start.id, end.id, PATH_PREFIX_TOKEN])

		aggregated_paths = []
		for path in paths:
			if len(path) == 1:
				continue
			example = list(prefix)
			for j in range(1, len(path)):
				example.append(path[j - 1].id)
				if len(example) > max_input_size:
					continue
				shortest_path_length = min([len(p) for p in paths if path[:j] == p[:j]])
				if distance_from_start != None and j != distance_from_start:
					continue
				if distance_from_end != None and shortest_path_length - j != distance_from_end:
					continue
				if distance_from_start == None and distance_from_end == None:
					# impose the same rejection sampling constraints as the training data distribution
					#num_predictions = total_predictions + len(aggregated_paths)
					#if num_predictions != 0 and distances_from_end[len(path) - j] / num_predictions >= MAX_FREQ_PER_BUCKET:
					#	continue
					pass
				index = 0
				while index < len(aggregated_paths):
					if aggregated_paths[index][0] == example:
						break
					index += 1
				if index == len(aggregated_paths):
					aggregated_paths.append((example[:], [], []))
				if len(path) == shortest_path_length:
					if path[j].id not in aggregated_paths[index][1]:
						aggregated_paths[index][1].append(path[j].id)
						#if len(aggregated_paths[index][1]) == 1:
						#	distances_from_end[len(path) - j] += 1
				if path[j].id not in aggregated_paths[index][2]:
					aggregated_paths[index][2].append(path[j].id)

		shuffle(aggregated_paths)
		for partial_path, best_next_steps, useful_next_steps in aggregated_paths:
			if len(best_next_steps) == 0:
				continue
			# TODO: this is for testing; why is the model not learning to generalize to this distribution? is it learning some spurious correlation?
			#new_current_vertex = choice([v for v in g if len(v.children) != 0])
			#partial_path[-1] = new_current_vertex.id
			current_vertex = next(v for v in g if v.id == partial_path[-1])
			valid_next_steps = [child.id for child in current_vertex.children]

			best_next_vertex = next(v for v in g if v.id == best_next_steps[0])
			if lookahead_steps != None and lookahead_depth(current_vertex, best_next_vertex, end) != lookahead_steps:
				continue
			#if lookahead_depth(current_vertex, best_next_vertex, end) != 1:
			#	continue

			# replace once vertex with the reserved vertex
			#best_vertex_id = choice(best_next_steps)
			#best_next_steps = [(b if b != best_vertex_id else RESERVED_INDICES[0]) for b in best_next_steps]
			#useful_next_steps = [(b if b != best_vertex_id else RESERVED_INDICES[0]) for b in useful_next_steps]
			#valid_next_steps = [(b if b != best_vertex_id else RESERVED_INDICES[0]) for b in valid_next_steps]
			#partial_path = [(b if b != best_vertex_id else RESERVED_INDICES[0]) for b in partial_path]

			input = [PADDING_TOKEN] * (max_input_size - len(partial_path)) + partial_path
			input = LongTensor(input).to(device)
			#import pdb; pdb.set_trace()
			#if len(best_next_steps) < len(useful_next_steps) and next(partial_path[i+2] for i in range(len(partial_path) - 2) if partial_path[i] == QUERY_PREFIX_TOKEN) not in valid_next_steps:
			#	import pdb; pdb.set_trace()
			if len(valid_next_steps) == 1 or len(best_next_steps) != 1:
				continue
			predictions, _ = model(input)
			predictions = predictions[-1, :]
			best_edge_counts.append(len(best_next_steps))
			useful_edge_counts.append(len(useful_next_steps))
			valid_edge_counts.append(len(current_vertex.children))
			graph_size_counts.append(len(g))
			if torch.argmax(predictions) in best_next_steps:
				best_predictions += 1
			if torch.argmax(predictions) in useful_next_steps:
				useful_predictions += 1
			if torch.argmax(predictions) in valid_next_steps:
				valid_predictions += 1
			else:
				pass
			total_predictions += 1

		if (print_progress and total_predictions != 0) or total_predictions >= num_samples:
			best_acc = best_predictions / total_predictions
			useful_acc = useful_predictions / total_predictions
			valid_acc = valid_predictions / total_predictions
			print("test accuracy: n = %d, best = %.2f±%.2f, useful = %.2f±%.2f, valid = %.2f±%.2f" % (total_predictions, best_acc, binomial_confidence_int(best_acc, total_predictions), useful_acc, binomial_confidence_int(best_acc, total_predictions), valid_acc, binomial_confidence_int(valid_acc, total_predictions)))

	print("average number of best edges = %.2f, useful edges = %.2f, valid edges = %.2f, vertices = %.2f" % (np.mean(best_edge_counts), np.mean(useful_edge_counts), np.mean(valid_edge_counts), np.mean(graph_size_counts)))
	stdout.flush()

class DummyDataset(Dataset):
	def __init__(self, inputs, outputs, device, x_type=LongTensor, y_type=LongTensor):
		self.x_data = x_type(inputs).to(device)
		self.y_data = y_type(outputs).to(device)

	def __len__(self):
		return len(self.x_data)

	def __getitem__(self, idx):
		src_seq = self.x_data[idx]
		tgt_seq = self.y_data[idx]
		return (src_seq, tgt_seq)

def unique(x):
	y = []
	for e in x:
		if e not in y:
			y.append(e)
	return y

def train(max_input_size, dataset_size, max_lookahead, seed_value, nlayers, hidden_dim, bidirectional, absolute_pos_emb, learnable_token_emb, toeplitz_attn):
	seed(seed_value)
	torch.manual_seed(seed_value)
	np.random.seed(seed_value)

	QUERY_PREFIX_TOKEN = max_input_size - 1
	PADDING_TOKEN = max_input_size - 2
	EDGE_PREFIX_TOKEN = max_input_size - 3
	PATH_PREFIX_TOKEN = max_input_size - 4
	num_generated = 0
	inputs = np.empty((dataset_size, max_input_size), dtype=np.int64)
	outputs = np.empty(dataset_size, dtype=np.int64)
	lookahead_step_histogram = [0] * max_input_size
	path_length_histogram = [0] * max_input_size
	MAX_FREQS_PER_BUCKET = [0] * max_input_size
	for i in range(max_lookahead + 1):
		MAX_FREQS_PER_BUCKET[i] = 1/(max_lookahead+1)
	MAX_FREQS_PER_BUCKET[max_lookahead] += 0.05
	#path_lengths = [0] * max_input_size
	#total_path_lengths = 0
	#MAX_PATH_LENGTH_BUCKET = 0.4
	valid_outputs = []
	train_filename = 'useful_path_results/train{}_inputsize{}_maxlookahead{}_seed{}.pkl'.format(dataset_size, max_input_size, max_lookahead, seed_value)
	if isfile(train_filename):
		# check if we've already generated the training data
		print("Loading training data from '{}'...".format(train_filename))
		stdout.flush()
		with open(train_filename, 'rb') as f:
			inputs, outputs, valid_outputs = pickle.load(f)
	else:
		# we haven't generated the training data yet, so generate it here
		while num_generated < dataset_size:
			while True:
				lookahead = choice([i for i in range(max_lookahead + 1) if num_generated == 0 or lookahead_step_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i]])
				num_vertices = randrange(6, (max_input_size-5) // 3)
				num_vertices = max(num_vertices, min(lookahead * 2 + 1 + randrange(0, 3), (max_input_size-5) // 3))
				g, start, end, paths = generate_example(num_vertices, 4, (max_input_size - 5) // 3, lookahead=lookahead)
				if paths != None and min([len(path) for path in paths]) > 1:
					break

			edges = []
			for vertex in g:
				for child in vertex.children:
					edges.append((vertex.id, child.id))
			shuffle(edges)

			prefix = []
			for source, target in edges:
				prefix.extend([EDGE_PREFIX_TOKEN, source, target])
			prefix.extend([QUERY_PREFIX_TOKEN, start.id, end.id, PATH_PREFIX_TOKEN])

			for path in paths:
				if len(path) == 1:
					continue
				#if total_path_lengths != 0 and path_lengths[len(path)] / total_path_lengths >= MAX_PATH_LENGTH_BUCKET:
				#	continue
				#path_lengths[len(path)] += 1
				#total_path_lengths += 1
				example = list(prefix)
				for j in range(1, len(path)):
					example.append(path[j - 1].id)
					if len(example) > max_input_size:
						#print('WARNING: Generated example is too long.')
						continue

					def has_path(start, end):
						stack = [start]
						visited = set()
						while len(stack) != 0:
							v = stack.pop()
							if v == end:
								return True
							for child in v.children:
								if child not in visited:
									visited.add(child)
									stack.append(child)
						return False

					lookahead_steps = lookahead_depth(path[j-1], path[j], end)
					useful_steps = [v for v in path[j-1].children if has_path(v, end)]

					if num_generated != 0 and lookahead_step_histogram[lookahead_steps] / num_generated >= MAX_FREQS_PER_BUCKET[lookahead_steps]:
						continue
					lookahead_step_histogram[lookahead_steps] += 1
					path_length_histogram[j] += 1

					inputs[num_generated,(max_input_size-len(example)):] = example
					inputs[num_generated,:(max_input_size-len(example))] = PADDING_TOKEN
					outputs[num_generated] = choice(useful_steps).id #path[j].id #choice([v.id for v in path[j - 1].children])
					valid_outputs.append([v.id for v in useful_steps]) #unique([other_path[j].id for other_path in paths if other_path[:j-1] == path[:j-1]])) #[v.id for v in path[j - 1].children])
					num_generated += 1
					if num_generated == dataset_size:
						break
				if num_generated == dataset_size:
					break

			if num_generated % 1000 == 0 or num_generated >= dataset_size:
				print("{} examples generated.".format(num_generated))
				#print("Path length histogram:")
				#print(', '.join(['%d:%.2f' % (i, path_lengths[i] / total_path_lengths + 1e-9) for i in range(len(path_lengths)) if path_lengths[i] != 0]))
				print("Lookahead steps histogram:")
				print(', '.join(['%d:%.2f' % (i, lookahead_step_histogram[i] / num_generated + 1e-9) for i in range(len(lookahead_step_histogram)) if lookahead_step_histogram[i] != 0]))
				print("Path length histogram:")
				print(', '.join(['%d:%.2f' % (i, path_length_histogram[i] / num_generated + 1e-9) for i in range(len(path_length_histogram)) if path_length_histogram[i] != 0]))
				stdout.flush()

		# save the generated training data to file
		with open(train_filename, 'wb') as f:
			pickle.dump((inputs, outputs, valid_outputs), f)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		#from sys import exit
		#exit(-1)
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	train_data = DummyDataset(inputs, outputs, device)
	train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

	# compute the checkpoint filenames and try to resume from the last one
	filename = 'useful_path_results/checkpoints_{}layer_inputsize{}_maxlookahead{}_seed{}_train{}'.format(nlayers, max_input_size, max_lookahead, seed_value, dataset_size)
	if bidirectional:
		filename += '_nomask'
	if not absolute_pos_emb:
		filename += '_noAPE'
	if learnable_token_emb:
		filename += '_learntokemb'
	if toeplitz_attn:
		filename += '_toeplitz'
	if isdir(filename):
		existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
	else:
		existing_epochs = []
		makedirs(filename)

	ntoken = max_input_size
	nhead = 1
	d_hid = max_input_size + hidden_dim
	dropout = 0
	if len(existing_epochs) == 0:
		model = Transformer(
				layers=nlayers,
				pad_idx=PADDING_TOKEN,
				words=max_input_size,
				seq_len=max_input_size,
				heads=nhead,
				dims=max(max_input_size,d_hid),
				rate=1,
				dropout=dropout,
				bidirectional=bidirectional,
				absolute_pos_emb=absolute_pos_emb,
				learn_token_emb=learnable_token_emb,
				diagonal_attn=toeplitz_attn)
		epoch = 0
		model.to(device)
	else:
		last_epoch = max(existing_epochs)
		epoch = last_epoch + 1
		model = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device)

	loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
	optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=1.0e-4)

	log_interval = 1
	eval_interval = 1
	save_interval = 1

	while True:
		epoch_loss = 0.0
		for idx, batch in enumerate(train_loader):
			model.train()
			optimizer.zero_grad()

			input, output = batch

			logits = model(input)
			regularization = None
			for p in model.parameters():
				if regularization == None:
					regularization = p.norm(1)
				else:
					regularization = regularization + p.norm(1)
			loss_val = loss_func(logits[:, -1, :], output) #+ 1.0 * regularization / sum([len(p) for p in model.parameters()])
			epoch_loss += loss_val.item()

			loss_val.backward()
			optimizer.step()

		if epoch % save_interval == 0:
			ckpt_filename = filename + '/epoch{}.pt'.format(epoch)
			print('saving to "{}".'.format(ckpt_filename))
			torch.save(model, ckpt_filename)
			print('done saving model.')
			stdout.flush()

		if epoch % log_interval == 0:
			print("epoch = {}, loss = {}".format(epoch, epoch_loss))
			stdout.flush()

		if epoch % eval_interval == 0:
			evaluate_model(model, max_input_size)

			training_indices = torch.randint(dataset_size, (400,))
			logits, _ = model(LongTensor(inputs[training_indices, :]).to(device))
			predictions = torch.argmax(logits[:, -1, :], 1)
			training_acc = sum([predictions[i] in valid_outputs[training_indices[i]] for i in range(400)]) / 400
			print("training accuracy: %.2f±%.2f" % (training_acc, binomial_confidence_int(training_acc, 400)))
			stdout.flush()

		epoch += 1

if __name__ == "__main__":
	import argparse
	def parse_bool_arg(v):
		if isinstance(v, bool):
			return v
		elif v.lower() in ('yes', 'true', 'y', 't', '1'):
			return True
		elif v.lower() in ('no', 'false', 'n', 'f', '0'):
			return False
		else:
			raise argparse.ArgumentTypeError('Boolean value expected.')

	parser = argparse.ArgumentParser()
	parser.add_argument("--max-input-size", type=int)
	parser.add_argument("--dataset-size", type=int)
	parser.add_argument("--max-lookahead", type=int)
	parser.add_argument("--nlayers", type=int)
	parser.add_argument("--hidden-dim", type=int)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--bidirectional", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--absolute-pos-emb", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--learn-tok-emb", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--toeplitz-attn", type=parse_bool_arg, required=True, metavar="'y/n'")
	args = parser.parse_args()

	train(
		max_input_size=args.max_input_size,
		dataset_size=args.dataset_size,
		max_lookahead=args.max_lookahead,
		seed_value=args.seed,
		nlayers=args.nlayers,
		hidden_dim=args.hidden_dim,
		bidirectional=args.bidirectional,
		absolute_pos_emb=args.absolute_pos_emb,
		learnable_token_emb=args.learn_tok_emb,
		toeplitz_attn=args.toeplitz_attn)
