from random import sample, randrange, choice, shuffle, seed, uniform, getstate, setstate, Random
from os import listdir, makedirs, rename, remove, popen, sched_getaffinity
from os.path import isfile, isdir
from sys import stdout
import pickle
import numpy as np
import torch
from torch import nn, Tensor, LongTensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from gpt2 import Transformer, TransformerLayer, ToeplitzMode, AblationMode
from Sophia import SophiaG
from vocab import VOCAB
import time
import multiprocessing
from mapping import map_tokens_to_natural_language, map_tokens_to_natural_language_batched
from mapping import create_custom_tokenizer

def build_module(name):
	from os import system
	if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I../ {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
		print(f"ERROR: Unable to compile `{name}.cpp`.")
		import sys
		sys.exit(1)
try:
	from os.path import getmtime
	from importlib.util import find_spec
	generator_spec = find_spec('generator')
	if generator_spec == None:
		raise ModuleNotFoundError
	if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
		print("C++ module `generator` is out-of-date. Compiling from source...")
		build_module("generator")
	import generator
except ModuleNotFoundError:
	print("C++ module `generator` not found. Compiling from source...")
	build_module("generator")
	import generator
except ImportError:
	print("Error loading C++ module `generator`. Compiling from source...")
	build_module("generator")
	import generator
print("C++ module `generator` loaded.")

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

def get_descendants(node):
	queue = [node]
	visited = []
	descendants = []
	while len(queue) != 0:
		current = queue.pop()
		visited.append(current)
		for child in current.children:
			if child not in descendants:
				descendants.append(child)
			if child in visited:
				continue
			queue.append(child)
	return descendants

def generate_graph_with_lookahead(num_vertices, max_num_parents, max_vertex_id, lookahead, num_paths):
	num_vertices = max(2, num_vertices, 1 + num_paths * lookahead)

	vertices = []
	for i in range(num_vertices):
		vertices.append(Node(i))

	vertices[1].parents.append(vertices[0])
	vertices[0].children.append(vertices[1])
	for i in range(1, lookahead):
		vertices[1 + i].parents.append(vertices[i])
		vertices[i].children.append(vertices[1 + i])
	if lookahead == 0:
		index = 2
	else:
		index = 1 + lookahead
		for j in range(num_paths - 1):
			vertices[index].parents.append(vertices[0])
			vertices[0].children.append(vertices[index])
			index += 1
			other_branch_length = lookahead + randrange(min(2, num_vertices - index - (num_paths - j - 1) * lookahead + 2))
			for i in range(1, other_branch_length):
				vertices[index].parents.append(vertices[index - 1])
				vertices[index - 1].children.append(vertices[index])
				index += 1

	num_prefix_vertices = randrange(num_vertices - index + 1)
	prev_vertex = vertices[0]
	for i in range(num_prefix_vertices):
		vertices[index].children.append(prev_vertex)
		prev_vertex.parents.append(vertices[index])
		prev_vertex = vertices[index]
		index += 1

	start = vertices[0]
	end = vertices[max(1, lookahead)]

	# sample some parent/ancestor vertices
	alpha = 0.5
	in_degrees = np.array([alpha + len(vertex.parents) for vertex in vertices[:num_vertices]])
	out_degrees = np.array([alpha + len(vertex.children) for vertex in vertices[:num_vertices]])
	for i in range(index, num_vertices):
		# sample the number of child and parent vertices
		num_children = randrange(0, max_num_parents)
		num_parents = randrange(0 if num_children != 0 else 1, max_num_parents)
		num_children = min(num_children, i)
		num_parents = min(num_parents, i)

		# sample the children of this new node
		probabilities = in_degrees[:index].copy()
		probabilities /= np.sum(probabilities)
		for child_id in np.random.choice(index, num_children, replace=False, p=probabilities):
			vertices[index].children.append(vertices[child_id])
			vertices[child_id].parents.append(vertices[index])
			in_degrees[child_id] += 1

		# to avoid creating a cycle, we have to remove any descendants from the possible parents
		descendants = get_descendants(vertices[index])
		probabilities = out_degrees[:index].copy()

		for descendant in descendants:
			probabilities[descendant.id] = 0
		total_probability = np.sum(probabilities)
		if total_probability != 0.0:
			probabilities /= total_probability
			num_parents = min(num_parents, index - len(descendants))

			# sample the parents of this new node
			for parent_id in np.random.choice(index, num_parents, replace=False, p=probabilities):
				vertices[parent_id].children.append(vertices[i])
				vertices[i].parents.append(vertices[parent_id])
				out_degrees[parent_id] += 1
		index += 1

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

def compute_paths(graph, start, end, get_shortest_paths):
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
					queue[j] = (queue[j][0], min(queue[j][1], distance + 1))
					found_child = True
					break
			if not found_child:
				queue.append((child, distance + 1))

	if end not in reverse_pointers:
		return None

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
				return None
			continue
		for next in forward_pointers[partial_path[-1]]:
			queue.append(partial_path + [next])

	return paths

def generate_example(num_vertices, max_num_parents, max_vertex_id, get_shortest_paths=True, lookahead=None, num_paths=None):
	if lookahead == None:
		graph = generate_graph(num_vertices, max_num_parents, max_vertex_id)

		# randomly select two vertices
		start = graph[randrange(len(graph) - 1)]
		while True:
			end = graph[randrange(len(graph) - 1)]
			if end != start:
				break
	else:
		graph, start, end = generate_graph_with_lookahead(num_vertices, max_num_parents, max_vertex_id, lookahead, num_paths)
		if graph == None:
			return None, None, None, None

	paths = compute_paths(graph, start, end, get_shortest_paths)
	if paths == None:
		return None, None, None, None

	return (graph, start, end, paths)

def generate_star_graph(num_spokes, spoke_length, max_vertex_id):
	num_vertices = 1 + num_spokes * spoke_length

	vertices = []
	for i in range(num_vertices):
		vertices.append(Node(i))

	vertices[1].parents.append(vertices[0])
	vertices[0].children.append(vertices[1])
	for i in range(1, spoke_length):
		vertices[1 + i].parents.append(vertices[i])
		vertices[i].children.append(vertices[1 + i])
	if spoke_length == 0:
		index = 2
	else:
		index = 1 + spoke_length
		for j in range(num_spokes - 1):
			vertices[index].parents.append(vertices[0])
			vertices[0].children.append(vertices[index])
			index += 1
			for i in range(1, spoke_length):
				vertices[index].parents.append(vertices[index - 1])
				vertices[index - 1].children.append(vertices[index])
				index += 1

	start = vertices[0]
	end = vertices[max(1, spoke_length)]

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

def binomial_confidence_int(p, n):
	return 1.96 * np.sqrt(p * (1.0 - p) / n)

def generate_star_graph_data(max_input_size, num_spokes, spoke_length, num_samples=1000):
	QUERY_PREFIX_TOKEN = (max_input_size-5) // 3 + 4
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	PATH_PREFIX_TOKEN = (max_input_size-5) // 3 + 1

	total_predictions = 0
	inputs = np.empty((num_samples, max_input_size), dtype=np.int64)
	outputs = np.empty(num_samples, dtype=np.int64)
	while total_predictions < num_samples:
		g, start, end = generate_star_graph(num_spokes, spoke_length, (max_input_size - 5) // 3)

		paths = compute_paths(g, start, end, get_shortest_paths=True)
		if paths == None:
			continue

		prefix = []
		for vertex in g:
			for child in vertex.children:
				prefix.extend([EDGE_PREFIX_TOKEN, vertex.id, child.id])
		prefix.extend([QUERY_PREFIX_TOKEN, start.id, end.id, PATH_PREFIX_TOKEN])

		prefix.append(start.id)
		input = [PADDING_TOKEN] * (max_input_size - len(prefix)) + prefix
		inputs[total_predictions,:] = input
		outputs[total_predictions] = paths[0][1].id
		total_predictions += 1
		if total_predictions == num_samples:
			break

	return inputs, outputs

def generate_eval_data(max_input_size, min_path_length=2, distance_from_start=-1, distance_from_end=-1, lookahead_steps=None, num_paths_at_fork=None, num_samples=1000):
	QUERY_PREFIX_TOKEN = (max_input_size-5) // 3 + 4
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	PATH_PREFIX_TOKEN = (max_input_size-5) // 3 + 1

	min_vertices = max(3, min_path_length)

	total_predictions = 0
	best_predictions = 0
	useful_predictions = 0
	valid_predictions = 0
	best_edge_counts = []
	useful_edge_counts = []
	valid_edge_counts = []
	graph_size_counts = []
	inputs = np.empty((num_samples, max_input_size), dtype=np.int64)
	outputs = np.empty(num_samples, dtype=np.int64)
	#distances_from_end = [0] * max_input_size
	#MAX_FREQ_PER_BUCKET = 0.30
	while total_predictions < num_samples:
		while True:
			num_vertices = randrange(min_vertices, (max_input_size - 5) // 3)
			if lookahead_steps != None:
				# first compute the maximum number of paths we can fit with the given lookahead
				if lookahead_steps == 0:
					num_paths = randrange(1, 3)
				elif num_paths_at_fork != None:
					num_paths = num_paths_at_fork
				else:
					max_num_paths = ((max_input_size - 5) // 3 - 1) // lookahead_steps
					num_paths = randrange(2, max_num_paths + 1)
				num_vertices = min(lookahead_steps * num_paths + 1 + randrange(0, 6), (max_input_size - 5) // 3)
			else:
				num_paths = None
			g, start, end, paths = generate_example(num_vertices, 4, (max_input_size - 5) // 3, get_shortest_paths=False, lookahead=lookahead_steps, num_paths=num_paths)
			if paths != None and min([len(path) for path in paths]) > (min(lookahead_steps, min_path_length) if lookahead_steps != None else min_path_length):
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
			for j in range(1, len(path)):
				example = prefix + [v.id for v in path[:j]]
				if len(example) > max_input_size:
					continue
				shortest_path_length = min([len(p) for p in paths if path[:j] == p[:j]])
				if distance_from_start != -1 and j != distance_from_start:
					continue
				if distance_from_end != -1 and shortest_path_length - j != distance_from_end:
					continue
				if distance_from_start == -1 and distance_from_end == -1:
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
			current_vertex = next(v for v in g if v.id == partial_path[-1])
			valid_next_steps = [child.id for child in current_vertex.children]

			best_next_vertex = next(v for v in g if v.id == best_next_steps[0])
			if lookahead_steps != None and lookahead_depth(current_vertex, best_next_vertex, end) != lookahead_steps:
				continue

			input = [PADDING_TOKEN] * (max_input_size - len(partial_path)) + partial_path
			if len(valid_next_steps) == 1 or len(best_next_steps) != 1 or len(useful_next_steps) != 1:
				continue
			inputs[total_predictions,:] = input
			outputs[total_predictions] = useful_next_steps[0]
			total_predictions += 1
			if total_predictions == num_samples:
				break

	return inputs, outputs

def evaluate_model(model, inputs, labels, outputs, nl, PADDING_TOKEN):
	device = next(model.parameters()).device
	inputs = torch.tensor(inputs)
	outputs = torch.tensor(outputs)
	labels = torch.tensor(labels)
	labels = labels.to(device)
	inputs = inputs.to(device)
	outputs = outputs.to(device)
	max_input_size = inputs.shape[1]

	if outputs.dim() == 2:
		loss_func = BCEWithLogitsLoss(reduction='mean')
	else:
		loss_func = CrossEntropyLoss(reduction='mean')

	logits, _ = model(inputs)
	
	if nl:
		loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
		loss = loss_func(logits[:, :-1, :].reshape(-1, logits.shape[2]), labels[:, 1:].reshape(-1)).item()
		
		training_acc = (labels[:, 1:] == torch.argmax(logits[:, :-1, :], dim=2)) * (labels[:, 1:] != PADDING_TOKEN)  # b x (l-1)
		padding_mask = (labels[:, 1:] == PADDING_TOKEN)
		training_acc = torch.all(torch.logical_or(padding_mask, training_acc), dim=1).float()  # b
		training_acc = torch.sum(training_acc).item() / labels.size(0)
		return training_acc, loss, None

	else:
		loss = loss_func(logits[:, -1, :], outputs).item()

		predictions = torch.argmax(logits[:, -1, :], 1)
		if outputs.dim() == 2:
			acc = torch.sum(torch.gather(outputs, 1, torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
		else:
			acc = sum(predictions == outputs).item() / len(predictions)
		return acc, loss, predictions

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

def generate_training_set(max_input_size, dataset_size, max_lookahead, reserved_inputs, distance_from_start, quiet=False):
	QUERY_PREFIX_TOKEN = (max_input_size-5) // 3 + 4
	PADDING_TOKEN = (max_input_size-5) // 3 + 3
	EDGE_PREFIX_TOKEN = (max_input_size-5) // 3 + 2
	PATH_PREFIX_TOKEN = (max_input_size-5) // 3 + 1

	num_generated = 0
	num_collisions = 0
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

	while num_generated < dataset_size:
		while True:
			lookahead = choice([i for i in range(max_lookahead + 1) if num_generated == 0 or lookahead_step_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i]])
			if lookahead == 0:
				num_paths = randrange(1, 3)
			else:
				max_num_paths = ((max_input_size - 5) // 3 - 1) // lookahead
				num_paths = randrange(2, max_num_paths + 1)
			num_vertices = min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) // 3)
			g, start, end, paths = generate_example(num_vertices, 4, (max_input_size - 5) // 3, lookahead=lookahead, num_paths=num_paths)
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
			for j in range(1, len(path)):
				if distance_from_start != -1 and j != distance_from_start:
					continue
				example = prefix + [v.id for v in path[:j]]
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

				# check if this input is reserved
				if tuple(example) in reserved_inputs:
					num_collisions += 1
					continue

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

		if not quiet and (num_generated % 1000 == 0 or num_generated >= dataset_size):
			print("{} examples generated.".format(num_generated))
			#print("Path length histogram:")
			#print(', '.join(['%d:%.2f' % (i, path_lengths[i] / total_path_lengths + 1e-9) for i in range(len(path_lengths)) if path_lengths[i] != 0]))
			print("Lookahead steps histogram:")
			print(', '.join(['%d:%.2f' % (i, lookahead_step_histogram[i] / num_generated + 1e-9) for i in range(len(lookahead_step_histogram)) if lookahead_step_histogram[i] != 0]))
			print("Path length histogram:")
			print(', '.join(['%d:%.2f' % (i, path_length_histogram[i] / num_generated + 1e-9) for i in range(len(path_length_histogram)) if path_length_histogram[i] != 0]))
			stdout.flush()

	return inputs, outputs, valid_outputs, num_collisions

def train(max_input_size, dataset_size, max_lookahead, seed_value, nlayers, hidden_dim, bidirectional, absolute_pos_emb, learnable_token_emb, toeplitz_attn, toeplitz_reg, toeplitz_pos_only, add_padding, ablate, pre_ln, curriculum_mode, looped, dfs, nl, nl2):
	generator.set_seed(seed_value)
	seed(seed_value)
	torch.manual_seed(seed_value)
	np.random.seed(seed_value)

	
	if nl:
		TRANSFORMER_LENGTH = max_input_size * 6
		tokenizer = create_custom_tokenizer(vocab=VOCAB, max_length=TRANSFORMER_LENGTH)
		ntoken = len(tokenizer)
		PADDING_TOKEN = tokenizer.pad_token_id
	else:
		PADDING_TOKEN = (max_input_size-5) // 3 + 3
		TRANSFORMER_LENGTH = max_input_size
		ntoken = (max_input_size-5) // 3 + 5
	
	
	BATCH_SIZE = 1024
	print('Number of available CPUs: {}'.format(len(sched_getaffinity(0))))
	stdout.flush()

	if curriculum_mode == 'n' and dataset_size != -1:
		print('ERROR: Curriculum learning is only supported with streaming training (i.e. dataset_size = -1).')
		stdout.flush()
		return

	# first reserve some data for OOD testing
	random_state = getstate()
	np_random_state = np.random.get_state()
	torch_random_state = torch.get_rng_state()

	reserved_inputs = set()
	NUM_TEST_SAMPLES = 10000
	dist_from_start = 1
	if dfs:
		# TODO: i think we could theoretically generate examples with `backtrack_distance = (max_input_size - 4) // 4 - 1`, but the rejection rate in the current rejection sampling method is too high to feasibly generate such samples
		max_backtrack_distance = (max_input_size - 4) // 4 - 2
		for backtrack_distance in [-1] + list(range(1, max_backtrack_distance + 1)):
			generator.set_seed(seed_value)
			print('Reserving OOD test data for lookahead = {}'.format(lookahead))
			stdout.flush();
			inputs,outputs,_,_ = generator.generate_dfs_training_set(max_input_size, NUM_TEST_SAMPLES, reserved_inputs, backtrack_distance, True)
			for i in range(inputs.shape[0]):
				reserved_inputs.add(tuple([x for x in inputs[i,:] if x != PADDING_TOKEN]))
			if backtrack_distance == -1:
				eval_inputs, eval_outputs = inputs, outputs
	else:
		max_test_lookahead = ((max_input_size - 5) // 3 - 1) // 2
		dist_from_start = 1 if add_padding else -1
		for lookahead in list(range(1, max_test_lookahead + 1)):
			gen_eval_start_time = time.perf_counter()
			setstate(random_state)
			np.random.set_state(np_random_state)
			torch.set_rng_state(torch_random_state)

			print('Reserving OOD test data for lookahead = {}'.format(lookahead))
			stdout.flush()
			# inputs,outputs = generate_eval_data(max_input_size, min_path_length=2, distance_from_start=dist_from_start, distance_from_end=-1, lookahead_steps=lookahead, num_paths_at_fork=None, num_samples=NUM_TEST_SAMPLES)
			max_edges = (max_input_size - 5) // 3
			inputs, outputs, labels, num_collisions = generator.generate_training_set(max_input_size, NUM_TEST_SAMPLES, lookahead, max_edges, reserved_inputs, dist_from_start, nl,True)

			print('Done. Throughput: {} examples/s'.format(NUM_TEST_SAMPLES / (time.perf_counter() - gen_eval_start_time)))
			for i in range(inputs.shape[0]):
				reserved_inputs.add(tuple([x for x in inputs[i,:] if x != PADDING_TOKEN]))

			if nl:
				eval_inputs, eval_outputs, eval_labels, eval_num_collisions = inputs, outputs, labels, num_collisions
				eval_inputs, eval_outputs, eval_labels = map_tokens_to_natural_language_batched(tokenizer, eval_inputs, eval_labels, max_input_size, TRANSFORMER_LENGTH)
			else:
				if nl2: 
					eval_inputs, eval_outputs, eval_labels = map_tokens_to_natural_language_batched(tokenizer, eval_inputs, eval_labels, max_input_size, TRANSFORMER_LENGTH)
				
	if BATCH_SIZE < eval_inputs.shape[0]:
		eval_inputs = eval_inputs[:BATCH_SIZE]
		eval_outputs = eval_outputs[:BATCH_SIZE]
		eval_labels = eval_labels[:BATCH_SIZE]
		# eval_num_collisions = eval_num_collisions[:BATCH_SIZE]

	train_filename = 'train{}_v3_inputsize{}_maxlookahead{}_{}seed{}.pkl'.format(dataset_size, max_input_size, max_lookahead, 'padded_' if add_padding else '', seed_value)
	prefix = 'dfs_results/' if dfs else 'useful_path_results_nl/'
	makedirs(prefix, exist_ok=True)
	if dataset_size != -1:
		train_path = prefix + train_filename
		if isfile(train_path):
			# check if we've already generated the training data
			print("Loading training data from '{}'...".format(train_path))
			stdout.flush()
			with open(train_path, 'rb') as f:
				inputs, outputs, valid_outputs = pickle.load(f)
		else:
			# we haven't generated the training data yet, so generate it here
			inputs, outputs, labels, _ = generator.generate_training_set(max_input_size, dataset_size, max_lookahead, reserved_inputs, 1 if add_padding else -1, False)

			# save the generated training data to file
			with open(train_path, 'wb') as f:
				pickle.dump((inputs, outputs, labels), f)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		#from sys import exit
		#exit(-1)
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	if dataset_size != -1:
		train_data = DummyDataset(inputs, outputs, device)
		train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

	# compute the checkpoint filenames and try to resume from the last one
	filename = prefix + 'checkpoints_v3_{}_{}layer_inputsize{}_maxlookahead{}_seed{}_train{}'.format(nl, nlayers, max_input_size, max_lookahead, seed_value, dataset_size if dataset_size != -1 else 'streaming')
	if bidirectional:
		filename += '_nomask'
	if not absolute_pos_emb:
		filename += '_noAPE'
	if learnable_token_emb:
		filename += '_learntokemb'
	if ablate == "none":
		filename += '_unablated'
	elif ablate == "attn_linear":
		filename += '_ablateattnlinear'
	if toeplitz_attn:
		filename += '_toeplitz'
		if toeplitz_pos_only:
			filename += 'pos'
	if toeplitz_reg != 0.0:
		filename += '_toeplitz'
		if toeplitz_pos_only:
			filename += 'pos'
		filename += str(toeplitz_reg)
	if not pre_ln:
		filename += '_postLN'
	if add_padding:
		filename += '_padded'
	if curriculum_mode == 'y':
		filename += '_curriculum'
	elif curriculum_mode == 'layerbylayer':
		filename += '_layercurriculum'
	elif curriculum_mode == 'layerbylayer2':
		filename += '_layercurriculum2'
	if looped:
		filename += '_looped'
	if dfs:
		filename += '_dfs'
	if isdir(filename):
		existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
	else:
		existing_epochs = []
		makedirs(filename)

	nhead = 1
	d_hid = ntoken + hidden_dim
	dropout = 0
	if ablate == "none":
		ablation_mode = AblationMode.NO_ABLATION
	elif ablate == "attn_linear":
		ablation_mode = AblationMode.ABLATE_ATTN_LINEAR
	elif ablate == "attn_linear_projv":
		ablation_mode = AblationMode.ABLATE_ATTN_LINEAR_PROJV
	if toeplitz_attn and toeplitz_pos_only:
		toeplitz = ToeplitzMode.LOWER_RIGHT
	elif toeplitz_attn and not toeplitz_pos_only:
		toeplitz = ToeplitzMode.BLOCK
	else:
		toeplitz = ToeplitzMode.NONE
	if len(existing_epochs) == 0:
		if curriculum_mode in ('layerbylayer','layerbylayer2'):
			initial_layers = min(3, nlayers)
		else:
			initial_layers = nlayers
		model = Transformer(
				layers=initial_layers,
				pad_idx=PADDING_TOKEN,
				words=ntoken,
				seq_len=TRANSFORMER_LENGTH,
				heads=nhead,
				dims=max(ntoken,d_hid),
				rate=1,
				dropout=dropout,
				bidirectional=bidirectional,
				absolute_pos_emb=absolute_pos_emb,
				learn_token_emb=learnable_token_emb,
				ablate=ablation_mode,
				toeplitz=toeplitz,
				pre_ln=pre_ln,
				looped=looped)
		epoch = 0
		model.to(device)
	else:
		last_epoch = max(existing_epochs)
		epoch = last_epoch + 1
		print("Loading model from '{}/epoch{}.pt'...".format(filename, last_epoch))
		stdout.flush()
		loaded_obj = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device)
		model, random_state, np_random_state, torch_random_state = loaded_obj
		setstate(random_state)
		np.random.set_state(np_random_state)
		torch.set_rng_state(torch_random_state.cpu())

	loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
	optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=1.0e-5, weight_decay=0.1)

	log_interval = 1
	eval_interval = 1
	save_interval = 1

	if curriculum_mode == 'n':
		initial_lookahead = max_lookahead
		initial_max_edges = (max_input_size - 5) // 3
	elif curriculum_mode == 'y':
		initial_lookahead = 1
		initial_max_edges = (max_input_size - 5) // 3
	elif curriculum_mode == 'layerbylayer':
		initial_lookahead = 2
		initial_max_edges = 5
	elif curriculum_mode == 'layerbylayer2':
		initial_lookahead = 2
		initial_max_edges = 5
	if hasattr(model, 'lookahead'):
		initial_lookahead = model.lookahead
	else:
		model.lookahead = initial_lookahead
	if hasattr(model, 'max_edges'):
		initial_max_edges = model.max_edges
	else:
		model.max_edges = initial_max_edges

	if dataset_size == -1:
		# we are doing streaming training, so use an IterableDataset
		from itertools import cycle
		from threading import Lock
		STREAMING_BLOCK_SIZE = 2 ** 18
		NUM_DATA_WORKERS = 2
		seed_generator = Random(seed_value)
		seed_generator_lock = Lock()
		seed_values = []

		def get_seed(index):
			if index < len(seed_values):
				return seed_values[index]
			seed_generator_lock.acquire()
			while index >= len(seed_values):
				seed_values.append(seed_generator.randrange(2 ** 32))
			seed_generator_lock.release()
			return seed_values[index]

		class StreamingDataset(torch.utils.data.IterableDataset):
			def __init__(self, offset, lookahead, max_edges):
				super(StreamingDataset).__init__()
				self.offset = offset
				self.lookahead = lookahead
				self.max_edges = max_edges

				self.multiprocessing_manager = multiprocessing.Manager()
				self.total_collisions = self.multiprocessing_manager.Value(int, 0)
				self.collisions_lock = self.multiprocessing_manager.Lock()

			def process_data(self, start):
				current = start
				worker_info = torch.utils.data.get_worker_info()
				worker_id = worker_info.id
				while True:
					worker_start_time = time.perf_counter()
					new_seed = get_seed(current)
					generator.set_seed(new_seed)
					seed(new_seed)
					torch.manual_seed(new_seed)
					np.random.seed(new_seed)

					generate_start_time = time.perf_counter()
					if dfs:
						inputs, outputs, labels, num_collisions = generator.generate_dfs_training_set(max_input_size, BATCH_SIZE, reserved_inputs, -1, True)
					else:
						inputs, outputs, labels, num_collisions = generator.generate_training_set(max_input_size, BATCH_SIZE, self.lookahead, self.max_edges, reserved_inputs, dist_from_start, nl, True)

						if nl:
							inputs, outputs, labels = map_tokens_to_natural_language_batched(tokenizer, inputs, labels, max_input_size, TRANSFORMER_LENGTH)
					if num_collisions != 0:
						with self.collisions_lock:
							self.total_collisions.value += num_collisions
						print('Total number of training examples generated that are in the test set: {}'.format(self.total_collisions.value))
						stdout.flush()

					worker_end_time = time.perf_counter()
					#print('[WORKER {}] yield = {}, throughput = {} examples/s, rank = {}'.format(worker_id, curreinpunt, BATCH_SIZE / (worker_end_time - worker_start_time), multiprocessing.current_process()._identity[0]))
					#print('[WORKER {}] time to get seed = {}s, time to generate data = {}s'.format(worker_id, generate_start_time - worker_start_time, worker_end_time - generate_start_time))
					#stdout.flush()
					yield inputs, outputs, labels
					current += NUM_DATA_WORKERS

			def __iter__(self):
				worker_info = torch.utils.data.get_worker_info()
				worker_id = worker_info.id
				return self.process_data(self.offset + worker_id)

		iterable_dataset = StreamingDataset(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.lookahead, model.max_edges)
		train_loader = DataLoader(iterable_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)

	while True:
		start_time = time.perf_counter()
		transfer_time = 0.0
		train_time = 0.0
		log_time = 0.0
		epoch_loss = 0.0
		num_batches = 0
		effective_dataset_size = (STREAMING_BLOCK_SIZE if dataset_size == -1 else dataset_size)
		reinit_data_loader = False
		for batch in (train_loader if dataset_size == -1 else cycle(train_loader)):

			# num_batches += 1
			# print("batch = {}".format(num_batches))
			# continue

			batch_start_time = time.perf_counter()
			model.train()
			optimizer.zero_grad()

			input, output, label = batch
			
				#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			train_start_time = time.perf_counter()
			transfer_time += train_start_time - batch_start_time

			input = input.to(device, non_blocking=True)
			output = output.to(device, non_blocking=True)
			label = label.to(device, non_blocking=True)

			logits = model(input)
			if nl: 
				loss_val = loss_func(logits[:, :-1, :].reshape(-1, logits.shape[-1]), label[:,1:].reshape(-1))
				# loss_val = loss_func(logits[:, :-1].view(-1, logits.shape[-1]), label[:, 1:].view(-1))
			else:
				loss_val = loss_func(logits[:, -1, :], label)

			if toeplitz_reg != 0.0:
				def compute_toeplitz_regularization(m):
					regularization = 0.0
					for i in range(-A.size(0) + 1, A.size(1)):
						regularization += torch.var(torch.diagonal(A, offset=i), unbiased=False)
					return regularization

				for transformer in model.transformers:
					P_q = next(v for k,v in transformer.attn.proj_q.named_parameters() if k == 'weight')
					P_k = next(v for k,v in transformer.attn.proj_k.named_parameters() if k == 'weight')
					A = torch.matmul(P_q.transpose(-2,-1),P_k)
					if not toeplitz_pos_only:
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,:ntoken])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,ntoken:d_hid])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,d_hid:])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,:ntoken])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,ntoken:d_hid])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,d_hid:])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,:ntoken])
						loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,ntoken:d_hid])
					loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,d_hid:])
			epoch_loss += loss_val.item()

			loss_val.backward()
			optimizer.step()

			#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			log_start_time = time.perf_counter()
			train_time += log_start_time - train_start_time

			num_batches += 1
			if num_batches == effective_dataset_size // BATCH_SIZE:
				#time4 = time.perf_counter()
				#print('[MAIN] Time to train: {}s'.format(time4 - time3))
				#stdout.flush()

				if epoch % log_interval == 0:
					elapsed_time = time.perf_counter() - start_time
					print("epoch = {}, training loss = {}".format(epoch, epoch_loss))
					if device.type == 'cuda':
						utilization = popen('nvidia-smi --query-gpu=utilization.gpu --format=csv').read().split('\n')[1]
						print("throughput = {} examples/s, GPU utilization = {}".format(effective_dataset_size / elapsed_time, utilization))
					else:
						print("throughput = {} examples/s".format(effective_dataset_size / elapsed_time))
					print("[PROFILE] Total batch time: {}s".format(elapsed_time))
					print("[PROFILE] Time to transfer data to GPU: {}s".format(transfer_time))
					print("[PROFILE] Time to train: {}s".format(train_time))
					print("[PROFILE] Time to log/save/validate: {}s".format(log_time))
					stdout.flush()
					start_time = time.perf_counter()
					transfer_time = 0.0
					train_time = 0.0
					log_time = 0.0

				if epoch % eval_interval == 0:
					model.eval()
					logits, _ = model(input)
					
					if nl:					
						training_acc = (label[:, 1:] == torch.argmax(logits[:, :-1, :], dim=2)) * (label[:, 1:] != PADDING_TOKEN)  # b x (l-1)
						padding_mask = (label[:, 1:] == PADDING_TOKEN)
						training_acc = torch.all(torch.logical_or(padding_mask, training_acc), dim=1).float()  # b
						training_acc = torch.sum(training_acc).item() / output.size(0)
					else:
						training_acc = torch.sum(torch.gather(output, 1, torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / output.size(0)

					print("training accuracy: %.2f±%.2f" % (training_acc, binomial_confidence_int(training_acc, output.size(0))))
					del input, output
					stdout.flush()

					test_acc,test_loss,_ = evaluate_model(model, eval_inputs, eval_labels, eval_outputs, nl, PADDING_TOKEN)
					print("test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, binomial_confidence_int(test_acc, 1000), test_loss))
					stdout.flush()
					#time6 = time.perf_counter()
					#print('[MAIN] Time to evaluate model: {}s'.format(time6 - time5))
					#stdout.flush()

					if curriculum_mode != 'n' and model.lookahead < max_lookahead and training_acc > 0.99:
						if model.max_edges < (max_input_size - 5) // 3:
							# increase the maximum number of edges by 1
							print("Increasing maximum number of edges to {}".format(model.max_edges + 1))
							model.max_edges += 1
						elif curriculum_mode in ('layerbylayer','layerbylayer2') and model.lookahead + 1 > 2 ** (len(model.transformers) - 2) and len(model.transformers) < nlayers and not model.looped:
							# add another layer to the model
							print("Increasing number of transformer layers to {}".format(len(model.transformers) + 1))
							if absolute_pos_emb:
								embedding_dim = max(ntoken,d_hid)+max_input_size
								position_dim = max_input_size
							else:
								embedding_dim = max(ntoken,d_hid)
								position_dim = 0
							with torch.no_grad():
								if curriculum_mode == 'layerbylayer':
									assert ablation_mode == AblationMode.NO_ABLATION, "Layer-by-layer curriculum learning is not supported with any ablation."
									new_layer = TransformerLayer(nhead, embedding_dim, ntoken, position_dim, 1, dropout, True, ablation_mode, toeplitz, pre_ln)
								elif curriculum_mode == 'layerbylayer2':
									import copy
									new_layer = copy.deepcopy(model.transformers[-2])
								linear_weight = torch.empty(new_layer.attn.linear.weight.shape)
								linear_weight.uniform_(-0.001,0.001)
								linear_bias = torch.empty(new_layer.attn.linear.bias.shape)
								linear_bias.uniform_(-0.001,0.001)
								new_layer.attn.linear.weight = nn.Parameter(linear_weight)
								new_layer.attn.linear.bias = nn.Parameter(linear_bias)
								ff_weight = torch.empty(new_layer.ff[3].weight.shape)
								ff_weight.uniform_(-0.001,0.001)
								ff_bias = torch.empty(new_layer.ff[3].bias.shape)
								ff_bias.uniform_(-0.001,0.001)
								new_layer.ff[3].weight = nn.Parameter(ff_weight)
								new_layer.ff[3].bias = nn.Parameter(ff_bias)
								new_layer.to(device)
								if curriculum_mode =='layerbylayer':
									model.transformers.append(new_layer)
								elif curriculum_mode == 'layerbylayer2':
									model.transformers.insert(len(model.transformers)-2, new_layer)

							model.lookahead = min(max_lookahead, 2 ** (len(model.transformers) - 1))
						else:
							model.lookahead += 1
						print("Training accuracy is sufficiently high. Training lookahead is {}.".format(model.lookahead))
						reinit_data_loader = True
						break

				if epoch % save_interval == 0:
					ckpt_filename = filename + '/epoch{}.pt'.format(epoch)
					print('saving to "{}".'.format(ckpt_filename))
					torch.save((model,getstate(),np.random.get_state(),torch.get_rng_state()), ckpt_filename)
					print('done saving model.')
					stdout.flush()

					#time5 = time.perf_counter()
					#print('[MAIN] Time to save model: {}s'.format(time5 - time4))
					#stdout.flush()

				#time7 = time.perf_counter()
				#print('[MAIN] Total time for epoch: {}s'.format(time7 - time1))
				#stdout.flush()
				epoch += 1
				num_batches = 0
				epoch_loss = 0.0
				if reinit_data_loader:
					break

			#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			log_end_time = time.perf_counter()
			log_time += log_end_time - log_start_time

		if reinit_data_loader:
			iterable_dataset = StreamingDataset(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model.lookahead, model.max_edges)
			train_loader = DataLoader(iterable_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)
			reinit_data_loader = False

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
	parser.add_argument("--toeplitz-reg", type=float, required=True, default=0.0)
	parser.add_argument("--toeplitz-pos-only", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--add-padding", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--ablate", type=str, default="none", choices=["none", "attn_linear", "attn_linear_projv"])
	parser.add_argument("--preLN", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--curriculum", type=str, required=True, choices=["y", "n", "layerbylayer", "layerbylayer2"])
	parser.add_argument("--looped", type=parse_bool_arg, default=False)
	parser.add_argument("--dfs", type=parse_bool_arg, default=False)
	parser.add_argument('--nl',type=parse_bool_arg, default=False )
	parser.add_argument('--nl2',type=parse_bool_arg, default=False )
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
		toeplitz_attn=args.toeplitz_attn,
		toeplitz_reg=args.toeplitz_reg,
		toeplitz_pos_only=args.toeplitz_pos_only,
		add_padding=args.add_padding,
		ablate=args.ablate,
		pre_ln=args.preLN,
		curriculum_mode=args.curriculum,
		looped=args.looped,
		dfs=args.dfs,
		nl=args.nl)
