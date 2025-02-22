from random import seed, randrange, getstate, Random
import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from Sophia import SophiaG
from sys import stdout
from train import binomial_confidence_int
import multiprocessing

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

class TransformerProber(nn.Module):
	def __init__(self, tfm_model, probe_layer):
		super().__init__()
		hidden_dim = tfm_model.ln_head.normalized_shape[0]
		self.model = tfm_model
		for param in tfm_model.parameters():
			param.requires_grad = False
		self.probe_layer = probe_layer
		n = tfm_model.positional_embedding.size(0)
		self.decoder = nn.Linear(hidden_dim + n * hidden_dim, 1)
		if probe_layer > len(tfm_model.transformers):
			raise Exception('probe_layer must be <= number of layers')

	def to(self, device):
		super().to(device)
		self.decoder.to(device)

	def forward(self, x: torch.Tensor):
		# Create masking tensor.
		mask = self.model.pad_masking(x, 0)
		if not self.model.bidirectional:
			mask = mask + self.model.future_masking(x, 0)

		# Use token embedding and positional embedding layers.
		#x = self.token_embedding(x) + self.positional_embedding(x, offset)
		'''print('input x:')
		print(x)'''
		x = self.model.token_embedding[x]
		if len(x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
		x = torch.cat((x, pos), -1)
		x = self.model.dropout_embedding(x)
		input = x.clone()

		'''print("embedded input:")
		for i in range(x.size(0)):
			print("  x[{},:] != 0: {}".format(i, torch.nonzero(x[i,:])[:,0].tolist()))'''

		# Apply transformer layers sequentially.
		present = []
		for i, transformer in enumerate(self.model.transformers):
			x = transformer(x, None, mask)

			if not self.training:
				present.append(x[1])
				x = x[0]
			if i + 1 == self.probe_layer:
				break

		dec_input = torch.cat((x, input.reshape(input.size(0), input.size(1) * input.size(2)).unsqueeze(1).repeat((1,input.size(1),1))), dim=2)
		return self.decoder(dec_input)


def evaluate_decoder(model, max_input_size):
	num_generated = 0
	num_examples = 1000
	inputs = torch.empty((num_examples, max_input_size), dtype=torch.int64)
	outputs = torch.empty((num_examples, max_input_size), dtype=torch.int64)
	while num_generated < num_examples:
		while True:
			g, start, end, paths = generate_example(randrange(3, 7), 4, max_input_size - 5, get_shortest_paths=False)
			if paths != None and min([len(path) for path in paths]) > 1:
				break

		prefix = []
		for vertex in g:
			for child in vertex.children:
				prefix.extend([EDGE_PREFIX_TOKEN, vertex.id, child.id])
		prefix.extend([QUERY_PREFIX_TOKEN, start.id, end.id, PATH_PREFIX_TOKEN])

		for path in paths:
			if len(path) == 1:
				continue
			example = list(prefix)
			for j in range(1, len(path)):
				example.append(path[j - 1].id)
				if len(example) > max_input_size:
					#print('WARNING: Generated example is too long.')
					continue
				inputs[num_generated,(max_input_size-len(example)):] = FloatTensor(example)
				inputs[num_generated,:(max_input_size-len(example))] = PADDING_TOKEN
				# compute the positions of matching source vertices
				outputs[num_generated,:] = FloatTensor([1.0 if (i > 0 and inputs[num_generated, i - 1] == EDGE_PREFIX_TOKEN and inputs[num_generated, i] == path[j - 1].id) else 0.0 for i in range(max_input_size)])
				num_generated += 1
				if num_generated == num_examples:
					break
			if num_generated == num_examples:
				break

	inputs = inputs.to(device)
	outputs = outputs.to(device)
	model.eval()
	logits = model(inputs)
	logits = logits[:, -1, :output.size(1)]
	sigmoid = Sigmoid()
	predictions = torch.round(sigmoid(logits))

	exact_match = 0
	partial_match = 0
	for i in range(num_examples):
		if torch.all(predictions[i,:] == outputs[i,:]):
			exact_match += 1
		if all(predictions[i,j] == outputs[i,j] for j in range(max_input_size) if outputs[i,j] == 0) and any(predictions[i,j] == outputs[i,j] for j in range(max_input_size) if outputs[i,j] == 1):
			partial_match += 1
	print("exact match accuracy = %.2f, partial match accuracy = %.2f" % (exact_match / num_examples, partial_match / num_examples))

if __name__ == "__main__":
	seed_value = 1
	seed(seed_value)
	torch.manual_seed(seed_value)
	np.random.seed(seed_value)

	from sys import argv, exit
	if len(argv) != 2:
		print("Usage: train_probe [checkpoint_filepath]")
		exit(1)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		#from sys import exit
		#exit(-1)
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	tfm_model, _, _, _ = torch.load(argv[1], map_location=device)
	for transformer in tfm_model.transformers:
		if not hasattr(transformer, 'pre_ln'):
			transformer.pre_ln = True
	model = TransformerProber(tfm_model, probe_layer=4)
	model.to(device)

	suffix = argv[1][argv[1].index('inputsize')+len('inputsize'):]
	max_input_size = int(suffix[:suffix.index('_')])

	# reserve some data for validation
	lookahead_steps = 10
	reachable_distance = 8
	start_vertex_index = 3
	exclude_start_vertex = True
	reserved_inputs = set()

	# we are doing streaming training, so use an IterableDataset
	from itertools import cycle
	from threading import Lock
	STREAMING_BLOCK_SIZE = 2 ** 13
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
		def __init__(self, offset):
			super(StreamingDataset).__init__()
			self.offset = offset

			self.multiprocessing_manager = multiprocessing.Manager()
			self.total_collisions = self.multiprocessing_manager.Value(int, 0)
			self.collisions_lock = self.multiprocessing_manager.Lock()

		def process_data(self, start):
			current = start
			worker_info = torch.utils.data.get_worker_info()
			worker_id = worker_info.id
			while True:
				new_seed = get_seed(current)
				generator.set_seed(new_seed)
				seed(new_seed)
				torch.manual_seed(new_seed)
				np.random.seed(new_seed)

				inputs, outputs, num_collisions = generator.generate_reachable_training_set(max_input_size, BATCH_SIZE, lookahead_steps, reserved_inputs, 1, reachable_distance, start_vertex_index, exclude_start_vertex)
				if num_collisions != 0:
					with self.collisions_lock:
						self.total_collisions.value += num_collisions
					print('Total number of training examples generated that are in the test set: {}'.format(self.total_collisions.value))
					stdout.flush()

				yield inputs, outputs
				current += NUM_DATA_WORKERS

		def __iter__(self):
			worker_info = torch.utils.data.get_worker_info()
			worker_id = worker_info.id
			return self.process_data(self.offset + worker_id)

	epoch = 0
	BATCH_SIZE = 2 ** 9
	iterable_dataset = StreamingDataset(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE)
	train_loader = DataLoader(iterable_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)

	loss_func = BCEWithLogitsLoss(reduction='mean')
	optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=1.0e-4)

	log_interval = 1
	eval_interval = 1
	save_interval = 10

	epoch_loss = 0.0
	num_batches = 0
	while True:
		for batch in train_loader:
			model.train()
			optimizer.zero_grad()

			input, output = batch
			input = input.to(device, non_blocking=True)
			output = output.to(device, non_blocking=True)

			logits = model(input)
			# only take the predictions on source vertices
			logits = logits[:,range(2,max_input_size-5,3),-1]
			output = output[:,range(2,max_input_size-5,3)]
			loss_val = loss_func(logits, output)
			epoch_loss += loss_val.item()

			loss_val.backward()
			optimizer.step()

			num_batches += 1
			if num_batches == STREAMING_BLOCK_SIZE // BATCH_SIZE:
				if epoch % save_interval == 0:
					ckpt_filename = 'probe/epoch{}.pt'.format(epoch)
					print('saving to "{}".'.format(ckpt_filename))
					torch.save((model,getstate(),np.random.get_state(),torch.get_rng_state()), ckpt_filename)
					print('done saving model.')
					stdout.flush()

				if epoch % log_interval == 0:
					print("epoch = {}, training loss = {}".format(epoch, epoch_loss))

				if epoch % eval_interval == 0:
					model.eval()
					logits = model(input)
					training_preds = (logits[:,range(2,max_input_size-5,3),-1] > 0.5)
					training_labels = (output == 1)
					total_preds = training_preds.size(0) * training_preds.size(1)
					training_acc = torch.sum(training_preds == training_labels).item() / total_preds
					try:
						print("training accuracy: %.2f±%.2f" % (training_acc, binomial_confidence_int(training_acc, total_preds)))
						true_positive_acc = torch.sum(training_preds[output == 1]).item() / torch.sum(output == 1).item()
						print("true positive rate: %.2f±%.2f" % (true_positive_acc, binomial_confidence_int(true_positive_acc, torch.sum(output == 1).item())))
						true_negative_acc = torch.sum(training_preds[output == 0] == 0).item() / torch.sum(output == 0).item()
						print("true negative rate: %.2f±%.2f" % (true_negative_acc, binomial_confidence_int(true_negative_acc, torch.sum(output == 0).item())))
						del input, output
					except ZeroDivisionError:
						pass
					stdout.flush()

					#test_acc,test_loss,_ = evaluate_model(model, eval_inputs, eval_outputs)
					#print("test accuracy = %.2f±%.2f, test loss = %f" % (test_acc, binomial_confidence_int(test_acc, 1000), test_loss))
					stdout.flush()

				epoch += 1
				num_batches = 0
				epoch_loss = 0.0
