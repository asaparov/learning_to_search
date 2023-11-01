from random import seed, randrange
import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from train import generate_example, DummyDataset
from Sophia import SophiaG

class TransformerProber(nn.Module):
	def __init__(self, tfm_model, probe_layer):
		super().__init__()
		hidden_dim = tfm_model.ln_head.normalized_shape[0]
		self.model = tfm_model
		for param in tfm_model.parameters():
			param.requires_grad = False
		self.probe_layer = probe_layer
		self.decoder = nn.Linear(hidden_dim, hidden_dim)

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

		return self.decoder(x)


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
	seed(1)
	torch.manual_seed(1)
	np.random.seed(1)

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

	tfm_model = torch.load(argv[1], map_location=device)
	model = TransformerProber(tfm_model, probe_layer=1)
	model.to(device)

	max_input_size = 24
	QUERY_PREFIX_TOKEN = max_input_size - 1
	PADDING_TOKEN = max_input_size - 2
	EDGE_PREFIX_TOKEN = max_input_size - 3
	PATH_PREFIX_TOKEN = max_input_size - 4
	dataset_size = 100000
	num_generated = 0
	inputs = np.empty((dataset_size, max_input_size), dtype=np.int64)
	outputs = np.empty((dataset_size, max_input_size), dtype=np.int64)
	path_lengths = [0] * max_input_size
	total_path_lengths = 0
	MAX_PATH_LENGTH_BUCKET = 0.4
	valid_outputs = []

	while num_generated < dataset_size:
		while True:
			g, start, end, paths = generate_example(randrange(3, 7), 4, max_input_size - 5)
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
			if total_path_lengths != 0 and path_lengths[len(path)] / total_path_lengths >= MAX_PATH_LENGTH_BUCKET:
				continue
			path_lengths[len(path)] += 1
			total_path_lengths += 1
			example = list(prefix)
			for j in range(1, len(path)):
				example.append(path[j - 1].id)
				if len(example) > max_input_size:
					#print('WARNING: Generated example is too long.')
					continue
				inputs[num_generated,(max_input_size-len(example)):] = example
				inputs[num_generated,:(max_input_size-len(example))] = PADDING_TOKEN
				# compute the positions of matching source vertices
				outputs[num_generated,:] = [1.0 if (i > 0 and inputs[num_generated, i - 1] == EDGE_PREFIX_TOKEN and inputs[num_generated, i] == path[j - 1].id) else 0.0 for i in range(max_input_size)]
				valid_outputs.append([v.id for v in path[j - 1].children])
				num_generated += 1
				if num_generated == dataset_size:
					break
			if num_generated == dataset_size:
				break

		if num_generated % 1000 == 0 or num_generated >= dataset_size:
			print("{} examples generated.".format(num_generated))
			print("Path length histogram:")
			print(', '.join(['%d:%.2f' % (i, path_lengths[i] / total_path_lengths + 1e-9) for i in range(len(path_lengths)) if path_lengths[i] != 0]))

	train_data = DummyDataset(inputs, outputs, device, x_type=LongTensor, y_type=FloatTensor)
	train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)

	loss_func = BCEWithLogitsLoss(reduction='mean')
	optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=1.0e-3)

	log_interval = 1
	eval_interval = 20
	save_interval = 100

	epoch = 0
	import pdb; pdb.set_trace()
	while True:
		epoch_loss = 0.0
		for idx, batch in enumerate(train_loader):
			model.train()
			optimizer.zero_grad()

			input, output = batch

			logits = model(input)
			loss_val = loss_func(logits[:, -1, :output.size(1)], output)
			epoch_loss += loss_val.item()

			loss_val.backward()
			optimizer.step()

		if epoch % save_interval == 0:
			filename = 'checkpoints_2layer_noprojv_nolinear/probe/epoch{}.pt'.format(epoch)
			if len(argv) <= 1 or filename != argv[1]:
				print('saving to "{}".'.format(filename))
				torch.save(model, filename)
				print('done saving model.')

		if epoch % log_interval == 0:
			print("epoch = {}, loss = {}".format(epoch, epoch_loss))

		if epoch % eval_interval == 0:
			evaluate_decoder(model, max_input_size)

			'''training_indices = torch.randint(dataset_size, (400,))
			logits, _ = model(LongTensor(inputs[training_indices, :]).to(device))
			predictions = torch.argmax(logits[:, -1, :], 1)
			training_acc = sum([predictions[i] in valid_outputs[training_indices[i]] for i in range(400)]) / 400
			print("training accuracy: %.2f±%.2f" % (training_acc, binomial_confidence_int(training_acc, 400)))'''

		epoch += 1
