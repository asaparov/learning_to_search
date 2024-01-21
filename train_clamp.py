import torch
from torch import nn, Tensor, LongTensor
from torch.utils.data import DataLoader
from gpt2 import TransformerLayer, ToeplitzMode, Past, FutureMasking
from Sophia import SophiaG
from typing import Tuple, List, Union
from sys import stdout
from os import listdir, makedirs, popen
from os.path import isdir
import time


class ContinuousTransformer(nn.Module):
	def __init__(self,
					layers: int,
					heads: int,
					dims: int,
					rate: int = 4,
					bidirectional: bool = True,
					ablate: bool = True,
					toeplitz: ToeplitzMode = ToeplitzMode.NONE):
		super().__init__()
		self.bidirectional = bidirectional
		self.future_masking = FutureMasking()

		self.transformers = nn.ModuleList([
			TransformerLayer(heads, dims, dims, 0, rate, 0, True, ablate, not ablate, toeplitz)
			for l in range(layers)])

	def to(self, device):
		super().to(device)

	def forward(self,
				x: torch.Tensor,
				use_grad_ckpt: bool = False
				) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
		# Create masking tensor.
		if not self.bidirectional:
			#mask = self.future_masking(x, 0)
			raise Exception("Not implemented")
		else:
			mask = torch.zeros(x.shape, dtype=torch.bool)

		# Apply transformer layers sequentially.
		present = []
		for i, transformer in enumerate(self.transformers):
			x = transformer(x, None, mask)

			if not self.training:
				present.append(x[1])
				x = x[0]

		return x if self.training else (x, present)


def generate_data(input_size, num_examples):
	from torch.distributions.cauchy import Cauchy
	m = Cauchy(0.0, 4.0)
	x = m.sample((num_examples, input_size, input_size))
	x[x > input_size] /= input_size
	x[x < 1.0] = 0.0
	x[:,:,0] = 1.1
	x = torch.gather(x, dim=-1, index=torch.argsort(torch.rand_like(x), dim=-1))
	ln = torch.nn.LayerNorm(input_size)
	return ln(x).detach()

def clamp_loss(inputs, predictions):
	num_examples = inputs.size(0)
	loss = torch.sum(torch.maximum(predictions[inputs < 1.0], torch.tensor(0.0)) ** 2 / num_examples)
	mask = (inputs >= 1.0)
	denom = torch.sum(mask, dim=-1)
	means = torch.sum(predictions * mask, dim=-1) / denom
	return loss + torch.sum(torch.sum(((predictions - means.unsqueeze(-1)) * mask) ** 2, dim=-1) / denom / num_examples)

def evaluate_model(model, inputs):
	predictions, _ = model(inputs)
	loss = clamp_loss(inputs, predictions).item()
	return loss, predictions

def train(max_input_size, seed_value, nlayers, bidirectional, toeplitz_attn, toeplitz_reg, toeplitz_pos_only, ablate, pre_ln):
	torch.manual_seed(seed_value)

	# first reserve some data for OOD testing
	torch_random_state = torch.get_rng_state()

	gen_eval_start_time = time.perf_counter()
	torch.set_rng_state(torch_random_state)

	NUM_TEST_SAMPLES = 10000
	eval_inputs = generate_data(max_input_size, num_examples=NUM_TEST_SAMPLES)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		#from sys import exit
		#exit(-1)
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')
	eval_inputs = eval_inputs.to(device)

	# compute the checkpoint filenames and try to resume from the last one
	filename = 'clamp_results/checkpoints_{}layer_inputsize{}_seed{}'.format(nlayers, max_input_size, seed_value)
	if bidirectional:
		filename += '_nomask'
	if not ablate:
		filename += '_unablated'
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
	if isdir(filename):
		existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
	else:
		existing_epochs = []
		makedirs(filename)

	if len(existing_epochs) == 0:
		if toeplitz_attn and toeplitz_pos_only:
			toeplitz = ToeplitzMode.LOWER_RIGHT
		elif toeplitz_attn and not toeplitz_pos_only:
			toeplitz = ToeplitzMode.BLOCK
		else:
			toeplitz = ToeplitzMode.NONE
		nhead = 1
		model = ContinuousTransformer(
					layers=nlayers,
					heads=nhead,
					dims=max_input_size,
					rate=1,
					bidirectional=bidirectional,
					ablate=ablate,
					toeplitz=toeplitz,
					pre_ln=pre_ln)
		epoch = 0
		model.to(device)
	else:
		last_epoch = max(existing_epochs)
		epoch = last_epoch + 1
		loaded_obj = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device)
		model, torch_random_state = loaded_obj
		torch.set_rng_state(torch_random_state.cpu())

	optimizer = SophiaG((p for p in model.parameters() if p.requires_grad), lr=1.0e-4, weight_decay=0.1)

	log_interval = 1
	eval_interval = 1
	save_interval = 1

	# we are doing streaming training, so use an IterableDataset
	from threading import Lock
	BATCH_SIZE = 2 ** 16
	STREAMING_BLOCK_SIZE = 2 ** 21
	NUM_DATA_WORKERS = 8
	seed_generator = torch.Generator()
	seed_generator.manual_seed(seed_value)
	seed_generator_lock = Lock()
	seed_values = []

	def get_seed(index):
		if index < len(seed_values):
			return seed_values[index]
		seed_generator_lock.acquire()
		while index >= len(seed_values):
			seed_values.append(torch.randint(2 ** 32, (1,), generator=seed_generator))
		seed_generator_lock.release()
		return seed_values[index]

	class StreamingDataset(torch.utils.data.IterableDataset):
		def __init__(self, offset):
			super(StreamingDataset).__init__()
			self.offset = offset

		def process_data(self, start):
			current = start
			worker_info = torch.utils.data.get_worker_info()
			worker_id = worker_info.id
			while True:
				new_seed = get_seed(current)
				torch.manual_seed(new_seed)

				inputs = generate_data(max_input_size, BATCH_SIZE)

				yield inputs
				current += NUM_DATA_WORKERS

		def __iter__(self):
			worker_info = torch.utils.data.get_worker_info()
			worker_id = worker_info.id
			return self.process_data(self.offset + worker_id)

	iterable_dataset = StreamingDataset(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE)
	train_loader = DataLoader(iterable_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)

	while True:
		start_time = time.perf_counter()
		transfer_time = 0.0
		train_time = 0.0
		log_time = 0.0
		epoch_loss = 0.0
		num_batches = 0
		effective_dataset_size = STREAMING_BLOCK_SIZE
		for batch in train_loader:
			batch_start_time = time.perf_counter()
			model.train()
			optimizer.zero_grad()

			input = batch
			input = input.to(device, non_blocking=True)

			#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			train_start_time = time.perf_counter()
			transfer_time += train_start_time - batch_start_time

			predictions = model(input)
			loss_val = clamp_loss(input, predictions)
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
			del input

			#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			log_start_time = time.perf_counter()
			train_time += log_start_time - train_start_time

			num_batches += 1
			if num_batches == effective_dataset_size // BATCH_SIZE:
				if epoch % save_interval == 0:
					ckpt_filename = filename + '/epoch{}.pt'.format(epoch)
					print('saving to "{}".'.format(ckpt_filename))
					torch.save((model,torch.get_rng_state()), ckpt_filename)
					print('done saving model.')
					stdout.flush()

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
					test_loss,_ = evaluate_model(model, eval_inputs)
					print("test loss = %f" % test_loss)
					stdout.flush()

				epoch += 1
				num_batches = 0
				epoch_loss = 0.0

			#if device.type == 'cuda':
			#	torch.cuda.synchronize(device)
			log_end_time = time.perf_counter()
			log_time += log_end_time - log_start_time

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
	parser.add_argument("--nlayers", type=int)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--bidirectional", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--toeplitz-attn", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--toeplitz-reg", type=float, required=True, default=0.0)
	parser.add_argument("--toeplitz-pos-only", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--ablate", type=parse_bool_arg, required=True, metavar="'y/n'")
	parser.add_argument("--preLN", type=parse_bool_arg, required=True, metavar="'y/n'")
	args = parser.parse_args()

	train(
		max_input_size=args.max_input_size,
		seed_value=args.seed,
		nlayers=args.nlayers,
		bidirectional=args.bidirectional,
		toeplitz_attn=args.toeplitz_attn,
		toeplitz_reg=args.toeplitz_reg,
		toeplitz_pos_only=args.toeplitz_pos_only,
		ablate=args.ablate,
		pre_ln=args.preLN)
