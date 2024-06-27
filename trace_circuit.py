from random import seed, randrange, shuffle
import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from train import generate_example, lookahead_depth, generate_eval_data
from gpt2 import TokenEmbedding
import math

def normalize_conditions(conditions):
	# normalize the list of conditions
	for i in range(len(conditions)):
		(row1,col1,sign1,row2,col2,sign2) = conditions[i]
		if row1 < row2:
			continue
		elif row1 == row2 and col1 < col2:
			continue
		elif row1 == row2 and col1 == col2 and sign1 < sign2:
			continue
		conditions[i] = (row2,col2,sign2,row1,col1,sign1)
	conditions.sort()

def compute_attention(layer_index: int, attn_layer, x: torch.Tensor, mask: torch.Tensor):
	n, d = x.shape[-2], x.shape[-1]
	k_params = {k:v for k,v in attn_layer.proj_k.named_parameters()}
	q_params = {k:v for k,v in attn_layer.proj_q.named_parameters()}
	P_k = k_params['weight']
	P_q = q_params['weight']
	U_k = torch.cat((P_k,k_params['bias'].unsqueeze(1)),1)
	U_q = torch.cat((P_q,q_params['bias'].unsqueeze(1)),1)
	A = torch.matmul(U_q.transpose(-2,-1),U_k)
	x_prime = torch.cat((x, torch.ones(x.shape[:-1] + (1,), device=device)), -1)
	QK = torch.matmul(torch.matmul(x_prime, A), x_prime.transpose(-2,-1)) / math.sqrt(d)
	attn_pre_softmax = QK + mask.type_as(QK) * QK.new_tensor(-1e9)
	attn = attn_layer.attn.dropout(attn_pre_softmax.softmax(-1))

	if attn_layer.proj_v:
		v = attn_layer.proj_v(x)
	else:
		v = x
	new_x = torch.matmul(attn, v)
	if attn_layer.linear:
		out = attn_layer.linear(new_x)
	else:
		out = new_x
	return out, attn_pre_softmax, attn, v, new_x, A

def forward(model, x: torch.Tensor, mask: torch.Tensor, start_layer: int, start_at_ff: bool, end_layer: int, perturbations, frozen_ops):
	# Apply transformer layers sequentially.
	layer_inputs = [None] * start_layer
	attn_inputs = [None] * start_layer
	attn_matrices = [None] * start_layer
	v_outputs = [None] * start_layer
	attn_linear_inputs = [None] * start_layer
	attn_outputs = [None] * start_layer
	attn_pre_softmax = [None] * start_layer
	A_matrices = [None] * start_layer
	ff_inputs = [None] * start_layer
	ff_parameters = [None] * start_layer
	current_layer = start_layer
	for layer_idx in range(start_layer,end_layer+1):
		transformer = model.transformers[layer_idx]
		# Layer normalizations are performed before the layers respectively.
		if perturbations != None:
			for (perturb_layer, perturb_index, perturb_vec) in perturbations:
				if current_layer == perturb_layer:
					if perturb_index == -1:
						x[0:,:] = perturb_vec[0:,:]
					else:
						x[perturb_index,:] = perturb_vec
		if start_at_ff:
			layer_inputs.append(None)
			attn_inputs.append(None)
			v_outputs.append(None)
			attn_matrices.append(None)
			attn_linear_inputs.append(None)
			A_matrices.append(None)
			start_at_ff = False
		else:
			layer_inputs.append(x)
			a = transformer.ln_attn(x)
			attn_inputs.append(a)
			if frozen_ops != None and layer_idx < len(frozen_ops):
				attn_matrix = frozen_ops[layer_idx][0]
				if transformer.attn.proj_v:
					v = transformer.attn.proj_v(a)
				else:
					v = x
				attn_linear_input = torch.matmul(attn_matrix, v)
				if transformer.attn.linear:
					a = transformer.attn.linear(attn_linear_input)
				else:
					a = attn_linear_input
				pre_softmax, A = None, None
			elif transformer.attn.proj_v:
				a, pre_softmax, attn_matrix, v, attn_linear_input, A = compute_attention(current_layer, transformer.attn, a, mask)
			else:
				a, pre_softmax, attn_matrix, v, attn_linear_input, A = compute_attention(current_layer, transformer.attn, x, mask)
			v_outputs.append(v)
			attn_matrices.append(attn_matrix)
			attn_pre_softmax.append(pre_softmax)
			attn_linear_inputs.append(attn_linear_input)
			attn_outputs.append(a)
			A_matrices.append(A)
			x = x + a

		ff_inputs.append(x)
		if transformer.ff:
			if False and frozen_ops != None and layer_idx < len(frozen_ops):
				ff_mask = frozen_ops[layer_idx][1]
				x = x + transformer.ff[3](transformer.ff[2](ff_mask * transformer.ff[0](transformer.ln_ff(x))))
			else:
				x = x + transformer.ff(transformer.ln_ff(x))
			ff_params = {k:v for k,v in transformer.ff.named_parameters()}
			ff_parameters.append((ff_params['0.weight'].T, ff_params['3.weight'].T, ff_params['0.bias'], ff_params['3.bias']))
		else:
			ff_parameters.append((None, None, None, None))
		#print(x[-1,:])
		current_layer += 1
	layer_inputs.append(x)
	token_dim = model.token_embedding.shape[0]

	if model.ln_head:
		x = model.ln_head(x)
	if model.positional_embedding is not None:
		x = x[...,:-model.positional_embedding.shape[0]]
	if type(model.token_embedding) == TokenEmbedding:
		x = model.token_embedding(x, transposed=True)
	else:
		x = torch.matmul(x, model.token_embedding.transpose(0, 1))

	prediction = torch.argmax(x[...,-1,:token_dim], dim=-1).tolist()

	return layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction

@torch.no_grad()
def activation_patch_output_logit(model, j, layer_input, mask, prediction, attn_matrices):
	n = layer_input.size(0)

	frozen_ops = []
	for k in range(j):
		frozen_ops.append((attn_matrices[k], None))
	frozen_ops.append(None)

	attn_matrix = attn_matrices[j]
	zero_output_logits = torch.empty(n*n, device=layer_input.device)
	max_output_logits = torch.empty(n*n, device=layer_input.device)

	# try zeroing attn_matrix[a,b] for all a,b
	new_attn_matrices = attn_matrix.repeat(n*n,1,1).detach()
	diag_indices = torch.cat((torch.arange(0,n*n).unsqueeze(1),torch.cartesian_prod(torch.arange(0,n),torch.arange(0,n))),dim=1)
	new_attn_matrices[tuple(diag_indices.T)] = 0.0

	# perform forward pass on other_inputs
	BATCH_SIZE = 1024
	for start in range(0, n*n, BATCH_SIZE):
		end = min(start + BATCH_SIZE, n*n)
		frozen_ops[j] = (new_attn_matrices[start:end,], None)
		perturb_layer_inputs, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, layer_input.repeat(end-start,1,1).detach(), mask, j, False, len(model.transformers)-1, None, frozen_ops)

		# try computing the attention dot product but with perturbed dst embeddings
		zero_output_logits[start:end] = perturb_layer_inputs[-1][:,-1,prediction]
		del perturb_attn_inputs
		del perturb_layer_inputs
	zero_output_logits = zero_output_logits.reshape((n,n))

	# try maxing attn_matrix[a,b] for all a,b
	new_attn_matrices = attn_matrix.repeat(n*n,1,1).detach()
	new_attn_matrices[tuple(diag_indices.T)] = torch.max(attn_matrix,dim=1).values.repeat((1,n)).T.flatten()
	new_attn_matrices /= torch.sum(new_attn_matrices,dim=-1).unsqueeze(-1).repeat((1,1,n))

	# perform forward pass on other_inputs
	BATCH_SIZE = 1024
	for start in range(0, n*n, BATCH_SIZE):
		end = min(start + BATCH_SIZE, n*n)
		frozen_ops[j] = (new_attn_matrices[start:end,], None)
		perturb_layer_inputs, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, layer_input.repeat(end-start,1,1).detach(), mask, j, False, len(model.transformers)-1, None, frozen_ops)

		# try computing the attention dot product but with perturbed dst embeddings
		max_output_logits[start:end] = perturb_layer_inputs[-1][:,-1,prediction]
		del perturb_attn_inputs
		del perturb_layer_inputs
	max_output_logits = max_output_logits.reshape((n,n))

	return zero_output_logits, max_output_logits

@torch.no_grad()
def perturb_attn_ops(model, i, j, dst, src, layer_input, mask, A_matrices, attn_matrices, attn_inputs):
	A = A_matrices[i][:-1,:-1]
	n = attn_inputs[i].size(0)

	frozen_ops = []
	for k in range(j):
		frozen_ops.append((attn_matrices[k], None))
	frozen_ops.append(None)

	attn_matrix = attn_matrices[j]
	zero_src_products = torch.empty(n*n, device=layer_input.device)
	zero_dst_products = torch.empty(n*n, device=layer_input.device)
	max_src_products = torch.empty(n*n, device=layer_input.device)
	max_dst_products = torch.empty(n*n, device=layer_input.device)

	# try zeroing attn_matrix[a,b] for all a,b
	new_attn_matrices = attn_matrix.repeat(n*n,1,1).detach()
	diag_indices = torch.cat((torch.arange(0,n*n).unsqueeze(1),torch.cartesian_prod(torch.arange(0,n),torch.arange(0,n))),dim=1)
	new_attn_matrices[tuple(diag_indices.T)] = 0.0

	# perform forward pass on other_inputs
	BATCH_SIZE = 1024
	for start in range(0, n*n, BATCH_SIZE):
		end = min(start + BATCH_SIZE, n*n)
		frozen_ops[j] = (new_attn_matrices[start:end,], None)
		_, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, layer_input.repeat(end-start,1,1).detach(), mask, j, False, i, None, frozen_ops)

		# try computing the attention dot product but with perturbed dst embeddings
		zero_src_products[start:end] = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))[:,dst,src]
		zero_dst_products[start:end] = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))[:,dst,src]
		del perturb_attn_inputs
	zero_src_products = zero_src_products.reshape((n,n))
	zero_dst_products = zero_dst_products.reshape((n,n))

	# try maxing attn_matrix[a,b] for all a,b
	new_attn_matrices = attn_matrix.repeat(n*n,1,1).detach()
	new_attn_matrices[tuple(diag_indices.T)] = torch.max(attn_matrix,dim=1).values.repeat((1,n)).T.flatten()
	new_attn_matrices /= torch.sum(new_attn_matrices,dim=-1).unsqueeze(-1).repeat((1,1,n))

	# perform forward pass on other_inputs
	BATCH_SIZE = 1024
	for start in range(0, n*n, BATCH_SIZE):
		end = min(start + BATCH_SIZE, n*n)
		frozen_ops[j] = (new_attn_matrices[start:end,], None)
		_, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, layer_input.repeat(end-start,1,1).detach(), mask, j, False, i, None, frozen_ops)

		# try computing the attention dot product but with perturbed dst embeddings
		max_src_products[start:end] = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))[:,dst,src]
		max_dst_products[start:end] = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))[:,dst,src]
		del perturb_attn_inputs
	max_src_products = max_src_products.reshape((n,n))
	max_dst_products = max_dst_products.reshape((n,n))

	return zero_src_products, zero_dst_products, max_src_products, max_dst_products

@torch.no_grad()
def perturb_residuals(model, i, j, dst, src, mask, A_matrices, attn_inputs, attn_outputs, ff_inputs):
	A = A_matrices[i][:-1,:-1]
	n = attn_inputs[i].size(0)
	d = attn_inputs[i].size(1)

	res_src_products = torch.empty(n, device=attn_inputs[i].device)
	res_dst_products = torch.empty(n, device=attn_inputs[i].device)

	# try zeroing the residual of token k, for all k
	new_ff_input = ff_inputs[j].repeat(n,1,1).detach()
	diag_indices = torch.cartesian_prod(torch.arange(0,n),torch.arange(0,d))
	diag_indices = torch.cat((diag_indices[:,0].unsqueeze(1),diag_indices),dim=-1)
	new_ff_input[tuple(diag_indices.T)] = attn_outputs[j].reshape(n*d)

	# perform forward pass on other_inputs
	_, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, new_ff_input.detach(), mask, j, True, i, None, None)
	res_src_products = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))[:,dst,src]
	res_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))[:,dst,src]
	del perturb_attn_inputs

	return res_src_products, res_dst_products

class NoUnusedVertexIDs(Exception):
    pass

@torch.no_grad()
def explain_attn_op(model, input, mask, A_matrices, attn_matrices, attn_inputs, ff_inputs, i, dst, src):
	n, d = attn_inputs[0].size(0), attn_inputs[0].size(1)
	EDGE_PREFIX_TOKEN = (n - 5) // 3 + 2
	edge_indices = [i + 1 for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]

	# create perturbed inputs where each input replaces every occurrence of a token value with an unused token value
	max_vertex_id = (n - 5) // 3
	try:
		unused_id = next(t for t in range(1, max_vertex_id + 5) if t not in input)
	except StopIteration:
		raise NoUnusedVertexIDs()
	other_inputs = input.repeat(max_vertex_id + 5 + n, 1)
	for j in range(max_vertex_id + 5):
		other_inputs[j,input == j] = unused_id

	# perform forward pass on other_inputs
	other_inputs = model.token_embedding[other_inputs]
	if len(other_inputs.shape) == 2:
		pos = model.positional_embedding
	else:
		pos = model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0:-2] + (-1,-1))
	other_inputs = torch.cat((other_inputs, pos), -1)
	other_inputs = model.dropout_embedding(other_inputs)

	# for the last n rows of `other_inputs`, perturb the position embeddings
	for j in range(n):
		#if j < edge_indices[-1] or j > edge_indices[-1] + 2:
		#	src_edge_index = edge_indices[-1]
		#else:
		#	src_edge_index = edge_indices[-2]
		#offset = (src_edge_index % 3) - (j % 3)
		#if offset > 1:
		#	offset -= 3
		other_inputs[max_vertex_id+5+j,j,d-n:] = 0 #other_inputs[max_vertex_id+5+j,src_edge_index-offset,d-n:].clone().detach()

	frozen_ops = []
	for j in range(i):
		frozen_ops.append((attn_matrices[j], model.transformers[j].ff[0](model.transformers[j].ln_ff(ff_inputs[j])) > 0.0))

	_, perturb_attn_inputs, _, _, _, _, _, _, _, _, _ = forward(model, other_inputs, mask, 0, False, i, None, frozen_ops)

	# try computing the attention dot product but with perturbed dst embeddings
	A = A_matrices[i][:-1,:-1]
	old_products = torch.matmul(torch.matmul(attn_inputs[i], A), attn_inputs[i].T)
	old_product = old_products[dst, src]
	new_src_products = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))
	new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))
	new_src_products = new_src_products[:,dst,src]
	new_dst_products = new_dst_products[:,dst,src]

	is_negative_copy = (attn_matrices[i][dst,src] < torch.median(attn_matrices[i][dst,edge_indices]))
	if is_negative_copy:
		src_dependencies = torch.nonzero(new_src_products > old_product + math.sqrt(d) / 3)
		dst_dependencies = torch.nonzero(new_dst_products > old_product + math.sqrt(d) / 3)
		src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=True).indices]
		dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=True).indices]
	else:
		src_dependencies = torch.nonzero(new_src_products < old_product - math.sqrt(d) / 3)
		dst_dependencies = torch.nonzero(new_dst_products < old_product - math.sqrt(d) / 3)
		src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=False).indices]
		dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=False).indices]

	return src_dependencies, dst_dependencies

class TransformerProber(nn.Module):
	def __init__(self, tfm_model, probe_layer):
		super().__init__()
		d = tfm_model.ln_head.normalized_shape[0]
		self.model = tfm_model
		for param in tfm_model.parameters():
			param.requires_grad = False
		self.probe_layer = probe_layer
		n = tfm_model.positional_embedding.size(0)
		self.A_ops = [nn.parameter.Parameter(torch.empty((d,d)))]
		if probe_layer > len(tfm_model.transformers):
			raise Exception('probe_layer must be <= number of layers')

	def reset_parameters(self) -> None:
		for A_op in self.A_ops:
			nn.init.kaiming_uniform_(A_op, a=math.sqrt(5))

	def to(self, device):
		super().to(device)
		for A_op in self.A_ops:
			A_op.to(device)

	def forward(self, x: torch.Tensor):
		d = self.model.ln_head.normalized_shape[0]
		n = self.model.positional_embedding.size(0)
		QUERY_PREFIX_TOKEN = (n - 5) // 3 + 4
		PADDING_TOKEN = (n - 5) // 3 + 3
		EDGE_PREFIX_TOKEN = (n - 5) // 3 + 2
		PATH_PREFIX_TOKEN = (n - 5) // 3 + 1

		other_inputs = x.repeat((3*n+1,1))

		# create perturbed inputs where each input replaces every occurrence of a token value with an unused token value
		max_vertex_id = (n - 5) // 3
		unused_id = next(t for t in range(1, max_vertex_id + 1) if t not in x)
		for j in range(max_vertex_id + 1):
			other_inputs[j,x == j] = unused_id

		# create perturbed inputs where each input swaps the position of one token
		other_inputs = self.model.token_embedding[other_inputs]
		if len(other_inputs.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0], -1, -1)
		other_inputs = torch.cat((other_inputs, pos), -1)
		other_inputs = self.model.dropout_embedding(other_inputs)

		# perturb the position embeddings
		edge_indices = [i + 1 for i in range(len(x)) if x[i] == EDGE_PREFIX_TOKEN]
		for j in range(n):
			if x[j] in (PADDING_TOKEN, QUERY_PREFIX_TOKEN, PATH_PREFIX_TOKEN):
				continue
			if j < edge_indices[-1] or j > edge_indices[-1] + 2:
				src_edge_index = edge_indices[-1]
			else:
				src_edge_index = edge_indices[-2]
			offset = (src_edge_index % 3) - (j % 3)
			if offset > 1:
				offset -= 3
			other_inputs[n+j,j,:d-n] = 0
			other_inputs[n+j,j,unused_id] = 1
			other_inputs[2*n+j,j,d-n:] = other_inputs[n+j,src_edge_index-offset,d-n:].clone().detach()

		# Create masking tensor.
		mask = self.model.pad_masking(x, 0)
		if not self.model.bidirectional:
			mask = mask + self.model.future_masking(x, 0)

		# perform forward pass on original input
		x = self.model.token_embedding[x]
		if len(x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
		x = torch.cat((x, pos), -1)
		x = self.model.dropout_embedding(x)
		_,attn_inputs,_,_,_,_,_,A_matrices,_,_,_ = forward(self.model, x, mask, 0, False, self.probe_layer, None, None)

		# perform forward pass on other_inputs
		_,perturb_attn_inputs,_,_,_,_,_,_,_,_,_ = forward(self.model, other_inputs, mask, 0, False, self.probe_layer, None, None)

		y = x.clone().detach()
		perturb_y = other_inputs.clone().detach()
		mixed_A_predictions = []
		mixed_A_labels = []
		for i in range(self.probe_layer + 1):
			left_mixed_a = torch.matmul(torch.matmul(perturb_y, self.A_ops[i]), y.transpose(-2,-1))
			right_mixed_a = torch.matmul(torch.matmul(y, self.A_ops[i]), perturb_y.transpose(-2,-1))
			mixed_A_predictions.append((left_mixed_a,right_mixed_a))

			A = A_matrices[i][:-1,:-1]
			left_mixed_a_label = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].T)
			right_mixed_a_label = torch.matmul(torch.matmul(attn_inputs[i], A), torch.permute(perturb_attn_inputs[i],(0,2,1)))
			mixed_A_labels.append((left_mixed_a_label,right_mixed_a_label))
			break

			a = torch.matmul(torch.matmul(y, self.A_ops[i]), y.transpose(-2,-1))
			if mask is not None:
				a += mask.type_as(a) * a.new_tensor(-1e9)
			a = a.softmax(-1)
			y += torch.matmul(a, y)

			perturb_a = torch.matmul(torch.matmul(perturb_y, self.A_ops[i]), perturb_y.transpose(-2,-1))
			if mask is not None:
				perturb_a += mask.type_as(perturb_a) * perturb_a.new_tensor(-1e9)
			perturb_a = perturb_a.softmax(-1)
			perturb_y += torch.matmul(perturb_a, perturb_y)

		return mixed_A_predictions, mixed_A_labels

class AlgorithmCase:
	def __init__(self, conditions, then, inputs):
		# `conditions` is a list of tuples (x_i,y_i,z_i,u_i,v_i,w_i), representing a conjunction of conditions checking if x_i[y_i] is large if `z_i == True`, or x_i[y_i] is small if `z_i == False`, and u_i[v_i] is large if `w_i == True`, or u_i[v_i] is small if `w_i == False`
		self.conditions = conditions
		# `then` is a list of tuples (s_i,d_i) representing a sequence of copy operations, where row s_i from the input is copied into row d_i of the output
		self.then = then
		self.inputs = inputs

		# normalize the list of conditions and then
		normalize_conditions(self.conditions)

	def __str__(self):
		s = 'if'
		first = True
		for (x,y,z,u,v,w) in self.conditions:
			if not first:
				s += ' and'
			first = False
			if z == True:
				s += ' (x[{},{}] is large'.format(x,y)
			else:
				s += ' (x[{},{}] is small'.format(x,y)
			if w == True:
				s += ' and x[{},{}] is large)'.format(u,v)
			else:
				s += ' and x[{},{}] is small)'.format(u,v)
		s += ':'
		for (src,dst) in self.then:
			s += '\n  copy x[{},:] into x[{},:]'.format(src,dst)
		return s

class AlgorithmStep:
	def __init__(self):
		self.cases = []
		self.residual_copies = []
		self.token_operations = {}
		self.position_operations = {}
		self.token_operation_inputs = {}
		self.position_operation_inputs = {}

	def add_case(self, conditions, then_instruction, input):
		normalize_conditions(conditions)
		for case in self.cases:
			if conditions == case.conditions:
				if then_instruction not in case.then:
					case.then.append(then_instruction)
				case.inputs.append(input)
				return
		self.cases.append(AlgorithmCase(conditions, [then_instruction], [input]))

	def add_residual_copy(self, row):
		if row not in self.residual_copies:
			self.residual_copies.append(row)

	def identify_operations(self, embedding_dim, token_dim, position_dim):
		self.token_operations = {}
		self.position_operations = {}
		self.token_operation_inputs = {}
		self.position_operation_inputs = {}
		for case in self.cases:
			for (x,y,z,u,v,w) in case.conditions:
				diff = y - v
				if y < token_dim and v < token_dim:
					if diff not in self.token_operations:
						self.token_operations[diff] = []
						self.token_operation_inputs[diff] = []
					if y not in self.token_operations[diff]:
						self.token_operations[diff].append(y)
					self.token_operation_inputs[diff].extend(case.inputs)
				elif y >= embedding_dim - position_dim and v >= embedding_dim - position_dim:
					if diff not in self.position_operations:
						self.position_operations[diff] = []
						self.position_operation_inputs[diff] = []
					if y not in self.position_operations[diff]:
						self.position_operations[diff].append(y)
					self.position_operation_inputs[diff].extend(case.inputs)


class ComputationNode:
	def __init__(self, layer, row_id):
		self.layer = layer
		self.row_id = row_id
		self.predecessors = []
		self.successors = []
		self.reachable = []
		self.op_explanations = []
		self.copy_directions = []

	def add_predecessor(self, predecessor):
		self.predecessors.append(predecessor)
		predecessor.successors.append(self)
		predecessor.op_explanations.append(None)
		predecessor.copy_directions.append(None)

	def __str__(self):
		return '{{layer:{},id:{}}}'.format(self.layer, self.row_id)

	def __repr__(self):
		return '{{layer:{},id:{}}}'.format(self.layer, self.row_id)

class TransformerTracer:
	def __init__(self, tfm_model):
		self.model = tfm_model
		self.model.eval()
		self.algorithm = [AlgorithmStep() for i in range(len(self.model.transformers))]

	'''def input_derivatives(self, x: torch.Tensor, mask: torch.Tensor, dx: float, start_layer: int, start_at_ff: bool, old_prediction: int, last_layer_output: torch.Tensor):
		dfdx = torch.empty(x.shape)
		for i in range(x.size(0)):
			for j in range(x.size(1)):
				new_x = x.clone().detach()
				new_x[i,j] += dx
				layer_inputs, _, _, _, _, _, _, _, _, _, _ = forward(self.model, new_x, mask, start_layer, start_at_ff)
				dfdx[i,j] = (layer_inputs[-1][-1,old_prediction] - last_layer_output[-1,old_prediction]) / dx
		return dfdx'''

	def input_derivative(self, start_layer: int, input_row_id: int, input_col_id: int, output_row_id: int, output_col_id: int, start_at_ff: bool, layer_inputs: torch.Tensor, attn_inputs: torch.Tensor, attn_pre_softmax: torch.Tensor, attn_matrices: torch.Tensor, ff_inputs: torch.Tensor):
		dydx = torch.zeros(layer_inputs[start_layer].shape)
		dydx[input_row_id,input_col_id] = 1.0

		for i in range(start_layer, len(self.model.transformers)):
			if not (i == start_layer and start_at_ff):
				mu = torch.mean(layer_inputs[i], dim=1).unsqueeze(1)
				dmudx = torch.mean(dydx, dim=1).unsqueeze(1)
				var = (self.model.transformers[i].ln_attn.eps + torch.var(layer_inputs[i], dim=1, correction=0)).unsqueeze(1)
				dybardx = (mu - layer_inputs[i])*torch.mean((layer_inputs[i] - mu)*(dydx - dmudx), dim=1).unsqueeze(1)/var + dydx - dmudx
				dybardx /= torch.sqrt(var)
				dybardx *= self.model.transformers[i].ln_attn.weight

				left = torch.matmul(self.model.transformers[i].attn.proj_q(attn_inputs[i]), torch.matmul(dybardx, self.model.transformers[i].attn.proj_k.weight.T).T)
				right = torch.matmul(torch.matmul(dybardx, self.model.transformers[i].attn.proj_q.weight.T), self.model.transformers[i].attn.proj_k(attn_inputs[i]).T)
				dadx = (left + right) / math.sqrt(attn_inputs[i].size(-1))

				shift = -torch.max(attn_pre_softmax[i], dim=1).values.unsqueeze(1)
				numerator = torch.sum(dadx * torch.exp(attn_pre_softmax[i] + shift), dim=1)
				denominator = torch.sum(torch.exp(attn_pre_softmax[i] + shift), dim=1)
				dsdx = attn_matrices[i] * (dadx - numerator / denominator)

				left = torch.matmul(attn_matrices[i], torch.matmul(dybardx, self.model.transformers[i].attn.proj_v.weight.T))
				right = torch.matmul(dsdx, self.model.transformers[i].attn.proj_v(attn_inputs[i]))
				dfdx = dydx + torch.matmul(left + right, self.model.transformers[i].attn.linear.weight.T)

				dydx = dfdx

			if self.model.transformers[i].ff:
				mu = torch.mean(layer_inputs[i], dim=1).unsqueeze(1)
				dmudx = torch.mean(dydx, dim=1).unsqueeze(1)
				var = (self.model.transformers[i].ln_ff.eps + torch.var(layer_inputs[i], dim=1, correction=0)).unsqueeze(1)
				dybardx = (mu - layer_inputs[i])*torch.mean((layer_inputs[i] - mu)*(dydx - dmudx), dim=1).unsqueeze(1)/var + dydx - dmudx
				dybardx /= torch.sqrt(var)
				dybardx *= self.model.transformers[i].ln_ff.weight

				mask = self.model.transformers[i].ff[0](self.model.transformers[i].ln_ff(ff_inputs[i])) > 0.0
				dfdx = dydx + torch.matmul(mask * torch.matmul(dybardx, self.model.transformers[i].ff[0].weight.T), self.model.transformers[i].ff[3].weight.T)

				dydx = dfdx

		return dydx[output_row_id, output_col_id]

	def input_derivatives(self, start_layer: int, input_row_id: int, output_row_id: int, output_col_id: int, start_at_ff: bool, layer_inputs: torch.Tensor, attn_inputs: torch.Tensor, attn_pre_softmax: torch.Tensor, attn_matrices: torch.Tensor, ff_inputs: torch.Tensor):
		out = torch.empty(layer_inputs[start_layer].size(1))
		for i in range(layer_inputs[start_layer].size(1)):
			out[i] = self.input_derivative(start_layer, input_row_id, i, output_row_id, output_col_id, start_at_ff, layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, ff_inputs)
		return out

	def trace2(self, x: torch.Tensor):
		n = x.shape[0]
		max_vertex_id = (n - 5) // 3
		QUERY_PREFIX_TOKEN = (n - 5) // 3 + 4
		PADDING_TOKEN = (n - 5) // 3 + 3
		EDGE_PREFIX_TOKEN = (n - 5) // 3 + 2
		PATH_PREFIX_TOKEN = (n - 5) // 3 + 1
		# Create masking tensor.
		mask = self.model.pad_masking(x, 0)
		if not self.model.bidirectional:
			mask = mask + self.model.future_masking(x, 0)

		# Use token embedding and positional embedding layers.
		input = x # store the input for keeping track the code paths executed by each input
		x = self.model.token_embedding[x]
		if len(x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
		x = torch.cat((x, pos), -1)
		x = self.model.dropout_embedding(x)
		d = x.shape[1]

		layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction = forward(self.model, x, mask, 0, False, len(self.model.transformers)-1, None, None)

		zero_output_logit_list = []
		max_output_logit_list = []
		for j in range(len(self.model.transformers)):
			zero_output_logits, max_output_logits = activation_patch_output_logit(self.model, j, layer_inputs[j], mask, prediction, attn_matrices)
			zero_output_logit_list.append(zero_output_logits)
			max_output_logit_list.append(max_output_logits)

		major_last_copies = torch.nonzero(zero_output_logit_list[len(self.model.transformers)-1] < torch.min(zero_output_logit_list[len(self.model.transformers)-1]) + 0.001)
		if major_last_copies.size(0) != 1 or major_last_copies[0,0] != n-1:
			print('WARNING: Last layer does not copy the answer from a single source token.')
			import pdb; pdb.set_trace()
			return None, None, None, None, prediction
		last_copy_src = major_last_copies[0,1]

		zero_src_product_list = []
		zero_dst_product_list = []
		max_src_product_list = []
		max_dst_product_list = []
		res_src_product_list = []
		res_dst_product_list = []
		for j in range(len(self.model.transformers)):
			zero_src_products, zero_dst_products, max_src_products, max_dst_products = perturb_attn_ops(self.model, len(self.model.transformers)-1, j, n-1, last_copy_src, layer_inputs[j], mask, A_matrices, attn_matrices, attn_inputs)
			zero_src_product_list.append(zero_src_products)
			zero_dst_product_list.append(zero_dst_products)
			max_src_product_list.append(max_src_products)
			max_dst_product_list.append(max_dst_products)

			#if j < len(self.model.transformers) - 1:
			#	res_src_products, res_dst_products = perturb_residuals(self.model, len(self.model.transformers)-1, j, n-1, last_copy_src, mask, A_matrices, attn_inputs, attn_outputs, ff_inputs)
			#	res_src_product_list.append(res_src_products)
			#	res_dst_product_list.append(res_dst_products)

		A = A_matrices[len(self.model.transformers)-1][:-1,:-1]
		old_products = torch.matmul(torch.matmul(attn_inputs[len(self.model.transformers)-1], A), attn_inputs[len(self.model.transformers)-1].T)
		old_product = old_products[n-1,last_copy_src]

		important_ops = []
		for j in range(len(self.model.transformers)):
			new_important_ops = []
			positive_copies = torch.nonzero(zero_output_logit_list[j] < layer_inputs[-1][-1,prediction] - math.sqrt(d) / 2)
			negative_copies = torch.nonzero(max_output_logit_list[j] < layer_inputs[-1][-1,prediction] - math.sqrt(d) / 2)
			if positive_copies.size(0) > negative_copies.size(0) or positive_copies.size(0) == 0:
				new_important_ops += [op.tolist() for op in negative_copies]
			if positive_copies.size(0) <= negative_copies.size(0) or negative_copies.size(0) == 0:
				new_important_ops += [op.tolist() for op in positive_copies]

			positive_copies = torch.nonzero(zero_src_product_list[j] < old_product - math.sqrt(d) / 2)
			negative_copies = torch.nonzero(max_src_product_list[j] < old_product - math.sqrt(d) / 2)
			if positive_copies.size(0) > negative_copies.size(0):
				new_important_ops += [c.tolist() for c in negative_copies if c.tolist() not in new_important_ops]
			else:
				new_important_ops += [c.tolist() for c in positive_copies if c.tolist() not in new_important_ops]

			positive_copies = torch.nonzero(zero_dst_product_list[j] < old_product - math.sqrt(d) / 2)
			negative_copies = torch.nonzero(max_dst_product_list[j] < old_product - math.sqrt(d) / 2)
			if positive_copies.size(0) > negative_copies.size(0):
				new_important_ops += [c.tolist() for c in negative_copies if c.tolist() not in new_important_ops]
			else:
				new_important_ops += [c.tolist() for c in positive_copies if c.tolist() not in new_important_ops]

			#if j < len(self.model.transformers) - 1:
			#	residual_copies = torch.nonzero(res_src_product_list[j] < old_product - math.sqrt(d) / 2)
			#	residual_copies = torch.cat((residual_copies,residual_copies),dim=-1)
			#	new_important_ops += [c.tolist() for c in residual_copies if c.tolist() not in new_important_ops]
			#	residual_copies = torch.nonzero(res_dst_product_list[j] < old_product - math.sqrt(d) / 2)
			#	residual_copies = torch.cat((residual_copies,residual_copies),dim=-1)
			#	new_important_ops += [c.tolist() for c in residual_copies if c.tolist() not in new_important_ops]

			important_ops.append(new_important_ops)

		forward_edges = []
		start_index = torch.sum(input == PADDING_TOKEN)
		for i in range((n - 5) // 3 + 1):
			forward_edges.append([])
		for i in range((n + 2) % 3, n-5, 3):
			if i >= start_index:
				forward_edges[input[i].item()].append(input[i+1].item())
		def path_length(start, end):
			if input[start] > (n - 5) // 3 or input[end] > (n - 5) // 3:
				return -1
			queue = [(input[start],0)]
			best_distance = -1
			while len(queue) != 0:
				current, distance = queue.pop()
				if current == input[end]:
					if best_distance == -1 or distance < best_distance:
						best_distance = distance
					continue
				for child in forward_edges[current]:
					queue.append((child,distance+1))
			return best_distance

		# reconstruct the circuit pathway for this input
		root = ComputationNode(len(self.model.transformers), n-1)
		nodes = [root]
		all_nodes = [root]
		for j in reversed(range(len(self.model.transformers))):
			# identify the operations whose destination is in relevant_indices
			new_nodes = []
			for node in nodes:
				new_node = ComputationNode(j, node.row_id)
				node.add_predecessor(new_node)
				new_nodes.append(new_node)
			for op in important_ops[j]:
				for node in nodes:
					if node.row_id == op[0]:
						try:
							predecessor = next(n for n in new_nodes if n.row_id == op[1])
						except StopIteration:
							predecessor = ComputationNode(j, int(op[1]))
							new_nodes.append(predecessor)
						node.add_predecessor(predecessor)

						# explain the copy from `predecessor` to `node`
						src_dependencies, dst_dependencies = explain_attn_op(self.model, input, mask, A_matrices, attn_matrices, attn_inputs, ff_inputs, predecessor.layer, node.row_id, predecessor.row_id)
						op_causes = []
						for src_dep in src_dependencies:
							if src_dep >= max_vertex_id+5 and src_dep+1 in dst_dependencies:
								# this is a position-based backward step
								if (int(src_dep)+1,int(src_dep)) not in op_causes:
									op_causes.append((int(src_dep)+1,int(src_dep)))
							elif src_dep >= max_vertex_id+5 and src_dep-1 in dst_dependencies:
								# this is a position-based forward step
								if (int(src_dep)-1,int(src_dep)) not in op_causes:
									op_causes.append((int(src_dep)-1,int(src_dep)))
							elif src_dep <= max_vertex_id and src_dep in dst_dependencies:
								# this is a token matching step
								if int(src_dep) not in op_causes:
									op_causes.append(int(src_dep))
						if max_vertex_id+5+n-4 in src_dependencies and max_vertex_id+5+n-3 in src_dependencies and max_vertex_id+5+n-1 in dst_dependencies:
							# this is a hard-coded copy from tokens that are reachable from both the start and goal vertices into the last token
							if (max_vertex_id+5+n-1, (max_vertex_id+5+n-3,max_vertex_id+5+n-4)) not in op_causes:
								op_causes.append((max_vertex_id+5+n-1, (max_vertex_id+5+n-3,max_vertex_id+5+n-4)))
						if max_vertex_id+5+n-3 in src_dependencies:
							# this is a hard-coded copy from tokens that are reachable from the goal vertex
							if (None, max_vertex_id+5+n-3) not in op_causes:
								op_causes.append((None, max_vertex_id+5+n-3))
						forward_dist = path_length(predecessor.row_id, node.row_id)
						backward_dist = path_length(node.row_id, predecessor.row_id)
						if node.row_id == predecessor.row_id:
							copy_direction = None
						elif forward_dist != -1:
							if backward_dist != -1:
								# this could be an effective forwards or backwards copy
								copy_direction = '*'
							else:
								# this is an effective backward step
								copy_direction = 'b'
						else:
							if backward_dist != -1:
								# this is an effective forward step
								copy_direction = 'f'
							else:
								copy_direction = None

						index = predecessor.successors.index(node)
						predecessor.op_explanations[index] = op_causes
						predecessor.copy_directions[index] = copy_direction
			nodes = new_nodes
			all_nodes += new_nodes

		for node in all_nodes:
			node.reachable = [int(input[node.row_id]),max_vertex_id+5+node.row_id]
		for node in reversed(all_nodes):
			for successor in node.successors:
				successor.reachable += [n for n in node.reachable if n not in successor.reachable]

		# identify the paths in the graph that can be explained by the path-merging algorithm
		path_merge_explainable = []
		for node in reversed(all_nodes):
			if node.layer == 0 and node.row_id != n-1:
				path_merge_explainable.append(node)
			elif node not in path_merge_explainable:
				continue

			# check if this is a path merge operation
			for k in range(len(node.successors)):
				successor = node.successors[k]
				if successor in path_merge_explainable:
					continue
				if node.op_explanations[k] == None:
					if node.row_id == successor.row_id:
						path_merge_explainable.append(successor)
					continue
				if node.copy_directions[k] == 'f' and any(e[0]+1 == e[1] and e[0]+1 in successor.reachable and e[1] in node.reachable for e in node.op_explanations[k] if type(e) == tuple and type(e[0]) == int):
					path_merge_explainable.append(successor)
					continue
				if node.copy_directions[k] == 'b' and any(e[0]-1 == e[1] and e[0]-1 in successor.reachable and e[1] in node.reachable for e in node.op_explanations[k] if type(e) == tuple and type(e[0]) == int):
					path_merge_explainable.append(successor)
					continue
				for explanation in node.op_explanations[k]:
					if any(explanation == input[r-(max_vertex_id+5)] for r in node.reachable if r >= max_vertex_id+5):
						path_merge_explainable.append(successor)
						break
				if (max_vertex_id+5+n-1,(max_vertex_id+5+n-3,max_vertex_id+5+n-4)) in node.op_explanations[k] and successor.row_id == n-1 and max_vertex_id+5+n-3 in node.reachable and max_vertex_id+5+n-4 in node.reachable and successor not in path_merge_explainable:
					path_merge_explainable.append(successor)
				if (None,max_vertex_id+5+n-3) in node.op_explanations[k] and successor.row_id == n-1 and max_vertex_id+5+n-3 in node.reachable and successor not in path_merge_explainable:
					path_merge_explainable.append(successor)

		return root, forward_edges, important_ops, path_merge_explainable, prediction

	def trace(self, x: torch.Tensor, other_x: torch.Tensor, quiet: bool = True):
		n = x.shape[0]
		QUERY_PREFIX_TOKEN = (n - 5) // 3 + 4
		PADDING_TOKEN = (n - 5) // 3 + 3
		EDGE_PREFIX_TOKEN = (n - 5) // 3 + 2
		PATH_PREFIX_TOKEN = (n - 5) // 3 + 1
		# Create masking tensor.
		mask = self.model.pad_masking(x, 0)
		if not self.model.bidirectional:
			mask = mask + self.model.future_masking(x, 0)

		# Use token embedding and positional embedding layers.
		input = x # store the input for keeping track the code paths executed by each input
		x = self.model.token_embedding[x]
		if len(x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
		x = torch.cat((x, pos), -1)
		x = self.model.dropout_embedding(x)
		d = x.shape[1]

		# Use token embedding and positional embedding layers.
		other_input = other_x # store the input for keeping track the code paths executed by each input
		other_x = self.model.token_embedding[other_x]
		if len(other_x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(other_x.shape[0], -1, -1)
		other_x = torch.cat((other_x, pos), -1)
		other_x = self.model.dropout_embedding(other_x)

		other_layer_inputs, other_attn_inputs, other_attn_pre_softmax, other_attn_matrices, other_v_outputs, other_attn_linear_inputs, other_attn_outputs, other_A_matrices, other_ff_inputs, other_ff_parameters, other_prediction = forward(self.model, other_x, mask, 0, False, len(self.model.transformers)-1, None, None)

		layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction = forward(self.model, x, mask, 0, False, len(self.model.transformers)-1, None, None)
		#[(3,65,other_layer_inputs[3][65,:]),(3,68,other_layer_inputs[3][68,:])]
		#[(4,23,other_layer_inputs[4][23,:]),(4,20,other_layer_inputs[4][20,:])]
		#[(5,89,other_layer_inputs[5][89,:])]
		#[(5,35,other_layer_inputs[5][35,:]),(5,38,other_layer_inputs[5][38,:])]
		#[(3,14,other_layer_inputs[3][14,:]),(3,71,other_layer_inputs[3][71,:])]
		#[(4,14,other_layer_inputs[4][14,:]),(4,41,other_layer_inputs[4][41,:]),(4,71,other_layer_inputs[4][71,:])]
		#[(3,29,other_layer_inputs[3][29,:]),(3,32,other_layer_inputs[3][32,:])]

		if not quiet:
			print("Model prediction: {}".format(prediction))
			print("Model prediction for other input: {}".format(other_prediction))

		def check_copy(i, dst, src, attn_inputs, attn_matrices):
			attn_input = attn_inputs[i]
			if not quiet:
				print('Attention layer {} is copying row {} into row {} with weight {} because:'.format(i,src,dst,attn_matrices[i][dst,src]))
			# determine why row `src` is being copied from
			attn_input_prime = torch.cat((attn_input, torch.ones((n,1))), 1)
			right_products = torch.matmul(attn_input_prime[dst,:], A_matrices[i]) * attn_input_prime[src,:]
			for right_index in torch.nonzero(right_products[:-1] > torch.max(right_products[:-1]) - 5.0).tolist():
				right_index = right_index[0]
				if not quiet:
					print('  Row {} at index {} has value {}'.format(src, right_index, attn_input_prime[src,right_index]))
				left_products = attn_input_prime[dst,:] * A_matrices[i][:,right_index].reshape(1,-1)[0]
				if attn_input_prime[src,right_index] > 0.0:
					left_indices = torch.nonzero(left_products > torch.max(left_products) - 5.0).tolist()
				else:
					left_indices = torch.nonzero(left_products < torch.min(left_products) - 5.0).tolist()
				for left_index in left_indices:
					left_index = left_index[0]
					if not quiet:
						print('  Row {} at index {} has value {}, and A[{},{}]={}'.format(dst, left_index, attn_input_prime[dst,left_index], left_index, right_index, A_matrices[i][left_index,right_index]))
			left_products = attn_input_prime[dst,:] * torch.matmul(A_matrices[i], attn_input_prime[src,:])
			for left_index in torch.nonzero(left_products[:-1] > torch.max(left_products[:-1]) - 5.0).tolist():
				left_index = left_index[0]
				if not quiet:
					print('  Row {} at index {} has value {}'.format(dst, left_index, attn_input_prime[dst,left_index]))

		def check_copy_contrastive(i, dst, src, other_src, attn_inputs, other_attn_inputs, attn_matrices):
			attn_input = attn_inputs[i]
			other_attn_input = other_attn_inputs[i]
			if not quiet:
				print('Attention layer {} is copying row {} into row {} with weight {} because:'.format(i,src,dst,attn_matrices[i][dst,src]))
			# determine why row `src` is being copied from
			attn_input_prime = torch.cat((attn_input, torch.ones((n,1))), 1)
			other_attn_input_prime = torch.cat((other_attn_input, torch.ones((n,1))), 1)
			right_products = torch.matmul(attn_input_prime[dst,:], A_matrices[i]) * (attn_input_prime[src,:] - other_attn_input_prime[other_src,:])
			import pdb; pdb.set_trace()
			for right_index in torch.nonzero(right_products[:-1] > torch.max(right_products[:-1]) - 2.0).tolist():
				right_index = right_index[0]
				if not quiet:
					print('  Row {} at index {} has value {}'.format(src, right_index, attn_input_prime[src,right_index]))
				left_products = attn_input_prime[dst,:] * A_matrices[i][:,right_index].reshape(1,-1)[0]
				if attn_input_prime[src,right_index] > 0.0:
					left_indices = torch.nonzero(left_products > torch.max(left_products) - 1.0).tolist()
				else:
					left_indices = torch.nonzero(left_products < torch.min(left_products) + 1.0).tolist()
				for left_index in left_indices:
					left_index = left_index[0]
					if not quiet:
						print('  Row {} at index {} has value {}, and A[{},{}]={}'.format(dst, left_index, attn_input_prime[dst,left_index], left_index, right_index, A_matrices[i][left_index,right_index]))
			left_products = attn_input_prime[dst,:] * torch.matmul(A_matrices[i], attn_input_prime[src,:] - other_attn_input_prime[other_src,:])
			for left_index in torch.nonzero(left_products[:-1] > torch.max(left_products[:-1]) - 2.0).tolist():
				left_index = left_index[0]
				if not quiet:
					print('  Row {} at index {} has value {}'.format(dst, left_index, attn_input_prime[dst,left_index]))

		def one_hot(n, k):
			v = torch.zeros(n)
			v[k] = 1.0
			return v

		def trace_activation(i, row, representations):
			# first undo the pre-attention layer norm
			vec = torch.zeros(d)
			for element, activation in representations:
				vec[element] = activation
			vec_layer_input = ((vec - self.model.transformers[i].ln_attn.bias) / self.model.transformers[i].ln_attn.weight) * torch.sqrt(torch.var(layer_inputs[i][row,:], correction=0) + self.model.transformers[i].ln_attn.eps) + torch.mean(layer_inputs[i][row,:])
			for j in range(len(representations)):
				element, _ = representations[j]
				representations[j] = element, vec_layer_input[element]

			# check to see how much the FF layer contributes to the representation
			ff_indices = []
			residual_indices = []
			for element, activation in representations:
				ff_output = layer_inputs[i][row,element] - ff_inputs[i-1][row,element]
				if activation > 0.5 and activation > torch.max(layer_inputs[i][row,:]) / 8:
					# this element has large positive activation in the output
					if ff_output > 0.2 * activation:
						# the FF layer contributes non-negligibly to the activation
						ff_indices.append((element, True))
					if ff_output < 0.8 * activation:
						# the residual connection contributes non-negligibly to the activation
						residual_indices.append(element)
				elif activation < -0.5 and activation < torch.min(layer_inputs[i][row,:]) / 8:
					# this element has large negative activation in the output
					if ff_output < 0.2 * activation:
						# the FF layer contributes non-negligibly to the activation
						ff_indices.append((element, False))
					if ff_output > 0.8 * activation:
						# the residual connection contributes non-negligibly to the activation
						residual_indices.append(element)
				else:
					# this element has small activation in the output
					if ff_inputs[i-1][row,element] > 0.5 and ff_inputs[i-1][row,element] > torch.max(ff_inputs[i-1][row,:]) / 8:
						ff_indices.append((element, False))
					elif ff_inputs[i-1][row,element] < -0.5 and ff_inputs[i-1][row,element] < torch.min(ff_inputs[i-1][row,:]) / 8:
						ff_indices.append((element, True))
					#else:
					#	ff_indices.append((element, None))
					residual_indices.append(element)

			new_ff_indices = []
			for element, sign in ff_indices:
				# check if the bias makes a significant contribution
				ff_output = layer_inputs[i][row,element] - ff_inputs[i-1][row,element]
				if sign == True and self.model.transformers[i-1].ff[3].bias[element] > 0.8 * ff_output:
					print('Output of FF layer {} at row {}, element {} is positive due to bias of FF3.'.format(i, row, element))
					continue
				elif sign == False and self.model.transformers[i-1].ff[3].bias[element] < 0.8 * ff_output:
					print('Output of FF layer {} at row {}, element {} is negative due to bias of FF3.'.format(i, row, element))
					continue

				ff0_input = self.model.transformers[i-1].ln_ff(ff_inputs[i-1][row,:])
				contributions = ff0_input.repeat((d,1))
				contributions[:,:] *= self.model.transformers[i-1].ff[0].weight
				contributions = torch.cat((contributions, self.model.transformers[i-1].ff[0].bias.unsqueeze(1)), 1)
				ff1_input = self.model.transformers[i-1].ff[0](ff0_input)
				contributions[ff1_input <= 0,:] = 0
				contributions = torch.sum(contributions.T * self.model.transformers[i-1].ff[3].weight[element,:], dim=1)
				if sign == True:
					# find the elements of contributions that are most positive and add their indices to new_ff_indices
					for index in torch.nonzero(contributions > torch.max(contributions) / 2).T[0].tolist():
						if index not in new_ff_indices:
							new_ff_indices.append(index)
				elif sign == False:
					# find the elements of contributions that are most negative and add their indices to new_ff_indices
					for index in torch.nonzero(contributions < torch.min(contributions) / 2).T[0].tolist():
						if index not in new_ff_indices:
							new_ff_indices.append(index)

			new_representations = []
			for index in residual_indices:
				new_representations.append((index, ff_inputs[i-1][row,index]))
			for index in new_ff_indices:
				if index not in residual_indices:
					new_representations.append((index, ff_inputs[i-1][row,index]))
			representations = new_representations

			# check to see how much the attention layer contributes to the representation
			attn_indices = []
			residual_indices = []
			for element, activation in representations:
				attn_output = ff_inputs[i-1][row,element] - layer_inputs[i-1][row,element]
				if activation > 0.5 and activation > torch.max(layer_inputs[i][row,:]) / 8:
					# this element has large positive activation in the output
					if attn_output > 0.2 * activation:
						# the attention layer contributes non-negligibly to the activation
						attn_indices.append((element, True))
					if attn_output < 0.8 * activation:
						# the residual connection contributes non-negligibly to the activation
						residual_indices.append(element)
				elif activation < -0.5 and activation < torch.min(layer_inputs[i][row,:]) / 8:
					# this element has large negative activation in the output
					if attn_output < 0.2 * activation:
						# the attention layer contributes non-negligibly to the activation
						attn_indices.append((element, False))
					if attn_output > 0.8 * activation:
						# the residual connection contributes non-negligibly to the activation
						residual_indices.append(element)
				else:
					# this element has small activation in the output
					if attn_inputs[i-1][row,element] > 0.5 and attn_inputs[i-1][row,element] > torch.max(attn_inputs[i-1][row,:]) / 8:
						attn_indices.append((element, False))
					elif attn_inputs[i-1][row,element] < -0.5 and attn_inputs[i-1][row,element] < torch.min(attn_inputs[i-1][row,:]) / 8:
						attn_indices.append((element, True))
					#else:
					#	attn_indices.append((element, None))
					residual_indices.append(element)

			new_attn_indices = []
			for element, sign in attn_indices:
				attn_output = ff_inputs[i-1][row,element] - layer_inputs[i-1][row,element]
				if sign == True and self.model.transformers[i-1].attn.linear.bias[element] > 0.8 * attn_output:
					print('Output of FF layer {} at row {}, element {} is positive due to bias of the post-attention linear layer.'.format(i, row, element))
					continue
				elif sign == False and self.model.transformers[i-1].attn.linear.bias[element] < 0.8 * attn_output:
					print('Output of FF layer {} at row {}, element {} is negative due to bias of the post-attention linear layer.'.format(i, row, element))
					continue

				# compute the contributions to this element from the proj_v bias
				bias_contributions = self.model.transformers[i-1].attn.proj_v.bias * self.model.transformers[i-1].attn.linear.weight[element,:]
				if sign == True and torch.max(bias_contributions) > 0.8 * attn_output:
					print('Output of FF layer {} at row {}, element {} is positive due to bias of the V-projection.'.format(i, row, element))
					continue
				elif sign == False and torch.min(bias_contributions) < 0.8 * attn_output:
					print('Output of FF layer {} at row {}, element {} is negative due to bias of the V-projection.'.format(i, row, element))
					continue

				# compute the contributions to this element from the attention input
				contributions = attn_matrices[i-1][row,:].unsqueeze(1).repeat(1,d) * attn_inputs[i-1] * torch.matmul(self.model.transformers[i-1].attn.linear.weight[element,:], self.model.transformers[i-1].attn.proj_v.weight).repeat(n,1)
				if sign == True:
					# find the elements of contributions that are most positive and add their indices to new_attn_indices
					for index in torch.nonzero(contributions > torch.max(contributions) / 2).tolist():
						if tuple(index) not in new_attn_indices:
							new_attn_indices.append(tuple(index))
				elif sign == False:
					# find the elements of contributions that are most negative and add their indices to new_attn_indices
					for index in torch.nonzero(contributions < torch.min(contributions) / 2).tolist():
						if tuple(index) not in new_attn_indices:
							new_attn_indices.append(tuple(index))

			new_representations = []
			for index in residual_indices:
				new_representations.append((row, index, attn_inputs[i-1][row,index]))
			for src_row, index in new_attn_indices:
				if (src_row, index) not in residual_indices:
					new_representations.append((src_row, index, attn_inputs[i-1][src_row,index]))
			representations = new_representations

			print('In the input to attention layer {}:'.format(i-1))
			for src_row, element, activation in representations:
				print('  Row {}, element {} has activation {}'.format(src_row, element, activation))

		def trace_activation_forward(representation, start_layer, end_layer, layer_inputs, ff_inputs):
			representation = representation.clone().detach()
			representations = [representation.clone().detach()]
			ff_representations = []
			attn_representations = []
			attn_out_representations = []
			bias_contribution = torch.zeros(representation[0].shape)
			bias_representations = [bias_contribution.clone().detach()]
			ff_bias_representations = []
			for i in range(start_layer, end_layer + 1):
				layer_norm_matrix = self.model.transformers[i].ln_attn.weight.unsqueeze(0).repeat((n,1)) / torch.sqrt(torch.var(layer_inputs[i], dim=1, correction=0) + self.model.transformers[i].ln_attn.eps).unsqueeze(1).repeat((1,d))
				layer_norm_bias = -torch.mean(layer_inputs[i], dim=1).unsqueeze(1) * layer_norm_matrix + self.model.transformers[i].ln_attn.bias
				attn_representation = representation * layer_norm_matrix
				attn_out_representation = torch.matmul(torch.matmul(attn_representation, self.model.transformers[i].attn.proj_v.weight.T), self.model.transformers[i].attn.linear.weight.T)
				attn_representations.append((attn_representation, bias_contribution * layer_norm_matrix + layer_norm_bias))
				attn_out_representations.append(attn_out_representation)
				representation += torch.matmul(torch.matmul(torch.matmul(attn_matrices[i], attn_representation), self.model.transformers[i].attn.proj_v.weight.T), self.model.transformers[i].attn.linear.weight.T)

				# compute the contribution from bias terms in the attention layer
				bias_contribution += torch.matmul(torch.matmul(torch.matmul(attn_matrices[i], bias_contribution * layer_norm_matrix), self.model.transformers[i].attn.proj_v.weight.T), self.model.transformers[i].attn.linear.weight.T)
				bias_contribution += torch.matmul(torch.matmul(torch.matmul(attn_matrices[i], layer_norm_bias), self.model.transformers[i].attn.proj_v.weight.T), self.model.transformers[i].attn.linear.weight.T)
				bias_contribution += torch.matmul(self.model.transformers[i].attn.proj_v.bias.unsqueeze(0), self.model.transformers[i].attn.linear.weight.T).repeat((n,1))
				bias_contribution += self.model.transformers[i].attn.linear.bias.unsqueeze(0).repeat((n,1))

				ff_representations.append(representation.clone().detach())
				ff_bias_representations.append(bias_contribution.clone().detach())

				if self.model.transformers[i].ff:
					layer_norm_matrix = self.model.transformers[i].ln_ff.weight.unsqueeze(0).repeat((n,1)) / torch.sqrt(torch.var(ff_inputs[i], dim=1, correction=0) + self.model.transformers[i].ln_ff.eps).unsqueeze(1).repeat((1,d))
					layer_norm_bias = -torch.mean(ff_inputs[i], dim=1).unsqueeze(1) * layer_norm_matrix + self.model.transformers[i].ln_ff.bias
					ff0_output = self.model.transformers[i].ff[0](self.model.transformers[i].ln_ff(ff_inputs[i]))
					representation += torch.matmul((ff0_output > 0) * torch.matmul(representation * layer_norm_matrix, self.model.transformers[i].ff[0].weight.T), self.model.transformers[i].ff[3].weight.T)

					# compute the contribution from the bias terms in the FF layer
					layer_norm_bias = -torch.mean(ff_inputs[i], dim=1).unsqueeze(1) * layer_norm_matrix + self.model.transformers[i].ln_ff.bias
					bias_contribution += torch.matmul((ff0_output > 0.0) * torch.matmul(bias_contribution * layer_norm_matrix, self.model.transformers[i].ff[0].weight.T), self.model.transformers[i].ff[3].weight.T)
					bias_contribution += torch.matmul((ff0_output > 0.0) * torch.matmul(layer_norm_bias, self.model.transformers[i].ff[0].weight.T), self.model.transformers[i].ff[3].weight.T)
					bias_contribution += torch.matmul((ff0_output > 0.0) * self.model.transformers[i].ff[0].bias.unsqueeze(0), self.model.transformers[i].ff[3].weight.T)
					bias_contribution += self.model.transformers[i].ff[3].bias.unsqueeze(0).repeat((n,1))
				representations.append(representation.clone().detach())
				bias_representations.append(bias_contribution.clone().detach())

			return attn_representations, attn_out_representations, ff_representations, representations, bias_representations, ff_bias_representations

		def check_copyr(i, dst, src, attn_inputs, attn_matrices, attn_representations):
			attn_input = attn_inputs[i]
			attn_representation, attn_bias_contribution = attn_representations[i]
			A = torch.matmul(self.model.transformers[i].attn.proj_q.weight.T, self.model.transformers[i].attn.proj_k.weight)
			left = torch.matmul(attn_representation[:,dst,:], A)
			products = torch.matmul(left, attn_representation[:,src,:].T)
			bias_product = torch.dot(attn_bias_contribution[dst,:],attn_bias_contribution[src,:])
			return products, bias_product

		def token_to_str(token_value):
			if token_value == PADDING_TOKEN:
				return 'P'
			elif token_value == EDGE_PREFIX_TOKEN:
				return 'E'
			elif token_value == QUERY_PREFIX_TOKEN:
				return 'Q'
			elif token_value == PATH_PREFIX_TOKEN:
				return 'A'
			else:
				return str(int(token_value))

		BOLD = '\033[1m'
		BLUE = '\033[94m'
		GREEN = '\033[92m'
		END = '\033[0m'

		def get_token_value(token):
			if token < n:
				return BOLD + GREEN + token_to_str(input[token]) + END

			position = token - n
			if position % 3 == 2:
				out = token_to_str(input[position-1]) + ' '
				out += BOLD + BLUE + token_to_str(input[position]) + END + ' '
				if position + 1 < n:
					out += token_to_str(input[position+1])
			elif position % 3 == 0:
				out = ''
				if position - 2 >= 0:
					out += token_to_str(input[position-2]) + ' '
				if position - 1 >= 0:
					out += token_to_str(input[position-1]) + ' '
				out += BOLD + BLUE + token_to_str(input[position]) + END
			elif position % 3 == 1:
				out = BOLD + BLUE + token_to_str(input[position]) + END + ' '
				if position + 1 < n:
					out += token_to_str(input[position+1]) + ' '
				if position + 2 < n:
					out += token_to_str(input[position+2])
			return out.strip()

		start_index = torch.sum(input == PADDING_TOKEN)
		num_edges = torch.sum(input == EDGE_PREFIX_TOKEN)

		forward_edges = []
		for i in range((n - 5) // 3 + 1):
			forward_edges.append([])
		for i in range(2, n-5, 3):
			if i >= start_index:
				forward_edges[input[i].item()].append(input[i+1].item())
		def path_length(start, end):
			if input[start] > (n - 5) // 3 or input[end] > (n - 5) // 3:
				return -1
			queue = [(input[start],0)]
			best_distance = -1
			while len(queue) != 0:
				current, distance = queue.pop()
				if current == input[end]:
					if best_distance == -1 or distance < best_distance:
						best_distance = distance
					continue
				for child in forward_edges[current]:
					queue.append((child,distance+1))
			return best_distance

		# find the major paths from the start vertex in the graph
		queue = [(input[-1].item(),None,0)]
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
		goal = input[torch.nonzero(input == QUERY_PREFIX_TOKEN)+2].item()
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
		correct_fork = forks[goal]
		other_fork = next(path for v,path in forks.items() if v != goal)

		def trace_contributions(start_layer, contributions, attn_inputs, attn_matrices, attn_representations, attn_out_representations, contributions_to_search):
			copy_graph = {}
			for i in reversed(range(start_layer + 1)):
				import pdb; pdb.set_trace()
				new_contributions = []
				inspected_copies = []
				for row, contribution in contributions:
					#representation_norms = torch.linalg.vector_norm(attn_out_representations[i][contribution,:,:], dim=-1)
					attn_products = torch.matmul(attn_out_representations[i][contribution,:,:], ff_representations[i][contribution,row,:])
					inputs = attn_products * attn_matrices[i][row,:]
					residual_product = torch.dot(representations[i][contribution,row,:], ff_representations[i][contribution,row,:])
					#if i == len(self.model.transformers) - 1:
						# TODO: do this automatically, maybe by doing perturbation analysis on the goal vertex?
					#	src_indices = torch.nonzero(inputs_with_residual > 0.25 * torch.max(inputs_with_residual)).squeeze(1).tolist()
					#else:
					#	src_indices = torch.nonzero(inputs_with_residual > 0.5 * torch.max(inputs_with_residual)).squeeze(1).tolist()

					copy_info = (None,None,None)
					if residual_product > 0.5 * max(residual_product, torch.max(inputs)):
						if (src_index,contribution) not in new_contributions:
							new_contributions.append((src_index,contribution))

						if contribution < n:
							print('Residual at layer {} is copying row {} the contribution from token at position {} ({})'.format(i, src_index, contribution, get_token_value(contribution)))
							copy_graph[(i,src_index)] = (i+1,row,0)
							copy_info = (0,contribution,contribution)
						else:
							print('Residual at layer {} is copying row {} the contribution from position {} ({})'.format(i, src_index, contribution - n, get_token_value(contribution - n)))
							copy_graph[(i,src_index)] = (i+1,row,0)
							copy_info = (0,contribution,contribution)

					if torch.max(inputs) > 0.5 * max(residual_product, torch.max(inputs)):
						src_indices = torch.nonzero(inputs > 0.5 * torch.max(inputs)).squeeze(1).tolist()

						# calculate why is each row in `src_indices` being copied into `row`
						for j in range(len(src_indices)):
							src_index = src_indices[j]
							if (src_index,row) in inspected_copies:
								continue
							inspected_copies.append((src_index,row))

							if (src_index,contribution) not in new_contributions:
								new_contributions.append((src_index,contribution))

							products, bias_product = check_copyr(i, row, src_index, attn_inputs, attn_matrices, attn_representations)
							print('Attention layer {} is copying token at {} into token at {} with weight {:.2f} because:'.format(i, src_index, row, attn_matrices[i][row,src_index]))
							max_product = torch.max(products)
							threshold = max_product - 10.0
							if max_product > 0.0 and max_product / 2 > threshold:
								threshold = max_product / 2

							# compute the decomposition of attn_inputs[i][row,:] in terms of the contributions
							#import pdb; pdb.set_trace()
							#attn_representation, _ = attn_representations[i]
							#solution = torch.linalg.lstsq(attn_representation[:,row,:].T / (torch.linalg.vector_norm(attn_representation[:,row,:], dim=-1) + 1e-9), attn_inputs[i][row,:], driver='gelss').solution

							for dst_contribution, src_contribution in torch.nonzero(products > threshold):
								if dst_contribution < n:
									print('  --> Token at {} has high contribution from token embedding at {} ({})'.format(row, dst_contribution, get_token_value(dst_contribution)))
									dst_contribution_index = dst_contribution
								else:
									print('  --> Token at {} has high contribution from position embedding {} (token: {})'.format(row, dst_contribution - n, get_token_value(dst_contribution - n)))
									dst_contribution_index = dst_contribution - n
								if src_contribution < n:
									print('  and token at {} has high contribution from token embedding at {} (token: {})'.format(src_index, src_contribution, get_token_value(src_contribution)))
									src_contribution_index = src_contribution
								else:
									print('  and token at {} has high contribution from position embedding {} (token: {})'.format(src_index, src_contribution - n, get_token_value(src_contribution - n)))
									src_contribution_index = src_contribution - n
								if dst_contribution_index % 3 == 2 and src_contribution_index % 3 == 0 and input[dst_contribution_index] == input[src_contribution_index] and input[dst_contribution_index] <= (n - 5) // 3:
									# this is a forwards step
									print(GREEN + '    This is a forwards step.' + END)
									copy_graph[(i,src_index)] = (i+1,row,1)
									if copy_info[0] == None:
										copy_info = (1,src_contribution,dst_contribution)
								elif dst_contribution_index % 3 == 0 and src_contribution_index % 3 == 2 and input[dst_contribution_index] == input[src_contribution_index] and input[dst_contribution_index] <= (n - 5) // 3:
									# this is a backwards step
									print(GREEN + '    This is a backwards step.' + END)
									copy_graph[(i,src_index)] = (i+1,row,-1)
									if copy_info[0] == None:
										copy_info = (-1,src_contribution,dst_contribution)
								elif dst_contribution_index == src_contribution_index - 1:
									print(GREEN + '    This is a backwards step.' + END)
									copy_graph[(i,src_index)] = (i+1,row,-1)
									if copy_info[0] == None:
										copy_info = (-1,src_contribution,dst_contribution)
								elif dst_contribution_index == src_contribution_index + 1:
									print(GREEN + '    This is a forwards step.' + END)
									copy_graph[(i,src_index)] = (i+1,row,1)
									if copy_info[0] == None:
										copy_info = (1,src_contribution,dst_contribution)
								elif dst_contribution_index == src_contribution_index:
									if (i,src_index) not in copy_graph:
										copy_graph[(i,src_index)] = (i+1,row,2)
									if copy_info[0] == None:
										copy_info = (2,src_contribution,dst_contribution)
								elif src_contribution >= n and dst_contribution >= n:
									# this is a hard-coded position-based copy
									print(GREEN + '    This is a hard-coded copy step (from position embedding {} to {}).'.format(src_contribution-n,dst_contribution-n) + END)
									if (i,src_index) not in copy_graph:
										copy_graph[(i,src_index)] = (i+1,row,(src_contribution-n,dst_contribution-n))
									if copy_info[0] == None:
										copy_info = (3,src_contribution,dst_contribution)
								else:
									if (i,src_index) not in copy_graph:
										copy_graph[(i,src_index)] = (i+1,row,None)
								#new_contribution = (src_index,src_contribution.item())
								#if new_contribution not in new_contributions:
								#	new_contributions.append(new_contribution)
								#new_contribution = (row,dst_contribution.item())
								#if new_contribution not in new_contributions:
								#	new_contributions.append(new_contribution)

						copy_type,src_contribution,dst_contribution = copy_info
						if copy_type not in (None, 0) and i > 0:
							if (i-1,row,dst_contribution.item()) not in contributions_to_search:
								contributions_to_search.append((i-1,row,dst_contribution.item()))
							if (i-1,src_index,src_contribution.item()) not in contributions_to_search:
								contributions_to_search.append((i-1,src_index,src_contribution.item()))
						elif type(copy_type) == tuple:
							_,_,copy_indices = copy_graph[(i,src_index)]
							src_pos,dst_pos = copy_indices
							if (i-1,row,n+dst_pos) not in contributions_to_search:
								contributions_to_search.append((i-1,row,n+dst_pos))
							if (i-1,src_index,n+src_pos) not in contributions_to_search:
								contributions_to_search.append((i-1,src_index,n+src_pos))
				contributions = new_contributions

			if (0,48) in copy_graph:
				# print the copy path
				current_layer, current_row = (0,48)
				while current_layer < start_layer:
					next_layer, next_row, copy_type = copy_graph[(current_layer,current_row)]
					if copy_type == 0:
						print('Copy from row {} to row {} (residual)'.format(current_row, next_row))
					elif copy_type == 1:
						print('Copy from row {} to row {} (forward step)'.format(current_row, next_row))
					elif copy_type == -1:
						print('Copy from row {} to row {} (backward step)'.format(current_row, next_row))
					elif copy_type == None:
						print('Copy from row {} to row {} (unknown step)'.format(current_row, next_row))
					current_layer, current_row = next_layer, next_row

		def trace_representation(target_layer, target_row, output_type, attn_src_row=None):
			contribution_list = []
			for i in range(target_layer + 1):
				representation = torch.zeros((n,n,d))
				for j in range(n):
					representation[j,j,:] = layer_inputs[i][j,:]
				attn_representations, _, _, representations, bias_representations, _ = trace_activation_forward(representation, i, target_layer, layer_inputs, ff_inputs)
				contribution_list.append((attn_representations, representations, bias_representations))

			if output_type == 'prediction':
				output_representation = torch.zeros((d))
				output_representation[prediction] = 1.0
			elif output_type == 'attn_dst':
				output_representation = attn_inputs[target_layer][target_row,:]
			elif output_type == 'attn_src':
				output_representation = attn_inputs[target_layer][attn_src_row,:]

			current_rows = [target_row]
			copy_graph = {}
			for i in reversed(range(target_layer + 1)):
				import pdb; pdb.set_trace()
				attn_representations, representations, bias_representations = contribution_list[i]
				if output_type == 'attn_dst':
					attn_representation, attn_bias_contribution = attn_representations[-1]
					A = torch.matmul(self.model.transformers[i].attn.proj_q.weight.T, self.model.transformers[i].attn.proj_k.weight)
					right = torch.matmul(A, attn_representation[:,attn_src_row,:].T)
					src_products = torch.matmul(output_representation, right)
				elif output_type == 'attn_src':
					attn_representation, attn_bias_contribution = attn_representations[-1]
					A = torch.matmul(self.model.transformers[i].attn.proj_q.weight.T, self.model.transformers[i].attn.proj_k.weight)
					left = torch.matmul(attn_representation[:,target_row,:], A)
					src_products = torch.matmul(left, output_representation)
				else:
					src_products = torch.matmul(representations[-1][:,target_row,:], output_representation)
				sorted_idx = np.argsort(src_products.detach().numpy())
				src_indices = reversed(next(sorted_idx[j:] for j in reversed(range(len(sorted_idx))) if torch.sum(torch.clamp(src_products[sorted_idx[j:]],min=0))/torch.sum(torch.clamp(src_products,min=0)) > 0.8))

				print("At the input to layer {}, the following rows contribute most to the final output: {}".format(i, ", ".join(["{}:{:.2f}".format(src_index,(src_products[src_index]/torch.sum(torch.clamp(src_products,min=0))).item()) for src_index in src_indices])))

				next_rows = []
				for row in current_rows:
					# we want to compute the important input rows that contribute to the output
					attn_products = attn_matrices[i][row,:] * src_products
					residual_product = src_products[row]
					products = np.concatenate((attn_products.detach().numpy(), [residual_product.detach().numpy()]))
					sorted_idx = np.argsort(products)
					indices = reversed(next(sorted_idx[j:] for j in reversed(range(len(sorted_idx))) if np.sum(np.maximum(products[sorted_idx[j:]],0))/np.sum(np.maximum(products,0)) > 0.8))

					for next_row in indices:
						if (i+1,row) not in copy_graph:
							copy_graph[(i+1,row)] = []
						if next_row not in copy_graph[(i+1,row)]:
							copy_graph[(i+1,row)].append(next_row)
						if next_row == n:
							# the residual connection is significant
							if row not in next_rows:
								next_rows.append(row)
							print("  Residual connection of layer {} copies contribution from row {}. (contribution: {:.2f})".format(i, row, products[next_row]/np.sum(np.maximum(products,0))))
						else:
							if next_row not in next_rows:
								next_rows.append(next_row)
							print("  Attention layer {} copies contribution from row {} to row {}. (contribution: {:.2f})".format(i, next_row, row, products[next_row]/np.sum(np.maximum(products,0))))

				current_rows = next_rows

		def classify_attn_op(i, dst, src, perturb_attn_inputs):
			# try computing the attention dot product but with perturbed dst embeddings
			A = A_matrices[i][:-1,:-1]
			old_product = torch.dot(torch.matmul(attn_inputs[i][dst,:], A), attn_inputs[i][src,:])
			new_src_products = torch.matmul(torch.matmul(attn_inputs[i][dst,:], A), perturb_attn_inputs[i][:,src,:].T)
			new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i][:,dst,:], A), attn_inputs[i][src,:])

			is_negative_copy = (attn_matrices[i][dst,src] < torch.median(attn_matrices[i][dst,edge_indices]))
			print('Layer {} copies token at {} into token at {} with weight {}.'.format(i, src, dst, attn_matrices[i][dst,src]))
			if is_negative_copy:
				src_dependencies = torch.nonzero(new_src_products > old_product + math.sqrt(d) / 2)
				dst_dependencies = torch.nonzero(new_dst_products > old_product + math.sqrt(d) / 2)
			else:
				src_dependencies = torch.nonzero(new_src_products < old_product - math.sqrt(d) / 2)
				dst_dependencies = torch.nonzero(new_dst_products < old_product - math.sqrt(d) / 2)
			print('  src dependencies: ' + ', '.join([get_token_value(dep) for dep in src_dependencies]))
			print('  dst dependencies: ' + ', '.join([get_token_value(dep) for dep in dst_dependencies]))

			'''for src_dep in src_dependencies:
				for dst_dep in dst_dependencies:
					if src_dep >= n and dst_dep >= n and src_dep-n < n-1 and input[src_dep-n+1] == input[dst_dep-n]:
						print('This is a forwards step. ({} and {})'.format(get_token_value(src_dep), get_token_value(dst_dep)))
					if src_dep >= n and dst_dep >= n and dst_dep-n < n-1 and input[dst_dep-n+1] == input[src_dep-n]:
						print('This is a backwards step. ({} and {})'.format(get_token_value(src_dep), get_token_value(dst_dep)))
					if src_dep < n and dst_dep < n and src_dep == dst_dep:
						print('This is a token matching step. ({})'.format(get_token_value(src_dep)))'''

			import pdb; pdb.set_trace()
			return src_dependencies, dst_dependencies

		def activation_patch_attn(i, dst, src):
			# create perturbed inputs where each input swaps the position of one edge
			edge_indices = [i + 1 for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]
			other_inputs = input.repeat((len(edge_indices), 1))
			swap_edge_idx = len(edge_indices) - 1
			for j in range(len(edge_indices)):
				if j == swap_edge_idx:
					continue
				# swap the current edge with the last edge
				other_inputs[j,edge_indices[j]] = input[edge_indices[swap_edge_idx]]
				other_inputs[j,edge_indices[j]+1] = input[edge_indices[swap_edge_idx]+1]
				other_inputs[j,edge_indices[swap_edge_idx]] = input[edge_indices[j]]
				other_inputs[j,edge_indices[swap_edge_idx]+1] = input[edge_indices[j]+1]

			# perform forward pass on other_inputs
			other_inputs = self.model.token_embedding[other_inputs]
			if len(other_inputs.shape) == 2:
				pos = self.model.positional_embedding
			else:
				pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0:-2] + (-1,-1))
			other_inputs = torch.cat((other_inputs, pos), -1)
			other_inputs = self.model.dropout_embedding(other_inputs)

			frozen_ops = []
			for j in range(i):
				frozen_ops.append((attn_matrices[j], self.model.transformers[j].ff[0](self.model.transformers[j].ln_ff(ff_inputs[j])) > 0.0))

			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = forward(self.model, other_inputs, mask, 0, False, i, None, frozen_ops)

			# try computing the attention dot product but with perturbed dst embeddings
			A = A_matrices[i][:-1,:-1]
			old_products = torch.matmul(torch.matmul(attn_inputs[i], A), attn_inputs[i].T)
			old_product = old_products[dst, src]
			import pdb; pdb.set_trace()
			new_src_products = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))
			new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))
			new_src_products = new_src_products[:,dst,src]
			new_dst_products = new_dst_products[:,dst,src]

			is_negative_copy = (attn_matrices[i][dst,src] < torch.median(attn_matrices[i][dst,edge_indices]))
			if is_negative_copy:
				src_dependencies = torch.nonzero(new_src_products > old_product + math.sqrt(d) / 2)
				dst_dependencies = torch.nonzero(new_dst_products > old_product + math.sqrt(d) / 2)
			else:
				src_dependencies = torch.nonzero(new_src_products < old_product - math.sqrt(d) / 2)
				dst_dependencies = torch.nonzero(new_dst_products < old_product - math.sqrt(d) / 2)
			print('  src dependencies: ' + ', '.join([get_token_value(edge_indices[dep]+n) for dep in src_dependencies]))
			print('  dst dependencies: ' + ', '.join([get_token_value(edge_indices[dep]+n) for dep in dst_dependencies]))

			import pdb; pdb.set_trace()

			# create perturbed inputs where each input replaces every occurrence of a token value with an unused token value
			max_vertex_id = (n - 5) // 3
			unused_id = next(t for t in range(1, max_vertex_id + 1) if t not in input)
			other_inputs = input.repeat(max_vertex_id + 1 + n, 1)
			for j in range(max_vertex_id + 1):
				other_inputs[j,input == j] = unused_id

			# perform forward pass on other_inputs
			other_inputs = self.model.token_embedding[other_inputs]
			if len(other_inputs.shape) == 2:
				pos = self.model.positional_embedding
			else:
				pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0:-2] + (-1,-1))
			other_inputs = torch.cat((other_inputs, pos), -1)
			other_inputs = self.model.dropout_embedding(other_inputs)

			# for the last n rows of `other_inputs`, perturb the position embeddings
			for j in range(n):
				if j < edge_indices[-1] or j > edge_indices[-1] + 2:
					src_edge_index = edge_indices[-1]
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[max_vertex_id+1+j,j,d-n:] = other_inputs[max_vertex_id+1+j,src_edge_index-offset,d-n:].clone().detach()

			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = forward(self.model, other_inputs, mask, 0, False, i, None, frozen_ops)

			# try computing the attention dot product but with perturbed dst embeddings
			A = A_matrices[i][:-1,:-1]
			old_products = torch.matmul(torch.matmul(attn_inputs[i], A), attn_inputs[i].T)
			old_product = old_products[dst, src]
			new_src_products = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))
			new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))
			import pdb; pdb.set_trace()
			new_src_products = new_src_products[:,dst,src]
			new_dst_products = new_dst_products[:,dst,src]

			is_negative_copy = (attn_matrices[i][dst,src] < torch.median(attn_matrices[i][dst,edge_indices]))
			if is_negative_copy:
				src_dependencies = torch.nonzero(new_src_products > old_product + math.sqrt(d) / 3)
				dst_dependencies = torch.nonzero(new_dst_products > old_product + math.sqrt(d) / 3)
				src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=True).indices]
				dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=True).indices]
			else:
				src_dependencies = torch.nonzero(new_src_products < old_product - math.sqrt(d) / 3)
				dst_dependencies = torch.nonzero(new_dst_products < old_product - math.sqrt(d) / 3)
				src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=False).indices]
				dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=False).indices]
			def dep_to_str(dep):
				if dep <= max_vertex_id:
					return BOLD + GREEN + str(dep.item()) + END
				else:
					return get_token_value(dep-max_vertex_id-1+n)
			print('  src dependencies: ' + ', '.join([dep_to_str(dep) for dep in src_dependencies]))
			print('  dst dependencies: ' + ', '.join([dep_to_str(dep) for dep in dst_dependencies]))

			import pdb; pdb.set_trace()

			'''new_attn_matrices = attn_pre_softmax[i].repeat(new_src_products.size(0),1,1)
			new_attn_matrices[:,dst,src] += (new_src_products - old_product) / math.sqrt(d)
			new_attn_matrices = self.model.transformers[i].attn.attn.dropout(new_attn_matrices.softmax(-1))
			import pdb; pdb.set_trace()

			new_frozen_ops = [None] * (i+1)
			new_frozen_ops[i] = (new_attn_matrices, None)
			perturb_layer_inputs, _, _, _, _, _, _, _, _, _, _ = forward(self.model, layer_inputs[i], mask, i, False, len(self.model.transformers)-1, None, new_frozen_ops)
			import pdb; pdb.set_trace()'''

			zero_src_product_list = []
			zero_dst_product_list = []
			max_src_product_list = []
			max_dst_product_list = []
			for j in range(i):
				new_inputs = layer_inputs[j].clone().detach()
				new_inputs[src,:] = perturb_layer_inputs[j][116,src,:]
				zero_src_products, zero_dst_products, max_src_products, max_dst_products = perturb_attn_ops(self.model, i, j, dst, src, new_inputs, mask, prediction, A_matrices, attn_matrices, attn_inputs)
				zero_src_product_list.append(zero_src_products)
				zero_dst_product_list.append(zero_dst_products)
				max_src_product_list.append(max_src_products)
				max_dst_product_list.append(max_dst_products)
			import pdb; pdb.set_trace()

			# create perturbed inputs where each input swaps the position of one token
			other_inputs = input.repeat((2*n,1))
			'''for j in range(n):
				if input[j] in (PADDING_TOKEN, QUERY_PREFIX_TOKEN, PATH_PREFIX_TOKEN):
					continue
				if j < edge_indices[-1] or j > edge_indices[-1] + 2:
					src_edge_index = edge_indices[-1]
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[j,j] = input[src_edge_index - offset]
				other_inputs[j,src_edge_index - offset] = input[j]'''

			other_inputs = self.model.token_embedding[other_inputs]
			if len(other_inputs.shape) == 2:
				pos = self.model.positional_embedding
			else:
				pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0], -1, -1)
			other_inputs = torch.cat((other_inputs, pos), -1)
			other_inputs = self.model.dropout_embedding(other_inputs)

			# perturb the position embeddings
			for j in range(n):
				if input[j] in (PADDING_TOKEN, QUERY_PREFIX_TOKEN, PATH_PREFIX_TOKEN):
					continue
				if j < edge_indices[-1] or j > edge_indices[-1] + 2:
					src_edge_index = edge_indices[-1]
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[j,j,:d-n] = other_inputs[j,src_edge_index-offset,:d-n].clone().detach()
				other_inputs[n+j,j,d-n:] = other_inputs[n+j,src_edge_index-offset,d-n:].clone().detach()

			# perform forward pass on other_inputs
			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = forward(self.model, other_inputs, mask, 0, False, i, None, frozen_ops)

			src_dependencies, dst_dependencies = classify_attn_op(i, dst, src, perturb_attn_inputs)

			if i > 1:
				for dep in src_dependencies:
					print('src dependency on ' + get_token_value(dep) + ' comes from:')
					queue = [(src,i-1)]
					copies = []
					while len(queue) != 0:
						(row,layer) = queue[0]
						del queue[0]
						abs_deviations = torch.sum(torch.abs(perturb_attn_inputs[layer][dep,:,:] - attn_inputs[layer][:,:]), dim=-1)
						weighted_deviations = abs_deviations * attn_matrices[layer][row,:]
						weighted_deviations[row] += abs_deviations[row]
						prev_rows = torch.nonzero(weighted_deviations > 0.4 * torch.max(weighted_deviations)).squeeze().tolist()
						if type(prev_rows) == int:
							prev_rows = [prev_rows]
						if (layer, prev_rows, row) in copies:
							continue
						copies.append((layer, prev_rows, row))
						print('  Layer {} copies rows {} into row {}.'.format(layer, prev_rows, row))
						if layer == 0:
							continue
						for prev_row in prev_rows:
							queue.append((prev_row,layer-1))
				for dep in dst_dependencies:
					print('dst dependency on ' + get_token_value(dep) + ' comes from:')
					queue = [(dst,i-1)]
					copies = []
					while len(queue) != 0:
						(row,layer) = queue[0]
						del queue[0]
						abs_deviations = torch.sum(torch.abs(perturb_attn_inputs[layer][dep,:,:] - attn_inputs[layer][:,:]), dim=-1)
						weighted_deviations = abs_deviations * attn_matrices[layer][row,:]
						weighted_deviations[row] += abs_deviations[row]
						prev_rows = torch.nonzero(weighted_deviations > 0.4 * torch.max(weighted_deviations)).squeeze().tolist()
						if type(prev_rows) == int:
							prev_rows = [prev_rows]
						if (layer, prev_rows, row) in copies:
							continue
						copies.append((layer, prev_rows, row))
						print('  Layer {} copies rows {} into row {}.'.format(layer, prev_rows, row))
						if layer == 0:
							continue
						for prev_row in prev_rows:
							queue.append((prev_row,layer-1))

			import pdb; pdb.set_trace()

		def activation_patch_attn_layer(i, attn_ops):
			# create perturbed inputs where each input replaces every occurrence of a token value with an unused token value
			edge_indices = [i + 1 for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]
			max_vertex_id = (n - 5) // 3
			unused_id = next(t for t in range(1, max_vertex_id + 1) if t not in input)
			other_inputs = input.repeat(max_vertex_id + 1 + n, 1)
			for j in range(max_vertex_id + 1):
				other_inputs[j,input == j] = unused_id

			# perform forward pass on other_inputs
			other_inputs = self.model.token_embedding[other_inputs]
			if len(other_inputs.shape) == 2:
				pos = self.model.positional_embedding
			else:
				pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0:-2] + (-1,-1))
			other_inputs = torch.cat((other_inputs, pos), -1)
			other_inputs = self.model.dropout_embedding(other_inputs)

			# for the last n rows of `other_inputs`, perturb the position embeddings
			for j in range(n):
				if j < edge_indices[-1] or j > edge_indices[-1] + 2:
					src_edge_index = edge_indices[-1]
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[max_vertex_id+1+j,j,d-n:] = other_inputs[max_vertex_id+1+j,src_edge_index-offset,d-n:].clone().detach()

			frozen_ops = []
			for j in range(i):
				frozen_ops.append((attn_matrices[j], self.model.transformers[j].ff[0](self.model.transformers[j].ln_ff(ff_inputs[j])) > 0.0))

			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = forward(self.model, other_inputs, mask, 0, False, i, None, frozen_ops)

			# try computing the attention dot product but with perturbed dst embeddings
			A = A_matrices[i][:-1,:-1]
			old_products = torch.matmul(torch.matmul(attn_inputs[i], A), attn_inputs[i].T)
			new_src_products = torch.matmul(torch.matmul(attn_inputs[i], A), perturb_attn_inputs[i].transpose(-2,-1))
			new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i], A), attn_inputs[i].transpose(-2,-1))

			for dst in edge_indices:
				potential_sources = torch.nonzero(attn_matrices[i][dst,:] > 0.7 * torch.max(attn_matrices[i][dst,:]))[0].tolist()
				if len(torch.nonzero(attn_matrices[i][dst,edge_indices] < 0.01)) < len(edge_indices) // 2:
					for src in torch.nonzero(attn_matrices[i][dst,edge_indices] < 0.01):
						new_src = edge_indices[src.item()]
						if new_src not in potential_sources:
							potential_sources.append(new_src)
				for src in potential_sources:
					print('Copy from {} to {}'.format(src, dst))
					old_product = old_products[dst,src]
					is_negative_copy = (attn_matrices[i][dst,src] < torch.median(attn_matrices[i][dst,edge_indices]))
					if is_negative_copy:
						src_dependencies = torch.nonzero(new_src_products[:,dst,src] > old_product + math.sqrt(d) / 3)
						dst_dependencies = torch.nonzero(new_dst_products[:,dst,src] > old_product + math.sqrt(d) / 3)
						src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=True).indices]
						dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=True).indices]
					else:
						src_dependencies = torch.nonzero(new_src_products[:,dst,src] < old_product - math.sqrt(d) / 3)
						dst_dependencies = torch.nonzero(new_dst_products[:,dst,src] < old_product - math.sqrt(d) / 3)
						src_dependencies = src_dependencies[torch.sort(new_src_products[src_dependencies],dim=0,descending=False).indices]
						dst_dependencies = dst_dependencies[torch.sort(new_dst_products[dst_dependencies],dim=0,descending=False).indices]
					src_dependencies = src_dependencies[0].tolist() if len(src_dependencies) != 0 else []
					dst_dependencies = dst_dependencies[0].tolist() if len(dst_dependencies) != 0 else []

					# remove dependencies that are explained by earlier attention operations
					if input[src] in src_dependencies and max_vertex_id+1+src in src_dependencies:
						dependent_indices = [src]
						for j in reversed(range(i)):
							for dependent_idx in dependent_indices:
								if max_vertex_id+1+dependent_idx in src_dependencies:
									src_dependencies.remove(max_vertex_id+1+dependent_idx)
								if dependent_idx in attn_ops[j]:
									for new_idx in attn_ops[j][dependent_idx]:
										if new_idx not in dependent_indices:
											dependent_indices.append(new_idx)
					if input[dst] in dst_dependencies and max_vertex_id+1+dst in dst_dependencies:
						dependent_indices = [dst]
						for j in reversed(range(i)):
							for dependent_idx in dependent_indices:
								if max_vertex_id+1+dependent_idx in dst_dependencies:
									dst_dependencies.remove(max_vertex_id+1+dependent_idx)
								if dependent_idx in attn_ops[j]:
									for new_idx in attn_ops[j][dependent_idx]:
										if new_idx not in dependent_indices:
											dependent_indices.append(new_idx)

					def dep_to_str(dep):
						if dep <= max_vertex_id:
							return BOLD + GREEN + str(dep) + END
						else:
							return get_token_value(dep-max_vertex_id-1+n)
					if len(src_dependencies) == 1 and len(dst_dependencies) == 1:
						# this attention operation only utilizes the representation of a single token or position in each of the source and destination tokens
						# TODO: `attn_ops` should differentiate between positive and negative copies
						if dst_dependencies[0] not in attn_ops[i]:
							attn_ops[i][dst_dependencies[0]] = []
						if src_dependencies[0] not in attn_ops[i][dst_dependencies[0]]:
							attn_ops[i][dst_dependencies[0]].append(src_dependencies[0])
					print('  src dependencies: ' + ', '.join([dep_to_str(dep) for dep in src_dependencies]))
					print('  dst dependencies: ' + ', '.join([dep_to_str(dep) for dep in dst_dependencies]))
			import pdb; pdb.set_trace()

		#activation_patch_attn(4, 89, 23)
		#activation_patch_attn(4, 35, 14)
		#activation_patch_attn(4, 35, 41)
		#activation_patch_attn(4, 35, 47)
		#activation_patch_attn(5, 89, 35)

		zero_src_product_list = []
		zero_dst_product_list = []
		max_src_product_list = []
		max_dst_product_list = []
		zero_output_logit_list = []
		max_output_logit_list = []
		import pdb; pdb.set_trace()
		for j in range(5+1):
			zero_src_products, zero_dst_products, max_src_products, max_dst_products, zero_output_logits, max_output_logits = perturb_attn_ops(self.model, 5, j, 89, 35, layer_inputs[j], mask, prediction, A_matrices, attn_matrices, attn_inputs)
			zero_src_product_list.append(zero_src_products)
			zero_dst_product_list.append(zero_dst_products)
			max_src_product_list.append(max_src_products)
			max_dst_product_list.append(max_dst_products)
			zero_output_logit_list.append(zero_output_logits)
			max_output_logit_list.append(max_output_logits)
		import pdb; pdb.set_trace()

		attn_ops = [{} for i in range(len(self.model.transformers))]
		activation_patch_attn_layer(0, attn_ops)
		activation_patch_attn_layer(1, attn_ops)
		activation_patch_attn_layer(2, attn_ops)
		activation_patch_attn_layer(3, attn_ops)
		activation_patch_attn_layer(4, attn_ops)

		'''probe_model = TransformerProber(self.model, 0)
		probe_model.to(device)
		probe_model.reset_parameters()
		from Sophia import SophiaG
		optimizer = SophiaG((p for p in probe_model.A_ops), lr=1.0e-1)
		loss_func = torch.nn.MSELoss()
		import pdb; pdb.set_trace()
		while True:
			probe_model.train()
			optimizer.zero_grad()

			torch.autograd.set_detect_anomaly(True)
			A_predictions, A_labels = probe_model(input)
			loss = sum([loss_func(A_predictions[i][0], A_labels[i][0]) + loss_func(A_predictions[i][1], A_labels[i][1]) for i in range(len(A_labels))])

			loss.backward()
			optimizer.step()
			print('loss: {}'.format(loss.item()))'''

		activation_patch_attn(5, 89, 35)
		#activation_patch_attn(4, 89, 23)
		#activation_patch_attn(4, 35, 14)
		#activation_patch_attn(4, 35, 41)
		#activation_patch_attn(3, 23, 11)
		#activation_patch_attn(3, 23, 23)
		#activation_patch_attn(3, 23, 32)
		#activation_patch_attn(3, 23, 88)
		#activation_patch_attn(2, 23, 11)
		#activation_patch_attn(1, 11, 11)
		#activation_patch_attn(0, 11, 12)

		#trace_activation(5, 35, [(40, attn_inputs[5][35,40]), (125, attn_inputs[5][35,125]), (128, attn_inputs[5][35,128])])
		#trace_representation(len(self.model.transformers)-1, n-1, 'prediction')
		#trace_representation(5, 89, 'attn_dst', 35)
		#trace_representation(4, 35, 'attn_dst', 14) # why is 14 copied into 35 at layer 4? from 'attn_dst', its because 14 contains contributions from rows 14 (6 -> 4), 88, 86 (placeholders before start/current vertex, 27), and from 'attn_src', its because 35 contains contributions from rows 35, 36 (26 -> 1), 21 (25 -> 19; heuristic?), 47 (4 -> 17). so is this a step forward since 14 having contribution from 14 (6 -> 4) is being copied into 35 having contribution from 35 (26 -> 1)? or is it step backward since 14 having contribution from (6 -> 4) is copied into 35 having contribution from (4 -> 17)?
		#  well how does 35 have contribution from 4 -> 17? it is copied into 89 by attention layer 0, so its not a backward step...
		#  but does this necessarily mean its a forward step? 6 -> 4 and 26 -> 1 are quite far from each other.
		representation = torch.zeros((2*n,n,d))
		for i in range(n):
			representation[i,i,input[i]] = 1.0
			representation[n+i,i,d-n+i] = 1.0
		attn_representations, attn_out_representations, ff_representations, representations, bias_representations, ff_bias_representations = trace_activation_forward(representation, 0, len(self.model.transformers) - 1, layer_inputs, ff_inputs)
		other_attn_representations, other_attn_out_representations, other_ff_representations, other_representations, other_bias_representations, other_ff_bias_representations = trace_activation_forward(representation, 0, len(self.model.transformers) - 1, other_layer_inputs, other_ff_inputs)
		prediction_index = int(torch.argmax(torch.linalg.vector_norm(representations[-1][:,:,:], dim=-1)[:n,n-1]))
		assert input[prediction_index] == prediction
		contributions_to_search = []
		#trace_contributions(len(self.model.transformers) - 1, [(n-1,prediction_index)], attn_inputs, attn_matrices, attn_representations, attn_out_representations, contributions_to_search)
		#trace_contributions(4, [(35,48)], attn_inputs, attn_matrices, attn_representations, attn_out_representations, contributions_to_search)
		#trace_contributions(4, [(89,23)], attn_inputs, attn_matrices, attn_representations, attn_out_representations, contributions_to_search)
		products, bias_product = check_copyr(2, 23, 11, attn_inputs, attn_matrices, attn_representations)
		other_products, other_bias_product = check_copyr(2, 23, 11, other_attn_inputs, other_attn_matrices, other_attn_representations)
		import pdb; pdb.set_trace()
		trace_contributions(1, [(11,23)], attn_inputs, attn_matrices, attn_representations, attn_out_representations, contributions_to_search)

		other_layer_inputs, other_attn_inputs, other_attn_pre_softmax, other_attn_matrices, other_v_outputs, other_attn_linear_inputs, other_attn_outputs, other_A_matrices, other_ff_inputs, other_ff_parameters, other_prediction = forward(self.model, x, mask, 0, False, [(4, 35, layer_inputs[4][35,:] - representations[4][30,35,:] + representations[3][30,35,:])], None)
		import pdb; pdb.set_trace()

		def compute_copy_distances(src_pos, dst_pos):
			copy_distances = []
			for k,copy_path in tracked_indices:
				if k != dst_pos or (src_pos != None and copy_path[0] != src_pos):
					continue
				copy_distance = []
				print(copy_path)
				for j in range(1, len(copy_path)):
					prev = copy_path[j - 1]
					curr = copy_path[j]
					length = path_length(curr, prev)
					if length == -1:
						length = path_length(prev, curr)
						if length == -1:
							length = None
						else:
							length = -length
					copy_distance.append(length)
				copy_distances.append(copy_distance)
			return copy_distances

		tracked_indices = [(j,[]) for j in range(n-5) if j % 3 != 1 and j >= start_index]
		tracked_indices += [(n-4,[]),(n-3,[]),(n-1,[])]
		#tracked_indices = [(n-3,[])]
		for i, transformer in enumerate(self.model.transformers):
			new_tracked_indices = []
			for tracked_index, src in tracked_indices:
				# determine the destination rows that this attention layer copies the row `tracked_index` into
				very_large = torch.nonzero(attn_matrices[i][:,tracked_index] >= 0.1)
				for dst_index in very_large:
					dst_index = dst_index.item()
					new_tracked_indices.append((dst_index,src+[tracked_index]))
				large = torch.nonzero(attn_matrices[i][:,tracked_index] >= 0.03)
				if len(large) > num_edges / 2:
					# this could be inverse encoded
					if torch.sum(large % 3 == 2) > 0.75 * len(large):
						# the large weights correspond to source vertices of edges
						small = [j for j in range(2, n, 3) if j >= start_index and attn_matrices[i][j,tracked_index] < 0.02]
						for dst_index in small:
							new_tracked_indices.append((dst_index,src+[tracked_index]))
					elif torch.sum(large % 3 == 0) > 0.75 * len(large):
						# the large weights correspond to target vertices of edges
						small = [j for j in range(0, n, 3) if j >= start_index and attn_matrices[i][j,tracked_index] < 0.02]
						for dst_index in small:
							new_tracked_indices.append((dst_index,src+[tracked_index]))
					else:
						small = torch.nonzero(attn_matrices[i][:,tracked_index] < 0.02)
						if len(small) < 0.25 * n:
							for dst_index in small:
								dst_index = dst_index.item()
								new_tracked_indices.append((dst_index,src+[tracked_index]))
			tracked_indices = new_tracked_indices

		print(compute_copy_distances(n-3, n-1))
		import pdb; pdb.set_trace()

		# work backwards and identify which inputs contributed most to the output
		representations = [(n-1,prediction,True)]
		if self.model.transformers[-1].ff:
			raise Exception('Not implemented')
		for i, transformer in reversed(list(enumerate(self.model.transformers))):
			ff_input = ff_inputs[i]
			ff_output = layer_inputs[i + 1] - ff_input
			# check if there is any significant signal coming from the bias vectors
			(weight0, weight3, bias0, bias3) = ff_parameters[i]
			if weight0 != None:
				#if torch.max(bias3) / torch.max(ff_output) > 0.3:
				#	print('Bias of second linear layer in FF layer {} has large component.'.format(i + 1))
				#if torch.max(bias0) / torch.max(ff_output) > 0.3:
				#	print('Bias of first linear layer in FF layer {} has large component.'.format(i + 1))
				#relevance = torch.matmul(relevance, torch.diag(torch.ones(n)) + torch.matmul(weight3, weight0))
				#relevance = torch.linalg.solve(torch.diag(torch.ones(n)) + torch.matmul(weight3, weight0), relevance, left=False)
				new_representations = []
				for row, element, sign in representations:
					replaced_representation = False
					if sign == True and element in torch.nonzero(ff_input[row] > 0.3 * torch.max(ff_input[row])):
						new_representations.append((row,element,True))
						replaced_representation = True
					if sign == False and element in torch.nonzero(ff_input[row] < 0.3 * torch.min(ff_input[row])):
						new_representations.append((row,element,False))
						replaced_representation = True
					if not replaced_representation:
						perturbed_output = self.model.transformers[i].ff(self.model.transformers[i].ln_ff(ff_input[row].repeat(d+1,1).fill_diagonal_(0.0)))[:,element]
						candidate_elements = torch.nonzero(torch.abs(perturbed_output - perturbed_output[-1]) > 0.2).tolist()
						for candidate_element in candidate_elements:
							candidate_element = candidate_element[0]
							if candidate_element in torch.nonzero(ff_input[row] > 0.3 * torch.max(ff_input[row])):
								if (row,candidate_element,True) not in new_representations:
									new_representations.append((row,candidate_element,True))
									replaced_representation = True
									if not quiet:
										print("FF layer {} causes row {} element {} to be {} because element {} is large".format(i,row,element,"large" if sign else "small",candidate_element))
							if candidate_element in torch.nonzero(ff_input[row] < 0.3 * torch.min(ff_input[row])):
								if (row,candidate_element,False) not in new_representations:
									new_representations.append((row,candidate_element,False))
									replaced_representation = True
									if not quiet:
										print("FF layer {} causes row {} element {} to be {} because element {} is small".format(i,row,element,"large" if sign else "small",candidate_element))
						if not replaced_representation:
							print("FF layer {} alters representation at row {}, element {}".format(i,row,element))
				representations = new_representations

			attn_input = attn_inputs[i]
			attn_output = ff_input - attn_input
			#relevance = torch.matmul(relevance, torch.diag(torch.ones(n)) + attn_matrices[i])
			#relevance = torch.linalg.solve(torch.diag(torch.ones(n)) + attn_matrices[i], relevance)
			new_representations = []
			visited_row_copies = []
			def add_representation(row, sign):
				for new_element in torch.nonzero(attn_input[row] > 0.5 * torch.max(attn_input[row])).tolist():
					new_element = new_element[0]
					if new_element == d:
						continue
					if (row,new_element,sign) not in new_representations:
						new_representations.append((row,new_element,sign))
			def add_representation_with_element(row, element, sign):
				if element == d:
					return
				if (row,element,sign) not in new_representations:
					new_representations.append((row,element,sign))
			def check_copy(i, row, j):
				attn_input = attn_inputs[i]
				if not quiet:
					print('Attention layer {} is copying row {} into row {} with weight {} because:'.format(i,j,row,attn_matrices[i][row,j]))
				# determine why row j is being copied from
				attn_input_prime = torch.cat((attn_input, torch.ones((n,1))), 1)
				right_products = torch.matmul(attn_input_prime[row,:], A_matrices[i]) * attn_input_prime[j,:]
				step_conditions = []
				for right_index in torch.nonzero(right_products[:-1] > torch.max(right_products[:-1]) - 1.0).tolist():
					right_index = right_index[0]
					if attn_input_prime[j,right_index] > 0.0:
						add_representation_with_element(j,right_index,True)
						right_sign = True
					else:
						add_representation_with_element(j,right_index,False)
						right_sign = False
					left_products = attn_input_prime[row,:] * A_matrices[i][:,right_index].reshape(1,-1)[0]
					if not quiet:
						print('  Row {} at index {} has value {}'.format(j, right_index, attn_input_prime[j,right_index]))
					if len(torch.nonzero(left_products > torch.max(left_products) - 1.0)) > 10:
						continue
					for left_index in torch.nonzero(left_products > torch.max(left_products) - 1.0).tolist():
						left_index = left_index[0]
						if not quiet:
							print('  Row {} at index {} has value {}, and A[{},{}]={}'.format(row, left_index, attn_input_prime[row,left_index], left_index, right_index, A_matrices[i][left_index,right_index]))
						if attn_input_prime[row,left_index] > 0.0:
							add_representation_with_element(row,left_index,True)
							left_sign = True
						else:
							add_representation_with_element(row,left_index,False)
							left_sign = False
						step_conditions.append((j,right_index,right_sign,row,left_index,left_sign))
				# identify the transformation of this attention layer
				self.algorithm[i].add_case(step_conditions, (j,row), input) #(j,row,attn_matrices[i][row,j]), input)
			import pdb; pdb.set_trace()
			for row, element, sign in representations:
				for j in range(n):
					if not attn_matrices[i][row,j] > 0.10:
						continue
					if sign == True and element not in torch.nonzero(attn_input[j,:] > 0.4 * torch.max(attn_input[j,:])):
						continue
					if sign == False and element not in torch.nonzero(attn_input[j,:] < 0.4 * torch.min(attn_input[j,:])):
						continue
					if (row,j) in visited_row_copies:
						continue
					visited_row_copies.append((row,j))

					add_representation_with_element(j,element,sign)
					check_copy(i, row, j)

				# check if the residual connection is copying the representation
				if sign == True and element in torch.nonzero(attn_input[row,:] > 0.4 * torch.max(attn_input[row,:])):
					add_representation_with_element(row,element,sign)
					if not quiet:
						print('Residual connection for attention layer {} is copying row {} into row {}.'.format(i,row,row))
					self.algorithm[i].add_residual_copy(row)
				if sign == False and element in torch.nonzero(attn_input[row,:] < 0.4 * torch.min(attn_input[row,:])):
					add_representation_with_element(row,element,sign)
					if not quiet:
						print('Residual connection for attention layer {} is copying row {} into row {}.'.format(i,row,row))
					self.algorithm[i].add_residual_copy(row)
			representations = new_representations

			if not quiet:
				print('At the input to attention layer {}, the representation is:'.format(i))
				for row, element, sign in representations:
					if sign == True:
						print('  In row {}, value at index {} is large.'.format(row,element))
					else:
						print('  In row {}, value at index {} is small.'.format(row,element))

		return prediction


if __name__ == "__main__":
	seed(1)
	torch.manual_seed(1)
	np.random.seed(1)

	torch.set_printoptions(sci_mode=False)
	from sys import argv, exit
	if len(argv) != 4:
		print("Usage: trace_circuit [checkpoint_filepath] [lookahead] [num_samples]")
		exit(1)

	if not torch.cuda.is_available():
		print("ERROR: CUDA device is not available.")
		#from sys import exit
		#exit(-1)
		device = torch.device('cpu')
	else:
		device = torch.device('cuda')

	filepath = argv[1]
	tfm_model, _, _, _ = torch.load(filepath, map_location=device)
	for transformer in tfm_model.transformers:
		if not hasattr(transformer, 'pre_ln'):
			transformer.pre_ln = True
	tracer = TransformerTracer(tfm_model)

	#input = [22, 21,  5, 19, 21, 11,  5, 21, 10,  3, 21,  4, 10, 21,  9,  4, 21,  9, 11, 23,  9,  3, 20,  9]
	#input = [22, 22, 22, 21, 14,  3, 21, 14,  6, 21, 18, 14, 21,  3,  1, 21,  6, 16, 23, 18,  1, 20, 18, 14]
	#input = [46, 45,  3, 19, 45, 18, 39, 45, 36, 15, 45, 24, 42, 45, 37,  3, 45, 37, 36, 45, 23, 32, 45,  8, 24, 45, 19, 30, 45, 15, 23, 45, 39, 40, 45, 40, 34, 45, 30, 18, 45, 32,  8, 47, 37, 34, 44, 37]
	#input = [46, 46, 46, 46, 46, 46, 46, 45, 31, 39, 45, 42,  4, 45, 21,  7, 45, 19, 20, 45, 13, 22, 45,  7, 42, 45, 20, 21, 45, 17, 19, 45, 17, 31, 45, 10, 14, 45, 39, 10, 45, 14, 13, 47, 17,  4, 44, 17]
	#input = [62, 62, 62, 62, 62, 61, 15,  8, 61, 11, 18, 61,  9,  5, 61, 19, 14, 61, 19, 17, 61,  1, 11, 61,  6,  7, 61, 10,  3, 61,  2,  1, 61, 13, 10, 61, 12,  4, 61, 17, 16, 61,  7, 12, 61, 14,  2, 61,  3,  9, 61, 16, 15, 61, 18,  6, 61,  8, 13, 63, 19,  4, 60, 19]
	#input = [44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 43, 21, 40, 43, 21, 22, 43, 22, 34, 43, 13,  3, 43, 31,  2, 43, 24, 13, 43,  4, 41, 43, 30, 31, 43, 17, 15, 43, 34, 38, 43,  3, 28, 43, 18, 17, 43, 14, 24, 43,  2,  4, 43, 32, 30, 43, 40, 18, 43, 15, 14, 43, 38, 32, 45, 21, 28, 42, 21]
	input = [31, 31, 31, 31, 31, 31, 31, 30,  7, 23, 30,  9, 22, 30,  6,  4, 30, 6, 10, 30, 25, 19, 30, 17,  9, 30, 17, 16, 30,  1, 14, 30, 11, 21, 30, 26,  1, 30, 12, 11, 30, 14,  6, 30, 15, 25, 30,  4, 17, 30, 24, 28, 30, 19,  8, 30, 27, 26, 30, 27, 12, 30, 27,  5, 30, 22, 24, 30, 8,  3, 30, 18, 15, 30,  3,  7, 30,  3, 10, 30,  3,  2, 30, 21, 18, 32, 27, 28, 29, 27]
	input = torch.LongTensor(input).to(device)
	other_input = input.clone().detach()
	other_input[-3] = 7
	#other_input[next(i for i in range(len(input)) if input[i] ==  1 and input[i+1] == 14) + 1] = 21
	#other_input[next(i for i in range(len(input)) if input[i] == 11 and input[i+1] == 21) + 1] = 14
	#other_input[next(i for i in range(len(input)) if input[i] ==  6 and input[i+1] ==   4) + 0] = 18
	#other_input[next(i for i in range(len(input)) if input[i] ==  6 and input[i+1] ==   4) + 1] = 15
	#other_input[next(i for i in range(len(input)) if input[i] == 18 and input[i+1] ==  15) + 0] =  6
	#other_input[next(i for i in range(len(input)) if input[i] == 18 and input[i+1] ==  15) + 1] =  4
	#other_input[next(i for i in range(len(input)) if input[i] == 17 and input[i+1] ==  9) + 0] = 25
	#other_input[next(i for i in range(len(input)) if input[i] == 17 and input[i+1] ==  9) + 1] = 19
	#other_input[next(i for i in range(len(input)) if input[i] == 25 and input[i+1] == 19) + 0] = 17
	#other_input[next(i for i in range(len(input)) if input[i] == 25 and input[i+1] == 19) + 1] =  9
	#tracer.trace2(input)
	#prediction = tracer.trace(input, other_input, quiet=False)

	suffix = filepath[filepath.index('inputsize')+len('inputsize'):]
	max_input_size = int(suffix[:suffix.index('_')])
	max_vertex_id = (max_input_size - 5) // 3

	NUM_SAMPLES = int(argv[3])
	inputs,outputs = generate_eval_data(max_input_size, min_path_length=1, distance_from_start=1, distance_from_end=-1, lookahead_steps=int(argv[2]), num_paths_at_fork=None, num_samples=NUM_SAMPLES)
	inputs = torch.LongTensor(inputs).to(device)

	def print_computation_graph(root, input):
		queue = [root]
		visited = []
		while len(queue) != 0:
			node = queue.pop()
			visited.append(node)
			for predecessor in node.predecessors:
				queue.insert(0, predecessor)
		printed = []
		for node in reversed(visited):
			for predecessor in node.predecessors:
				if predecessor in printed:
					continue
				printed.append(predecessor)
				print('Layer {}: copy from {} ({}) into {} ({}); explanations: {}, direction: {}, src reachability: {}'.format(predecessor.layer, predecessor.row_id, input[predecessor.row_id], node.row_id, input[node.row_id], predecessor.op_explanations[predecessor.successors.index(node)], predecessor.copy_directions[predecessor.successors.index(node)], sorted(predecessor.reachable)))

		from analyze import print_graph
		print_graph(input.cpu().detach().numpy())

	#root, forward_edges, important_ops, path_merge_explainable, prediction = tracer.trace2(inputs[1,:])
	#print_computation_graph(root, inputs[1,:])
	#import pdb; pdb.set_trace()

	total = 0
	num_correct_vs_explainable = torch.zeros((2,2))
	num_correct_vs_explainable_recall = torch.zeros((2,2))
	aggregated_copy_directions = []
	aggregated_example_copy_directions = []
	aggregated_dist_copy_directions = []
	aggregated_op_explanations = []
	for i in range(len(tfm_model.transformers)):
		aggregated_copy_directions.append({})
		aggregated_example_copy_directions.append({})
		aggregated_dist_copy_directions.append({})
		aggregated_op_explanations.append({})
	aggregated_path_merge_lengths = [{} for i in range(len(tfm_model.transformers))]

	from functools import cmp_to_key
	def compare(x, y):
		if x == None:
			if y != None:
				return -1
			return 0
		elif y == None:
			return 1
		elif type(x) == str:
			if type(y) != str:
				return -1
			elif x < y:
				return -1
			elif x > y:
				return 1
			return 0
		elif type(y) == str:
			return 1
		elif type(x) == tuple:
			if type(y) != tuple:
				return -1
			if len(x) < len(y):
				return -1
			elif len(x) > len(y):
				return 1
			for i in range(len(x)):
				c = compare(x[i], y[i])
				if c != 0:
					return c
		elif type(y) == tuple:
			return 1
		elif x < y:
			return -1
		elif x > y:
			return 1
		return 0

	def explanation_list_to_str(explanations):
		if type(explanations) == tuple:
			tokens = [str(e) for e in explanations if e <= max_vertex_id]
			positions = [str(e - (max_vertex_id + 5)) for e in explanations if e >= max_vertex_id + 5]
			if len(tokens) == 0:
				if len(positions) == 0:
					raise Exception('Explanation tuple is empty.')
				elif len(positions) == 1:
					return 'position ' + ','.join(positions)
				else:
					return 'positions ' + ','.join(positions)
			elif len(tokens) == 1:
				if len(positions) == 0:
					return 'token ' + ','.join(tokens)
				elif len(positions) == 1:
					return 'token ' + ','.join(tokens) + ' and position ' + ','.join(positions)
				else:
					return 'token ' + ','.join(tokens) + ' and positions ' + ','.join(positions)
			else:
				if len(positions) == 0:
					return 'tokens ' + ','.join(tokens)
				elif len(positions) == 1:
					return 'tokens ' + ','.join(tokens) + ' and position ' + ','.join(positions)
				else:
					return 'tokens ' + ','.join(tokens) + ' and positions ' + ','.join(positions)
		elif type(explanations) == int:
			if explanations <= max_vertex_id:
				return 'token ' + str(explanations)
			else:
				return 'position ' + str(explanations - (max_vertex_id + 5))
		else:
			raise Exception('Invalid explanation type.')

	def explanation_to_str(explanation):
		if type(explanation) == str:
			return explanation
		elif type(explanation) == tuple:
			if explanation[1] == None:
				return 'dst activation at ' + explanation_list_to_str(explanation[0])
			elif explanation[0] == None:
				return 'src activation at ' + explanation_list_to_str(explanation[1])
			else:
				return 'src activation at ' + explanation_list_to_str(explanation[1]) + '; dst activation at ' + explanation_list_to_str(explanation[0])
		elif type(explanation) == int:
			return 'matching ' + explanation_list_to_str(explanation)
		else:
			raise Exception('Invalid explanation type.')

	def print_summary():
		print('\nAggregated results:')
		for i in range(len(aggregated_copy_directions)):
			total_copy_directions = sum(aggregated_copy_directions[i].values())
			total_example_copy_directions = sum(aggregated_example_copy_directions[i].values())
			total_dist_copy_directions = sum(aggregated_dist_copy_directions[i].values())
			total_op_explanations = sum(aggregated_op_explanations[i].values())
			print(' Layer {} copy directions:'.format(i))
			for copy_direction in sorted(aggregated_copy_directions[i].keys()):
				print('  {}: {} / {}'.format(copy_direction, aggregated_copy_directions[i][copy_direction], total_copy_directions))
			print(' Copy directions by distance from start node:')
			for copy_direction in sorted(aggregated_dist_copy_directions[i].keys(), key=cmp_to_key(compare)):
				print('  {}: {} / {}'.format(copy_direction, aggregated_dist_copy_directions[i][copy_direction], total_dist_copy_directions))
			#print(' Copy directions per example:')
			#for copy_direction in sorted(aggregated_example_copy_directions[i].keys()):
			#	print('  {}: {} / {}'.format(copy_direction, aggregated_example_copy_directions[i][copy_direction], total_example_copy_directions))
			print(' Layer {} op explanations:'.format(i))
			for explanation in sorted(aggregated_op_explanations[i].keys(), key=cmp_to_key(compare)):
				print('  {}: {} / {}'.format(explanation_to_str(explanation), aggregated_op_explanations[i][explanation], total_op_explanations))
			print(' Layer {} path merge lengths:'.format(i))
			for merge_pattern in sorted(aggregated_path_merge_lengths[i].keys()):
				print('  {}: {}'.format(merge_pattern, aggregated_path_merge_lengths[i][merge_pattern]))

	from sys import stdout
	for i in range(NUM_SAMPLES):
		try:
			root, forward_edges, important_ops, path_merge_explainable, prediction = tracer.trace2(inputs[i,:])
		except NoUnusedVertexIDs:
			print('Input has no unused vertex IDs. Skipping...')
			continue

		if path_merge_explainable != None:
			# filter out nodes that are not along the path from the input to the root
			new_path_merge_explainable = []
			queue = [root]
			while len(queue) != 0:
				node = queue.pop()
				if node not in path_merge_explainable:
					continue
				new_path_merge_explainable.append(node)
				for predecessor in node.predecessors:
					queue.append(predecessor)
			path_merge_explainable = new_path_merge_explainable

			# check that the final node's activations contain the path from the start vertex to the goal vertex
			def is_reachable(input, start, end, subgraph):
				if input[start] > max_vertex_id or input[end] > max_vertex_id:
					return False
				queue = [input[start]]
				while len(queue) != 0:
					current = queue.pop()
					if current == input[end]:
						return True
					for child in forward_edges[current]:
						if child in subgraph:
							queue.append(child)
				return False
			if root in path_merge_explainable:
				vertices = [i-(max_vertex_id+5) for i in root.reachable if i >= max_vertex_id+5]
				subgraph = [int(inputs[i,v]) for v in vertices]
				if not is_reachable(inputs[i,:], max_input_size-4, max_input_size-3, subgraph):
					path_merge_explainable.remove(root)

		if path_merge_explainable != None and root in path_merge_explainable:
			def shortest_distances(input, start, subgraph=None):
				if input[start] > max_vertex_id:
					return {input[start]:0}
				queue = [(input[start],0)]
				distances = {}
				while len(queue) != 0:
					current, distance = queue.pop()
					if current in distances and distance >= distances[current]:
						continue
					distances[current] = distance
					for child in forward_edges[current]:
						if subgraph == None or child in subgraph:
							queue.append((child,distance+1))
				return distances

			def reachable_path_length(input, reachable):
				vertices = [i-(max_vertex_id+5) for i in reachable if i >= max_vertex_id+5]
				subgraph = [int(input[v]) for v in vertices]
				longest_distance = 0
				for start in vertices:
					longest_distance = max(longest_distance, max(shortest_distances(input, start, subgraph).values()))
				return longest_distance

			path_merge_lengths = [{} for i in range(len(tfm_model.transformers))]
			def record_path_length(node, successor):
				src_path_length = reachable_path_length(inputs[i,:], node.reachable)
				dst_path_length = reachable_path_length(inputs[i,:], successor.reachable) - src_path_length
				if (src_path_length,dst_path_length) not in path_merge_lengths[node.layer]:
					path_merge_lengths[node.layer][(src_path_length,dst_path_length)] = 0
				path_merge_lengths[node.layer][(src_path_length,dst_path_length)] += 1

			example_copy_directions = [{} for i in range(len(tfm_model.transformers))]
			for node in path_merge_explainable:
				for k in range(len(node.successors)):
					successor = node.successors[k]
					if successor not in path_merge_explainable:
						continue
					record_path_length(node, successor)
					if node.copy_directions[k] != None:
						distances_from_start = shortest_distances(inputs[i,:], max_input_size-4)
						if int(inputs[i,node.row_id]) == int(inputs[i,max_input_size-4]):
							distance = 0
						elif int(inputs[i,node.row_id]) in distances_from_start:
							distance = distances_from_start[int(inputs[i,node.row_id])]
						else:
							distances_to_start = shortest_distances(inputs[i,:], node.row_id)
							if int(inputs[i,max_input_size-4]) in distances_to_start:
								distance = -distances_to_start[int(inputs[i,max_input_size-4])]
							else:
								distance = None
						for direction in node.copy_directions[k]:
							if direction not in aggregated_copy_directions[node.layer]:
								aggregated_copy_directions[node.layer][direction] = 1
							else:
								aggregated_copy_directions[node.layer][direction] += 1
							if direction not in example_copy_directions[node.layer]:
								example_copy_directions[node.layer][direction] = 1
							else:
								example_copy_directions[node.layer][direction] += 1
							if (distance,direction) not in aggregated_dist_copy_directions[node.layer]:
								aggregated_dist_copy_directions[node.layer][(distance,direction)] = 1
							else:
								aggregated_dist_copy_directions[node.layer][(distance,direction)] += 1
					if node.op_explanations[k] == None:
						# this is a residual copy
						node.op_explanations[k] = ['residual']
					for explanation in node.op_explanations[k]:
						if explanation not in aggregated_op_explanations[node.layer]:
							aggregated_op_explanations[node.layer][explanation] = 1
						else:
							aggregated_op_explanations[node.layer][explanation] += 1
			for l in range(len(example_copy_directions)):
				k = tuple(sorted(example_copy_directions[l].items()))
				if len(k) == 0:
					continue
				if k not in aggregated_example_copy_directions[l]:
					aggregated_example_copy_directions[l][k] = 1
				else:
					aggregated_example_copy_directions[l][k] += 1

			for l in range(len(path_merge_lengths)):
				for k,v in path_merge_lengths[l].items():
					if k not in aggregated_path_merge_lengths[l]:
						aggregated_path_merge_lengths[l][k] = 0
					aggregated_path_merge_lengths[l][k] += v

		num_explainable_edges = 0
		if path_merge_explainable != None:
			# compute the number of edges in `important_ops` that are in `path_merge_explainable`
			for node in path_merge_explainable:
				for successor in node.successors:
					if [node.row_id, successor.row_id] in important_ops[node.layer]:
						num_explainable_edges += 1
		if important_ops != None:
			num_important_edges = sum([len(l) for l in important_ops])
		else:
			num_important_edges = 0

		is_explainable = (root != None and root in path_merge_explainable)
		is_correct = (prediction == outputs[i])
		num_correct_vs_explainable[int(is_correct),int(is_explainable)] += 1
		if important_ops != None:
			if num_explainable_edges > num_important_edges:
				import pdb; pdb.set_trace()
			num_correct_vs_explainable_recall[int(is_correct),int(is_explainable)] += num_explainable_edges / num_important_edges
		total += 1
		print('[iteration {}]'.format(i))
		print('  Accuracy: {}'.format(torch.sum(num_correct_vs_explainable[1,:]) / total))
		print('  Fraction of inputs explainable by path-merging algorithm: {}'.format(torch.sum(num_correct_vs_explainable[:,1]) / total))
		print('  Fraction of inputs that are correct and explainable: {}'.format(torch.sum(num_correct_vs_explainable[1,1]) / total))
		print('  Fraction of inputs that are incorrect and explainable: {}'.format(torch.sum(num_correct_vs_explainable[0,1]) / total))
		print('  Fraction of inputs that are correct and unexplainable: {}'.format(torch.sum(num_correct_vs_explainable[1,0]) / total))
		print('  Fraction of inputs that are incorrect and unexplainable: {}'.format(torch.sum(num_correct_vs_explainable[0,0]) / total))
		#print('  "Recall" of inputs that are correct and explainable: {}'.format(torch.sum(num_correct_vs_explainable_recall[1,1]) / num_correct_vs_explainable[1,1]))
		#print('  "Recall" of inputs that are incorrect and explainable: {}'.format(torch.sum(num_correct_vs_explainable_recall[0,1]) / num_correct_vs_explainable[0,1]))
		#print('  "Recall" of inputs that are correct and unexplainable: {}'.format(torch.sum(num_correct_vs_explainable_recall[1,0]) / num_correct_vs_explainable[1,0]))
		#print('  "Recall" of inputs that are incorrect and unexplainable: {}'.format(torch.sum(num_correct_vs_explainable_recall[0,0]) / num_correct_vs_explainable[0,0]))
		if (i + 1) % 10 == 0:
			print_summary()
		stdout.flush()

	print_summary()
