from random import seed, randrange, shuffle
import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from train import generate_example, lookahead_depth
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


class TransformerTracer:
	def __init__(self, tfm_model):
		self.model = tfm_model
		self.model.eval()
		self.algorithm = [AlgorithmStep() for i in range(len(self.model.transformers))]

	def compute_attention(self, layer_index: int, attn_layer, x: torch.Tensor, mask: torch.Tensor):
		n, d = x.shape[0], x.shape[1]
		k_params = {k:v for k,v in attn_layer.proj_k.named_parameters()}
		q_params = {k:v for k,v in attn_layer.proj_q.named_parameters()}
		P_k = k_params['weight']
		P_q = q_params['weight']
		U_k = torch.cat((P_k,k_params['bias'].unsqueeze(1)),1)
		U_q = torch.cat((P_q,q_params['bias'].unsqueeze(1)),1)
		A = torch.matmul(U_q.transpose(-2,-1),U_k)
		x_prime = torch.cat((x, torch.ones((n,1))), 1)
		QK = torch.matmul(torch.matmul(x_prime, A), x_prime.transpose(-2,-1)) / math.sqrt(d)
		attn_pre_softmax = QK + mask.type_as(QK) * QK.new_tensor(-1e4)
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

	def forward(self, x: torch.Tensor, mask: torch.Tensor, start_layer: int, start_at_ff: bool, perturbations):
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
		for transformer in self.model.transformers[start_layer:]:
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
				if transformer.attn.proj_v:
					a, pre_softmax, attn_matrix, v, attn_linear_input, A = self.compute_attention(current_layer, transformer.attn, a, mask)
				else:
					a, pre_softmax, attn_matrix, v, attn_linear_input, A = self.compute_attention(current_layer, transformer.attn, x, mask)
				v_outputs.append(v)
				attn_matrices.append(attn_matrix)
				attn_pre_softmax.append(pre_softmax)
				attn_linear_inputs.append(attn_linear_input)
				attn_outputs.append(a)
				A_matrices.append(A)
				x = x + a

			ff_inputs.append(x)
			if transformer.ff:
				x = x + transformer.ff(transformer.ln_ff(x))
				ff_params = {k:v for k,v in transformer.ff.named_parameters()}
				ff_parameters.append((ff_params['0.weight'].T, ff_params['3.weight'].T, ff_params['0.bias'], ff_params['3.bias']))
			else:
				ff_parameters.append((None, None, None, None))
			#print(x[-1,:])
			current_layer += 1
		layer_inputs.append(x)
		token_dim = self.model.token_embedding.shape[0]

		if self.model.ln_head:
			x = self.model.ln_head(x)
		if self.model.positional_embedding is not None:
			if len(x.shape) == 2:
				x = x[:,:-self.model.positional_embedding.shape[0]]
			else:
				x = x[:,:,:-self.model.positional_embedding.shape[0]]
		if type(self.model.token_embedding) == TokenEmbedding:
			x = self.model.token_embedding(x, transposed=True)
		else:
			x = torch.matmul(x, self.model.token_embedding.transpose(0, 1))

		prediction = torch.argmax(x[-1,:token_dim]).tolist()

		return layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction

	'''def input_derivatives(self, x: torch.Tensor, mask: torch.Tensor, dx: float, start_layer: int, start_at_ff: bool, old_prediction: int, last_layer_output: torch.Tensor):
		dfdx = torch.empty(x.shape)
		for i in range(x.size(0)):
			for j in range(x.size(1)):
				new_x = x.clone().detach()
				new_x[i,j] += dx
				layer_inputs, _, _, _, _, _, _, _, _, _, _ = self.forward(new_x, mask, start_layer, start_at_ff)
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

	def trace(self, x: torch.Tensor, other_x: torch.Tensor, quiet: bool = True):
		n = x.shape[0]
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
			pos = self.model.positional_embedding.unsqueeze(0).expand(n, -1, -1)
		x = torch.cat((x, pos), -1)
		x = self.model.dropout_embedding(x)
		d = x.shape[1]

		# Use token embedding and positional embedding layers.
		other_input = other_x # store the input for keeping track the code paths executed by each input
		other_x = self.model.token_embedding[other_x]
		if len(other_x.shape) == 2:
			pos = self.model.positional_embedding
		else:
			pos = self.model.positional_embedding.unsqueeze(0).expand(n, -1, -1)
		other_x = torch.cat((other_x, pos), -1)
		other_x = self.model.dropout_embedding(other_x)

		other_layer_inputs, other_attn_inputs, other_attn_pre_softmax, other_attn_matrices, other_v_outputs, other_attn_linear_inputs, other_attn_outputs, other_A_matrices, other_ff_inputs, other_ff_parameters, other_prediction = self.forward(other_x, mask, 0, False, None)

		layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction = self.forward(x, mask, 0, False, None)
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

		def trace_activation(i, row, vec):
			# first undo the pre-attention layer norm
			vec_layer_input = ((vec - self.model.transformers[i].ln_attn.bias) / self.model.transformers[i].ln_attn.weight) * torch.sqrt(torch.var(layer_inputs[i][row,:], correction=0) + self.model.transformers[i].ln_attn.eps) + torch.mean(layer_inputs[i][row,:])

			# check to see how much the FF layer contributes to the representation
			vec = torch.zeros(attn_inputs[i][row,:].shape)
			vec[40] = vec_layer_input[40]
			vec[125] = vec_layer_input[125]
			vec[138] = vec_layer_input[138]
			vec_layer_input = vec
			vec_relu_out = torch.linalg.solve(self.model.transformers[i-1].ff[3].weight.T, (vec_layer_input - self.model.transformers[i-1].ff[3].bias).unsqueeze(1).T, left=False)[0]
			vec_relu_out = vec_relu_out * (self.model.transformers[i-1].ff[2](self.model.transformers[i-1].ff[1](self.model.transformers[i-1].ff[0](self.model.transformers[i-1].ln_ff(ff_inputs[i-1][row,:])))) > 0.0)
			vec_ff_in = torch.linalg.solve(self.model.transformers[i-1].ff[0].weight.T, (vec_relu_out - self.model.transformers[i-1].ff[0].bias).unsqueeze(1).T, left=False)[0]
			print(torch.nonzero(self.model.transformers[i-1].ln_ff(ff_inputs[i-1][row,:])*vec_ff_in > 3.0))
			import pdb; pdb.set_trace()

		vec = torch.zeros(attn_inputs[5][35,:].shape)
		vec[40] = attn_inputs[5][35,40]
		vec[125] = attn_inputs[5][35,125]
		vec[138] = attn_inputs[5][35,138]
		trace_activation(5, 35, vec)

		PADDING_TOKEN = (n - 5) // 3 + 3
		EDGE_PREFIX_TOKEN = (n - 5) // 3 + 2
		start_index = torch.sum(input == PADDING_TOKEN)
		num_edges = torch.sum(input == EDGE_PREFIX_TOKEN)

		forward_edges = []
		for i in range((n - 5) // 3 + 1):
			forward_edges.append([])
		for i in range(2, n-5, 3):
			if i >= start_index:
				forward_edges[input[i].item()].append(input[i+1].item())
		def path_length(start, end):
			if input[start] > (90 - 5) // 3 or input[end] > (90 - 5) // 3:
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
	if len(argv) != 2:
		print("Usage: trace_circuit [checkpoint_filepath]")
		exit(1)

	if True or not torch.cuda.is_available():
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
	tracer = TransformerTracer(tfm_model)

	#input = [22, 21,  5, 19, 21, 11,  5, 21, 10,  3, 21,  4, 10, 21,  9,  4, 21,  9, 11, 23,  9,  3, 20,  9]
	#input = [22, 22, 22, 21, 14,  3, 21, 14,  6, 21, 18, 14, 21,  3,  1, 21,  6, 16, 23, 18,  1, 20, 18, 14]
	#input = [46, 45,  3, 19, 45, 18, 39, 45, 36, 15, 45, 24, 42, 45, 37,  3, 45, 37, 36, 45, 23, 32, 45,  8, 24, 45, 19, 30, 45, 15, 23, 45, 39, 40, 45, 40, 34, 45, 30, 18, 45, 32,  8, 47, 37, 34, 44, 37]
	#input = [46, 46, 46, 46, 46, 46, 46, 45, 31, 39, 45, 42,  4, 45, 21,  7, 45, 19, 20, 45, 13, 22, 45,  7, 42, 45, 20, 21, 45, 17, 19, 45, 17, 31, 45, 10, 14, 45, 39, 10, 45, 14, 13, 47, 17,  4, 44, 17]
	#input = [62, 62, 62, 62, 62, 61, 15,  8, 61, 11, 18, 61,  9,  5, 61, 19, 14, 61, 19, 17, 61,  1, 11, 61,  6,  7, 61, 10,  3, 61,  2,  1, 61, 13, 10, 61, 12,  4, 61, 17, 16, 61,  7, 12, 61, 14,  2, 61,  3,  9, 61, 16, 15, 61, 18,  6, 61,  8, 13, 63, 19,  4, 60, 19]
	#input = [44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 43, 21, 40, 43, 21, 22, 43, 22, 34, 43, 13,  3, 43, 31,  2, 43, 24, 13, 43,  4, 41, 43, 30, 31, 43, 17, 15, 43, 34, 38, 43,  3, 28, 43, 18, 17, 43, 14, 24, 43,  2,  4, 43, 32, 30, 43, 40, 18, 43, 15, 14, 43, 38, 32, 45, 21, 28, 42, 21]
	input = [31, 31, 31, 31, 31, 31, 31, 30,  7, 23, 30,  9, 22, 30,  6,  4, 30, 6, 10, 30, 25, 19, 30, 17,  9, 30, 17, 16, 30,  1, 14, 30, 11, 21, 30, 26,  1, 30, 12, 11, 30, 14,  6, 30, 15, 25, 30, 24, 28, 30,  4, 17, 30, 19,  8, 30, 27, 26, 30, 27, 12, 30, 27,  5, 30, 22, 24, 30, 8,  3, 30, 18, 15, 30,  3,  7, 30,  3, 10, 30,  3,  2, 30, 21, 18, 32, 27, 28, 29, 27]
	input = torch.LongTensor(input).to(device)
	other_input = input.clone().detach()
	#other_input[-3] = 7
	other_input[next(i for i in range(len(input)) if input[i] ==  1 and input[i+1] == 14) + 1] = 21
	other_input[next(i for i in range(len(input)) if input[i] == 11 and input[i+1] == 21) + 1] = 14
	#other_input[next(i for i in range(len(input)) if input[i] ==   6 and input[i+1] ==   4) + 0] = 18
	#other_input[next(i for i in range(len(input)) if input[i] ==   6 and input[i+1] ==   4) + 1] = 15
	#other_input[next(i for i in range(len(input)) if input[i] ==  18 and input[i+1] ==  15) + 0] =  6
	#other_input[next(i for i in range(len(input)) if input[i] ==  18 and input[i+1] ==  15) + 1] =  4
	prediction = tracer.trace(input, other_input, quiet=False)
	import pdb; pdb.set_trace()

	token_dim = tfm_model.token_embedding.shape[0]
	position_dim = (tfm_model.positional_embedding.shape[0] if tfm_model.positional_embedding != None else 0)
	embedding_dim = tfm_model.token_embedding.shape[1] + position_dim

	min_path_length = 2
	lookahead_steps = 9
	distance_from_start = None
	distance_from_end = None

	QUERY_PREFIX_TOKEN = position_dim - 1
	PADDING_TOKEN = position_dim - 2
	EDGE_PREFIX_TOKEN = position_dim - 3
	PATH_PREFIX_TOKEN = position_dim - 4
	min_vertices = max(3, min_path_length)
	num_samples = 10000
	total_predictions = 0
	while total_predictions < num_samples:
		while True:
			num_vertices = randrange(min_vertices, (position_dim - 5) // 3)
			if lookahead_steps != None:
				num_vertices = max(num_vertices, min(lookahead_steps * 2 + 1 + randrange(0, 3), (position_dim - 5) // 3))
			g, start, end, paths = generate_example(num_vertices, 4, (position_dim - 5) // 3, get_shortest_paths=False, lookahead=lookahead_steps)
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
				if len(example) > position_dim:
					continue
				shortest_path_length = min([len(p) for p in paths if path[:j] == p[:j]])
				if distance_from_start != None and j != distance_from_start:
					continue
				if distance_from_end != None and shortest_path_length - j != distance_from_end:
					continue
				if distance_from_start == None and distance_from_end == None:
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

			input = [PADDING_TOKEN] * (position_dim - len(partial_path)) + partial_path
			input = torch.LongTensor(input).to(device)
			prediction = tracer.trace(input)
			total_predictions += 1


	print('\n\nReconstructed algorithm:')
	FREQUENCY_THRESHOLD = 0.01
	for i in range(len(tracer.algorithm)):
		tracer.algorithm[i].identify_operations(embedding_dim, token_dim, position_dim)
		total_inputs = sum([len(case.inputs) for case in tracer.algorithm[i].cases])
		print('\nLayer {}:'.format(i))
		singleton_position_diffs = []
		singleton_token_diffs = []
		for diff,row_indices in tracer.algorithm[i].token_operations.items():
			if len(row_indices) == 1:
				singleton_token_diffs.append(diff)
				continue
			frequency = len(tracer.algorithm[i].token_operation_inputs[diff]) / total_inputs
			if frequency < FREQUENCY_THRESHOLD:
				continue
			row_indices.sort()
			if diff == 0:
				other_t = 't'
			elif diff > 0:
				other_t = 't+' + str(diff)
			else:
				other_t = 't-' + str(-diff)
			print('for any token t in ' + str(row_indices) + ':')
			print('  for row i where x[i,t] is large:')
			print('    for row j where x[j,{}] is large:'.format(other_t))
			print('      copy x[j,:] into x[i,:]')
			print(' [frequency: %.5f]' % frequency)
		for diff,row_indices in tracer.algorithm[i].position_operations.items():
			if len(row_indices) == 1:
				singleton_position_diffs.append(diff)
				continue
			frequency = len(tracer.algorithm[i].position_operation_inputs[diff]) / total_inputs
			if frequency < FREQUENCY_THRESHOLD:
				continue
			row_indices.sort()
			if diff == 0:
				other_p = 'p'
			elif diff > 0:
				other_p = 'p+' + str(diff)
			else:
				other_p = 'p-' + str(-diff)
			print('for any position p in ' + str(row_indices) + ':')
			print('  for row i where x[i,p] is large:')
			print('    for row j where x[j,{}] is large:'.format(other_p))
			print('      copy x[j,:] into x[i,:]')
			print(' [frequency: %.5f]' % frequency)
		for case in tracer.algorithm[i].cases:
			if any([y-v in singleton_token_diffs for (x,y,z,u,v,w) in case.conditions if y < token_dim and v < token_dim]) or any([y-v in singleton_position_diffs for (x,y,z,u,v,w) in case.conditions if y >= embedding_dim - position_dim and v >= embedding_dim - position_dim]):
				frequency = len(case.inputs) / total_inputs
				if frequency < FREQUENCY_THRESHOLD:
					continue
				print(str(case))
				print(' [frequency: %.5f]' % frequency)
		for row in tracer.algorithm[i].residual_copies:
			print('residual copy x[{},:] into x[{},:]'.format(row,row))
