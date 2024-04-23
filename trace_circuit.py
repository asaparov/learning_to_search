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
		n, d = x.shape[-2], x.shape[-1]
		k_params = {k:v for k,v in attn_layer.proj_k.named_parameters()}
		q_params = {k:v for k,v in attn_layer.proj_q.named_parameters()}
		P_k = k_params['weight']
		P_q = q_params['weight']
		U_k = torch.cat((P_k,k_params['bias'].unsqueeze(1)),1)
		U_q = torch.cat((P_q,q_params['bias'].unsqueeze(1)),1)
		A = torch.matmul(U_q.transpose(-2,-1),U_k)
		x_prime = torch.cat((x, torch.ones(x.shape[:-1] + (1,))), -1)
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

	def forward(self, x: torch.Tensor, mask: torch.Tensor, start_layer: int, start_at_ff: bool, end_layer: int, perturbations):
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
		for transformer in self.model.transformers[start_layer:end_layer+1]:
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

		other_layer_inputs, other_attn_inputs, other_attn_pre_softmax, other_attn_matrices, other_v_outputs, other_attn_linear_inputs, other_attn_outputs, other_A_matrices, other_ff_inputs, other_ff_parameters, other_prediction = self.forward(other_x, mask, 0, False, len(self.model.transformers)-1, None)

		layer_inputs, attn_inputs, attn_pre_softmax, attn_matrices, v_outputs, attn_linear_inputs, attn_outputs, A_matrices, ff_inputs, ff_parameters, prediction = self.forward(x, mask, 0, False, len(self.model.transformers)-1, None)
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

			# TODO: this assumes this is a positive copy; implement corresponding logic for negative copies
			sorted_src_products = torch.sort(torch.clamp(new_src_products, max=old_product))
			sorted_dst_products = torch.sort(torch.clamp(new_dst_products, max=old_product))
			src_gaps = torch.nonzero(sorted_src_products.values[1:] - sorted_src_products.values[:-1] > 0.05 * torch.abs(old_product))
			dst_gaps = torch.nonzero(sorted_dst_products.values[1:] - sorted_dst_products.values[:-1] > 0.05 * torch.abs(old_product))
			print('Layer {} copies token at {} into token at {} with weight {}.'.format(i, src, dst, attn_matrices[i][dst,src]))
			if len(src_gaps) == 0:
				# TODO: all `new_src_products` are very close to old_product
				import pdb; pdb.set_trace()
				raise Exception('Not implemented.')
			else:
				src_dependencies = sorted_src_products.indices[:src_gaps[-1]+1]
				print('  src dependencies: ' + ', '.join([get_token_value(dep) for dep in src_dependencies]))
			if len(dst_gaps) == 0:
				# TODO: all `new_dst_products` are very close to old_product
				import pdb; pdb.set_trace()
				raise Exception('Not implemented.')
			else:
				dst_dependencies = sorted_dst_products.indices[:dst_gaps[-1]+1]
				print('  dst dependencies: ' + ', '.join([get_token_value(dep) for dep in dst_dependencies]))

			for src_dep in src_dependencies:
				for dst_dep in dst_dependencies:
					#if src_dep < n - 1 and input[src_dep+1] == input[dst_dep]:
					#	pass
					if src_dep < n and dst_dep < n and src_dep == dst_dep:
						print('This is a token matching step.')

			return src_dependencies, dst_dependencies

		def activation_path_attn(i, dst, src):
			# create perturbed inputs where each input swaps the position of one edge
			edge_indices = [i + 1 for i in range(len(input)) if input[i] == EDGE_PREFIX_TOKEN]
			other_inputs = input.repeat((len(edge_indices), 1))
			for j in range(len(edge_indices)):
				if j != len(edge_indices) - 1:
					# swap the current edge with the last edge
					other_inputs[j,edge_indices[j]] = input[edge_indices[-1]]
					other_inputs[j,edge_indices[j]+1] = input[edge_indices[-1]+1]
					other_inputs[j,edge_indices[-1]] = input[edge_indices[j]]
					other_inputs[j,edge_indices[-1]+1] = input[edge_indices[j]+1]
				else:
					# swap the current edge with the 2nd to last edge
					other_inputs[j,edge_indices[j]] = input[edge_indices[-2]]
					other_inputs[j,edge_indices[j]+1] = input[edge_indices[-2]+1]
					other_inputs[j,edge_indices[-2]] = input[edge_indices[j]]
					other_inputs[j,edge_indices[-2]+1] = input[edge_indices[j]+1]

			# perform forward pass on other_inputs
			other_inputs = self.model.token_embedding[other_inputs]
			if len(other_inputs.shape) == 2:
				pos = self.model.positional_embedding
			else:
				pos = self.model.positional_embedding.unsqueeze(0).expand(other_inputs.shape[0], -1, -1)
			other_inputs = torch.cat((other_inputs, pos), -1)
			other_inputs = self.model.dropout_embedding(other_inputs)

			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = self.forward(other_inputs, mask, 0, False, i, None)

			# try computing the attention dot product but with perturbed dst embeddings
			A = A_matrices[i][:-1,:-1]
			old_product = torch.dot(torch.matmul(attn_inputs[i][dst,:], A), attn_inputs[i][src,:])
			new_src_products = torch.matmul(torch.matmul(attn_inputs[i][dst,:], A), perturb_attn_inputs[i][:,src,:].T)
			new_dst_products = torch.matmul(torch.matmul(perturb_attn_inputs[i][:,dst,:], A), attn_inputs[i][src,:])

			# TODO: this assumes this is a positive copy; implement corresponding logic for negative copies
			sorted_src_products = torch.sort(torch.clamp(new_src_products, max=old_product))
			sorted_dst_products = torch.sort(torch.clamp(new_dst_products, max=old_product))
			src_gaps = torch.nonzero(sorted_src_products.values[1:] - sorted_src_products.values[:-1] > 0.02 * torch.abs(old_product))
			dst_gaps = torch.nonzero(sorted_dst_products.values[1:] - sorted_dst_products.values[:-1] > 0.02 * torch.abs(old_product))
			print('Layer {} copies token at {} into token at {} with weight {}.'.format(i, src, dst, attn_matrices[i][dst,src]))
			if len(src_gaps) == 0:
				# TODO: all `new_src_products` are very close to old_product
				import pdb; pdb.set_trace()
				raise Exception('Not implemented.')
			else:
				src_dependencies = sorted_src_products.indices[:src_gaps[-1]+1]
				print('  src dependencies: ' + ', '.join([get_token_value(edge_indices[dep]+n) for dep in src_dependencies]))
			if len(dst_gaps) == 0:
				# TODO: all `new_dst_products` are very close to old_product
				import pdb; pdb.set_trace()
				raise Exception('Not implemented.')
			else:
				dst_dependencies = sorted_dst_products.indices[:dst_gaps[-1]+1]
				print('  dst dependencies: ' + ', '.join([get_token_value(edge_indices[dep]+n) for dep in dst_dependencies]))

			import pdb; pdb.set_trace()

			# create perturbed inputs where each input swaps the position of one token
			last_edge_index = edge_indices[-1]
			other_inputs = input.repeat((2*n,1))
			'''for j in range(n):
				if input[j] in (PADDING_TOKEN, QUERY_PREFIX_TOKEN, PATH_PREFIX_TOKEN):
					continue
				if j < last_edge_index or j > last_edge_index + 2:
					src_edge_index = last_edge_index
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[j,j] = input[src_edge_index - offset]
				other_inputs[j,src_edge_index - offset] = input[j]'''

			# perform forward pass on other_inputs
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
				if j < last_edge_index or j > last_edge_index + 2:
					src_edge_index = last_edge_index
				else:
					src_edge_index = edge_indices[-2]
				offset = (src_edge_index % 3) - (j % 3)
				if offset > 1:
					offset -= 3
				other_inputs[j,j,:d-n] = other_inputs[j,src_edge_index - offset,:d-n].clone().detach()
				other_inputs[n+j,j,d-n:] = other_inputs[n+j,src_edge_index - offset,d-n:].clone().detach()

			perturb_layer_inputs, perturb_attn_inputs, perturb_attn_pre_softmax, perturb_attn_matrices, perturb_v_outputs, perturb_attn_linear_inputs, perturb_attn_outputs, perturb_A_matrices, perturb_ff_inputs, perturb_ff_parameters, perturb_prediction = self.forward(other_inputs, mask, 0, False, i, None)

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

		#activation_path_attn(5, 89, 35)
		#activation_path_attn(4, 89, 23)
		activation_path_attn(4, 35, 14)
		#activation_path_attn(4, 35, 41)
		#activation_path_attn(3, 23, 11)
		#activation_path_attn(3, 23, 23)
		#activation_path_attn(3, 23, 32)
		#activation_path_attn(3, 23, 88)
		#activation_path_attn(1, 11, 11)
		#activation_path_attn(0, 11, 12)

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

		other_layer_inputs, other_attn_inputs, other_attn_pre_softmax, other_attn_matrices, other_v_outputs, other_attn_linear_inputs, other_attn_outputs, other_A_matrices, other_ff_inputs, other_ff_parameters, other_prediction = self.forward(x, mask, 0, False, [(4, 35, layer_inputs[4][35,:] - representations[4][30,35,:] + representations[3][30,35,:])])
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
	input = [31, 31, 31, 31, 31, 31, 31, 30,  7, 23, 30,  9, 22, 30,  6,  4, 30, 6, 10, 30, 25, 19, 30, 17,  9, 30, 17, 16, 30,  1, 14, 30, 11, 21, 30, 26,  1, 30, 12, 11, 30, 14,  6, 30, 15, 25, 30,  4, 17, 30, 24, 28, 30, 19,  8, 30, 27, 26, 30, 27, 12, 30, 27,  5, 30, 22, 24, 30, 8,  3, 30, 18, 15, 30,  3,  7, 30,  3, 10, 30,  3,  2, 30, 21, 18, 32, 27, 28, 29, 27]
	input = torch.LongTensor(input).to(device)
	other_input = input.clone().detach()
	#other_input[-3] = 7
	#other_input[next(i for i in range(len(input)) if input[i] ==  1 and input[i+1] == 14) + 1] = 21
	#other_input[next(i for i in range(len(input)) if input[i] == 11 and input[i+1] == 21) + 1] = 14
	#other_input[next(i for i in range(len(input)) if input[i] ==  6 and input[i+1] ==   4) + 0] = 18
	#other_input[next(i for i in range(len(input)) if input[i] ==  6 and input[i+1] ==   4) + 1] = 15
	#other_input[next(i for i in range(len(input)) if input[i] == 18 and input[i+1] ==  15) + 0] =  6
	#other_input[next(i for i in range(len(input)) if input[i] == 18 and input[i+1] ==  15) + 1] =  4
	other_input[next(i for i in range(len(input)) if input[i] == 17 and input[i+1] ==  9) + 0] = 25
	other_input[next(i for i in range(len(input)) if input[i] == 17 and input[i+1] ==  9) + 1] = 19
	other_input[next(i for i in range(len(input)) if input[i] == 25 and input[i+1] == 19) + 0] = 17
	other_input[next(i for i in range(len(input)) if input[i] == 25 and input[i+1] == 19) + 1] =  9
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
