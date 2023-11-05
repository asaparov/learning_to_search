from random import seed, randrange, shuffle
import numpy as np
import torch
from torch import nn, LongTensor, FloatTensor
from train import generate_example, lookahead_depth
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

	def compute_attention(self, attn_layer, x: torch.Tensor, mask: torch.Tensor):
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
		attn = QK
		attn += mask.type_as(attn) * attn.new_tensor(-1e4)
		attn = attn_layer.attn.dropout(attn.softmax(-1))

		v = x
		return torch.matmul(attn, v), attn, A

	def trace(self, x: torch.Tensor, quiet: bool = True):
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

		# Apply transformer layers sequentially.
		attn_inputs = []
		attn_matrices = []
		A_matrices = []
		ff_inputs = []
		ff_parameters = []
		for i, transformer in enumerate(self.model.transformers):
			# Layer normalizations are performed before the layers respectively.
			attn_inputs.append(x)
			a = transformer.ln_attn(x)
			if transformer.attn.proj_v:
				a, attn_matrix, A = self.compute_attention(transformer.attn, a, mask)
			else:
				a, attn_matrix, A = self.compute_attention(transformer.attn, x, mask)
			attn_matrices.append(attn_matrix)
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
		attn_inputs.append(x)
		prediction = torch.argmax(x[-1,:]).tolist()

		# work backwards and identify which inputs contributed most to the output
		representations = [(n-1,prediction,True)]
		for i, transformer in reversed(list(enumerate(self.model.transformers))):
			ff_input = ff_inputs[i]
			ff_output = attn_inputs[i + 1] - ff_input
			# check if there is any significant signal coming from the bias vectors
			(weight0, weight3, bias0, bias3) = ff_parameters[i]
			if weight0 != None:
				#if torch.max(bias3) / torch.max(ff_output) > 0.3:
				#	print('Bias of second linear layer in FF layer {} has large component.'.format(i + 1))
				#if torch.max(bias0) / torch.max(ff_output) > 0.3:
				#	print('Bias of first linear layer in FF layer {} has large component.'.format(i + 1))
				#relevance = torch.matmul(relevance, torch.diag(torch.ones(n)) + torch.matmul(weight3, weight0))
				#relevance = torch.linalg.solve(torch.diag(torch.ones(n)) + torch.matmul(weight3, weight0), relevance, left=False)
				# TODO: for now, we assume FF layers are approximately identities; but just in case, check here if they make significant changes to the representations
				for row, element, sign in representations:
					if not quiet and sign == True and element not in torch.nonzero(ff_input[row] > 0.4 * torch.max(ff_input[row])):
						print('WARNING: FF layer {} alters the representation at row {}.'.format(i,row))
					if not quiet and sign == False and element not in torch.nonzero(ff_input[row] < 0.4 * torch.min(ff_input[row])):
						print('WARNING: FF layer {} alters the representation at row {}.'.format(i,row))

			attn_input = attn_inputs[i]
			attn_output = ff_input - attn_input
			#relevance = torch.matmul(relevance, torch.diag(torch.ones(n)) + attn_matrices[i])
			#relevance = torch.linalg.solve(torch.diag(torch.ones(n)) + attn_matrices[i], relevance)
			new_representations = []
			visited_row_copies = []
			def add_representation(row, sign):
				if not sign:
					return
				for new_element in torch.nonzero(attn_input[row] > 0.5 * torch.max(attn_input[row])).tolist():
					new_element = new_element[0]
					if (row,new_element,sign) not in new_representations:
						if row == 12:
							import pdb; pdb.set_trace()
						new_representations.append((row,new_element,sign))
			def add_representation_with_element(row, element, sign):
				if not sign:
					return
				if (row,element,sign) not in new_representations:
					new_representations.append((row,element,sign))
			#import pdb; pdb.set_trace()
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
					#add_representation(j,sign)
					if not quiet:
						print('Attention layer {} is copying row {} into row {} with weight {} because:'.format(i,j,row,attn_matrices[i][row,j]))
					# determine why row j is being copied from
					attn_input_prime = torch.cat((attn_input, torch.ones((n,1))), 1)
					right_products = torch.matmul(attn_input_prime[row,:], A_matrices[i]) * attn_input_prime[j,:]
					step_conditions = []
					for right_index in torch.nonzero(right_products > torch.max(right_products) - 1.0).tolist():
						right_index = right_index[0]
						if attn_input_prime[j,right_index] > 0.0:
							add_representation_with_element(j,right_index,True)
							right_sign = True
						else:
						#	add_representation_with_element(j,right_index,False)
							right_sign = False
						left_products = attn_input_prime[row,:] * A_matrices[i][:,right_index].reshape(1,-1)[0]
						if len(torch.nonzero(left_products > torch.max(left_products) - 1.0)) > 10:
							continue
						if not quiet:
							print('  Row {} at index {} has value {}'.format(j, right_index, attn_input_prime[j,right_index]))
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


if __name__ == "__main__":
	seed(1)
	torch.manual_seed(1)
	np.random.seed(1)

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

	tfm_model = torch.load(argv[1], map_location=device)
	tracer = TransformerTracer(tfm_model)

	#input = [22, 21,  5, 19, 21, 11,  5, 21, 10,  3, 21,  4, 10, 21,  9,  4, 21,  9, 11, 23,  9,  3, 20,  9]
	#input = [22, 22, 22, 21, 14,  3, 21, 14,  6, 21, 18, 14, 21,  3,  1, 21,  6, 16, 23, 18,  1, 20, 18, 14]
	#input = [46, 45,  3, 19, 45, 18, 39, 45, 36, 15, 45, 24, 42, 45, 37,  3, 45, 37, 36, 45, 23, 32, 45,  8, 24, 45, 19, 30, 45, 15, 23, 45, 39, 40, 45, 40, 34, 45, 30, 18, 45, 32,  8, 47, 37, 34, 44, 37]
	#input = [46, 46, 46, 46, 46, 46, 46, 45, 31, 39, 45, 42,  4, 45, 21,  7, 45, 19, 20, 45, 13, 22, 45,  7, 42, 45, 20, 21, 45, 17, 19, 45, 17, 31, 45, 10, 14, 45, 39, 10, 45, 14, 13, 47, 17,  4, 44, 17]
	#input = torch.LongTensor(input).to(device)
	#tracer.trace(input, quiet=False)
	#import pdb; pdb.set_trace()

	token_dim = 48
	position_dim = 48
	embedding_dim = 96

	min_path_length = 2
	lookahead_steps = None
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
			g, start, end, paths = generate_example(randrange(min_vertices, 14), 4, position_dim - 5, get_shortest_paths=False, lookahead=lookahead_steps)
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
			tracer.trace(input)
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
