import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

plt.style.use('ggplot')
use_negative_encoding = True

def plot_matrix(X, filename):
	fig = plt.gcf()
	ax = plt.gca()
	fig.set_size_inches(3 * X.size(1) / max_input_size, 3 * X.size(0) / max_input_size, forward=True)
	cmax = max(1.0, torch.max(X))
	cax = ax.matshow(np.array(X), vmin=-cmax, vmax=cmax, cmap='RdBu')
	fig.colorbar(cax)
	plt.grid(False)
	fig.savefig(filename, dpi=256)
	plt.clf()

max_input_size = 12
QUERY_PREFIX_TOKEN = max_input_size - 1
PADDING_TOKEN = max_input_size - 2
EDGE_PREFIX_TOKEN = max_input_size - 3
PATH_PREFIX_TOKEN = max_input_size - 4

input = [EDGE_PREFIX_TOKEN, 3, 0, EDGE_PREFIX_TOKEN, 2, 3, QUERY_PREFIX_TOKEN, 2, 0, PATH_PREFIX_TOKEN, 3]
padding_size = max_input_size - len(input)
padded_input = [PADDING_TOKEN] * (max_input_size - len(input)) + input
x = torch.LongTensor(padded_input)

mask = torch.ones((max_input_size, max_input_size))
mask = mask.triu(1)
mask[:,:padding_size] = 1

# embed the input
positional_embedding = torch.diag(torch.ones(max_input_size))
token_embedding = torch.diag(torch.ones(max_input_size))
x = token_embedding[x]
x = torch.cat((x, positional_embedding), -1)

plot_matrix(x, "embedding.png")

# first attention layer
C = 100
K_proj = torch.zeros(max_input_size*2, max_input_size*2)
K_proj[:(max_input_size-5),:(max_input_size-5)] = C * torch.diag(torch.ones(max_input_size-5))
if use_negative_encoding:
	K_proj = -K_proj
K_proj[max_input_size:(2*max_input_size-5),max_input_size:(2*max_input_size-5)] = 2*C * torch.diag(torch.ones(max_input_size-5))
K_proj[-1,-1] = -C
Q_proj = torch.diag(torch.ones(2*max_input_size))
K = torch.matmul(x, K_proj)
Q = torch.matmul(x, Q_proj)
a = torch.matmul(Q, K.transpose(0,1)) / math.sqrt(K.size(-1))
#a += mask * -1e4
a = a.softmax(-1)
attn1_out = torch.matmul(a, x)

plot_matrix(a, "attention1.png")
plot_matrix(attn1_out, "attn1_out_before_residuals.png")

x = x + attn1_out

plot_matrix(x, "attn1_out_after_residuals.png")

# nonlinearity (feedforward layer)
sigmoid = torch.nn.Sigmoid()
nonlinearity_out = x * sigmoid(x)
x = x + nonlinearity_out

plot_matrix(x, "nonlinearity_out_after_residuals.png")

# second attention layer
Q_proj = torch.zeros(max_input_size*2, max_input_size*2)
Q_proj[max_input_size:,max_input_size:] = C * torch.diag(torch.ones(max_input_size))
if use_negative_encoding:
	Q_proj = -Q_proj
K_proj = torch.zeros(max_input_size*2, max_input_size*2)
for i in range(max_input_size, max_input_size*2 - 1):
	K_proj[i+1,i] = 1.0
K = torch.matmul(x, K_proj)
Q = torch.matmul(x, Q_proj)
a = torch.matmul(Q, K.transpose(0,1)) / math.sqrt(K.size(-1))
#a += mask * -1e4
a = a.softmax(-1)
attn2_out = torch.matmul(a, x)

plot_matrix(a, "attention2.png")
plot_matrix(attn2_out, "attn2_out_before_residuals.png")

x = x + attn2_out

plot_matrix(x, "attn2_out_after_residuals.png")
