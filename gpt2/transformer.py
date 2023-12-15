import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import LayerNorm
from gpt2 import AttentionLayer, Past, PadMasking, FutureMasking, PositionalEmbedding, TokenEmbedding, PositionwiseFeedForward
from typing import Optional, Tuple, List, Union


class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 heads: int,
                 dims: int,
                 token_dims: int,
                 pos_dims: int,
                 rate: int,
                 dropout: float = 0.1,
                 feedforward: bool = True,
                 sparse_v: bool = False,
                 linear: bool = True,
                 diagonal: bool = False):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, token_dims, pos_dims, dropout, sparse_v, linear, diagonal)
        self.ln_attn = LayerNorm(dims)
        if feedforward:
            self.ff = PositionwiseFeedForward(dims, rate, dropout)
            self.ln_ff = LayerNorm(dims)
        else:
            self.ff = None

    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        if self.attn.proj_v:
            a, past = self.attn(a, a, a, past, mask)
        else:
            a, past = self.attn(x, x, x, past, mask)

        '''print("attention layer output:")
        mean = torch.mean(a)
        std = torch.std(a)
        for i in range(a.size(0)):
            print("  a[{},:] > mean(a) + 0.8*std(a): {}".format(i, torch.nonzero(a[i,:] > mean + 0.8*std)[:,0].tolist()))'''

        x = x + a
        if self.ff:
            x = x + self.ff(self.ln_ff(x))

        '''print("feedforward layer + residual connection output:")
        mean = torch.mean(x)
        std = torch.std(x)
        for i in range(x.size(0)):
            print("  x[{},:] > mean(x) + 0.4*std(x): {}".format(i, torch.nonzero(x[i,:] > mean + 0.4*std)[:,0].tolist()))'''

        return x if self.training else (x, past)


class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 layers: int,
                 pad_idx: int,
                 words: int,
                 seq_len: int,
                 heads: int,
                 dims: int,
                 rate: int = 4,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 absolute_pos_emb: bool = True,
                 learn_token_emb: bool = False,
                 diagonal_attn: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        #self.positional_embedding = PositionalEmbedding(seq_len, dims)
        if learn_token_emb:
            self.token_embedding = TokenEmbedding(words, dims)
        else:
            self.token_embedding = torch.zeros((words, dims))
            self.token_embedding[:words,:words] = torch.diag(torch.ones(words))

        if absolute_pos_emb:
            self.positional_embedding = torch.diag(torch.ones(seq_len))
        else:
            self.positional_embedding = None
        self.dropout_embedding = nn.Dropout(dropout)

        if absolute_pos_emb:
            embedding_dim = dims + seq_len
            token_dim = words
            position_dim = seq_len
        else:
            embedding_dim = dims
            token_dim = words
            position_dim = 0
        self.transformers = nn.ModuleList([
            TransformerLayer(heads, embedding_dim, token_dim, position_dim, rate, dropout, l != layers - 1, True, False, diagonal_attn)
            for l in range(layers)])
        self.ln_head = LayerNorm(embedding_dim)

        def get_slopes(n):
            import math
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
            else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

        self.slopes = torch.Tensor(get_slopes(heads))
        #In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in the paper).
        #If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        #This works because the softmax operation is invariant to translation, and our bias functions are always linear.
        #self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(heads, -1, -1)
        #self.alibi = self.alibi.view(heads, 1, seq_len)
        #self.alibi = self.alibi.repeat(words//seq_len, 1, 1)  # batch_size, 1, 1

    def to(self, device):
        super().to(device)
        self.token_embedding = self.token_embedding.to(device)
        if self.positional_embedding != None:
            self.positional_embedding = self.positional_embedding.to(device)

    def forward(self,
                x: torch.Tensor,
                past: Optional[List[Past]] = None,
                use_grad_ckpt: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding and positional embedding layers.
        #x = self.token_embedding(x) + self.positional_embedding(x, offset)
        '''print('input x:')
        print(x)'''
        if type(self.token_embedding) == TokenEmbedding:
            x = self.token_embedding(x)
        else:
            x = self.token_embedding[x]
        if self.positional_embedding is not None:
            if len(x.shape) == 2:
                pos = self.positional_embedding
            else:
                pos = self.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat((x, pos), -1)
        x = self.dropout_embedding(x)
        #import pdb; pdb.set_trace()

        '''print("embedded input:")
        for i in range(x.size(0)):
            print("  x[{},:] != 0: {}".format(i, torch.nonzero(x[i,:])[:,0].tolist()))'''

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        if self.positional_embedding is not None:
            if len(x.shape) == 2:
                x = x[:,:-self.positional_embedding.shape[0]]
            else:
                x = x[:,:,:-self.positional_embedding.shape[0]]
        if type(self.token_embedding) == TokenEmbedding:
            x = self.token_embedding(x, transposed=True)
        else:
            x = torch.matmul(x, self.token_embedding.transpose(0, 1))
        '''print('prediction:')
        print(x[-1,:])'''
        #import pdb; pdb.set_trace()

        return x if self.training else (x, present)
