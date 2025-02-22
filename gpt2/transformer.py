import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import LayerNorm
from gpt2 import AttentionLayer, Past, PadMasking, FutureMasking, PositionalEmbedding, RotaryPositionalEmbedding, TokenEmbedding, PositionwiseFeedForward, ToeplitzMode
from typing import Optional, Tuple, List, Union
from enum import Enum


class AblationMode(Enum):
    NO_ABLATION = 0,
    ABLATE_ATTN_LINEAR = 1,
    ABLATE_ATTN_LINEAR_PROJV = 2

class PositionEmbedding(Enum):
    NONE = 0,
    ABSOLUTE = 1,
    ROTARY = 2


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
                 ablate: AblationMode = AblationMode.NO_ABLATION,
                 toeplitz: ToeplitzMode = ToeplitzMode.NONE,
                 pre_ln: bool = True):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, token_dims, pos_dims, dropout, ablate == AblationMode.ABLATE_ATTN_LINEAR_PROJV, ablate == AblationMode.NO_ABLATION, toeplitz)
        self.ln_attn = LayerNorm(dims)
        if feedforward:
            self.ff = PositionwiseFeedForward(dims, rate, dropout)
            self.ln_ff = LayerNorm(dims)
        else:
            self.ff = None
        self.pre_ln = pre_ln

    def forward(self,
                x: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None,
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # If pre_ln == True, layer normalizations are performed before the layers respectively.
        a = (self.ln_attn(x) if self.pre_ln else x)
        if self.attn.proj_v:
            a, past = self.attn(a, a, a, past, mask)
        else:
            a, past = self.attn(x, x, x, past, mask)

        x = x + a
        if not self.pre_ln:
             x = self.ln_attn(x)
        if self.ff:
            if self.pre_ln:
                x = x + self.ff(self.ln_ff(x))
            else:
                x = x + self.ff(x)
            if not self.pre_ln:
                x = self.ln_ff(x)

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
                 pos_emb: PositionEmbedding = PositionEmbedding.ABSOLUTE,
                 learn_token_emb: bool = False,
                 ablate: AblationMode = AblationMode.ABLATE_ATTN_LINEAR_PROJV,
                 toeplitz: ToeplitzMode = ToeplitzMode.NONE,
                 pre_ln: bool = True,
                 looped: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()
        self.looped = looped

        if learn_token_emb:
            self.token_embedding = TokenEmbedding(words, dims)
        else:
            self.token_embedding = torch.zeros((words, dims))
            self.token_embedding[:words,:words] = torch.diag(torch.ones(words))

        if pos_emb == PositionEmbedding.ABSOLUTE:
            self.positional_embedding = torch.diag(torch.ones(seq_len))
        elif pos_emb == PositionEmbedding.ROTARY:
            self.positional_embedding = RotaryPositionalEmbedding(dims, seq_len)
        else:
            self.positional_embedding = None
        self.dropout_embedding = nn.Dropout(dropout)

        if pos_emb == PositionEmbedding.ABSOLUTE:
            embedding_dim = dims + seq_len
            token_dim = words
            position_dim = seq_len
        else:
            embedding_dim = dims
            token_dim = words
            position_dim = 0
        self.transformers = nn.ModuleList([
            TransformerLayer(heads, embedding_dim, token_dim, position_dim, rate, dropout, True if ablate == AblationMode.NO_ABLATION else l != layers - 1, ablate, toeplitz, pre_ln)
            for l in range(layers)])
        if pre_ln:
            self.ln_head = LayerNorm(embedding_dim)
        else:
            self.ln_head = None

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
        if type(self.positional_embedding) == torch.Tensor:
            if len(x.shape) == 2:
                pos = self.positional_embedding
            else:
                pos = self.positional_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat((x, pos), -1)
        elif self.positional_embedding != None:
            x = self.positional_embedding(x)
        x = self.dropout_embedding(x)
        #import pdb; pdb.set_trace()

        '''print("embedded input:")
        for i in range(x.size(0)):
            print("  x[{},:] != 0: {}".format(i, torch.nonzero(x[i,:])[:,0].tolist()))'''

        # Apply transformer layers sequentially.
        present = []
        if self.looped:
            x = self.transformers[0](x, past[0] if past is not None else None, mask)
            if not self.training:
                present.append(x[1])
                x = x[0]
            for i in range(1, len(self.transformers)-2):
                x = self.transformers[1](x, past[i] if past is not None else None, mask)
                if not self.training:
                    present.append(x[1])
                    x = x[0]
            x = self.transformers[-2](x, past[len(self.transformers)-2] if past is not None else None, mask)
            if not self.training:
                present.append(x[1])
                x = x[0]
            x = self.transformers[-1](x, past[len(self.transformers)-2] if past is not None else None, mask)
            if not self.training:
                present.append(x[1])
                x = x[0]

        else:
            for i, transformer in enumerate(self.transformers):
                x = transformer(x, past[i] if past is not None else None, mask)

                if not self.training:
                    present.append(x[1])
                    x = x[0]

        if self.ln_head:
            x = self.ln_head(x)
        if type(self.positional_embedding) == torch.Tensor:
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
