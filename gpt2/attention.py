import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

Past = Tuple[torch.Tensor, torch.Tensor]


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))

        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)
        x = self.dropout(x.softmax(-1))

        return torch.matmul(x, v)


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """
    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split the tensors to multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one.
        return (super().forward(q, k, v, mask)
                .transpose(-3, -2)
                .contiguous()
                .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads)))


def toeplitz(c, r):
    vals = torch.cat((r, c))
    shape = len(c) + 1, len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)

class DiagonalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device':device, 'dtype':dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonals = nn.parameter.Parameter(torch.empty(out_features + in_features - 1, **factory_kwargs))
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.diagonals, -math.sqrt(5), math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = toeplitz(self.diagonals[:self.out_features-1], self.diagonals[self.out_features-1:])
        return nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class AttentionLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., query_len, past_len + kv_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., query_len, dims)
    output 2 (*)    float           (..., past_len + kv_len, dims)
    ===========================================================================
    """
    def __init__(self, heads: int, dims: int, dropout: float = 0.1, sparse_v: bool = False, linear: bool = True, diagonal: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        if diagonal:
            self.proj_q = DiagonalLinear(dims, dims)
            self.proj_k = None
        else:
            self.proj_q = nn.Linear(dims, dims)
            self.proj_k = nn.Linear(dims, dims)
        if sparse_v:
            self.proj_v = None
        else:
            self.proj_v = nn.Linear(dims, dims)
        if linear:
            self.linear = nn.Linear(dims, dims)
        else:
            self.linear = None

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                past: Optional[Past] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Past]:
        #import pdb; pdb.set_trace()
        q = self.proj_q(q)
        if self.proj_k:
            k = self.proj_k(k)
        if self.proj_v:
            v = self.proj_v(v)

        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.attn(q, k, v, mask)
        if self.linear:
            x = self.linear(x)
        return x, (k, v)
