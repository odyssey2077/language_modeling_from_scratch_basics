import torch
import torch.nn as nn
import math
from einops import rearrange, einsum
from cs336_basics.nn.functions import scaled_dot_product_attention, softmax


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features        
        self.device = device
        self.dtype = dtype
        std_dev = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(data=nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype=dtype, device=device), 
                                                              mean=0., std=std_dev, a=-3 * std_dev, b=3 * std_dev))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.embedding_table = nn.Parameter(data=nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device), 
                                                              mean=0., std=1.0, a=-3.0, b=3.0))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        indices_tensor = token_ids.to(self.embedding_table.device).long()
        return self.embedding_table[indices_tensor]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(data=torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(self.eps + torch.sum(torch.square(x), dim=-1) / self.d_model)
        rms = rearrange(rms, "... -> ... 1")
        result = x / rms * self.g
        return result.to(in_dtype)
    

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = round(d_model * 8 / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        return self.w2(w1_x * torch.sigmoid(w1_x) * w3_x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # [max_seq_len, 1]
        i_values = torch.arange(max_seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        # [1, d_k / 2]
        d_values = torch.arange(1, d_k / 2 + 1, dtype=torch.float32, device=device).unsqueeze(0)
        # [max_seq_len, d_k / 2]
        denominator = torch.pow(theta, (2 * (d_values - 1))/ d_k)
        # [max_seq_len, d_k / 2]
        theta_i_k = i_values / denominator
        # [max_seq_len, d_k / 2]
        self.register_buffer('cos_theta_i_k', torch.cos(theta_i_k), persistent=False)
        # [max_seq_len, d_k / 2]
        self.register_buffer('sin_theta_i_k', torch.sin(theta_i_k), persistent=False)

    # x [..., seq_len, d_k]
    # token_positions [..., seq_len]
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # [..., seq_len, d_k / 2]
        selected_cos_theta_i_k = self.cos_theta_i_k[token_positions]
        # [..., seq_len, d_k / 2]        
        selected_sin_theta_i_k = self.sin_theta_i_k[token_positions]   
        x = rearrange(x, '... (g e) -> e ... g', e=2)
        x_even, x_odd = x[0], x[1]
        result_even = x_even * selected_cos_theta_i_k - x_odd * selected_sin_theta_i_k
        result_odd = x_even * selected_sin_theta_i_k + x_odd * selected_cos_theta_i_k
        return rearrange(torch.stack([result_even, result_odd], dim=-1), '... g e -> ... (g e)')
    

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, max_seq_length: int | None = None, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0, "invalid d_model num_head configuration"
        self.num_heads = num_heads
        self.d_k = d_model / num_heads
        self.d_v = d_model / num_heads
        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_o = Linear(d_model, d_model, device, dtype)
        if max_seq_length is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_length, device)
        else:
            self.rope = None


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = rearrange(Q, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)
        K = rearrange(K, '... seq_len (num_heads d_k) -> ... num_heads seq_len d_k', num_heads=self.num_heads)        
        V = rearrange(V, '... seq_len (num_heads d_v) -> ... num_heads seq_len d_v', num_heads=self.num_heads)

        if self.rope is not None:
          Q = self.rope(Q,token_positions)                  
          K = self.rope(K, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).to(self.w_q.device)
        return self.w_o(rearrange(scaled_dot_product_attention(Q, K, V, self.d_k, mask), '... num_heads seq_len d_v -> ... seq_len (num_heads d_v)'))
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, max_seq_length: int | None = None, theta: float | None = None):
        super().__init__()
        self.mha = CausalMultiHeadAttention(d_model, num_heads, device, dtype, max_seq_length, theta)
        self.ffn = FFN(d_model, d_ff, device, dtype)
        self.mha_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        mha_output = self.mha(self.mha_norm(x), token_positions) + x
        return self.ffn(self.ffn_norm(mha_output)) + mha_output
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None, theta: float | None = None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, device, dtype, context_length, theta) for _ in range(num_layers)])
        self.output_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.o_linear = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)

        x = self.output_norm(x)
        x = self.o_linear(x)
        return x