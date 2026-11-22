import torch
import torch.nn as nn
from einops import rearrange, einsum
import math
import random
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.w, std=std, a = -3 * std, b = 3 * std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_in d_out -> ... d_out") 

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.normal(mean=0, std=1, size=(num_embeddings, embedding_dim), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = ((x ** 2).sum(dim=-1) / self.gain.shape[0] + self.eps) ** 0.5
        return (x / rms.unsqueeze(-1) * self.gain).to(in_dtype)

class SwiGLUFF(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        if not d_ff:
            d_ff = max(64, int(8 / 3 * d_model // 64) * 64)
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        return self.w2(silu * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        theta_ik = torch.arange(0, max_seq_len, device=device, dtype=dtype).repeat(d_k // 2, 1) / \
            ((theta ** ((2 * torch.arange(1, d_k // 2 + 1, device=device, dtype=dtype) - 2) / d_k)).repeat(max_seq_len, 1)).transpose(0, 1)
        cos = torch.cos(theta_ik)
        sin = torch.sin(theta_ik)  # d_k, max_seq_len
        cache = torch.stack((torch.stack((cos, -sin), dim=2), torch.stack((sin, cos), dim=2)), dim=2)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if len(token_positions.shape) == 1:
            token_positions = rearrange(token_positions, 'pos -> 1 pos')
        rope = rearrange(self.cache[:, token_positions, :, :], "... dk_2 batch seq two1 two2 -> ... batch seq dk_2 two1 two2")
        x = rearrange(x, '... seq (dk_2 two) -> ... seq dk_2 two 1', two=2)
        rotated = einsum(rope, x, '... seq dk_2 two1 two, ... seq dk_2 two one -> ... seq dk_2 two1 one')
        if rotated.shape[0] == 1:
            rotated = rearrange(rotated, '1 ... -> ...')
        return rearrange(rotated, '... seq dk_2 two one -> ... seq (dk_2 two one)')

def softmax(x, dim):
    max_val, _ = x.max(dim)
    x = torch.exp(x - max_val.unsqueeze(dim))
    sum_val = x.sum(dim)
    return x / sum_val.unsqueeze(dim)

def scaled_dot_product_attention(K, Q, V, mask):
    S = Q @ K.transpose(-1, -2) / math.sqrt(Q.shape[-1])
    if mask is not None:
        S = S.masked_fill(~mask, float('-inf'))
    S = softmax(S, -1)
    return S @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, with_rope=False, theta=10000, max_seq_len=100000, device=None, dtype=None):
        super().__init__()
        self.q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o = Linear(d_model, d_model, device=device, dtype=dtype)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.device = device
        if with_rope:
            self.rope = RoPE(theta, self.d_head, max_seq_len, device=device, dtype=dtype)

    def __split_head(self, x):
        return rearrange(x, "... seq (num_heads d_heads) -> ... num_heads seq d_heads", d_heads=self.d_head)

    def forward(self, x, pos=None):
        q = self.__split_head(self.q(x))
        k = self.__split_head(self.k(x))
        v = self.__split_head(self.v(x))
        if hasattr(self, "rope"):
            if not pos:
                pos = torch.arange(0, x.shape[-2])
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = scaled_dot_product_attention(k, q, v, torch.ones((x.shape[-2], x.shape[-2]), dtype=torch.bool, device=self.device).tril())
        x = rearrange(x, '... num_heads seq d_heads -> ... seq (num_heads d_heads)')
        return self.o(x)

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len, with_rope=True, device=device, dtype=dtype)
        self.ff = SwiGLUFF(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, context_len, num_layers, theta, device=None, dtype=None):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.Sequential(*[Block(d_model, num_heads, d_ff, context_len, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.emb(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.linear(x)
        return x

def cross_entropy(logits, targets):
    max_logits = logits.max(-1)[0].unsqueeze(-1) # batch, seq, 1
    base = torch.log(torch.exp(logits - max_logits).sum(-1)) + max_logits.squeeze(-1) # batch, seq
    picked = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (-picked + base).sum() / targets.numel()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, eps, betas, device=None):
        defaults = {"lr": lr, "decay": weight_decay, "eps": eps, "beta_m": betas[0], "beta_v": betas[1]}
        self.device = device
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            decay = group['decay']
            eps = group['eps']
            beta_m = group['beta_m']
            beta_v = group['beta_v']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 1)
                if 'm' in state:
                    m = state['m']
                    v = state['v']
                else: 
                    m = torch.zeros(p.shape, device=self.device, dtype=p.dtype)
                    v = torch.zeros(p.shape, device=self.device, dtype=p.dtype)
                lr_updated = lr * math.sqrt(1 - beta_v ** t) / (1 - beta_m ** t)
                p.data -= lr * decay * p.data
                m = beta_m * m + (1 - beta_m) * p.grad.data
                v = beta_v * v + (1 - beta_v) * p.grad.data ** 2
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1
                p.data -= lr_updated * m / (torch.sqrt(v) + eps)

def get_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + 1 / 2 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate

def gradient_clipping(params, max_l2_norm, eps=1e-6):
    norm_accum = 0
    for p in params:
        if p.grad is None:
            continue
        norm_accum += torch.linalg.vector_norm(p.grad) ** 2
    norm_accum = math.sqrt(norm_accum)
    if norm_accum < max_l2_norm:
        return
    clip_by = max_l2_norm / (norm_accum + eps)
    for p in params:
        if p.grad is None:
            continue
        p.grad.mul_(clip_by)

def load_data(x, batch_size, context_length, device):
    batches = torch.randint(0, len(x) - context_length, (batch_size,))
    x_sample = []
    y_sample = []
    for i in batches:
        x_sample.append(x[i: i + context_length])
        y_sample.append(x[i + 1: i + context_length + 1])
    return torch.tensor(np.array(x_sample), device=device, dtype=torch.long), torch.tensor(np.array(y_sample), device=device, dtype=torch.long)

def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    d = {"iteration": iteration, "model_state": model_state, "optimizer_state": optimizer_state}
    torch.save(d, out)

def load_checkpoint(src, model, optimizer):
    d = torch.load(src)
    model.load_state_dict(d['model_state'])
    if optimizer:
        optimizer.load_state_dict(d['optimizer_state'])
    return d['iteration']

def generate(prompt, tokenizer, lm, max_gen, temp, prob, context_len, eot='<|endoftext|>', device=None):
    encoded = tokenizer.encode(prompt)
    eot_idx = tokenizer.encode(eot)[0]
    input = torch.tensor(encoded, device=device).unsqueeze(0)
    output = []
    with torch.no_grad():
        gen_len = 0
        while gen_len <= max_gen:
            gen_len += 1

            logits = lm(input)
            probs, indicies = torch.sort(softmax(logits[0, -1] / temp, 0), descending=True)
            endpos = 0
            prob_sum = 0
            while prob_sum < prob:
                prob_sum += probs[endpos]
                endpos += 1
            probs = probs[:endpos]
            indicies = indicies[:endpos]
            idx = indicies[torch.multinomial(probs, 1)]
            input = torch.cat((input, idx.unsqueeze(0)), dim=1)
            output.append(idx.item())
            if len(input) > context_len:
                input = input[len(input) - context_len:]
            if idx == eot_idx:
                break
    return tokenizer.decode(output)


        


