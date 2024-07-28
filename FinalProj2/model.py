import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.resid_dropout(self.out_proj(attn_output))
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.src_token_emb = nn.Embedding(config.src_vocab_size, config.n_embd)
        self.tgt_token_emb = nn.Embedding(config.tgt_vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.tgt_vocab_size, bias=False)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, tgt_padding_mask=None):
        B, T_src = src.size()
        token_embeddings = self.src_token_emb(src)
        position_embeddings = self.pos_emb(torch.arange(0, T_src, device=src.device))
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)

        logits = self.head(x)

        if tgt is not None:
            B, T_tgt = tgt.size()
            if T_tgt > T_src:
                tgt = tgt[:, :T_src]
            elif T_tgt < T_src:
                padding = torch.full((tgt.size(0), T_src - T_tgt), fill_value=self.config.tgt_PAD_IDX, device=tgt.device)
                tgt = torch.cat([tgt, padding], dim=1)

            logits = logits.reshape(B * T_src, -1)
            tgt = tgt.reshape(B * T_src)

            loss = F.cross_entropy(logits, tgt, ignore_index=self.config.tgt_PAD_IDX)
            return logits, loss
        else:
            return logits, None

    def generate(self, src, max_new_tokens):
        self.eval()
        #testing, rsc_mask = none
        src_mask = (src != self.config.src_PAD_IDX).unsqueeze(1).unsqueeze(2).to(src.device)

        B, T_src = src.size()
        token_embeddings = self.src_token_emb(src)
        position_embeddings = self.pos_emb(torch.arange(0, T_src, device=src.device))
        x = self.drop(token_embeddings + position_embeddings)
        
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)

        start_token = self.config.src_START_IDX
        end_token = self.config.src_END_IDX
        
        generated_tokens = torch.full((B, 1), start_token, dtype=torch.long, device=src.device)
        
        for _ in range(max_new_tokens):
            tgt_emb = self.tgt_token_emb(generated_tokens)
            pos_emb = self.pos_emb(torch.arange(0, generated_tokens.size(1), device=src.device))
            tgt_emb = self.drop(tgt_emb + pos_emb)

           
            look_ahead_mask = self.create_look_ahead_mask(generated_tokens.size(1)).to(src.device)
            
            for block in self.blocks:
                tgt_emb = block(tgt_emb, look_ahead_mask)
            tgt_emb = self.ln_f(tgt_emb)
            
            logits = self.head(tgt_emb[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            # Check if any sequence in the batch has generated the end token
            if (next_token == end_token).any():
                break
        
        return generated_tokens[:, 1:]

    @staticmethod
    def create_look_ahead_mask(size):
        mask = torch.triu(torch.ones(size, size), diagonal=1) == 0
        return mask
