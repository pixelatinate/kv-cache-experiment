# Source: Sebastian Raschka, "Build a Large Language Model (From Scratch)"
# https://github.com/rasbt/LLMs-from-scratch  (Apache License 2.0)
# File: ch04/03_kv-cache/gpt_ch04.py, with the original comments removed
# so you can write your own. Code is unchanged.
#
# ANSWER KEY: the original file in your clone still has Raschka's comments:
#   LLMs-from-scratch/ch04/03_kv-cache/gpt_ch04.py
#
# HOW TO ANNOTATE
#   - Fill in each "# NOTE:" in your own words: what the block does and why.
#   - For lines that change data, write the SHAPE after the line. Use:
#       b = 1 (batch), num_tokens = 4, d_out = emb_dim = 768,
#       num_heads = 12, head_dim = 64, vocab_size = 50257
#   - Reading order tip: start at main() at the bottom and follow the calls
#     upward. The file is written bottom-up; it runs top-down from main().
#   - Chapter 3 parts: annotate now. Chapter 4 parts: annotate while you
#     read Chapter 4 (it doubles as your Chapter 4 study).
#
# This file runs as-is:  python gpt_annotate.py

import time
import tiktoken
import torch
import torch.nn as nn


#####################################
# CHAPTER 3: Multi-head attention
#####################################

# NOTE: What is this class for, in one sentence?
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # NOTE: Why must d_out divide evenly by num_heads?
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # NOTE: What are these four layers? Which ones does training adjust?
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # NOTE: What does this matrix look like for context_length = 4?
        #       (Draw it.) Why is it built once here instead of in forward()?
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # NOTE:
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        # NOTE: (shape before → after)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # NOTE: (shape before → after) Why swap these two dimensions?
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # NOTE:
        attn_scores = queries @ keys.transpose(2, 3)

        # NOTE: Why slice [:num_tokens, :num_tokens]?
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # NOTE: Why -inf, and what does softmax turn -inf into?
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # NOTE:
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # NOTE:
        context_vec = (attn_weights @ values).transpose(1, 2)

        # NOTE:
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# CHAPTER 4: The rest of the GPT model
#####################################

# NOTE:
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# NOTE: (Hint: compare with ReLU from Welch Labs.)
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# NOTE: What are the shapes going in, in the middle, and out?
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# NOTE: Describe one transformer block in one sentence.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # NOTE:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # NOTE:
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


# NOTE:
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # NOTE: (You know these from Chapter 2.)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # NOTE:
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # NOTE:
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # NOTE: Write the shape after every line below.
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# NOTE: Where is the wasted work in this loop? (This is the bridge to
#       the KV cache project.)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):

        # NOTE:
        idx_cond = idx[:, -context_size:]

        # NOTE:
        with torch.no_grad():
            logits = model(idx_cond)

        # NOTE: (shape before → after)
        logits = logits[:, -1, :]

        # NOTE:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # NOTE:
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


#####################################
# Running it (START READING HERE)
#####################################

def main():
    # NOTE: Which of these numbers have you already seen? What does each control?
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # NOTE: Is this model trained? What will the output text look like?
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    start_context = "Hello, I am"

    # NOTE: (Chapter 2) What shape is encoded_tensor, and what does unsqueeze(0) add?
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # NOTE: What is being timed? (You'll change max_new_tokens on Sunday.)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
