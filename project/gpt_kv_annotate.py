# Source: Sebastian Raschka, "Build a Large Language Model (From Scratch)"
# https://github.com/rasbt/LLMs-from-scratch  (Apache License 2.0)
# File: ch04/03_kv-cache/gpt_with_kv_cache.py. This is a copy of my
# annotated gpt_annotate.py (my notes kept as-is) with Raschka's KV cache
# changes added. Code matches his file.
#
# KV CACHE MARKERS (search for these):
#   # NEW (KV cache)     = a block Raschka added. Fill in the # NOTE: lines.
#   # CHANGED (KV cache) = an existing line he edited. Fill in the # NOTE:.
#   # WAS:               = the original line from gpt_ch04.py, for comparison.
#
# ANSWER KEY: the original file in your clone still has Raschka's comments:
#   LLMs-from-scratch/ch04/03_kv-cache/gpt_with_kv_cache.py
#
# HOW TO ANNOTATE
#   - Fill in each "# NOTE:" in your own words: what the block does and why.
#   - For lines that change data, write the SHAPE after the line. Use:
#       b = 1 (batch), num_tokens = 4, d_out = emb_dim = 768,
#       num_heads = 12, head_dim = 64, vocab_size = 50257
#   - Reading order tip: start at main() at the bottom and follow the calls
#     upward. The file is written bottom-up; it runs top-down from main().
#
# This file runs as-is:  python gpt_kv_annotate.py

import time
import tiktoken
import torch
import torch.nn as nn

# Chapter 3: This class takes each token's vector and returns a context-aware version of it, in the same shape, by running 12 attention heads in parallel and combining their results.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # d_out must be divisible by num_heads because each of the heads must calculate the same number of dimensions.
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # Each token's 768-sized vector passes through 768 query-weighted neurons for producing that token's 768-number query.
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)      # ... same process but for key-weighted...
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # ... same process but for value-weighted...
        self.out_proj = nn.Linear(d_out, d_out)                 # A final 768 layer that blends the 12 heads' outputs into one combined vector per token.
        self.dropout = nn.Dropout(dropout)                      # Randomly zeroes 10% of the attention weights, during training only

        # This block of code builds the causal mask. It's necessary so that each token can only look at itself and earlier tokens, otherwise the model will learn to just "cheat" and look at future tokens. 
        #   1 = hide (a future token). It's built once here and applied in forward().
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )

        # NEW (KV cache)
        #   cache_k and cache_v start as None (empty), and ptr_current_pos starts at 0. After 5 tokens, cache_k holds (1, 5, 12, 64).
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    # forward() passes x through the layers in order. 
    # CHANGED (KV cache): new use_cache argument.
    # During cached generation, num_tokens is 4 on the first call (prefill, the whole prompt), then 1 on every call after that (decode). It doesn't grow; the cache does.
    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        # Each token's vector gets run through the key, value, or query neurons to get their key, value, or query. 
        # The shape is (1, 4, 768) for each matrix. 1 is number of batches we're testing, 4 is the number of tokens in the input (from "Hello, I am"), and there are 768 dimensions in each token's embedding.
        # CHANGED (KV cache): keys/values renamed keys_new/values_new.
        # keys_new and values_new hold only this call's keys and values. They get added to the cache, and then keys/values means the whole cache.
        #   queries keeps its name because queries are never cached: only the current token's query is needed.
        keys_new = self.W_key(x)        # Run every token's vector through the key neurons to get each token's key.
        values_new = self.W_value(x)    # Same, but for values. 
        queries = self.W_query(x)   # Same, but for queries. 

        # Reshape each token's 768 numbers into 12 slices of 64, one slice per head to prepare it for transposition.
        #   No numbers change; they're just regrouped. Shape: (1, 4, 768) → (1, 4, 12, 64)
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim)
        values_new = values_new.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # NEW (KV cache)
        # If we're using the cache, when there's no cache, we first set the size to (1, 4, 12, 64). On the next step, keys_new is (1, 1, 12, 64) for the one new token, and the cache grows to (1, 5, 12, 64), and so on, for each subsequent token in the sequence. 
        # If we're not using the cache, every step feeds in the whole sequence, so the second dimension of keys grows from 5, to 6, so on. It's recomputed from scratch every time instead of being stored. 
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            keys, values = self.cache_k, self.cache_v
        else:
            keys, values = keys_new, values_new

        # This transposition is necessary because @ does matrix multiplication on the last two dimensions, treating every dimension before as "do this separately for each one".
        #   After the transposition, the last two dimensions are (tokens, 64). So queries @ keys.T will produce 12 score tables that are 4 rows long and 4 columns wide--we're calculating each token's attemtion to every other token and itself.
        #   If we didn't transpose, we'd be comparing heads with each other, which is not what we want to do.  
        #   The shape will go from (1, 4, 12, 64) to (1, 12, 4, 64).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Each head compares every query with every key. There will be one score per (token, token) pair.
        #   (1, 12, 4, 64) @ (1, 12, 64, 4) becomes (1, 12, 4, 4)
        attn_scores = queries @ keys.transpose(2, 3)
        
        # NEW (KV cache)
        #   The mask slice now needs two sizes, rows for queries and columns for keys, because there is 1 query but 5 keys. attn_scores is (1, 12, 1, 5).
        #   Also, during decode, the mask is still applied but hides nothing, because future tokens don't exist yet. Without cache, decode 1 would build a full 5x5 table with 25 scores per head, and only the last row would be used. 
        #   With the cache, we only compute the last row. 
        num_tokens_Q = queries.shape[-2]
        num_tokens_K = keys.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K 
            ]
            self.ptr_current_pos += num_tokens_Q                # ptr_current_pos is the position of the current query token, so it picks which mask row to use. For the 5th token, that would be row 4. During decode, that row hides nothing. 

        # The stored mask is 1024 rows and 1024 columns because it's built for the maximum context length. The input only has 4 tokens, so this cuts out the top-left 4 by 4 corner for us to use as a mask.
        # CHANGED (KV cache): now the else branch; slices with num_tokens_Q, num_tokens_K.
        # WAS: mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        # Masked scores are set to −inf because softmax turns −inf into exactly 0 weight. Setting them to 0 wouldn't work because e^0 = 1, so the future token would still be visible. 
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Take the size of the last dimension of keys, which is 64, and square root before dividing the scores by it (√64 = 8). A dot-product of two 64-number vectors will add up 64 products, so the score will get too big.
        #   Big scores make softmax become all-or-nothing, which makes slopes too flat and backpropagation difficult. 
        #   dim = -1 normalizes across the last dimension, the 4 keys, so each row adds up to 1.
        #   Then attn_weights zeroes out 10% of the values to the model doesn't overrely on some attention weights during training/
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights) 

        # This line takes the weights of (1, 12, 4, 4), matrix multiplies it with (1, 12, 4, 64), and the vector becomes (1, 12, 4, 64) again. 
        #   Then the transposition swaps the values so that the matrix becomes (1, 4, 12, 64) again, so that we have batch, tokens, heads, and slices again.
        #   Each head blends the value vectors using its attention weights, giving each token a 64-number context slice per head.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # This block basically just glues each of the 12 tokens 64 slices back into a 768-number vector so the shape is (1, 4, 768) again.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        # context_vec ends up being the same shape as x, so the process can be repeated. 
        return context_vec

    # NEW (KV cache)
    # This is called once at the start of each new generation, before prefill (through model.reset_kv_cache()), so that the cache clears and new values don't build on top of the previous run's values. 
    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


# Chapter 4: As each token's 768 numbers pass through the 12 transformer blocks, the numbers from the transformer edits can drift widely. 
#   One token's vector might end up with values around 50, another's around 0.01. Layer norm rescales each token's 768 numbers to an average of 0 and a spread of 1, keeping their pattern, then applies a learned scale and shift. This keeps values in a steady range through the 12 blocks, so training stays stable."
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                     # We add this very small number to the variance so you never divide by exactly 0.
        self.scale = nn.Parameter(torch.ones(emb_dim))      # A 768-sized vector of learned values for each dimension. Starts at 1, so no change at first.
        self.shift = nn.Parameter(torch.zeros(emb_dim))     # Same as above, but for shift instead of scale. Starts at 0, so no change at first.

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)                 # Step 1: Find the average of each token's 768 numbers. Shape: (1, 4, 1), one average per token.
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # Step 2: Measure the spread. Take each number's distance from the average, square it,
                                                            #   and average those squares to get the variance. Shape: (1, 4, 1), one variance per token.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)    # Step 3: Subtract the average from each number, so the list's average becomes zero.
                                                            #   Then divide by the square root of the variance (the standard deviation, or the
                                                            #   typical distance from the average) to get a spread of 1. The pattern is kept,
                                                            #   only the overall size changes. Shape: (1, 4, 768).
        return self.scale * norm_x + self.shift             # Step 4: Multiply by the learned scale and add the learned shift. Starts as no change
                                                            #   (scale = 1, shift = 0), but training can adjust each of the 768 positions.


# While ReLU is common, we're not going to use it here. ReLU outputs 0 for values less than 0, and whatever the value is for values greater than zero. 
#   However, this leads to a sharp corner when transitioning from 0 to nonzero values. GELU results in smoother functions, which is better for backprop.
#   Also, GELU results in small, nonzero outputs for small values less than 0, which allows those values to contribute to learning. 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# nn.Linear is a row of neurons. An input x is multiplied by its weight, and a bias is added.
# nn.Sequential bundles the three layers (Linear>GELU>Linear) into one object so that forward() can be a single line.
# This class processes each token's vector separately. There is no mixing between tokens, because that is attention's job.
#   It expands each token's 768 numbers to 3072 so our model can check more patterns, applies a non-linear activation, and then shrinks the number of neurons back to 768 so the next layer can use the same code.
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),      # nn.Linear(768, 3072) means that we have 3072 neurons that are each taking in 768 input numbers. The neurons multiply each of the 768 dimensions by a weight, add them up, and add a bias. Each neuron outputs one number, so 768 numbers go in, and 3072 come out.
            GELU(),                                             # Non-linear activation. Without it, the two linear layers would collapse into one (linear after linear is still linear), so the model could only learn straight-line patterns (the XOR problem). 
                                                                # GELU is a smooth version of ReLU, so it also has usable slopes everywhere for backprop.
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),      # Shrink the matrix back to 768 dimensions so our shape is a suitable size to do the next matrix algebra operation on the next block. 
        )

    def forward(self, x):
        return self.layers(x)   # Passes x through the three layers in order.

# This class sets up a transformer block, which includes a multihead attention layer and a feed forward network. 
# Output is the same shape as the input, so 12 of these can be stacked.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # All of these variables are defined in the main function.
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])

        # The transformer block will consist of a feed forward network, two layers of normalization, a dropout layer, and a multihead attention layer. 
        self.ff = FeedForward(cfg)                          
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # CHANGED (KV cache): new use_cache argument, passed through to attention.
    def forward(self, x, use_cache=False):
        # Attention: Tokens exchange information in this block.
        shortcut = x                # Save the input. After the layer runs, it's added back, so the layer only learns an edit. It also gives the calculus a frictionless highway to pass learning signals all the way through a massive neural network without getting trapped or flattened out through the layers.
        x = self.norm1(x)           # After saving the shortcut, we normalize x to manage the data's spread. 
        # WAS: x = self.att(x)
        # NEW (KV cache)
        # The feed-forward network doesn't need to use a cache, because it's not mixing the values of any tokens together. 
        x = self.att(x, use_cache=use_cache)             # The attention layer, where each token gathers information from itself and earlier tokens. Returns context-aware vectors, same shape (1, 4, 768).
        x = self.drop_shortcut(x)   # During training we randomly zero 10% of the layer's output before adding the shortcut back.
        x = x + shortcut            # We then add the shortcut data back in. 

        # Feed-Forward Nework: Each token processes what it gathered.
        shortcut = x                # Save the current x, after attention, as the new shortcut.
        x = self.norm2(x)           # We apply the second normalization layer. 
        x = self.ff(x)              # The feed-forward layer, which processes each token's vector separately by expanding, running GELU, and shrinking. No mixing occurs between tokens.
        x = self.drop_shortcut(x)   # Again, we zero out some of the data. 
        x = x + shortcut            # Same as above. 

        return x

# Built from the config settings. Takes token IDs (batch, tokens) and returns logits, which are 50,257 raw scores for each position (batch, tokens, 50257).
# Token IDs are used to look up token and position vectors and add them to 12 transformer blocks. A final layer normalization is run, and a score is generated for every vocabulary token.
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Build the two lookup tables that turn token IDs into vectors and a dropout layer.
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])              # Create the token lookup table of vocab_size rows and emb_dim columns. There is one learned vector per token (row), and the numbers start as random.
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])          # Create a positional embedding of context_length rows and emb_dim columns so the model knows where each token sits in the sequence. There is one learned vector per slot.
        self.drop_emb = nn.Dropout(cfg["drop_rate"])                                # Create a layer that randomly zeroes out a fraction of the input embedding values during training.

        # Create transformer blocks based on n_layers with their own weights. nn.Sequential means that block 1's output goes into block 2 and so on.
        # WAS: self.trf_blocks = nn.Sequential(
        #          *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # NEW (KV cache)
        # We switched from nn.Sequential here to nn.ModuleList, because nn.Sequential can only hand each block x. It doesn't pass extra arguments like use_cache=True. 
        # current_pos counts how many tokens have gone through the model so far so the next token gets the right position number. 
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.current_pos = 0

        self.final_norm = LayerNorm(cfg["emb_dim"])                                 # Create a final normalization layer of size emb_dim. One last layer norm to clean up each token's vector before scoring.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)    # 50,257 neurons, one per vocabulary token. Each reads a token's 768 numbers and outputs one score, which makes the logits.

    # CHANGED (KV cache): new use_cache argument.
    def forward(self, in_idx, use_cache=False):
        # in_idx starts as (1, 4) 
        batch_size, seq_len = in_idx.shape                                      # batch_size is 1 and seq_len is 4. 
        tok_embeds = self.tok_emb(in_idx)                                       # Shape: (1, 4, 768). Each of the 4 IDs is swapped for its 768-number row.

        # WAS: pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # Shape: (4, 768). torch.arange(4) = [0, 1, 2, 3], the slot numbers, so it fetches 4 position rows
        # NEW (KV cache)
        # We can't use arange(seq_len) when using a cache, because it always starts at 0. Instead, we want to start at the token we're decoding. 
        #   The model keeps a running count of which token we're on through current_pos. 
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)                         # Looking up one ID returns one row of 768 numbers, shape (1, 768). unsqueeze(0) adds the batch dimension, giving (1, 1, 768). That matches tok_embeds (1, 1, 768), so the two can be added.

        x = tok_embeds + pos_embeds                                             # Shape: (1, 4, 768). The 4 position rows are added to every sequence in the batch using broadcasting.                                     
        x = self.drop_emb(x)                                                    # Shape: (1, 4, 768). Zeroing some values doesn't change the shape.

        # WAS: x = self.trf_blocks(x)                                                  # Shape: (1, 4, 768). Every block keeps the shape. That's why 12 can be chained.
        # NEW (KV cache)
        # This loop passes use_cache to each block, which nn.Sequential can't do. 
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)                                                  # Shape: (1, 4, 768). Layer norm changes values, not shape.
        logits = self.out_head(x)                                               # Shape: (1, 4, 50257). nn.Linear changes only the last number from 768 to 50,257. One score per position.
        return logits

    # NEW (KV cache)
    # Each of the 12 transformer blocks has an attention layer that keeps its own cache_k and cache_v. A reset is necessary so that the the new prompt's keys don't get added to the old ones, 
    #   and so that current_pos starts counting at 0 again on the next run too. The reset sets the cache_k and cache_v to None in each transformer block, sets its mask pointer to 0, and resets the current_pos to 0. 
    #   It runs once before prefill. 
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


# This loop is somewhat wasteful, because each pass re-runs every earlier token through all 12 blocks, recomputing keys and values that are identical to the last pass. 
#   This is because the causal mask means earlier tokens can't see later ones. Only the last position's output is used. The KV cache stores those keys and values, so each pass only has to process the new token.
#   idx is a (batch, n_tokens) array of indices.
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # CHANGED (KV cache): Raschka removed model.eval() from this function in this file.
    # Removing model.eval() doesn't matter, because main() calls model.eval. 
    for _ in range(max_new_tokens) :                            # The loop runs max_new_tokens times, one new token per pass.
        idx_cond = idx[:, -context_size:]                       # If the sequence is longer than the context window of 1,024, keep only the last 1,024 tokens.

        # Logits are raw scores for every vocabulary token, at every position. No softmax needed because argmax simply needs the highest score.
        # No training happens because we're in the output stage, otherwise known as inference, so skip the bookkeeping backprop uses. That saves memory and time.
        with torch.no_grad():
            logits = model(idx_cond)

        # Keep only the last position's scores, the prediction for the next token: (batch, 50257).
        logits = logits[:, -1, :]

        # The model identifies which token has the largest score, or logit, of coming next. 
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Append the new token to the sequence. It becomes part of the input on the next pass.
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# NEW (KV cache)
# This loop is less wasteful, because it uses a KV cache. The cache stores keys and values, and each step feeds only the new token through the model rather than the whole sequence. 
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()                                                        # Turns the dropout layer off
    ctx_len = context_size or model.pos_emb.num_embeddings              # Use context size if one was provided, otherwise, fall back to the number of rows in the position table. 

    # Again, we're running the model to generate text now, so we turn off backpropagation bookkeeping. 
    with torch.no_grad():
        if use_cache:
            # Prefill: This line runs the whole prompt through the model in one pass and fills the cache in all 12 layers. The input shape is (1, 4).
            # If we're using the cache, we empty it for each new prompt and calculate the logits based only on the last tokens that fit in the context window. 
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for step in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)   # argmax chooses the token ID with the highest score. 
                idx = torch.cat([idx, next_idx], dim=1)                 # Then add the next token to our sequence, which is one row of IDs, growing with each step. 
                # We don't need a forward call for our last token, because our response is complete. A forward call would compute scores for token 201, when we only need to return 200 tokens. 
                if step + 1 < max_new_tokens:                           
                    logits = model(next_idx, use_cache=True)            # Decode: Only 1 new token (1,1) goes through our model now. Earlier keys and values are retrieved from the cache. 
        else:
            # This is the no-cache path, which allows our one function to run with and without a cache. This is to check that both processes produce identical text. 
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx

# Start reading here. 
def main():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # The number of distinct tokens the model knows.
        "context_length": 1024,     # The max number of tokens the model can look at at once. It matches the rows in the position table.
        "emb_dim": 768,             # The number of numbers used to describe each token, or an embedding. Tokens used in similar ways will end up with similar embeddings.
        "n_heads": 12,              # The number of heads. Each head runs an attention calculation separately, using its own 768/12 = 64-number slice of the queries, keys and values. Each head can focus on a different relationship, and their outputs are joined back into 768.
        "n_layers": 12,             # The number of stacked transformer blocks. Each block consists of an attention and feed forward network. 
        "drop_rate": 0.1,           # The dropout rate. Randomly zeroes out some of the attention weights, the embeddings, and some of the transformer block's outputs during training because we want to strengthen the other values and not have the model over rely on some of the connections. 
        "qkv_bias": False           # Setting to false so that query, key, and value only multiply by weights without help from the bias. Many modern LLMs skip it.
    }

    torch.manual_seed(123)              # Sets the rng so the random starting weights are the same on every run.
    model = GPTModel(GPT_CONFIG_124M)   # Define our model according to the class. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # cuda is not available on my mac, I don't think? 
    model.to(device)                    # Use CPU
    model.eval()                        # Command to switch the model into inference mode, so it turns off the dropout. Every value is used and results are repeatable.

    start_context = "Hello, I am"       # Pass in our context. 

    # The tokenizer turns the text into 4 token IDs. torch.tensor makes those a tensor of shape (4,). unsequeeze then adds the batch dimension, making it (1, 4).
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    # Times the loop below. 
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    # Runs the model 200 times. Each pass predicts one token ID and adds it to the end. It returns the 4 prompt IDs plus 200 new ones, shape (1, 204). This call is the part being timed.
    # WAS: 
    # token_ids = generate_text_simple(
    #   model=model,
    #   idx=encoded_tensor,
    #   max_new_tokens=200,
    #   context_size=GPT_CONFIG_124M["context_length"]
    # )

    # NEW: (KV cache)
    # context_size was removed from this call because generate_text_simple_cached falls back to pos_emb.num_embeddings, which is 1,024, the same number main() used to pass in.
    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )

    # This handles the timing of our llm. 
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
