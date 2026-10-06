# experiment.py: What the KV cache actually buys you
#
# Times Raschka's two UNMODIFIED scripts at several output lengths:
#   gpt_ch04.py            -> generate_text_simple         (no cache)
#   gpt_with_kv_cache.py   -> generate_text_simple_cached  (with cache)
# and checks that both produce identical tokens.
#
# Run from the repo root folder, using the book repo's venv:
#   ~/Documents/GitHub/LLMs-from-scratch/.venv/bin/python project/experiment.py
#
# Outputs (in project/):
#   results.csv           every individual run
#   kv_cache_speed.png    the chart
#   a markdown table printed to the terminal, to paste into README.md

import csv
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch

# Python only imports files from certain folders. This line adds Raschka's
# ch04/03_kv-cache folder to that list, so `import gpt_ch04` works without
# copying or editing his files.
RASCHKA_DIR = Path.home() / "Documents/GitHub/LLMs-from-scratch/ch04/03_kv-cache"
sys.path.insert(0, str(RASCHKA_DIR))

import gpt_ch04            # noqa: E402  (no cache)
import gpt_with_kv_cache   # noqa: E402  (with cache)

PROJECT_DIR = Path(__file__).parent

######################################################################
# Experiment settings
######################################################################

TOKEN_COUNTS = [50, 100, 200, 400]   # max_new_tokens values to test
RUNS = 3                             # timed runs per setting; we report the median
PROMPT = "Hello, I am"

GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }


######################################################################
# Helpers
######################################################################

def build_model(module):
    """Build a fresh, untrained GPT model from one of Raschka's files.

    `module` is gpt_ch04 or gpt_with_kv_cache. Both files have a GPTModel class.
    """
    torch.manual_seed(123) 
    model = module.GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model


def encode_prompt(text):
    """Turn the prompt into a (1, num_tokens) tensor of token IDs."""
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def time_one_run(generate_fn, model, idx, max_new_tokens):
    """Run generate_fn once. Return (seconds, token_ids)."""
    # TODO: Record the time, call generate_fn(model, idx, max_new_tokens),
    #       record the time again. (Same pattern as Raschka's main(),
    #       minus the CUDA lines; we're on CPU.)
    start = time.time()
    token_ids = generate_fn(model, idx, max_new_tokens)
    seconds = time.time() - start
    return seconds, token_ids


# These two wrappers give both versions the same call signature, so
# time_one_run doesn't need to know which one it's timing.

def generate_no_cache(model, idx, max_new_tokens):
    return gpt_ch04.generate_text_simple(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=GPT_CONFIG_124M["context_length"],
    )


def generate_with_cache(model, idx, max_new_tokens):
    return gpt_with_kv_cache.generate_text_simple_cached(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
    )


def tokens_per_sec(num_tokens, seconds):
    # TODO: Decide which token count to divide by and write it in
    #       README.md (Method > "How tokens/sec is calculated"):
    #         - new tokens only (max_new_tokens), or
    #         - the whole output (prompt + new), like Raschka's script.
    #       main() below passes in whichever you choose.
    return num_tokens / seconds


######################################################################
# The experiment
######################################################################

def main():
    print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} CPU threads")

    idx = encode_prompt(PROMPT)
    prompt_len = idx.shape[1]

    model_no_cache = build_model(gpt_ch04)
    model_cache = build_model(gpt_with_kv_cache)

    # Warm-up: one short throwaway run each, so one-time startup costs
    # don't land in whichever version happens to be timed first.
    time_one_run(generate_no_cache, model_no_cache, idx, 10)
    time_one_run(generate_with_cache, model_cache, idx, 10)

    rows = []       # one dict per individual run, for results.csv
    summary = []    # one dict per token count, for the table and chart

    for n in TOKEN_COUNTS:
        times_no_cache = []
        times_cache = []

        for run in range(1, RUNS + 1):
            t_nc, ids_nc = time_one_run(generate_no_cache, model_no_cache, idx, n)
            t_c, ids_c = time_one_run(generate_with_cache, model_cache, idx, n)

            # TODO: Check that both versions produced exactly the same token
            #       IDs. Hint: torch.equal(a, b) returns True or False.
            identical = torch.equal(ids_nc, ids_c)

            print(f"{n:>4} tokens, run {run}: no cache {t_nc:6.2f} s | "
                  f"cache {t_c:6.2f} s | identical: {identical}")

            times_no_cache.append(t_nc)
            times_cache.append(t_c)
            rows.append({"new_tokens": n, "run": run,
                         "no_cache_s": round(t_nc, 3), "cache_s": round(t_c, 3),
                         "identical": identical})

        med_nc = statistics.median(times_no_cache)
        med_c = statistics.median(times_cache)

        # TODO: Pick the token count for tokens/sec (see tokens_per_sec above):
        #       n, or n + prompt_len.
        count = n

        summary.append({
            "new_tokens": n,
            "runs_no_cache": times_no_cache, "med_no_cache": med_nc,
            "tps_no_cache": tokens_per_sec(count, med_nc),
            "runs_cache": times_cache, "med_cache": med_c,
            "tps_cache": tokens_per_sec(count, med_c),
            "speedup": med_nc / med_c,
        })

    save_csv(rows)
    print_markdown_table(summary)
    plot(summary)


######################################################################
# Output
######################################################################

def save_csv(rows):
    path = PROJECT_DIR / "results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {path}")


def print_markdown_table(summary):
    """Print rows in the same column order as README.md section 6.2."""
    fmt = lambda ts: ", ".join(f"{t:.2f}" for t in ts)
    print("\n| New tokens | No cache: runs (s) | No cache: median (s) | No cache (tok/s) "
          "| Cache: runs (s) | Cache: median (s) | Cache (tok/s) | Speedup |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in summary:
        print(f"| {s['new_tokens']} | {fmt(s['runs_no_cache'])} | {s['med_no_cache']:.2f} "
              f"| {s['tps_no_cache']:.1f} | {fmt(s['runs_cache'])} | {s['med_cache']:.2f} "
              f"| {s['tps_cache']:.1f} | {s['speedup']:.1f}× |")


def plot(summary):
    x = [s["new_tokens"] for s in summary]
    series = [
        ("With KV cache", [s["tps_cache"] for s in summary], "#2a78d6"),
        ("No cache", [s["tps_no_cache"] for s in summary], "#eb6834"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for label, y, color in series:
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
        # Direct label at the end of each line
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#0b0b0b")

    ax.set_xticks(x)
    ax.set_xlim(0, x[-1] * 1.25)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("New tokens generated", color="#52514e")
    ax.set_ylabel("Tokens per second", color="#52514e")
    ax.set_title("GPT-2 124M (untrained), CPU: tokens/sec vs. output length",
                 loc="left", fontsize=11, color="#0b0b0b")
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.tick_params(colors="#52514e", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=9)

    fig.tight_layout()
    path = PROJECT_DIR / "kv_cache_speed.png"
    fig.savefig(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
