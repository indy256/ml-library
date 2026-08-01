"""
Minimal GPT-2 training script — ClimbMix dataset version.

Same model and training loop as train_gpt2.py, but instead of
Tiny Shakespeare it trains on real web data: the ClimbMix dataset
(hosted on Hugging Face as karpathy/climbmix-400b-shuffle). The full
dataset is ~400 billion tokens split into thousands of ~92 MB "shard"
files; we download and train on the first num_shards of them (each is
~100 million tokens — one shard alone is ~300x more text than Tiny
Shakespeare).

The model is GPT-2's architecture with three deviations, each of which
was measured to reach a given loss in fewer steps and is explained where
it appears below: rotary position embeddings instead of a learned
position table, QK-norm on the queries and keys, and GPT-2's own
1/sqrt(2 * n_layer) init on the projections that write back into the
residual stream. Together with a retuned Adam learning rate they reach
the previous version's 700-step loss at step 314, for 7.7% more time per
step — about 2.07x less wall clock to the same place. Checkpoints record
which architecture wrote them and refuse to load into a different one.

Each shard is downloaded ONCE and cached on disk as
data/shard_00000.parquet (~92 MB). Tokenization into GPT-2 token ids is
NOT cached: shards are tokenized in memory as training reaches them, on
a background thread so the GPU never waits for one, which means two
shards are in RAM at a time — the one being trained on and the one being
prepared behind it. Bumping num_shards later only downloads the NEW
shards.

Every eval_interval steps the script prints the loss and a short sample
of text generated from sample_prompt, so progress is visible as text and
not only as a falling number. The loss printed is the average over the
steps since the last report, which the training loop has already
computed — measuring it costs no extra work.

Training checkpoints itself every eval_interval steps: the model
weights, the optimizer state, and the exact reading position in the
data are saved to gpt2_climbmix_checkpoint.pt. The script can therefore
be aborted (Ctrl-C, crash, reboot) at any time — running it again
continues from the last checkpoint instead of starting over. Delete the
checkpoint file to restart training from scratch. The write happens on a
background thread, so training pauses only for the ~11 ms it takes to
copy the state to host memory, not for the ~420 ms of serializing.

Usage:
    python train_gpt2_climbmix.py              # train on 1 shard
    python train_gpt2_climbmix.py 4            # train on the first 4 shards
"""

import concurrent.futures
import os
import sys
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from muon import SingleDeviceMuonWithAuxAdam

# GPT-2 tokens can decode to any Unicode character, which the default
# Windows console encoding may not be able to print. Switch stdout to
# UTF-8 and replace anything unprintable instead of crashing.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------
# Config — all the knobs that control the model size and training.
#
# Same small GPT-2 as before. The real GPT-2 (124M parameters) uses the
# exact same architecture and tokenizer — only bigger numbers:
# n_layer=12, n_head=12, n_embd=768, block_size=1024.
# ----------------------------------------------------------------------
block_size = 1024      # context length: how many tokens the model can
                      # "see" at once when predicting the next one
n_layer = 8          # how many transformer blocks are stacked on top of
                      # each other (deeper = smarter but slower)
n_head = 8           # each attention layer is split into this many
                      # independent "heads" that each look for different
                      # patterns in the text. Must divide n_embd evenly —
                      # 512 / 8 = 64 numbers per head, the same head size
                      # the real GPT-2 uses.
n_embd = 512          # embedding size: every token is represented
                      # internally as a list of this many numbers
dropout = 0.0         # during training, randomly zero out this fraction of
                      # values, which fights memorization. Turned OFF here:
                      # dropout earns its keep when a model sees the same
                      # text over and over, but this run streams ~1.3 billion
                      # tokens of web text and never revisits any of it, so a
                      # 45M-parameter model has nothing to memorize. Leaving
                      # it on would only cost speed — every dropout is an
                      # extra random mask to generate, write and read back.

num_shards = 1000      # how many ClimbMix shards to train on. Each shard is
                      # ~100M tokens (~92 MB download, ~200 MB of RAM once
                      # tokenized). Can also be set from the command line,
                      # see Usage.

batch_size = 64       # how many text snippets we train on simultaneously.
                      # Bigger batches give the GPU more independent work per
                      # kernel launch, so throughput improves — but with
                      # diminishing returns, and past ~64 it reverses on this
                      # machine as the activations stop fitting comfortably
                      # in memory (96 was measurably SLOWER than 64).
                      # Note this also doubles the tokens each step consumes,
                      # so max_steps steps now cover twice as much text.
learning_rate = 3e-3  # how big a step Adam takes on each update. Adam only
                      # handles the embeddings and the 1D parameters (see
                      # muon_lr below), and THAT is why this is ten times the
                      # usual 3e-4. The old value was tuned when AdamW drove
                      # every parameter in the model, including the hidden
                      # matrices; back then 6e-4 measured worse than 3e-4 and
                      # the number looked settled. Handing the hidden matrices
                      # to Muon changed what this group is, and the leftover
                      # group — embeddings and 1D params — wants a much larger
                      # step. Measured at 700 steps, held-out loss:
                      #
                      #     3e-4  4.7126   <- the old value
                      #     1e-3  4.4362
                      #     3e-3  4.3740   <- broad flat optimum
                      #     5e-3  4.3875
                      #     1e-2  4.4400
                      #
                      # Same U-shape Muon's own LR has. 3e-3 alone is worth
                      # 1.44x fewer steps to the old loss.
muon_lr = 0.02        # the same, for the hidden weight matrices, which are
                      # trained by Muon instead (see muon.py). Not comparable
                      # to the number above: Muon's update is orthogonal, so
                      # its scale is set entirely by the learning rate rather
                      # than by the gradient's magnitude, and the useful range
                      # is a hundred times larger.
                      #
                      # This is a broad flat optimum with a cliff above it —
                      # at 700 steps, 0.01 -> 4.1623, 0.02 -> 4.1578,
                      # 0.03 -> 4.1729, 0.04 -> 4.4016, 0.06 diverges
                      # outright. The flat part is genuinely flat: 0.01 and
                      # 0.02 differ by less than the 0.006 that two identical
                      # runs differ by, so this is 0.02 because it measured
                      # best, not because 0.01 was wrong. Do not go past 0.03.
max_steps = 2000000      # total number of training updates
eval_interval = 50   # print the loss, sample, and checkpoint every N steps.
                      # Reporting is free — the loss printed is the average of
                      # the training losses already computed over the window
                      # (see the report_loss note in main), so this interval
                      # controls how often we checkpoint and how smooth the
                      # printed number is, not how much work is spent measuring.

# Every eval also prints a short piece of text the model generates from
# this prompt, so learning is visible as text and not just as a number:
# the first samples are gibberish, then words appear, then sentences.
# Kept short because generation runs one token at a time and each token
# costs a full forward pass — 100 tokens is a couple of seconds, which is
# nothing spread over eval_interval steps.

# sample_prompt = "What is the distance between Berlin and Paris?"
sample_prompt = "What is poetry?"
sample_tokens = 100

# This script trains on a CUDA GPU with bfloat16 support, and says so
# directly rather than through a `device` variable: there is no CPU or
# float16 fallback. The dtype decisions below are what make it fast, and
# a fallback path would be a second, untested set of them.

# Fix the random seed so the run is reproducible: same numbers every time.
torch.manual_seed(1337)

# --- dtypes ------------------------------------------------------------
# There are exactly two, and the split is the whole story:
#
#   PARAMETERS and GRADIENTS are float32 — they accumulate across steps,
#     so their low bits are the only ones that carry information forward.
#   ACTIVATIONS are bfloat16 — every one of them is recomputed from
#     scratch next step, so precision lost there does not accumulate.
#
# bfloat16 halves the bytes to move and gets the tensor-core paths, so
# the matmuls are ~2x faster; it has the same exponent range as float32
# (only fewer mantissa bits), so unlike float16 it needs no loss scaling.
#
# Autocast (in the training loop) gets only half of the second rule; the
# other half has to be said explicitly, because autocast does NOT get it
# right on its own. It picks the dtype of individual OPERATIONS, and keeps
# layer_norm and embedding on a float32 list (along with reductions and
# transcendentals generally, where the extra bits are worth having). That
# is the right choice locally and the wrong one globally: every LayerNorm hands
# the next block a float32 tensor, so x between the blocks is 134 MB
# instead of 67 MB, and autocast then has to insert a cast back to
# bfloat16 in front of every single matmul. The profile showed both halves
# of that bill — 15 ms/step of pure `_to_copy` kernels, and a LayerNorm
# backward moving twice the bytes it needed to. So LayerNorm and the
# embedding lookup below each end with an explicit .to(torch.bfloat16),
# which is what actually pins the residual stream to 16 bits.
#
# This model is memory-bound, not arithmetic-bound: the elementwise
# kernels in the transformer body already run at the GPU's measured 203
# GB/s, so bytes moved IS time. Pinning the stream to bfloat16 halves the
# bytes and was worth ~8% of end-to-end throughput, with a loss curve that
# matches the float32 stream to four decimal places (checked over 400
# steps on real data).

# torch.compile traces the model once and generates fused GPU kernels for
# it. This model is dominated by memory traffic rather than arithmetic
# (see the padded vocab note below), and fusing chains like
# "layernorm -> matmul -> gelu" into one kernel is exactly what removes
# that traffic. Costs ~1-2 minutes of compilation on the first step, then
# pays it back many times over. Set to False to debug with plain eager.
use_compile = True

# ----------------------------------------------------------------------
# Data: download ClimbMix shards and pre-tokenize them (both cached)
# ----------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")  # the GPT-2 tokenizer, 50257 tokens
vocab_size = enc.n_vocab

# The model's output layer is by far the most expensive part of a small
# GPT: it turns every one of the 65,536 tokens in a batch into 50,257
# scores — two thirds of all the arithmetic in a training step happens in
# this one layer. 50257 is an awkward size for a GPU — tensor cores work on tiles
# of 8/16/64 numbers, so a row of 50257 leaves a ragged remainder that
# forces slower fallback code. Rounding the layer up to 50304 (= 64 * 786)
# adds 47 unused columns, ~0.1% more arithmetic, and makes the whole
# output layer measurably faster. The extra token ids simply never appear
# in the data; training pushes their scores down and generation masks
# them out, so the model behaves exactly as if they weren't there.
model_vocab_size = (vocab_size + 63) // 64 * 64

# num_shards can be overridden from the command line, e.g.
#   python train_gpt2_climbmix.py 4
if len(sys.argv) > 1:
    num_shards = int(sys.argv[1])

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def shard_path(shard):
    """Where the raw downloaded shard number `shard` is stored on disk.

    The dataset names its files shard_00000.parquet, shard_00001.parquet,
    ... so we zero-pad the shard number to 5 digits the same way.
    """
    return os.path.join(DATA_DIR, f"shard_{shard:05d}.parquet")


def download_shard(shard):
    """Download one raw shard from Hugging Face (skipped if a previous
    run already downloaded it)."""
    parquet_path = shard_path(shard)
    if os.path.exists(parquet_path):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    url = f"{BASE_URL}/{os.path.basename(parquet_path)}"
    print(f"downloading {url} ...")
    def progress(blocks, block_size_bytes, total):
        done = blocks * block_size_bytes
        print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    # Download to a temporary name first, then rename. If the download
    # dies halfway, we won't mistake a half-file for a finished one.
    urllib.request.urlretrieve(url, parquet_path + ".tmp", progress)
    os.replace(parquet_path + ".tmp", parquet_path)
    print()


def load_shard(shard, verbose=True):
    """Tokenize one downloaded shard; return its token ids.

    Tokenization is not cached — it runs in memory every time a shard is
    needed. Returns one numpy array with the whole shard as a single
    token stream. uint16 works because every GPT-2 token id is < 50257,
    which fits in 16 bits — half the memory of the usual 32-bit ints.

    verbose=False silences the progress output, which is what the
    background prefetch below uses: it runs while the training loop is
    rewriting its own status line, and two writers on one line garble
    both.
    """
    parquet_path = shard_path(shard)
    # Parquet is a compressed table format;
    # each shard has a "text" column where each row is one document (a
    # web page, article, etc.). Instead of decompressing every document
    # at once (several GB of peak memory), we read a few thousand
    # documents at a time, tokenize them, and keep only the compact
    # uint16 token ids.
    if verbose:
        print(f"tokenizing {parquet_path} ...")
    eot = enc.eot_token  # id 50256
    n_docs = 0
    n_tokens = 0
    chunks = []
    for batch in pq.ParquetFile(parquet_path).iter_batches(
        columns=["text"], batch_size=8192
    ):
        texts = batch.column("text").to_pylist()
        # encode_ordinary_batch uses multiple CPU threads, which
        # matters at this size (hundreds of MB of text overall).
        all_ids = enc.encode_ordinary_batch(texts)

        # Join the documents into one long token stream, with the
        # special <|endoftext|> token between them so the model can
        # learn where one document ends and an unrelated one begins.
        total = sum(len(ids) for ids in all_ids) + len(all_ids)
        tokens = np.empty(total, dtype=np.uint16)
        pos = 0
        for ids in all_ids:
            tokens[pos : pos + len(ids)] = ids
            pos += len(ids)
            tokens[pos] = eot
            pos += 1
        chunks.append(tokens)

        n_docs += len(texts)
        n_tokens += total
        if verbose:
            print(f"\r  {n_docs:,} documents, {n_tokens:,} tokens", end="")
    if verbose:
        print()
    return np.concatenate(chunks)


# Download every shard up front (cached on disk, so this is instant on
# later runs). Tokenization is deferred: shards are tokenized one at a
# time as training reaches them, so we never hold the whole dataset in
# memory at once.
for shard in range(num_shards):
    download_shard(shard)
print(f"dataset: {num_shards} shard(s), ~100M tokens each")


# How many tokens one batch consumes: batch_size back-to-back chunks of
# block_size tokens (plus one extra token for the shifted targets).
batch_tokens = batch_size * block_size

# Batches are read deterministically, not sampled: get_batch consumes the
# shards one by one — all of shard 0 first, then shard 1, and so on,
# wrapping back to shard 0 after the last — and within each shard it reads
# consecutive chunks front to back. There is no train/val split: every
# batch comes from the same stream, whether the caller uses it to train on
# or to measure loss with. So two counters describe the whole reader:
# which shard it is currently reading, and the next read position in it.
cur_shard = 0
cur_pos = 0

# The tokens of the shard currently being read: tokenized lazily on the
# first batch of the run and replaced (not kept) when we move on to the
# next shard.
cur_tokens = None

# Tokenizing a shard takes ~6 seconds, and a shard lasts ~840 steps, so
# doing it inline stalls the GPU for 6 seconds roughly every 7 minutes —
# about 1% of the run spent with the GPU idle. It is pure CPU work, and
# during training the CPU has nothing to do but queue kernels, so the
# next shard is tokenized on a background thread while the current one is
# still being trained on. By the time the reader needs it, it is ready.
#
# The cost is holding two shards in RAM instead of one (~110 MB each as
# uint16), which is what the module docstring's "one at a time" claim
# used to buy. That trade is worth it at this size.
_loader = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="shard-prefetch")
_prefetch_shard = None      # which shard _prefetch_future is tokenizing
_prefetch_future = None


def take_shard(shard):
    """Return the tokens of `shard`, and start tokenizing the one after it.

    If the background thread was already working on this shard (the
    normal case) we just wait for it, which by then costs nothing.
    """
    global _prefetch_shard, _prefetch_future
    if _prefetch_future is not None and _prefetch_shard == shard:
        tokens = _prefetch_future.result()
    else:
        # No prefetch, or it guessed wrong (only happens on the first
        # shard of a run and after a resume lands mid-stream).
        if _prefetch_future is not None:
            _prefetch_future.result()  # let it finish, then drop it
        tokens = load_shard(shard)
    _prefetch_shard = (shard + 1) % num_shards
    _prefetch_future = _loader.submit(load_shard, _prefetch_shard, verbose=False)
    return tokens


def get_batch():
    """Grab the next batch of examples.

    Returns two tensors of shape (batch_size, block_size):
      x = chunks of tokens (the input)
      y = the same chunks shifted one token to the right (the answer)

    So at every position, the model's target is simply "the next token".
    """
    global cur_shard, cur_pos, cur_tokens
    # Keep reading the current shard front to back; once it doesn't have a
    # full batch left, move on to the next shard (wrapping back to shard 0
    # after the last) and start from its beginning.
    if cur_tokens is None:  # first batch of the run
        cur_tokens = take_shard(cur_shard)
    if cur_pos + batch_tokens + 1 > len(cur_tokens):
        cur_shard = (cur_shard + 1) % num_shards
        cur_pos = 0
        # Drop the finished shard before taking the next one, so the two
        # arrays alive at once are the new shard and the one being
        # prefetched behind it, never three.
        cur_tokens = None
        cur_tokens = take_shard(cur_shard)
    d, pos = cur_tokens, cur_pos
    cur_pos += batch_tokens
    # The whole batch is one contiguous run of tokens, so we can take it in
    # a single slice and just reinterpret its shape — no per-chunk slicing
    # or stacking. (.astype(int64) because the shard stores compact uint16,
    # but PyTorch embedding layers want 64-bit indices.) One extra token is
    # read at the end so the targets can be the same run shifted by one.
    #
    # .pin_memory() puts the batch in a region of RAM the GPU can read
    # directly, which lets .to(non_blocking=True) hand the copy to the
    # DMA engine and return immediately: the CPU goes straight on to
    # queueing the next step's work instead of waiting for the transfer.
    buf = torch.from_numpy(d[pos : pos + batch_tokens + 1].astype(np.int64)).pin_memory()
    # x = the tokens, y = the same tokens shifted right by one, so at every
    # position the target is simply "the next token".
    x = buf[:-1].view(batch_size, block_size)
    y = buf[1:].view(batch_size, block_size)
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


def train_position():
    """Describe where the NEXT batch will be read from."""
    if cur_tokens is None:  # nothing has been read yet this run
        return f"shard {cur_shard}, token {cur_pos:,}"
    return f"shard {cur_shard}, token {cur_pos:,} ({100 * cur_pos / len(cur_tokens):.1f}% into the shard)"


# ----------------------------------------------------------------------
# Model: the GPT-2 architecture (identical to the previous scripts)
#
# A GPT is a stack of identical "transformer blocks". Each block does two
# things: (1) attention — lets every token look at the tokens before it
# and gather information from them, and (2) an MLP — lets each token
# "think about" what it gathered.
#
# Shape notation used in comments below:
#   B = batch size (how many text chunks at once)
#   T = time / sequence length (how many tokens in each chunk)
#   C = channels (n_embd, the size of each token's internal vector)
# ----------------------------------------------------------------------
class LayerNorm(nn.LayerNorm):
    """nn.LayerNorm that hands back bfloat16.

    Identical arithmetic to nn.LayerNorm — same parameters, same
    normalization, same float32 internals — with one cast bolted onto the
    end so the result does not silently widen the residual stream back to
    float32. See the dtypes comment above for why that one cast is worth
    ~8% of the step. torch.compile folds it into the LayerNorm's own
    kernel, so it costs nothing to apply.
    """

    def forward(self, x):
        return super().forward(x).to(torch.bfloat16)


# ----------------------------------------------------------------------
# Rotary position embeddings (RoPE)
#
# The original GPT-2 tells a token where it is by ADDING a learned
# "position vector" to it before the first block (the wpe table). RoPE
# instead ROTATES each query and key by an angle proportional to its
# position, inside every attention layer. Because a dot product only
# sees the angle BETWEEN two vectors, rotating both by their positions
# makes q . k depend on how far apart the two tokens are rather than on
# where each one sits absolutely -- which is the thing attention
# actually wants to know, and it no longer has to spend capacity
# learning it.
#
# Measured at 700 steps, held-out loss, everything else fixed: 4.3740
# with the learned table, 4.2454 with RoPE. It also deletes the wpe
# table (0.5M parameters) outright.
#
# The tables are built ONCE here, in bfloat16. Rebuilding
# them on every forward, and letting them widen the rotation to float32,
# measured materially slower for exactly the same loss — the rotation is
# pure memory traffic, so doubling its width doubles its cost.
#
# RoPE and QK-norm together cost +7.7% per step (502.2 -> 540.9 ms,
# measured by alternating both configs in one process; sequential runs
# on this box disagree with each other by 7% and cannot resolve this).
# Against 2.23x fewer steps that is a large net win.
# ----------------------------------------------------------------------
head_dim = n_embd // n_head


def _rope_tables():
    """cos/sin of every (position, frequency) pair, (block_size, head_dim/2).

    Frequencies fall off geometrically, so the first pairs of channels
    spin fast (they resolve neighbouring tokens) and the last ones barely
    move over the whole context (they encode coarse, long-range position).
    """
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, head_dim, 2, device="cuda").float() / head_dim))
    angles = torch.outer(
        torch.arange(block_size, device="cuda").float(), inv_freq)
    return angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)


rope_cos, rope_sin = _rope_tables()


def apply_rope(x):
    """Rotate q or k — shape (B, n_head, T, head_dim) — by position.

    Treats the head's channels as head_dim/2 planes (channel i paired
    with channel i + head_dim/2) and turns each plane by its angle. That
    "two halves" pairing is a convention; any consistent one works, as
    long as q and k use the same.
    """
    seqlen = x.size(-2)
    cos, sin = rope_cos[:seqlen], rope_sin[:seqlen]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def norm_heads(x):
    """Scale each head's vector to unit RMS (QK-norm).

    Applied to q and k before attention. It stops any single position
    from producing a huge query or key, which is what makes attention
    logits blow up and the softmax saturate early in training. Worth
    4.2454 -> 4.1754 at 700 steps on top of RoPE.

    Written out rather than called as F.rms_norm: the functional form
    does not fuse into the surrounding kernels here, and measured 2%
    slower end-to-end (551.8 vs 540.9 ms/step) for identical loss —
    4.1757 against 4.1754, which is inside run-to-run noise. The mean is
    taken in float32 because it is a reduction over squared values.
    """
    return x * torch.rsqrt(
        x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(torch.bfloat16)


class CausalSelfAttention(nn.Module):
    """The heart of the transformer.

    For each token position, attention computes three vectors:
      query (q): "what am I looking for?"
      key   (k): "what do I contain?"
      value (v): "what information do I give if someone looks at me?"

    Each position compares its query against every other position's key
    to decide "how much should I pay attention to that token?", then
    takes a weighted average of their values.

    "Causal" means a token may only look at tokens BEFORE it, never
    after — because at generation time the future doesn't exist yet.
    """

    def __init__(self):
        super().__init__()
        # One linear layer produces q, k and v all at once (it outputs
        # 3*n_embd numbers, which we split into three parts below).
        # A linear layer is just a matrix multiply: output = input @ W + b.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # A final linear layer to mix the heads' outputs back together.
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding size
        # Compute q, k, v for every position, then split into three tensors.
        q, k, v = self.c_attn(x).split(n_embd, dim=2)
        # Split the embedding into n_head smaller pieces ("heads"), so each
        # head can attend to different patterns independently.
        # Shape change: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        # Normalize each head's query and key, then rotate them by their
        # position. Both act only on q and k — never on v, which carries
        # content rather than addressing. See norm_heads and apply_rope.
        q, k = norm_heads(q), norm_heads(k)
        q, k = apply_rope(q), apply_rope(k)
        # PyTorch's built-in attention: softmax(q @ k^T / sqrt(d)) @ v.
        # is_causal=True applies the "no looking at the future" mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Undo the head split: glue the heads back into one vector per position.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    """A small two-layer neural network applied to each position separately.

    After attention has gathered information from other tokens, the MLP
    processes that information. It expands the vector to 4x its size,
    applies a nonlinearity (GELU — a smooth version of "zero out negatives"),
    and shrinks it back down.
    """

    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)    # expand: 512 -> 2048
        self.c_proj = nn.Linear(4 * n_embd, n_embd)  # shrink: 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block: attention, then MLP.

    Two details that make deep networks trainable:
      - LayerNorm (ln_1, ln_2): rescales values to a standard range before
        each sub-layer, which keeps the numbers well-behaved.
      - Residual connections (the "x +" parts): each sub-layer ADDS its
        result to its input rather than replacing it. Gradients can then
        flow straight through the network, so even deep stacks train well.
    """

    def __init__(self):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = LayerNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # communicate: look at other positions
        x = x + self.mlp(self.ln_2(x))   # compute: process what was gathered
        return x


class GPT(nn.Module):
    """The full model: embeddings -> transformer blocks -> prediction head."""

    def __init__(self):
        super().__init__()
        # Token embedding: a learned lookup table that maps each token id
        # to a vector of n_embd numbers. This vector is the model's internal
        # "meaning" of that token, and it improves during training.
        # With the 50,257-token vocabulary this table is by far the largest
        # part of the model (50257 * 512 ≈ 26M of the ~45M parameters).
        # It is built at the padded size for the speed reason explained
        # where model_vocab_size is defined.
        self.wte = nn.Embedding(model_vocab_size, n_embd)
        # There is no position embedding table here. Attention alone has no
        # idea about word order, but RoPE supplies that inside each
        # attention layer instead of adding a learned vector up front —
        # see the RoPE section above for why that trains faster.
        self.drop = nn.Dropout(dropout)
        # The stack of transformer blocks — the actual "brain".
        self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
        self.ln_f = LayerNorm(n_embd)  # one final normalization
        # The prediction head: converts each position's final vector into
        # vocab_size scores ("logits") — one score per possible next token.
        self.lm_head = nn.Linear(n_embd, model_vocab_size, bias=False)
        # Weight tying (a GPT-2 trick): the table that turns ids INTO vectors
        # and the layer that turns vectors back into token scores share
        # the same weights. Saves parameters and slightly improves quality.
        self.lm_head.weight = self.wte.weight

        # Initialize all weights with small random values (std 0.02, as in
        # GPT-2). Good starting values matter for stable training.
        self.apply(self._init_weights)

        # The two matrices per block that write BACK into the residual
        # stream get a smaller init than everything else. Every block adds
        # its output to the stream rather than replacing it, so with n_layer
        # blocks all starting at the same scale the stream's variance grows
        # with depth, and the early steps are spent undoing that. Scaling
        # these down by 1/sqrt(2 * n_layer) — two residual writes per block —
        # starts the stream at a depth-independent scale instead. This is
        # GPT-2's own rule; it was simply missing here. Worth 4.7126 ->
        # 4.6560 at 700 steps on its own.
        resid_std = 0.02 / (2 * n_layer) ** 0.5
        for block in self.blocks:
            for proj in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(proj.weight, mean=0.0, std=resid_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Run tokens through the model.

        idx:     (B, T) tensor of token ids — the input text.
        targets: (B, T) tensor of the correct next tokens, or None.

        Returns (logits, loss), but only ever one of the two: with targets
        it returns (None, loss), and without them — during generation — it
        returns (logits, None). The comment further down explains why the
        logits must not escape this method during training.
        """
        B, T = idx.shape
        # Look up each token's vector. Where the token SITS is not added
        # here — RoPE applies it inside every attention layer instead.
        # (.to(torch.bfloat16) for the same reason as in LayerNorm:
        # nn.Embedding is on autocast's float32 list, so this is where the
        # residual stream's width is established for the whole forward pass.)
        x = self.drop(self.wte(idx).to(torch.bfloat16))
        # Pass through every transformer block in order.
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # Generation: there is no loss to compute, so the caller wants
            # the scores themselves. For every position, produce a score for
            # each possible next token. Shape (B, T, vocab_size).
            return self.lm_head(x), None

        # Training. The logits are by far the largest tensor in the entire
        # step: B*T*50304 numbers is 6.1 GiB in bfloat16, more than 100x the
        # size of every other activation in the model put together. All we
        # actually want out of them is one number, the loss — so they must
        # NOT be returned. Any tensor that leaves this method is an output of
        # the compiled graph, which torch.compile must write to memory in
        # full and keep alive until the backward pass is done. Returning it
        # "just in case" costs ~15% of training throughput.
        #
        # This layer is also where the step spends most of its time, and
        # these two lines are the price of writing it in plain PyTorch:
        # torch.compile fuses everything it can here, but it cannot fuse
        # the softmax into the matmul that feeds it (the matmul is cuBLAS,
        # which the compiler treats as a black box), so the logits are
        # written once, read once to reduce them, and read again in the
        # backward — six passes over 6.1 GiB per step, ~185 ms of the step
        # at this chip's 203 GB/s.
        #
        # Hand-written Triton kernels used to do that fusion (three passes
        # instead of six, 112k tok/s instead of 93k). They were removed in
        # favour of these two lines. What was measured before removing
        # them, at this model's shapes, output layer forward+backward:
        #
        #     plain PyTorch, compiled     285 ms
        #     Triton, softmax in the matmul kernels
        #                                 175 ms
        #
        # and the plain version is not obviously improvable from here:
        # its reduction kernels already run at 180-205 GB/s of the 203
        # available, and the only formulations that cut a pass (chunking
        # with recompute, hand-picked backward matmul layouts, a manual
        # logsumexp) all measured slower end to end. Fusing a reduction
        # into a matmul needs a matmul kernel of one's own, and
        # torch.compile will not generate one on this GPU — inductor's
        # Triton GEMM templates require 68 SMs and a GB10 has 48.
        logits = self.lm_head(x)
        # Cross-entropy loss: measures how badly the predicted scores
        # match the true next tokens. Lower = better. It flattens
        # (B, T, vocab_size) into (B*T, vocab_size) because the loss
        # function treats every position as an independent prediction.
        loss = F.cross_entropy(logits.view(-1, model_vocab_size), targets.view(-1))
        return None, loss

    @torch.no_grad()  # generation is not training, so skip gradient tracking
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text by predicting one token at a time.

        Start from some prompt (idx), predict the next token, append it,
        and repeat. This loop IS how all GPT models produce text.
        """
        for _ in range(max_new_tokens):
            # The model can only see block_size tokens, so if the text
            # got longer than that, keep only the most recent tokens.
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # We only care about the prediction at the LAST position
            # (the next token). Drop the padding columns the output layer
            # was rounded up to — they are not real tokens and the
            # tokenizer could not decode them.
            logits = logits[:, -1, :vocab_size].float()
            # Softmax turns raw scores into probabilities that sum to 1.
            # Temperature <1 makes the model more conservative, >1 makes
            # it more random.
            probs = F.softmax(logits / temperature, dim=-1)
            # Randomly pick one token, weighted by those probabilities.
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append the chosen token and continue the loop.
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
# There used to be an estimate_loss() here that ran ten extra forward
# passes over ten fresh batches every eval. It was measuring something the
# training loop already knows.
#
# Every training step computes the loss of its batch BEFORE the optimizer
# updates on it, and the reader only ever moves forward through the
# shards, so that number is already a loss on text the model has never
# been trained on — exactly what the separate eval was buying. Averaging
# the losses the loop produces anyway therefore costs nothing, and it
# averages over a full eval_interval of batches instead of ten, so the
# printed number is steadier than the one it replaces.
#
# It was worth 6% of the run's wall clock, and another 17% of the batches
# read: those ten batches per fifty steps were consumed for measurement
# and then thrown away, never trained on.
#
# Two things to know about the number. It is an average over the window
# rather than a reading at the current weights, so while the loss is
# falling steeply it lags by about half a window. And if dropout is ever
# turned back on it will read slightly high, because training batches are
# measured with dropout active.


def sample_text(model, prompt, max_new_tokens):
    """Continue `prompt` for max_new_tokens tokens and return the text.

    Generation runs on the uncompiled model: it feeds a sequence that grows
    by one token each round, and a compiled model would recompile itself for
    every new length. eval() switches the model out of training mode, which
    matters if dropout is ever turned back on: the training loop leaves the
    model in train mode, and sampling with dropout on would randomly zero
    values and degrade the text.
    """
    context = torch.tensor([enc.encode(prompt)], dtype=torch.long, device="cuda")
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(context, max_new_tokens=max_new_tokens)
    model.train()
    return enc.decode(out[0].tolist())


# Where the training checkpoint lives. Unlike the final model file (which
# holds only the weights), the checkpoint holds everything needed to
# CONTINUE training, so an aborted run doesn't have to start over.
CKPT_PATH = "gpt2_climbmix_checkpoint.pt"

# A short description of the architecture that produced a checkpoint.
# `optimizer_name` below already guards against resuming someone else's
# optimizer state; this guards the same way against resuming someone
# else's MODEL. Switching to RoPE deleted the wpe table and changed what
# attention does, so a checkpoint written before that change cannot be
# continued — and a state dict that merely has one fewer key is exactly
# the kind of mismatch load_state_dict would be happy to complain about
# obscurely, or that a future change might make it not complain about at
# all. Anything that alters the meaning of the stored weights belongs in
# this string.
ARCH = (f"rope-qknorm-L{n_layer}-H{n_head}-E{n_embd}"
        f"-T{block_size}-V{model_vocab_size}")


# The checkpoint is 517 MB, and handing torch.save a dict of GPU tensors
# made it a 1.2-second stall every eval — 4% of the whole run, spent with
# the GPU idle. Almost none of that was the disk or the device-to-host
# copy (a bulk copy of this much data takes 9 ms): it was allocating
# 517 MB of fresh pageable host memory and faulting it in, once per save.
#
# So the host side is allocated exactly once and reused. _CKPT_MIRROR
# holds a pinned CPU tensor per state entry; a save copies into those
# (11 ms) and hands the mirror to a background thread, which does the
# actual serializing and writing (~420 ms) while training carries on.
_CKPT_MIRROR = {}
_writer = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ckpt-writer")
_write_future = None


def _mirror_to_cpu(obj, seen, path=()):
    """Deep-copy a state dict onto the persistent pinned host buffers.

    Walks the same nested structure torch.save would, replacing every
    tensor with its CPU mirror and leaving everything else (ints, floats,
    the param_groups bookkeeping) alone.

    `seen` maps a source tensor's address to the mirror already made for
    it in this pass, so two keys holding the SAME tensor come back as one
    object. That matters here: wte.weight and lm_head.weight are tied, and
    mirroring them separately would break the sharing that lets torch.save
    store the 103 MB embedding once instead of twice.
    """
    if torch.is_tensor(obj):
        key = (obj.data_ptr(), obj.shape, obj.dtype)
        if key in seen:
            return seen[key]
        buf = _CKPT_MIRROR.get(path)
        if buf is None or buf.shape != obj.shape or buf.dtype != obj.dtype:
            buf = torch.empty(obj.shape, dtype=obj.dtype, device="cpu").pin_memory()
            _CKPT_MIRROR[path] = buf
        buf.copy_(obj, non_blocking=True)
        seen[key] = buf
        return buf
    if isinstance(obj, dict):
        return {k: _mirror_to_cpu(v, seen, path + (k,)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_mirror_to_cpu(v, seen, path + (i,))
                         for i, v in enumerate(obj))
    return obj


def _write_checkpoint(payload):
    """Serialize to disk. Runs on the background thread."""
    # Same crash-safety trick as the data files: write to a .tmp name and
    # rename, so an abort mid-save can't corrupt the previous checkpoint.
    torch.save(payload, CKPT_PATH + ".tmp")
    os.replace(CKPT_PATH + ".tmp", CKPT_PATH)


def save_checkpoint(model, optimizer, step):
    """Snapshot everything needed to resume training later.

    Besides the model weights this stores:
      - the optimizer state: Muon keeps a momentum buffer per hidden
        matrix and Adam keeps two running averages per remaining weight,
        and training quality suffers if those are reset
      - the step number, so the loop continues counting where it stopped
      - the data-loader counters, so we keep reading the shards from the
        same position instead of re-training on the same text
      - the random number generator state (used by dropout), making the
        resumed run behave as if it had never been interrupted

    Returns as soon as the state has been copied to host memory; the disk
    write finishes in the background. See _CKPT_MIRROR above.
    """
    global _write_future
    # The previous write reads the same buffers we are about to overwrite,
    # so it has to have finished. At one save per eval_interval this never
    # actually waits — the write takes ~0.4 s and a window is ~28 s — but
    # it is what makes reusing the buffers safe rather than a race.
    if _write_future is not None:
        _write_future.result()
    checkpoint = _mirror_to_cpu({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, seen={})
    # Which optimizer wrote that state. An optimizer state dict is
    # just numbered groups of tensors with no record of what produced
    # them, so without this a checkpoint from a different optimizer
    # either fails to load with a shape error or, worse, loads
    # something meaningless. See load_checkpoint.
    checkpoint["optimizer_name"] = type(optimizer).__name__
    checkpoint["arch"] = ARCH
    checkpoint["step"] = step
    checkpoint["cur_shard"] = cur_shard
    checkpoint["cur_pos"] = cur_pos
    checkpoint["rng_state"] = torch.get_rng_state().clone()
    checkpoint["cuda_rng_state"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    # The copies above were queued with non_blocking=True onto pinned
    # memory, so they are still in flight; the background thread must not
    # start reading until they have landed.
    torch.cuda.synchronize()
    _write_future = _writer.submit(_write_checkpoint, checkpoint)


def finish_checkpoint():
    """Block until any in-flight checkpoint write has hit the disk."""
    if _write_future is not None:
        _write_future.result()


def load_checkpoint(model, optimizer):
    """Restore the last checkpoint, if one exists.

    Returns the step number to resume training at — 0 when there is no
    checkpoint (i.e. this is a fresh training run).
    """
    global cur_shard, cur_pos
    if not os.path.exists(CKPT_PATH):
        return 0
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    # Checkpoints written before the switch to Muon hold AdamW state, which
    # cannot be resumed into the current optimizer — different parameter
    # groups holding different buffers. Say so plainly instead of letting
    # load_state_dict raise something obscure, because the fix is a
    # decision the person running this has to make, not a bug.
    want = type(optimizer).__name__
    got = ckpt.get("optimizer_name", "AdamW")
    if got != want:
        raise SystemExit(
            f"{CKPT_PATH} was written by {got}, but this script now trains "
            f"with {want}. Their optimizer states are not interchangeable.\n"
            f"Move or delete the checkpoint to start a fresh run:\n"
            f"    mv {CKPT_PATH} {CKPT_PATH}.adamw"
        )
    # Same idea, for the model. Checkpoints written before the switch to
    # RoPE + QK-norm hold a wpe table this model no longer has, and their
    # attention weights were trained against different arithmetic, so the
    # numbers in them do not mean what this architecture would read them
    # as. Refuse rather than guess.
    got_arch = ckpt.get("arch", "learned-wpe (pre-RoPE)")
    if got_arch != ARCH:
        raise SystemExit(
            f"{CKPT_PATH} was written by a different architecture:\n"
            f"    checkpoint: {got_arch}\n"
            f"    this script: {ARCH}\n"
            f"Weights from one cannot be continued under the other.\n"
            f"Move the checkpoint aside to start a fresh run:\n"
            f"    mv {CKPT_PATH} {CKPT_PATH}.pre-rope"
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    s, pos = ckpt["cur_shard"], ckpt["cur_pos"]
    # num_shards may have been changed since the checkpoint was written:
    # if the shard we were reading no longer exists, start over from
    # shard 0 (which the counters already point at).
    if s < num_shards:
        cur_shard, cur_pos = s, pos
    torch.set_rng_state(ckpt["rng_state"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    print(f"resuming from {CKPT_PATH} at step {ckpt['step']}")
    return ckpt["step"] + 1


def main():
    model = GPT().cuda()  # build the model and move it onto the GPU
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab: {vocab_size}, params: {n_params/1e6:.2f}M")

    # The compiled model is what we train through; `model` stays the plain
    # module underneath. They share the same weights, so anything that
    # inspects weights — checkpointing, generation — uses `model` directly
    # and doesn't have to know compilation happened at all.
    if use_compile:
        print("compiling the model (first step will take a minute) ...")
    fast_model = torch.compile(model) if use_compile else model

    # The optimizer's job: nudge every weight in the direction that reduces
    # the loss, step after step. Two different rules do that here, split by
    # what kind of thing the parameter is.
    #
    # Muon takes the hidden weight matrices — the four matrices inside each
    # transformer block, 25.2M parameters — because those are linear maps,
    # and orthogonalizing their updates is worth 2.33x in steps-to-loss for
    # +5% wallclock. muon.py explains the mechanism and shows the numbers.
    #
    # Adam takes the rest: the token embedding (which is also the output
    # head, since they're tied above) and every 1D parameter — biases and
    # LayerNorm gains. None of those are linear maps, and Muon must not be
    # applied to them. There is no position embedding in that list any
    # more; RoPE replaced it.
    #
    # This group's learning rate is the single biggest lever in the file
    # and the easiest one to leave stale — see the note on `learning_rate`.
    #
    # The split is by "2D and inside a block", which picks out exactly the
    # four matrices per block and nothing else. Sorting the rest by identity
    # rather than by name is what keeps the tied embedding/head from being
    # counted twice.
    hidden_matrices = [p for p in model.blocks.parameters() if p.ndim >= 2]
    hidden_ids = {id(p) for p in hidden_matrices}
    other_params = [p for p in model.parameters() if id(p) not in hidden_ids]
    optimizer = SingleDeviceMuonWithAuxAdam([
        dict(params=hidden_matrices, lr=muon_lr, momentum=0.95,
             weight_decay=0.0, use_muon=True),
        dict(params=other_params, lr=learning_rate, betas=(0.9, 0.95),
             eps=1e-8, weight_decay=0.01, use_muon=False),
    ])

    # If a previous run was aborted, pick up exactly where it left off.
    start_step = load_checkpoint(model, optimizer)

    # Show where in the data the next training batch will come from — handy
    # for confirming that a resumed run really continues where it stopped.
    print(f"train position: {train_position()}")

    # Throughput is measured over the steps since the last eval: each
    # training step consumes batch_size * block_size tokens, so tokens/s
    # = tokens consumed in the window / wall time of the window. The
    # window restarts after every eval so the pause for printing the
    # sample (and starting the checkpoint) doesn't drag the number down.
    window_start = time.time()
    window_tokens = 0
    # The losses of the steps in the current window, which is where the
    # reported loss now comes from — see the note above sample_text.
    window_loss = 0.0
    window_steps = 0

    for step in range(start_step, max_steps + 1):
        # Periodically print the loss so we can watch learning progress,
        # and save a checkpoint so an aborted run loses at most
        # eval_interval steps of work.
        if step % eval_interval == 0:
            if step > start_step:
                print()  # move off the in-place progress line (see below)
            if window_steps:
                print(f"step {step:5d} | loss {window_loss / window_steps:.4f}")
            else:
                # First eval of the run: no steps have been taken yet, so
                # there is nothing to average.
                print(f"step {step:5d} | loss pending "
                      f"(reported every {eval_interval} steps)")
            print("--- sample ---")
            print(sample_text(model, sample_prompt, sample_tokens))
            print("--------------")
            if step > 0:
                save_checkpoint(model, optimizer, step)
            # Begin the window with the GPU verified idle, so window_start is
            # a real boundary and not a moment with work still in flight.
            torch.cuda.synchronize()
            window_start = time.time()
            window_tokens = 0
            window_loss = 0.0
            window_steps = 0

        # --- one training step: the core loop of ALL deep learning ---
        x, y = get_batch()                       # 1. get a batch of examples
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = fast_model(x, y)           # 2. forward pass: predict and measure error
        optimizer.zero_grad(set_to_none=True)    # 3. clear old gradients from last step
        loss.backward()                          # 4. backward pass: compute, for every weight,
                                                 #    how it should change to reduce the loss
        optimizer.step()                         # 5. actually update the weights

        # After every batch, report the training speed and the updated
        # data position. \r rewrites one console line in place instead of
        # printing thousands of lines.
        #
        # The synchronize is what makes the number trustworthy. Everything
        # above only QUEUES work on the GPU and returns in about a
        # millisecond, so without waiting here the elapsed time would cover
        # steps the GPU has not actually finished: the first lines after
        # every eval each claimed several million tok/s, and the window
        # average stayed ~13% too high even 50 steps later. Waiting costs
        # essentially nothing — the GPU is the bottleneck, and the CPU has
        # nothing to do but queue the next step.
        torch.cuda.synchronize()
        window_tokens += batch_size * block_size
        # The GPU has just been waited on, so this .item() is a 4-byte
        # read of a result that has already landed, not a second stall.
        window_loss += loss.item()
        window_steps += 1
        tok_per_s = window_tokens / (time.time() - window_start)
        print(f"\rstep {step:5d} | {tok_per_s:,.0f} tok/s | {train_position()}", end="")

    print()  # move off the last in-place progress line
    # Let the last background checkpoint write land before exiting.
    finish_checkpoint()
    # Save the trained weights to disk so they can be loaded later.
    torch.save(model.state_dict(), "gpt2_climbmix_model.pt")
    print("saved model to gpt2_climbmix_model.pt")

    # Show off: generate 200 tokens (roughly 800 characters) from the
    # trained model, continuing from a starting prompt. Note this is a
    # base model, not a chat assistant: it wasn't trained to ANSWER
    # questions, only to CONTINUE text the way web documents continue.
    print("--- sample ---")
    print(sample_text(model, sample_prompt, max_new_tokens=200))


if __name__ == "__main__":
    main()
