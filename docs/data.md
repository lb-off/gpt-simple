# Data pipeline

GPT-Simple trains on causal-LM token streams. It supports two input
formats and handles tokenization, long-document windowing, sequence
packing, and (optionally) curriculum mixing.

## Two formats

| Format | What it is | Use it when |
| ------ | ---------- | ----------- |
| `pretokenized` (default) | Memory-mapped `.bin/.idx` shards, tokenized ahead of time. | Real training runs — no per-step tokenization or JSON parsing overhead, and it supports curriculum learning and deterministic resume. |
| `jsonl` | Raw `.jsonl` files (one `{"text": ...}` per line), tokenized on the fly. | Quick experiments and small datasets. No curriculum support. |

Set the format with `data.format`. See [Configuration](configuration.md)
for all data fields.

## Pretokenizing

Convert raw text into binary shards once, up front:

```bash
gpt-simple tokenize \
  --input_dir ./raw_data \
  --output_dir ./data/tokenized \
  --tokenizer_path gpt2 \
  --max_length 2048 \
  --overlap_size 256 \
  --num_workers 8
```

Inputs are `.jsonl` or `.jsonl.gz` files with a `text` field. Each output
shard is a `.bin` + `.idx` pair.

### Binary format

- **`.bin`** — a flat array of token IDs (`uint16`, or `uint32` for large
  vocabularies). Documents and windows are concatenated back-to-back,
  each terminated with an end-of-document (EOD) token.
- **`.idx`** — an index with a 16-byte header (magic `GPTS`, version,
  dtype code, document count), an `int64` offsets array (token-level
  start of each entry, plus a final total-count sentinel), and a
  per-entry `uint16` overlap-prefix length (how many leading tokens are
  windowing overlap and must be masked in the loss).

At training time the `.bin` is memory-mapped, so shards far larger than
RAM stream with constant memory.

## Document windowing

Documents longer than `max_length` are split into windows. Consecutive
windows can share an `overlap_size`-token overlap so the model still sees
local context across the cut. The overlapping prefix of each later window
is masked out of the loss (label `-100`) to avoid training on the same
tokens twice. `overlap_size` must be at most half of `max_length`.

## Sequence packing

With `packing: true`, multiple shorter documents are packed into a single
`max_length` sequence to minimize padding (length-binned greedy packing).
Within a packed sequence:

- positions are continuous (RoPE is not reset at document boundaries);
- the token after each EOD is masked in the loss;
- with `attention_mode` of `sdpa_mask` or `flex`, per-token `doc_ids`
  prevent attention across document boundaries (see
  [Architecture](architecture.md)).

## Curriculum learning

A curriculum trains through ordered phases, each mixing data buckets at
chosen ratios for a token budget (pretokenized format only). Buckets are
subdirectories of `data.path`:

```
<data.path>/
├── train/
│   ├── web/    *.bin *.idx
│   ├── code/   ...
│   └── math/   ...
└── val/
    └── ...
```

```yaml
data:
  path: ./data/tokenized
  format: pretokenized
  curriculum:
    - duration_tokens: 5_000_000_000
      mix: {web: 0.6, code: 0.2, math: 0.1, wiki: 0.1}
    - duration_tokens: 5_000_000_000
      mix: {web: 0.3, code: 0.3, math: 0.2, wiki: 0.2}
```

Bucket selection uses a counter-based PRNG so the exact mix is
reproducible and resumable.

## Bucket exhaustion

When a phase asks for more of a bucket than exists, that bucket runs dry.
What happens then is controlled by `data.allow_bucket_exhaustion`, which
expresses *intent* and is enforced at two stages:

| | `allow_bucket_exhaustion: false` (default) | `allow_bucket_exhaustion: true` |
| --- | --- | --- |
| **Validation** (`gpt-simple validate`) | A *predicted* shortfall (curriculum demand > inventory) is a blocking error. | The shortfall is a warning; the run proceeds. |
| **Runtime** (a bucket *actually* runs dry) | The trainer **halts**: it saves a checkpoint and reports status `halted`, rather than silently changing the mix. | The loader **drops** the exhausted bucket and **renormalizes** the remaining weights; training continues. |

A single bucket emptying never silently alters your mix. In a correctly
validated default run no bucket should exhaust at all, so the runtime halt
is a safety net for an estimate that was off or for resume-time drift. The
halt is coordinated across ranks and fires at the first sign of drift (the
first worker slot to drain the bucket).

To continue past a halt **with** a renormalized mix, resume with the flag
set:

```bash
gpt-simple train --config config.yaml --data.allow_bucket_exhaustion true
```

The `halted` status is terminal: the auto-resume orchestrators treat it as
a stop (they do *not* resubmit) so the run doesn't loop on the same
exhaustion — see [Orchestration](orchestration.md). Note this is the same
flag you would already need to set to pass validation for a deliberate
drain. A *budget* mismatch (curriculum total ≠ tokens the loop consumes)
is governed separately by `allow_budget_mismatch`.

## Deterministic resume

The pretokenized path resumes the data stream exactly. Each emitted item
carries a small cursor describing the dataset position after it; the
training loop commits per-worker cursors at checkpoint time and restores
them on resume. Because progress is tracked **per file** (not per
global step), a run can resume with a different `world_size` or
`num_workers` and still consume every document exactly once. Details in
[Checkpointing & resume](checkpointing-and-resume.md).

---

Authoritative source: `src/gpt_simple/pretokenize.py`,
`src/gpt_simple/data.py`.
