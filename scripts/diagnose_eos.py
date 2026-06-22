#!/usr/bin/env python3
"""
Read-only EOS / document-boundary diagnostic for gpt_simple.

NOTHING here writes to your data, checkpoints, or config. It only reads
the tokenizer, the pretokenized .bin/.idx shards, builds in-memory batches
through the real dataloader code, and (optionally) runs forward/generate on
a checkpoint.

Two phases
----------
  data   (CPU, fast, login/prepost node OK)
         Check 1  tokenizer special-token resolution
         Check 2  EOD presence/placement inside the .bin shards
         Check 3  whether EOD is ever a *training target* (label masking)

  model  (needs a forward pass; for a 2.3B model use a GPU node)
         Check 4  p(EOS) the trained model assigns at true document ends
         Check 5  stop-reason stats over a set of prompts
                  (fraction that terminate via EOS vs. hit max_new_tokens)

Why the split: checks 1-3 alone confirm or refute the "EOD is masked out of
the loss, so the model can never learn to emit it" hypothesis, and they need
no GPU. Checks 4-5 quantify the downstream effect on the actual checkpoint.

Usage
-----
  # cheap part, anywhere:
  python scripts/diagnose_eos.py data --config path/to/config.yaml
  python scripts/diagnose_eos.py data \
      --tokenizer /path/to/tokenizer --data-root /path/to/pretokenized \
      --max-length 2048

  # model part, on a GPU node:
  python scripts/diagnose_eos.py model --run /path/to/run_output_dir
  python scripts/diagnose_eos.py model --checkpoint /path/.../checkpoint-XXXX \
      --data-root /path/to/pretokenized

Either pass --config (a training YAML/JSON) to auto-fill tokenizer / data-root
/ max-length, or pass them explicitly. Explicit flags override the config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Make `gpt_simple` importable whether or not it's pip-installed.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gpt_simple.pretokenize import read_idx  # noqa: E402
from gpt_simple.tokenizer import SimpleLLMTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _np_dtype(dtype_code: int):
    return np.uint16 if dtype_code == 2 else np.uint32


def _safe_decode(tok: SimpleLLMTokenizer, tid: Optional[int]) -> str:
    if tid is None:
        return "<None>"
    try:
        return repr(tok.decode([int(tid)], skip_special_tokens=False))
    except Exception as e:  # pragma: no cover - defensive
        return f"<decode error: {e}>"


def _list_bin_files(data_root: Path, split: str, max_per_bucket: int) -> dict:
    """Return {bucket_name: [bin_path, ...]} for a split, capped per bucket."""
    split_dir = data_root / split
    out: dict = {}
    if not split_dir.is_dir():
        return out
    for bucket_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        bins = sorted(bucket_dir.glob("*.bin"))[:max_per_bucket]
        if bins:
            out[bucket_dir.name] = bins
    return out


def _resolve_paths_from_config(cfg_path: Optional[str]):
    """Pull (tokenizer, data_root, max_length) from a training config if given."""
    if not cfg_path:
        return None, None, None
    from gpt_simple.config import Config

    cfg = Config.from_file(cfg_path)
    return cfg.data.tokenizer, cfg.data.path, cfg.data.max_length


# ---------------------------------------------------------------------------
# Check 1 — tokenizer special-token resolution
# ---------------------------------------------------------------------------

def check_tokenizer(tokenizer_path: str) -> SimpleLLMTokenizer:
    _hr("CHECK 1 — tokenizer special-token resolution")
    tok = SimpleLLMTokenizer(tokenizer_path)
    underlying = type(tok.tokenizer).__name__
    print(f"tokenizer path     : {tokenizer_path}")
    print(f"underlying class   : {underlying}")
    print(f"vocab_size         : {tok.vocab_size}")
    print(f"eos_token_id       : {tok.eos_token_id}   decode={_safe_decode(tok, tok.eos_token_id)}")
    print(f"eod_token_id       : {tok.eod_token_id}   decode={_safe_decode(tok, tok.eod_token_id)}")
    print(f"bos_token_id       : {tok.bos_token_id}   decode={_safe_decode(tok, tok.bos_token_id)}")
    print(f"pad_token_id       : {tok.pad_token_id}   decode={_safe_decode(tok, tok.pad_token_id)}")

    print("\ninterpretation:")
    if tok.eod_token_id is None:
        print("  !! eod_token_id is None — pretokenize would have no real EOD to append.")
        print("     This alone would break document-boundary learning. Investigate the")
        print("     tokenizer's special-token config first.")
    if tok.eos_token_id is not None and tok.eod_token_id == tok.eos_token_id:
        print("  - eod == eos (expected: this repo aliases eod_token_id = eos_token_id).")
    if (
        tok.pad_token_id is not None
        and tok.eod_token_id is not None
        and tok.pad_token_id == tok.eod_token_id
    ):
        print("  !! pad_token_id == eod_token_id — padding mask and EOD mask collide;")
        print("     this compounds any EOD-learning problem. Worth flagging.")
    return tok


# ---------------------------------------------------------------------------
# Check 2 — EOD presence / placement in the .bin shards
# ---------------------------------------------------------------------------

def check_eod_in_data(tok: SimpleLLMTokenizer, bin_files_by_bucket: dict) -> None:
    _hr("CHECK 2 — EOD presence and placement inside the .bin shards")
    eod = tok.eod_token_id
    if eod is None:
        print("eod_token_id is None — skipping (see Check 1).")
        return

    print(f"Looking for EOD id = {eod} in shard token streams.\n")
    grand_tokens = grand_windows = grand_eod = grand_end = 0

    for bucket, bins in bin_files_by_bucket.items():
        b_tokens = b_windows = b_eod = b_end = 0
        for bp in bins:
            idx_path = bp.with_suffix(".idx")
            if not idx_path.exists():
                print(f"  [skip] missing idx for {bp.name}")
                continue
            dtype_code, offsets, overlap_lengths = read_idx(idx_path)
            arr = np.memmap(str(bp), dtype=_np_dtype(dtype_code), mode="r")
            total = int(offsets[-1])
            stream = np.asarray(arr[:total])
            n_windows = len(overlap_lengths)

            eod_count = int((stream == eod).sum())
            last_positions = offsets[1:] - 1  # last token index of each window
            last_positions = last_positions[last_positions >= 0]
            ends_with_eod = int((stream[last_positions] == eod).sum())

            b_tokens += total
            b_windows += n_windows
            b_eod += eod_count
            b_end += ends_with_eod
            del arr

        interior = b_eod - b_end  # EOD tokens NOT at a window-final position
        frac = (b_eod / b_tokens) if b_tokens else 0.0
        print(f"  bucket {bucket!r}: {len(bins)} shard(s)")
        print(f"    tokens                 : {b_tokens:,}")
        print(f"    windows (idx entries)  : {b_windows:,}")
        print(f"    EOD tokens             : {b_eod:,}  ({frac:.6%} of tokens)")
        print(f"    windows ending in EOD  : {b_end:,}  / {b_windows:,}")
        print(f"    EOD in interior pos    : {interior:,}  (expect ~0)")
        grand_tokens += b_tokens
        grand_windows += b_windows
        grand_eod += b_eod
        grand_end += b_end

    print("\ntotals:")
    print(f"  tokens={grand_tokens:,}  windows={grand_windows:,}  "
          f"EOD={grand_eod:,}  windows_ending_in_EOD={grand_end:,}")
    print("\ninterpretation:")
    print("  - EOD should appear ~once per document; long docs split into multiple")
    print("    windows carry EOD only on their final window, so")
    print("    (windows ending in EOD) <= (windows) and ~= (number of documents).")
    if grand_eod == 0:
        print("  !! ZERO EOD tokens in the data — boundary marker never written.")
    elif grand_end < grand_windows * 0.05:
        print("  !! Very few windows end in EOD — most sequences have no terminator.")


# ---------------------------------------------------------------------------
# Check 3 — is EOD ever a training TARGET? (the smoking-gun check)
# ---------------------------------------------------------------------------

def check_label_masking(
    tok: SimpleLLMTokenizer,
    bin_files_by_bucket: dict,
    max_length: int,
    n_items: int,
    packing: bool,
) -> None:
    _hr("CHECK 3 — is EOD ever a training target? (label masking)")
    from gpt_simple.data import PreTokenizedDataset

    eod = tok.eod_token_id
    pad = tok.pad_token_id
    if eod is None:
        print("eod_token_id is None — skipping (see Check 1).")
        return

    print(f"Building real training items via PreTokenizedDataset "
          f"(max_length={max_length}, packing={packing}).")
    print("For each item we count, AFTER the model's internal label shift:")
    print("  - EOD tokens present in input_ids")
    print("  - positions whose *target* (shift_labels) is EOD and is NOT masked")
    print("    -> these are the only positions that teach the model to emit EOD.\n")

    # Pick the bucket with the most shards so packing has material to work with.
    bucket = max(bin_files_by_bucket, key=lambda b: len(bin_files_by_bucket[b]))
    bins = bin_files_by_bucket[bucket]
    print(f"Using bucket {bucket!r} ({len(bins)} shard(s)).\n")

    ds = PreTokenizedDataset(
        bin_files=bins,
        max_length=max_length,
        seed=42,
        pad_token_id=pad if pad is not None else 0,
        eod_token_id=eod,
        attention_mode="causal",
        pack_sequences=packing,
        shard_by_rank=False,
    )

    import torch

    total_input_eod = 0
    total_eod_targets_trained = 0   # shift_labels == eod AND != -100
    total_eod_targets_masked = 0    # input had EOD but its target slot is -100
    items_seen = 0

    for item in ds:
        if items_seen >= n_items:
            break
        items_seen += 1
        input_ids = item["input_ids"]
        labels = item["labels"]

        # Replicate the model's internal shift: target for prediction at pos i
        # is labels[i+1]; logits[:-1] align with labels[1:].
        shift_labels = labels[1:]

        n_eod_in_input = int((input_ids == eod).sum())
        total_input_eod += n_eod_in_input

        eod_targets = (shift_labels == eod)
        total_eod_targets_trained += int(eod_targets.sum())

        # How many EOD tokens sit at a position whose target slot was masked?
        # (i.e. EOD present in input but never appears as an un-masked target)
        masked_eod = ((input_ids[1:] == eod) & (labels[1:] == -100))
        total_eod_targets_masked += int(masked_eod.sum())

    print(f"items inspected               : {items_seen}")
    print(f"EOD tokens seen in input_ids  : {total_input_eod}")
    print(f"EOD as an UN-masked target    : {total_eod_targets_trained}   <-- key number")
    print(f"EOD present but target masked : {total_eod_targets_masked}")

    print("\ninterpretation:")
    if total_input_eod > 0 and total_eod_targets_trained == 0:
        print("  !! CONFIRMS the hypothesis: EOD appears in the inputs but is NEVER an")
        print("     un-masked target. The model receives zero gradient teaching it to")
        print("     emit EOD, so it cannot learn to terminate. (See data.py")
        print("     _finalize_sequence: labels[eod_positions] = -100.)")
    elif total_eod_targets_trained > 0:
        print(f"  - EOD IS a training target {total_eod_targets_trained} time(s); the")
        print("    masking hypothesis is NOT the whole story. Look to Checks 4-5 and")
        print("    to undertraining / decoding settings.")
    else:
        print("  - No EOD seen in inputs at all for these items — revisit Check 2 and")
        print("    whether these shards actually contain document terminators.")


# ---------------------------------------------------------------------------
# Check 4 — p(EOS) the trained model assigns at true document ends
# ---------------------------------------------------------------------------

def check_peos_at_doc_ends(
    checkpoint: str,
    data_root: Path,
    n_docs: int,
    ctx_cap: int,
) -> None:
    _hr("CHECK 4 — p(EOS) the trained model assigns at true document ends")
    import torch
    from gpt_simple.generate import load_for_inference

    model, tok, mcfg = load_for_inference(checkpoint)
    device = next(model.parameters()).device
    eod = tok.eod_token_id
    if eod is None:
        print("eod_token_id is None — skipping.")
        return
    print(f"checkpoint loaded on {device}; eod id = {eod}\n")

    bins_by_bucket = _list_bin_files(data_root, "val", max_per_bucket=2)
    if not bins_by_bucket:
        bins_by_bucket = _list_bin_files(data_root, "train", max_per_bucket=1)
    if not bins_by_bucket:
        print(f"No .bin shards found under {data_root}/(val|train). Skipping.")
        return

    p_end = []   # p(EOD) at the true end of a document (last content token -> EOD)
    p_mid = []   # p(EOD) at a random interior position (baseline)
    rank_end = []

    docs_used = 0
    for bucket, bins in bins_by_bucket.items():
        for bp in bins:
            if docs_used >= n_docs:
                break
            dtype_code, offsets, overlap_lengths = read_idx(bp.with_suffix(".idx"))
            arr = np.memmap(str(bp), dtype=_np_dtype(dtype_code), mode="r")
            n_windows = len(overlap_lengths)
            order = np.random.RandomState(0).permutation(n_windows)
            for wi in order:
                if docs_used >= n_docs:
                    break
                seg = np.asarray(arr[offsets[wi]:offsets[wi + 1]])
                if len(seg) < 8 or seg[-1] != eod:
                    continue  # only windows that truly end in EOD
                content = seg[:-1]  # strip EOD; predict it from the content
                if len(content) > ctx_cap:
                    content = content[-ctx_cap:]
                ids = torch.tensor(content[None, :].astype(np.int64), device=device)
                logits = model.forward(input_ids=ids, use_cache=False,
                                        return_dict=True)["logits"][0]
                probs = torch.softmax(logits[-1].float(), dim=-1)
                p_end.append(float(probs[eod]))
                # rank of EOD among all tokens (0 = argmax)
                rank_end.append(int((probs > probs[eod]).sum()))

                # baseline: an interior position
                if len(content) > 4:
                    mid = len(content) // 2
                    probs_mid = torch.softmax(logits[mid].float(), dim=-1)
                    p_mid.append(float(probs_mid[eod]))
                docs_used += 1
            del arr

    if not p_end:
        print("Found no windows ending in EOD to score. Revisit Check 2.")
        return

    pe = np.array(p_end)
    rk = np.array(rank_end)
    print(f"documents scored                 : {len(pe)}")
    print(f"p(EOS) at true doc end  mean     : {pe.mean():.3e}")
    print(f"                        median   : {np.median(pe):.3e}")
    print(f"                        max      : {pe.max():.3e}")
    print(f"EOS rank at doc end     median   : {int(np.median(rk))} (0 = model's top choice)")
    print(f"fraction where EOS is top-1      : {(rk == 0).mean():.2%}")
    if p_mid:
        pm = np.array(p_mid)
        print(f"p(EOS) at interior pos  mean     : {pm.mean():.3e}  (baseline)")

    print("\ninterpretation:")
    print("  - If a model has learned to stop, p(EOS) at a TRUE document end should be")
    print("    substantial (often top-1) and far above the interior baseline.")
    if pe.mean() < 1e-3 and (rk == 0).mean() < 0.01:
        print("  !! p(EOS) is ~0 even at genuine document ends and EOS is essentially")
        print("     never the top choice -> consistent with 'EOS never learned'")
        print("     (structural, matches the Check 3 masking finding) rather than mere")
        print("     undertraining (which would leave it small-but-nonzero and rising).")


# ---------------------------------------------------------------------------
# Check 5 — stop-reason statistics over prompts
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Once upon a time, in a village surrounded by dark pine forests,",
    "def fib(n):",
    "-- Select the ten most recent orders with customer names\nSELECT",
    "The Fibonacci sequence begins 1, 1, 2, 3, 5, 8, and",
    "Here is a simple recipe for chocolate chip cookies:\n\nIngredients:",
    "import numpy as np\n\n",
    "The capital of France is",
    "# Configuration\n[generation_1]\ntemperature =",
]


def check_stop_reasons(
    checkpoint: str,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: int,
    doc_mode: bool = False,
) -> None:
    label = ("complete documents (fair stop-rate probe)"
             if doc_mode else "open-ended prompts")
    _hr(f"CHECK 5 — stop-reason statistics over {label}")
    import statistics
    import torch
    from gpt_simple.generate import load_for_inference

    model, tok, mcfg = load_for_inference(checkpoint)
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    pad = tok.pad_token_id
    # temperature == 0 means greedy; pass 1.0 to generate() to avoid a
    # divide-by-zero in its temperature scaling, and disable sampling.
    do_sample = temperature > 0
    gen_temp = temperature if temperature > 0 else 1.0
    print(f"checkpoint loaded on {device}; eos id = {eos}; "
          f"max_new_tokens={max_new_tokens}, "
          f"temp={temperature}{' (greedy)' if not do_sample else ''}, "
          f"top_k={top_k}, top_p={top_p}\n")
    if seed is not None:
        torch.manual_seed(seed)

    stopped_on_eos = 0
    hit_cap = 0
    stop_offsets: List[int] = []
    for i, prompt in enumerate(prompts):
        # In doc mode the prompt IS a complete document; encode without added
        # special tokens so it matches the training distribution (raw content,
        # with EOD appended only during packing).
        ids = tok.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        plen = ids.shape[1]
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=gen_temp,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad,
            eos_token_id=eos,
        )[0]
        new_ids = out[plen:]
        new_len = int(new_ids.shape[0])
        # offset (1-based) of the first EOS in the continuation, if any
        first_eos = None
        if eos is not None:
            hits = (new_ids == int(eos)).nonzero(as_tuple=True)[0]
            if hits.numel() > 0:
                first_eos = int(hits[0]) + 1
        last_is_eos = eos is not None and int(out[-1]) == int(eos)
        if last_is_eos:
            stopped_on_eos += 1
            reason = "EOS"
            if first_eos is not None:
                stop_offsets.append(first_eos)
        else:
            hit_cap += 1
            reason = "MAX"
        snippet = (prompt[-40:] if doc_mode else prompt[:40])
        print(f"  [{i}] new_tokens={new_len:>4}  stop={reason}  "
              f"first_eos@={first_eos}  prompt…={snippet!r}")

    n = len(prompts)
    print(f"\nstopped on EOS : {stopped_on_eos}/{n}  ({stopped_on_eos / n:.0%})")
    print(f"hit max_new    : {hit_cap}/{n}  ({hit_cap / n:.0%})")
    if stop_offsets:
        print(f"tokens-to-EOS (stopped) median: {int(statistics.median(stop_offsets))}"
              f"  (smaller = stops promptly at the ending)")

    print("\ninterpretation:")
    if doc_mode:
        print("  - Prompts are COMPLETE documents, so a model that learned to terminate")
        print("    should emit EOS within the first few generated tokens (high stop rate,")
        print("    small tokens-to-EOS).")
        if stopped_on_eos == 0:
            print("  - 0% here is meaningful but read it WITH Check 4: early in training EOS")
            print("    is often only rank ~2 at true doc ends, so sampling can still miss it.")
            print("    Re-run on later checkpoints (and/or --temperature 0); the stop rate")
            print("    should climb as p(EOS) at document ends sharpens.")
    else:
        print("  - These are OPEN-ENDED prompts (document beginnings), so a low stop rate is")
        print("    EXPECTED and is NOT decisive on its own — the model legitimately has more")
        print("    to say. The authoritative signal is CHECK 4: p(EOS) at TRUE document ends")
        print("    vs the interior baseline. For a fair stop-rate test pass --docs-file with")
        print("    complete documents (optionally --temperature 0 for greedy).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)

    # shared
    def add_common(p):
        p.add_argument("--config", default=None,
                       help="training YAML/JSON to auto-fill tokenizer/data-root/max-length")
        p.add_argument("--tokenizer", default=None, help="tokenizer path (overrides config)")
        p.add_argument("--data-root", default=None,
                       help="pretokenized root (contains train/ and val/)")
        p.add_argument("--max-length", type=int, default=None)

    pdata = sub.add_parser("data", help="CPU checks 1-3 (tokenizer, EOD-in-data, label masking)")
    add_common(pdata)
    pdata.add_argument("--split", default="train", choices=["train", "val"])
    pdata.add_argument("--max-shards-per-bucket", type=int, default=1)
    pdata.add_argument("--n-items", type=int, default=200,
                       help="how many packed training items to inspect in Check 3")
    pdata.add_argument("--no-packing", action="store_true",
                       help="inspect items with packing disabled (sequential)")

    pmodel = sub.add_parser("model", help="GPU checks 4-5 (p(EOS), stop-reason stats)")
    add_common(pmodel)
    grp = pmodel.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run", help="run output_dir (latest checkpoint auto-selected)")
    grp.add_argument("--checkpoint", help="specific checkpoint dir")
    pmodel.add_argument("--n-docs", type=int, default=64, help="docs to score in Check 4")
    pmodel.add_argument("--ctx-cap", type=int, default=1024,
                        help="max context tokens fed when scoring p(EOS)")
    pmodel.add_argument("--prompts-file", default=None,
                        help="newline-separated OPEN-ENDED prompts for Check 5 (else built-in)")
    pmodel.add_argument("--docs-file", default=None,
                        help="newline-separated COMPLETE documents (one per line). Fed whole as "
                             "prompts for a fair stop-rate probe: a model that learned to "
                             "terminate should emit EOS within a few tokens. Takes precedence "
                             "over --prompts-file.")
    pmodel.add_argument("--max-new-tokens", type=int, default=256)
    pmodel.add_argument("--temperature", type=float, default=0.8)
    pmodel.add_argument("--top-k", type=int, default=None)
    pmodel.add_argument("--top-p", type=float, default=None)
    pmodel.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    cfg_tok, cfg_root, cfg_ml = _resolve_paths_from_config(args.config)
    tokenizer_path = args.tokenizer or cfg_tok
    data_root = args.data_root or cfg_root
    max_length = args.max_length or cfg_ml or 2048

    if args.phase == "data":
        if not tokenizer_path:
            ap.error("need --tokenizer or --config (with data.tokenizer)")
        tok = check_tokenizer(tokenizer_path)
        if not data_root:
            print("\n(no --data-root/--config: skipping Checks 2-3, which need shards)")
            return
        bins = _list_bin_files(Path(data_root), args.split, args.max_shards_per_bucket)
        if not bins:
            print(f"\nNo .bin shards under {data_root}/{args.split}. "
                  "Checks 2-3 skipped.")
            return
        check_eod_in_data(tok, bins)
        check_label_masking(
            tok, bins, max_length, args.n_items, packing=not args.no_packing,
        )

    elif args.phase == "model":
        checkpoint = args.checkpoint or args.run
        # A checkpoint bundles its own tokenizer; resolve it from there unless
        # the user overrode it via --tokenizer / --config.
        if not tokenizer_path:
            from gpt_simple.generate import _resolve_checkpoint, _resolve_tokenizer_dir
            ckpt_dir = _resolve_checkpoint(Path(checkpoint))
            tokenizer_path = str(_resolve_tokenizer_dir(ckpt_dir, None))
            print(f"(auto-resolved tokenizer from checkpoint: {tokenizer_path})")
        if not data_root:
            print("(no --data-root/--config: Check 4 will be skipped)")
        # Check 1 again for the record (cheap), so the model report is self-contained.
        check_tokenizer(tokenizer_path)
        if data_root:
            check_peos_at_doc_ends(checkpoint, Path(data_root), args.n_docs, args.ctx_cap)
        if args.docs_file:
            prompts = [ln.rstrip("\n") for ln in Path(args.docs_file).read_text().splitlines() if ln.strip()]
            doc_mode = True
        elif args.prompts_file:
            prompts = [ln.rstrip("\n") for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
            doc_mode = False
        else:
            prompts = _DEFAULT_PROMPTS
            doc_mode = False
        check_stop_reasons(
            checkpoint, prompts, args.max_new_tokens, args.temperature,
            args.top_k, args.top_p, args.seed, doc_mode=doc_mode,
        )


if __name__ == "__main__":
    main()
