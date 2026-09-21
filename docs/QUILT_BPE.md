# Quilt BPE: tokenizer training as receipted decisions

Adapts gpu_bpe for the quilt ecosystem. Training a BPE tokenizer *is* a
decision process: at each rank, given global pair statistics, the trainer
admits (pair → new token id) or stops. This module books those decisions as
hash-chained receipts so a tokenizer's provenance — "trained on corpus X,
with merges admitted in exactly this order, under these statistics" — is
verifiable by any quilt substrate, and binds to the artifact it produced.

## The mapping (5-opcode family)

| BPE concept | quilt concept |
|---|---|
| corpus snapshot (bytes, pretokenizer regex) | **BIND** — `bind_corpus()` |
| each merge decision | **EFFECT** — pair, count, new token id, rank |
| multi-GPU per-rank statistics | **EFFECT** with `local_counts` preserved verbatim — disagreement is data, cited not deleted |
| exported vocab (`.pkl`) | **VIEW** — artifact path + sha-256 under the ledger head |
| merge-log gap / duplicate id / unbound corpus | **REFUSED** — named, booked, visible |
| merge log → final vocab | derived state derived — `replay_vocab()` reproduces `mergeable_ranks` |

## One chain across the 4quilt family

Every ledger in the family shares the recipe: canonical JSON
(`sort_keys`, tight separators) → fnv1a-32 row hashes → chain from genesis
`"0"*8`. The **envelope core** is family-owned — `tick, ts, op, actor,
payload, chain_prev, row_hash` — and producers add context fields freely
(laya4quilt carries `engine`, tagseq2tagseq4quilt carries `fabric_state`);
the row hash binds *every field present*, so producer context is exactly as
tamper-evident as the core. (Drift this checker exposed and doctrine now
codifies, per annals: storage owns facts about records; producers own
vocabularies.)

`quilt_bpe.verify_chain(rows)` verifies **any** family's rows without
knowing what they mean. The test suite freezes real rows from both sibling
repos as fixtures and checks them here unmodified — that is the cross-repo
contract, exercised in CI, not just claimed in a doc.

## Usage

```python
from quilt_bpe import MergeLedger, receipt_from_merge_log, replay_vocab, verify_chain

# from a training run (train_on_CPU.bpe_train accepts merge_log=[...]):
led = receipt_from_merge_log(merge_log, actor="run-7",
                             corpus_sha256=..., char_count=..., pat_str=...)
ok, bad_row, why = led.verify()
table = replay_vocab(merge_log)          # must equal the trainer's ranks
assert table == trainer_mergeable_ranks  # or the receipt is lying
```

```python
# checking someone else's ledger (any family):
ok, bad, why = verify_chain(their_rows)
```

```bash
# end-to-end on a real (tiny) training run, no network:
python examples/receipt_tiny_train.py --check
python tests/test_quilt_bpe.py          # 44 checks, no pytest required
```

## What this guards

- **"Trained on Fineweb" claims**: the BIND row binds the corpus hash; a
  swapped corpus changes the chain's first hash and every hash after it.
- **Reproducibility**: same merge log + `replay_vocab` = same tokenizer.
  A `.pkl` that disagrees with its receipt is a broken claim, and the chain
  says whose.
- **Multi-GPU divergence**: per-rank counts preserved verbatim at each
  decision — you can audit *which GPU disagreed* about a merge, after the
  fact, from the receipt alone.
- **Log tampering**: a merge log with a gap or duplicated id books REFUSED
  rows instead of crashing — the receipt records the lie rather than dying
  before writing it down.

## Not yet wired (honest scope)

- `train_on_GPU.py` / `train_on_many_GPUs.py` do not yet emit merge logs
  (no GPU on the node where this was built). The CPU path is instrumented
  and the receipt API accepts logs from any script — the GPU scripts need
  the same ~6-line hook, verified on hardware with CUDA.
