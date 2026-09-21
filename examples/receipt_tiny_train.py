"""End-to-end quilt receipt over a real (tiny) BPE training run.

Trains on a synthetic corpus (no network), books every merge decision into a
quilt_bpe.MergeLedger, then proves the receipt honest two ways:
  1. verify() replays the chain clean;
  2. replay_vocab() over the merge log reproduces the trainer's own
     mergeable_ranks table exactly — the receipt is not just well-formed,
     it binds the artifact.

Run: python examples/receipt_tiny_train.py --check
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quilt_bpe import MergeLedger, receipt_from_merge_log, replay_vocab, sha256_hex, verify_chain
from train_on_CPU import bpe_train

PAT = r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks and the fox runs. "
    "pack my box with five dozen liquor jugs! "
    "how vexingly quick daft zebras jump; "
) * 40


def main():
    merge_log: list = []
    ranks = bpe_train(data=CORPUS, vocab_size=300, pat_str=PAT, merge_log=merge_log)
    corpus_bytes = CORPUS.encode("utf-8")

    print("trained: %d mergeable ranks, %d merge events" % (len(ranks), len(merge_log)))

    led = receipt_from_merge_log(
        merge_log, actor="example", corpus_sha256=sha256_hex(corpus_bytes),
        char_count=len(corpus_bytes), pat_str=PAT)
    ok, bad, why = led.verify()
    print("ledger: %d rows, verify=%s" % (len(led.rows), ok))

    replayed = replay_vocab(merge_log)
    agree = replayed == ranks
    print("replay == shipped table: %s" % agree)

    # cross-check the receipt with the generic family verifier too
    generic_ok, _, _ = verify_chain(led.rows)
    print("generic verify_chain agrees: %s" % generic_ok)

    first = merge_log[0]
    last = merge_log[-1]
    print("first merge: id=%d pair=%s+%s count=%d" % (
        first["new_token_id"], first["pair"][0], first["pair"][1], first["count"]))
    print("last merge:  id=%d pair=%s+%s count=%d" % (
        last["new_token_id"], last["pair"][0], last["pair"][1], last["count"]))

    if "--check" in sys.argv:
        if not (ok and agree and generic_ok):
            print("CHECK FAILED")
            return 1
        print("CHECK OK: %d merges receipted, chain verifies, replay reproduces the table" % len(merge_log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
