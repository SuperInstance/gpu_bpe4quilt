"""Quilt-native receipts for BPE tokenizer training.

Every merge a tokenizer trainer commits is a *decision*: given global pair
statistics, admit (pair -> new token id) at this rank or refuse. This module
books those decisions as hash-chained receipts in the quilt 5-opcode family
envelope, so a tokenizer's provenance is verifiable by any quilt substrate:
BIND the corpus, EFFECT per merge (multi-GPU divergence preserved verbatim),
VIEW exports, and named REFUSED rows when a merge log lies.

Family doctrine (shared with laya4quilt / tagseq2tagseq4quilt):
- chain: fnv1a-32 over canonical JSON, genesis "0"*8 — integrity, not
  security; the algorithm is swappable, checker plurality is the guarantee.
- one row shape: tick/ts/op/actor/engine/payload/chain_prev/row_hash.
  The envelope is family-owned; the payload schema is producer-owned.
- refusals are named and booked, never silent.
- a view describes the ledger as observed: the chain head is captured
  *before* the VIEW row is booked.
- preservation is plural: exports come in jsonl + canon.

The generic ``verify_chain`` checks ANY family's rows (payloads
uninterpreted — annals doctrine) — that is the cross-repo contract.
"""
from __future__ import annotations

import hashlib
import json
import time

FNV1A_OFFSET = 0x811C9DC5
FNV1A_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF
GENESIS = "0" * 8
REQUIRED_ENVELOPE = ("tick", "ts", "op", "actor", "payload", "chain_prev", "row_hash")
# Family doctrine, after the drift this checker exposed: the envelope core is
# family-owned (7 fields above); producers MAY add context fields — laya4quilt
# carries `engine` (who served the decision), tagseq2tagseq4quilt carries
# `fabric_state` (state hash at booking) — and the row hash binds every field
# present, so producer context is just as tamper-evident as the core.
# (annals doctrine: storage owns facts about records; producers own vocabularies.)

PRODUCER = {"tool": "gpu_bpe4quilt", "vocabulary": "merge-ledger/v1"}


def fnv1a32(data: bytes) -> int:
    h = FNV1A_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV1A_PRIME) & MASK32
    return h


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_hash(row: dict) -> str:
    return "%08x" % fnv1a32(canonical(row).encode("utf-8"))


def verify_chain(rows: list) -> tuple:
    """Generic family verifier: replay any quilt ledger's rows.

    Payloads and producer fields are never interpreted (annals doctrine — the
    storage layer owns facts about records, producers own vocabularies).
    Checks core-envelope completeness, canonical-JSON stability, chain
    linkage, and row hashes over every field present.
    Returns (ok, bad_row_hash_or_None, reason_or_None).
    """
    prev = GENESIS
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED_ENVELOPE if k not in row]
        if missing:
            return False, row.get("row_hash"), "MISSING_ENVELOPE_FIELDS:%s" % ",".join(missing)
        try:
            if not isinstance(json.loads(canonical(row)), dict):
                return False, row.get("row_hash"), "NOT_CANONICAL_JSON"
        except (TypeError, ValueError):
            return False, row.get("row_hash"), "NOT_CANONICAL_JSON"
        if row["chain_prev"] != prev:
            return False, row.get("row_hash"), "CHAIN_BREAK_AT_ROW_%d" % i
        body = {k: v for k, v in row.items() if k != "row_hash"}  # every field present binds
        if row["row_hash"] != _row_hash(body):
            return False, row.get("row_hash"), "ROW_HASH_MISMATCH"
        prev = row["row_hash"]
    return True, None, None


class MergeLedger:
    """Hash-chained decision ledger for one BPE training run.

    A merge log that skips an id, duplicates one, or books before a corpus
    BIND gets named REFUSED rows — a fabricated "tokenizer trained on X"
    claim must leave visible gaps, not silent ones.
    """

    def __init__(self, actor: str, corpus_sha256: str, char_count: int,
                 pat_str: str, clock=None, engine: str = "bpe-trainer"):
        self.actor = actor
        self.clock = clock if clock is not None else time.time
        self.corpus = {"kind": "corpus", "producer": PRODUCER,
                       "corpus_sha256": corpus_sha256, "char_count": char_count,
                       "pat_str": pat_str, "base_vocab": 256}
        self.engine = engine
        self.rows: list = []
        self._next_id = 256
        self._bound = False

    def _book(self, op: str, payload: dict) -> dict:
        ts = float(self.clock())
        row = {"tick": len(self.rows) + 1, "ts": ts, "op": op, "actor": self.actor,
               "engine": self.engine, "payload": payload, "chain_prev": self._head()}
        row["row_hash"] = _row_hash(row)
        self.rows.append(row)
        return row

    def _head(self) -> str:
        return self.rows[-1]["row_hash"] if self.rows else GENESIS

    def bind_corpus(self) -> dict:
        if self._bound:
            return self._book("REFUSED", {"kind": "bind-corpus", "reason": "CORPUS_ALREADY_BOUND",
                                          "producer": PRODUCER})
        self._bound = True
        return self._book("BIND", self.corpus)

    def book_merge(self, new_token_id: int, pair: tuple, count: int,
                   stats_top: dict | None = None, local_counts: dict | None = None) -> dict:
        if not self._bound:
            return self._book("REFUSED", {"kind": "merge", "new_token_id": new_token_id,
                                          "reason": "NO_CORPUS_BIND", "producer": PRODUCER})
        if new_token_id != self._next_id:
            return self._book("REFUSED", {"kind": "merge", "new_token_id": new_token_id,
                                          "expected": self._next_id,
                                          "reason": "MERGE_SEQUENCE_GAP", "producer": PRODUCER})
        payload = {"kind": "merge", "producer": PRODUCER, "new_token_id": new_token_id,
                   "pair": [list(pair)[0].hex() if isinstance(pair[0], bytes) else pair[0],
                            list(pair)[1].hex() if isinstance(pair[1], bytes) else pair[1]],
                   "count": int(count)}
        if stats_top:
            payload["stats_top"] = stats_top
        if local_counts:
            # Multi-GPU: every rank's local count for the chosen pair is
            # preserved verbatim — disagreement is data, cited not deleted.
            payload["local_counts"] = {str(k): int(v) for k, v in sorted(local_counts.items())}
        self._next_id += 1
        return self._book("EFFECT", payload)

    def view_export(self, artifact_path: str, artifact_sha256: str, note: str = "") -> dict:
        head = self._head()
        payload = {"kind": "export", "producer": PRODUCER, "artifact": artifact_path,
                   "artifact_sha256": artifact_sha256, "note": note,
                   "merges_booked": self._next_id - 256,
                   "ledger_chain_head": head}
        return self._book("VIEW", payload)

    def verify(self) -> tuple:
        return verify_chain(self.rows)

    def export(self) -> dict:
        head = self._head()
        self._book("VIEW", {"kind": "ledger-export", "producer": PRODUCER,
                            "rows": len(self.rows), "ledger_chain_head": head})
        return {"jsonl": [canonical(r) for r in self.rows],
                "canon": {"head": head, "rows": len(self.rows),
                          "merges": self._next_id - 256}}


def receipt_from_merge_log(merge_events: list, actor: str, corpus_sha256: str,
                           char_count: int, pat_str: str, clock=None,
                           local_counts_by_step: dict | None = None) -> MergeLedger:
    """Convert a training script's merge log into a verified ledger.

    ``merge_events`` items: dicts with new_token_id, pair (bytes or hex
    strings), count. ``local_counts_by_step`` optionally maps
    new_token_id -> {rank: local_count} for multi-GPU divergence rows.
    Gaps and duplicates become REFUSED rows instead of raising — the receipt
    records the lie rather than dying before writing it down.
    """
    led = MergeLedger(actor, corpus_sha256, char_count, pat_str, clock=clock)
    led.bind_corpus()
    for ev in merge_events:
        locals_for_step = (local_counts_by_step or {}).get(ev["new_token_id"])
        led.book_merge(ev["new_token_id"], ev["pair"], ev["count"],
                       local_counts=locals_for_step)
    return led


def replay_vocab(merge_events: list) -> dict:
    """Derive mergeable_ranks from a merge log — derived state derived.

    The replayed table must reproduce the trainer's final vocabulary; a
    receipt + replay that disagrees with the shipped tokenizer is a broken
    claim, and the chain says whose.
    """
    ranks = {bytes([i]): i for i in range(256)}
    for ev in sorted(merge_events, key=lambda e: e["new_token_id"]):
        a, b = ev["pair"]
        if isinstance(a, str):
            a, b = bytes.fromhex(a), bytes.fromhex(b)
        ranks[a + b] = ev["new_token_id"]
    return ranks
