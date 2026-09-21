"""Tests for quilt_bpe. Plain-python accumulator (repo has no pytest deps).

Run: python tests/test_quilt_bpe.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quilt_bpe import (GENESIS, MergeLedger, canonical, fnv1a32, receipt_from_merge_log,
                       replay_vocab, sha256_hex, verify_chain)

FIXED_TS = 1000.0
PASS = 0
FAIL = 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL %s: got %r want %r" % (name, got, want))


def clock():
    return FIXED_TS


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------- primitives
check("fnv1a offset basis", "%08x" % fnv1a32(b""), "811c9dc5")
check("canonical key order", canonical({"b": 1, "a": [2, 3]}), '{"a":[2,3],"b":1}')
check("genesis", GENESIS, "0" * 8)

# ---------------------------------------------------------------- ledger basics
led = MergeLedger(actor="test", corpus_sha256="ab" * 32, char_count=64,
                  pat_str=r"\w+|\S", clock=clock)
row0 = led.bind_corpus()
check("bind op", row0["op"], "BIND")
check("bind genesis prev", row0["chain_prev"], GENESIS)
check("bind records corpus", row0["payload"]["corpus_sha256"], "ab" * 32)
check("bind names producer", row0["payload"]["producer"]["tool"], "gpu_bpe4quilt")

double_bind = led.bind_corpus()
check("double bind refused", double_bind["op"], "REFUSED")
check("double bind reason", double_bind["payload"]["reason"], "CORPUS_ALREADY_BOUND")

# ---------------------------------------------------------------- merge booking
m1 = led.book_merge(256, (b"a", b"b"), count=42)
check("merge 256 booked", m1["op"], "EFFECT")
check("merge pair hex", m1["payload"]["pair"], ["61", "62"])
check("next head chained", m1["chain_prev"], led.rows[-2]["row_hash"])  # REFUSED row sits between

gap = led.book_merge(258, (b"x", b"y"), count=1)
check("gap refused", gap["op"], "REFUSED")
check("gap reason", gap["payload"]["reason"], "MERGE_SEQUENCE_GAP")
check("gap names expected", gap["payload"]["expected"], 257)

m2 = led.book_merge(257, (b"ab", b"c"), count=7)
check("merge 257 after refusal", m2["op"], "EFFECT")
check("sequence continues", m2["payload"]["new_token_id"], 257)

# ---------------------------------------------------------------- multi-GPU divergence
div = MergeLedger(actor="mgpu", corpus_sha256="cd" * 32, char_count=128,
                  pat_str=r"\w+|\S", clock=clock)
div.bind_corpus()
d0 = div.book_merge(256, (b"a", b"b"), count=100, local_counts={0: 30, 1: 40, 2: 30})
check("divergence row effect", d0["op"], "EFFECT")
check("locals preserved verbatim", d0["payload"]["local_counts"], {"0": 30, "1": 40, "2": 30})
check("global count kept", d0["payload"]["count"], 100)

# ---------------------------------------------------------------- views
head_before = led.rows[-1]["row_hash"]
v = led.view_export("tokenizers/v300.pkl", sha256_hex(b"pickle-bytes"), note="smoke")
check("view op", v["op"], "VIEW")
check("view head captured pre-booking", v["payload"]["ledger_chain_head"], head_before)
check("view artifact", v["payload"]["artifact_sha256"], sha256_hex(b"pickle-bytes"))

# ---------------------------------------------------------------- verify + tamper
ok, bad, why = led.verify()
check("verify ok", ok, True)
victim = led.rows[1]
victim["payload"]["count"] = 9999
ok, bad, why = led.verify()
check("tamper detected", ok, False)
check("tamper pinned at row", bad, victim["row_hash"])

# canonical stability: verify does not depend on dict insertion order
led2 = MergeLedger(actor="test", corpus_sha256="ab" * 32, char_count=64,
                   pat_str=r"\w+|\S", clock=clock)
led2.bind_corpus()
led2.book_merge(256, (b"a", b"b"), count=42)
led2.book_merge(257, (b"ab", b"c"), count=7)
check("same inputs same chain", led2.rows[0]["row_hash"],
      MergeLedger(actor="test", corpus_sha256="ab" * 32, char_count=64,
                  pat_str=r"\w+|\S", clock=clock).bind_corpus()["row_hash"])

# ---------------------------------------------------------------- merge log receipts
log = [{"new_token_id": 256, "pair": (b"a", b"b"), "count": 10},
       {"new_token_id": 257, "pair": (b"ab", b"c"), "count": 4}]
rl = receipt_from_merge_log(log, actor="cli", corpus_sha256="ef" * 32, char_count=32,
                            pat_str=r"\w+|\S", clock=clock)
check("receipt rows", len(rl.rows), 3)  # BIND + 2 EFFECT
check("receipt verifies", rl.verify()[0], True)

ranks = replay_vocab(log)
check("replay base byte", ranks[b"a"], 97)
check("replay merge 256", ranks[b"ab"], 256)
check("replay merge 257", ranks[b"abc"], 257)
check("replay deterministic", replay_vocab(list(reversed(log))), ranks)

# tampered log: id 258 out of sequence → REFUSED row visible in receipt
bad_log = [{"new_token_id": 256, "pair": (b"a", b"b"), "count": 10},
           {"new_token_id": 258, "pair": (b"x", b"y"), "count": 1}]
bl = receipt_from_merge_log(bad_log, actor="cli", corpus_sha256="ef" * 32, char_count=32,
                            pat_str=r"\w+|\S", clock=clock)
refused = [r for r in bl.rows if r["op"] == "REFUSED"]
check("tampered log refuses", len(refused), 1)
check("refusal reason", refused[0]["payload"]["reason"], "MERGE_SEQUENCE_GAP")

# ---------------------------------------------------------------- export plural
ex = led2.export()
check("export jsonl rows", len(ex["jsonl"]), len(led2.rows))
check("export canon head", ex["canon"]["head"], led2.rows[-2]["row_hash"])  # head captured pre-VIEW
check("export books view", led2.rows[-1]["payload"]["kind"], "ledger-export")

# ---------------------------------------------------------------- CROSS-REPO: one recipe
laya_rows = json.load(open(os.path.join(FIXTURES, "laya4quilt_rows.json")))
ok, bad, why = verify_chain(laya_rows)
check("laya4quilt rows verify here", ok, True)
check("laya verify no bad row", bad, None)

fabric_rows = json.load(open(os.path.join(FIXTURES, "tagseq_fabric_rows.json")))
ok, bad, why = verify_chain(fabric_rows)
check("tagseq fabric rows verify here", ok, True)

# cross-repo tamper: corrupt a laya payload, expect pin at its row
laya_tampered = json.loads(canonical(laya_rows[0]))
laya_tampered["payload"]["answers"]["q1"]["choice"] = "FORGED"
laya_tampered["payload"]["usage"]["FORGED"] = True
ok, bad, why = verify_chain([laya_tampered])
check("cross-repo tamper detected", ok, False)
check("cross-repo tamper reason", why, "ROW_HASH_MISMATCH")

# envelope completeness is family-owned
broken = dict(laya_rows[0])
del broken["chain_prev"]
ok, bad, why = verify_chain([broken])
check("missing envelope field named", why, "MISSING_ENVELOPE_FIELDS:chain_prev")

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
