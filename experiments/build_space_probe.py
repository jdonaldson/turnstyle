"""
build_space_probe.py — export the SmolLM2-360M demanded-keys + guard-table probe
as a small npz artifact for the requirement-dag Hugging Face Space.

What it does (one MPS process, ~2 min):
  1. collect mean-pooled features for the 201-requirement corpus (goal_probe v2)
  2. pick the layer by wording-transfer (v0 -> v1), same rule as goal_probe
  3. report held-out (variant-2) metrics with train-on-v01 heads  -> README numbers
  4. refit ALL heads on the full corpus at that layer               -> deployment heads
  5. evaluate clause-span ARM ASSIGNMENT (which demanded key runs in which fork
     world) by re-scoring the key heads on span-pooled features — the Space needs
     this and goal_probe never probed it; spans come from closed-class splits
     (", " after "If...", ";", "otherwise") so no semantic keywords are involved
  6. write experiments/agentforce_space/probe_artifact.npz

Run:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/build_space_probe.py
"""
from __future__ import annotations
import json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goal_probe import (GOALABLE, KEYS, KEY_OF, CHECKS, COND_SURFACES,
                        build_corpus, MODELS)

MODEL_KEY = "smollm360"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentforce_space")


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[MODEL_KEY]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    return model, tok, dev


def mean_feats_all_layers(model, tok, dev, texts):
    import torch
    feats = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt").input_ids.to(dev)
            out = model(ids, output_hidden_states=True)
            feats.append(np.stack([h[0].float().mean(dim=0).cpu().numpy()
                                   for h in out.hidden_states]))
            if i % 50 == 0:
                print(f"  [{i}/{len(texts)}]", flush=True)
    return np.asarray(feats)              # [n, L+1, d]


def token_feats_at_layer(model, tok, dev, text, layer):
    """Per-token hidden states at one layer + char offsets."""
    import torch
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"].to(dev)
    offsets = enc["offset_mapping"][0].tolist()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    h = out.hidden_states[layer][0].float().cpu().numpy()   # [T, d]
    return h, offsets


def arm_char_spans(req: str):
    """Char spans of the two fork arms. Closed-class structure only:
    arm1 = between the first ',' and ';'; arm2 = after 'otherwise'."""
    lo = req.find(",") + 1
    semi = req.find(";")
    oth = req.lower().find("otherwise")
    if semi < 0 or oth < 0:
        return None
    return (lo, semi), (oth + len("otherwise"), len(req))


def span_mean(h, offsets, span):
    a, b = span
    rows = [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
    return h[rows].mean(axis=0) if rows else h.mean(axis=0)


# --- tiny numpy logistic head (mirror of the sklearn pipeline) ---------------
def fit_head(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    return {"mean": sc.mean_, "scale": sc.scale_,
            "coef": clf.coef_, "intercept": clf.intercept_,
            "classes": clf.classes_}

def apply_binary(head, x):
    z = (x - head["mean"]) / head["scale"]
    return 1.0 / (1.0 + np.exp(-(z @ head["coef"][0] + head["intercept"][0])))


def main():
    reqs = build_corpus()
    texts = [r["req"] for r in reqs]
    variant = np.array([r["variant"] for r in reqs])
    is_cond = np.array([r["guard"] is not None for r in reqs])
    y_keys = np.array([[k in r["keys"] for k in KEYS] for r in reqs])
    y_check = np.array([CHECKS.index(r["guard"]["check"]) if r["guard"] else -1 for r in reqs])
    y_pol = np.array([bool(r["guard"]["polarity"]) if r["guard"] else False for r in reqs])

    print(f"### building Space artifact — {MODELS[MODEL_KEY]}", flush=True)
    model, tok, dev = load_model()
    feats = mean_feats_all_layers(model, tok, dev, texts)
    n_layers = feats.shape[1]

    # layer by wording transfer (v0 -> v1), key heads
    v0, v1 = variant == 0, variant == 1
    scores = []
    for L in range(n_layers):
        accs = []
        for j in range(len(KEYS)):
            h = fit_head(feats[v0, L], y_keys[v0, j])
            accs.append(((apply_binary(h, feats[v1, L]) > .5) == y_keys[v1, j]).mean())
        scores.append(float(np.mean(accs)))
    best_L = int(np.argmax(scores))
    print(f"best layer L{best_L} (v0->v1 {scores[best_L]:.3f})", flush=True)

    # held-out report (train v01, eval v2) — the honest numbers for the card
    tr, te = variant < 2, variant == 2
    pred = np.zeros_like(y_keys[te])
    for j in range(len(KEYS)):
        h = fit_head(feats[tr, best_L], y_keys[tr, j])
        pred[:, j] = apply_binary(h, feats[te, best_L]) > .5
    held_set = float((pred == y_keys[te]).all(axis=1).mean())
    print(f"held-out keys set-exact (train v01): {held_set:.3f}", flush=True)

    # deployment heads: fit on ALL data at best_L
    art = {"layer": best_L, "model": MODELS[MODEL_KEY]}
    npz = {}
    for j, k in enumerate(KEYS):
        h = fit_head(feats[:, best_L], y_keys[:, j])
        for f in ("mean", "scale", "coef", "intercept"):
            npz[f"key_{k}_{f}"] = h[f]
    for name, y, rows in (("presence", is_cond, slice(None)),
                          ("polarity", y_pol[is_cond], is_cond),
                          ("check", y_check[is_cond], is_cond)):
        h = fit_head(feats[rows, best_L], y)
        for f in ("mean", "scale", "coef", "intercept"):
            npz[f"{name}_{f}"] = h[f]

    # arm assignment eval: span-pooled key-head probabilities on cond reqs
    print("evaluating clause-span arm assignment ...", flush=True)
    key_heads = {k: {f: npz[f"key_{k}_{f}"] for f in ("mean", "scale", "coef", "intercept")}
                 for k in KEYS}
    n_ok = n_tot = 0
    for r in reqs:
        if r["guard"] is None:
            continue
        spans = arm_char_spans(r["req"])
        if spans is None:
            continue
        h, offsets = token_feats_at_layer(model, tok, dev, r["req"], best_L)
        x1, x2 = span_mean(h, offsets, spans[0]), span_mean(h, offsets, spans[1])
        # gold arm membership by construction: the corpus's frozenset loses arm
        # order, so recover which goal's surface sits in arm-1 via the KNOWN
        # surface tables (build-time eval only; the APP never string-matches).
        from goal_probe import GOAL_SURFACES
        m = re.match(r"If [^,]+, (.+); otherwise (.+)\.", r["req"])
        arm1_text = m.group(1) if m else ""
        arm1_goal = None
        for a in r["goals"]:
            if any(s in arm1_text for s in GOAL_SURFACES[a]):
                arm1_goal = a
        if arm1_goal is None:
            continue
        arm2_goal = next(a for a in r["goals"] if a != arm1_goal)
        k1, k2 = KEY_OF[arm1_goal], KEY_OF[arm2_goal]
        p_k1_arm1 = apply_binary(key_heads[k1], x1[None, :])[0]
        p_k1_arm2 = apply_binary(key_heads[k1], x2[None, :])[0]
        p_k2_arm1 = apply_binary(key_heads[k2], x1[None, :])[0]
        p_k2_arm2 = apply_binary(key_heads[k2], x2[None, :])[0]
        # assign k1 to the higher-prob span, k2 to the other (2x2 argmax)
        correct = (p_k1_arm1 + p_k2_arm2) >= (p_k1_arm2 + p_k2_arm1)
        n_ok += bool(correct); n_tot += 1
    arm_acc = n_ok / max(n_tot, 1)
    print(f"arm assignment (span-pooled key heads): {arm_acc:.3f} ({n_ok}/{n_tot})", flush=True)

    meta = {"layer": best_L, "model": MODELS[MODEL_KEY],
            "keys": KEYS, "goalable": GOALABLE, "checks": CHECKS,
            "key_of": KEY_OF,
            "held_out_set_exact": held_set, "arm_assign_acc": arm_acc,
            "transfer_acc": scores[best_L]}
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "probe_artifact.npz"), **npz)
    with open(os.path.join(OUT_DIR, "probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    size = os.path.getsize(os.path.join(OUT_DIR, "probe_artifact.npz")) // 1024
    print(f"wrote {OUT_DIR}/probe_artifact.npz ({size} KB) + probe_meta.json")


if __name__ == "__main__":
    main()
