"""
frame_decompose.py — read attention deliveries in SEMANTIC coordinates.
First application: is Mistral's L30 H18 (the top weekday->end deliverer,
whose OV top-1 for Monday was 'Tuesday') transporting the weekday IDENTITY,
the weekday ANGLE, or the computed SUCCESSOR?

For each of the 84 "They met on a {wd} in {mo}." sentences, capture the
head's residual contribution (delta) at the final token. Then:

  1. OV SUCCESSOR COUNT (vocab space, absolute): delta @ W_U -> among the 7
     weekday tokens, is top-1 the sentence's weekday (identity), its
     successor, predecessor, or other? Counted over all 84.
  2. ANGLE ORDERING (ring plane): fit the weekday cos/sin plane on the
     END-TOKEN STATES at the landing layer (hs[L+1], where the weekday
     marginal is hollow); project per-weekday MEAN deltas -> 7 angles.
     Alignment resultant R vs day order (up to rotation/reflection):
     high R = the head transports the COORDINATE.
  3. PHASE-ANCHORED SHIFT: the plane's phase is fixed by the states
     themselves (psi(wd) = angle of the mean end-token state). The circular
     mean of phi(wd) - psi(wd), in units of days: ~0 = identity transport,
     ~+1 day = successor, ~-1 = predecessor. (An unanchored plane cannot
     distinguish these — identity and successor differ by a global
     rotation.)
  4. MONTH-PLANE SPECIFICITY control: weekday deliveries should not align
     with month angles.
  5. SCALAR FRAME TABLE: project deltas onto the bundled FrameLibrary
     directions (delta/scale @ direction, in coord_scale sigma units).
     CAVEAT printed: frames are fit at their own best layers; projecting an
     L30 delta onto a mid-stack direction is an approximation.

Contrast head: L24 H5 (the weekday-CLASS deliverer — OV writes the whole
weekday set) gets the same battery; prediction = weaker day-specific angle
structure than L30 H18.

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python \
        experiments/frame_decompose.py
Writes experiments/results/frame_decompose.json.
"""
from __future__ import annotations
import json, os

import numpy as np

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
      "Sunday"]
MO = ["January", "February", "March", "April", "May", "June", "July",
      "August", "September", "October", "November", "December"]
TEMPLATE = "They met on a {wd} in {mo}."
HEADS = [(30, 18, "top wd deliverer"), (24, 5, "wd-CLASS head (contrast)")]


def ridge_plane(X, Y, lam_scale=1.0):
    """Orthonormal readout plane (cols of Q) for X->Y in standardized space."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    K = Xc @ Xc.T
    lam = lam_scale * np.trace(K) / n + 1e-6
    A = np.linalg.solve(K + lam * np.eye(n), Y - Y.mean(0))
    W = Xc.T @ A
    Q, _ = np.linalg.qr(W)
    return Q


def ang(v2):
    return float(np.arctan2(v2[1], v2[0]))


def circ_mean(angles):
    z = np.mean(np.exp(1j * np.asarray(angles)))
    return float(np.angle(z)), float(np.abs(z))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.frame_library import load_library
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16).to(dev)
    d_model = mdl.config.hidden_size
    n_heads = mdl.config.num_attention_heads
    d_head = d_model // n_heads
    W_U = mdl.lm_head.weight.detach().float().cpu().numpy()
    lib = load_library(mdl)
    print(f"frame library: {sorted(lib.frames) if lib else 'MISSING'}",
          flush=True)

    wd_tok = [tok.encode(" " + w, add_special_tokens=False)[0] for w in WD]

    layers_needed = sorted({L for L, _, _ in HEADS})
    o_store = {}
    hooks = [mdl.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
        lambda m, inp, L=L: o_store.__setitem__(L, inp[0].detach().float()))
        for L in layers_needed]

    # ---- collect: deltas per head + landing-layer end-token states -------
    pairs = [(a, b) for a in WD for b in MO]
    deltas = {(L, h): [] for L, h, _ in HEADS}
    states = {L + 1: [] for L, _, _ in HEADS}
    W_O = {L: mdl.model.layers[L].self_attn.o_proj.weight
           .detach().float().cpu().numpy() for L in layers_needed}
    print(f"collecting {len(pairs)} sentences...", flush=True)
    for a, b in pairs:
        enc = tok(TEMPLATE.format(wd=a, mo=b), return_tensors="pt")
        o_store.clear()
        with torch.no_grad():
            out = mdl(**{k: v.to(dev) for k, v in enc.items()},
                      output_hidden_states=True)
        for L, h, _ in HEADS:
            oin = o_store[L][0, -1].cpu().numpy().reshape(n_heads, d_head)
            deltas[(L, h)].append(
                oin[h] @ W_O[L][:, h * d_head:(h + 1) * d_head].T)
        for Ls in states:
            states[Ls].append(
                out.hidden_states[Ls][0, -1].float().cpu().numpy())
    for hk in hooks:
        hk.remove()

    wd_idx = np.array([WD.index(a) for a, b in pairs])
    th_wd = 2 * np.pi * wd_idx / 7
    th_mo = 2 * np.pi * np.array([MO.index(b) for a, b in pairs]) / 12
    results: dict = {"model": MODEL, "heads": {}}

    for L, h, label in HEADS:
        print(f"\n================ L{L} H{h} ({label}) ================")
        X = np.array(states[L + 1], dtype=np.float32)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xz = (X - mu) / sd
        D = np.array(deltas[(L, h)], dtype=np.float32)
        Dz = D / sd                                    # delta: scale only

        # 1. OV successor count
        counts = {"identity": 0, "successor": 0, "predecessor": 0, "other": 0}
        for i, d in enumerate(D):
            lg = d @ W_U.T
            top = int(np.argmax(lg[wd_tok]))
            diff = (top - wd_idx[i]) % 7
            counts["identity" if diff == 0 else
                   "successor" if diff == 1 else
                   "predecessor" if diff == 6 else "other"] += 1
        print(f"OV weekday-token top-1 over 84: {counts}")

        # planes at the landing layer
        Q_wd = ridge_plane(Xz, np.stack([np.cos(th_wd), np.sin(th_wd)], 1))
        Q_mo = ridge_plane(Xz, np.stack([np.cos(th_mo), np.sin(th_mo)], 1))

        # per-weekday mean states and mean deltas -> angles
        phi, psi, expl = [], [], []
        for wdi in range(7):
            m = wd_idx == wdi
            dbar = Dz[m].mean(0)
            sbar = (Xz[m] - Xz.mean(0)).mean(0)
            p = dbar @ Q_wd
            phi.append(ang(p)); psi.append(ang(sbar @ Q_wd))
            expl.append(float((p ** 2).sum() / (dbar ** 2).sum()))
        phi, psi = np.array(phi), np.array(psi)
        th7 = 2 * np.pi * np.arange(7) / 7

        # 2. ordering (up to rotation/reflection)
        _, R_fwd = circ_mean(phi - th7)
        _, R_rev = circ_mean(phi + th7)
        # 3. phase-anchored shift vs the states' own angles
        shift, R_shift = circ_mean(phi - psi)
        shift_days = shift / (2 * np.pi / 7)
        # 4. month specificity
        phim = []
        for moi in range(12):
            m = np.array([MO.index(b) for a, b in pairs]) == moi
            phim.append(ang(Dz[m].mean(0) @ Q_mo))
        _, Rm_fwd = circ_mean(np.array(phim) - 2 * np.pi * np.arange(12) / 12)
        _, Rm_rev = circ_mean(np.array(phim) + 2 * np.pi * np.arange(12) / 12)

        print(f"angle ordering:   R_fwd={R_fwd:.2f} R_rev={R_rev:.2f} "
              f"(1.0 = perfectly ordered landing)")
        print(f"phase shift vs states: {shift_days:+.2f} days "
              f"(R={R_shift:.2f})  [0=identity, +1=successor]")
        print(f"in-plane explained fraction of delta: "
              f"mean {np.mean(expl):.2%}")
        print(f"month-plane control: R_fwd={Rm_fwd:.2f} R_rev={Rm_rev:.2f}")

        # 5. scalar frames
        frame_rows = []
        if lib:
            for name, fr in sorted(lib.frames.items()):
                axd = fr.axis
                # delta projection: scale-standardize (no mean shift — this
                # is a difference vector), dot the direction, sigma units
                proj = (D / axd.scale) @ axd.direction / (fr.coord_scale or 1.0)
                frame_rows.append((name, float(np.abs(proj).mean()),
                                   int(axd.layer)))
            frame_rows.sort(key=lambda r: -r[1])
            top = ", ".join(f"{n}({v:.2f}σ, fit@L{fl})"
                            for n, v, fl in frame_rows[:5])
            print(f"scalar frames (mean |Δσ|, top 5): {top}")

        results["heads"][f"L{L}H{h}"] = {
            "label": label, "ov_counts": counts,
            "R_fwd": round(R_fwd, 2), "R_rev": round(R_rev, 2),
            "shift_days": round(shift_days, 2), "R_shift": round(R_shift, 2),
            "expl_frac": round(float(np.mean(expl)), 4),
            "month_R": [round(Rm_fwd, 2), round(Rm_rev, 2)],
            "phi": [round(float(x), 2) for x in phi],
            "psi": [round(float(x), 2) for x in psi],
            "frames_top": [(n, round(v, 3), fl)
                           for n, v, fl in frame_rows[:8]]}

    print("\nCAVEAT: scalar frames are fit at their own best layers "
          "(shown); projecting L24/L30 deltas onto them is approximate.")
    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "frame_decompose.json")
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
