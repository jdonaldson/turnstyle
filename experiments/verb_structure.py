"""If verb space isn't K discrete clusters, what IS its geometry?

Hypothesis: like the affect frame family, verb space is a CONTINUOUS multi-dimensional
feature space organized by graded verb-semantic property axes (physicality, dynamism,
force, volition, sociality, valence). Fit each axis contrastively, measure mutual
independence (cosine) + how much verb-cloud variance the axes capture, interpret the
extremes of each, and read the PCA spectrum (a few dominant gradients vs a flat continuum).
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "experiments")

GERUNDS = sorted(set((
    "running walking jumping flying swimming climbing crawling sprinting jogging skipping "
    "marching hiking dancing leaping gliding eating drinking chewing swallowing tasting "
    "sipping gulping biting licking building writing painting drawing sculpting knitting "
    "sewing carving molding designing composing baking cooking frying roasting boiling "
    "grilling steaming breaking smashing crushing burning tearing shattering demolishing "
    "ripping cracking exploding talking shouting whispering arguing singing chatting yelling "
    "speaking explaining preaching mumbling thinking remembering learning calculating "
    "reasoning imagining understanding memorizing planning analyzing seeing hearing watching "
    "listening observing smelling staring glancing hitting kicking punching slapping stabbing "
    "poking pushing pulling grabbing throwing catching cleaning washing scrubbing mopping "
    "sweeping wiping bathing brushing polishing crying laughing smiling frowning weeping "
    "sighing sleeping breathing yawning sneezing coughing snoring buying selling trading "
    "paying lending borrowing healing nursing teaching helping guiding rescuing driving "
    "sailing rowing cycling parking").split()))

# property axes: (high pole words, low pole words)
AXES = {
    "physicality": ("lifting carrying throwing kicking hitting digging running".split(),
                    "thinking imagining remembering reasoning believing understanding".split()),
    "dynamism":    ("running jumping flying sprinting leaping racing climbing".split(),
                    "sitting resting lying sleeping standing waiting".split()),
    "force":       ("smashing crushing punching slamming pounding shattering".split(),
                    "touching patting stroking tapping brushing sipping".split()),
    "volition":    ("building planning designing writing cooking deciding".split(),
                    "sneezing coughing yawning breathing shivering blinking".split()),
    "sociality":   ("talking arguing teaching greeting chatting debating".split(),
                    "sleeping daydreaming cleaning reading staring resting".split()),
    "valence":     ("healing helping building rescuing nurturing comforting".split(),
                    "destroying stabbing burning attacking hurting smashing".split()),
}
TMPL = "the act of {w}"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def collect(mdl, tok, dev, words, layer):
    import torch
    out = []
    for w in words:
        sent = TMPL.format(w=w); cs = sent.rfind(w); ce = cs + len(w)
        enc = tok(sent, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            hs = mdl(**enc, output_hidden_states=True).hidden_states
        idxs = [k for k, (s, e) in enumerate(offs) if e > cs and s < ce] or [-1]
        out.append(hs[layer][0, idxs[-1]].float().cpu().numpy())
    return np.array(out)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    L = int(0.55 * mdl.config.num_hidden_layers)

    # all words needed (gerunds + pole words), shared standardization
    polewords = sorted({w for hi, lo in AXES.values() for w in hi + lo})
    allw = sorted(set(GERUNDS) | set(polewords))
    A = {w: v for w, v in zip(allw, collect(mdl, tok, dev, allw, L))}
    M = np.array([A[w] for w in allw]); mu, sd = M.mean(0), M.std(0) + 1e-6

    def z(words):
        return (np.array([A[w] for w in words]) - mu) / sd

    dirs = {}
    for name, (hi, lo) in AXES.items():
        d = z(hi).mean(0) - z(lo).mean(0)
        dirs[name] = d / (np.linalg.norm(d) + 1e-12)
    names = list(dirs)

    print(f"verb structure @L{L}: {len(GERUNDS)} gerunds, {len(names)} property axes\n")
    print("=== axis mutual |cos| (independent dimensions?) ===")
    print("            " + " ".join(f"{n[:7]:>7s}" for n in names))
    for a in names:
        print(f"  {a:11s} " + " ".join(f"{abs(float(dirs[a]@dirs[b])):7.2f}" for b in names))

    G = z(GERUNDS)                                  # (N,H) standardized gerunds
    B = np.array([dirs[n] for n in names])          # (k,H) axis basis
    proj = G @ B.T                                  # (N,k) gerund coords on axes
    # variance of gerund cloud captured by the k-axis subspace
    Q, _ = np.linalg.qr(B.T)
    capt = float((np.linalg.norm(G @ Q, "fro") ** 2) / (np.linalg.norm(G, "fro") ** 2))
    print(f"\nproperty-axis subspace captures {capt*100:.0f}% of gerund-cloud variance "
          f"(k={len(names)} of {len(GERUNDS)})")

    print("\n=== extremes per axis (what each dimension sorts by) ===")
    for i, n in enumerate(names):
        order = np.argsort(proj[:, i])
        lo = [GERUNDS[j] for j in order[:6]]; hi = [GERUNDS[j] for j in order[-6:]][::-1]
        print(f"  {n:11s} HIGH: {' '.join(hi)}")
        print(f"  {'':11s} LOW : {' '.join(lo)}")

    # PCA spectrum: gradient (slow decay) vs clusters (few dominant)
    lam = np.linalg.svd(G - G.mean(0), compute_uv=False) ** 2
    frac = lam / lam.sum()
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    print(f"\n=== PCA spectrum (top-10 variance fractions) ===  PR={pr:.1f}")
    print("  " + " ".join(f"{f*100:.0f}%" for f in frac[:10]))
    # which property each top PC aligns with
    _, _, Vt = np.linalg.svd(G - G.mean(0), full_matrices=False)
    print("  top PCs vs property axes (|cos|):")
    for p in range(4):
        al = {n: abs(float(Vt[p] @ dirs[n])) for n in names}
        top = max(al, key=al.get)
        print(f"    PC{p+1} ~ {top} ({al[top]:.2f})  [{' '.join(f'{n}:{al[n]:.2f}' for n in names)}]")


if __name__ == "__main__":
    main()
