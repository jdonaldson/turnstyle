"""Self-directed sparsity probing: give the model a topic, let it propose nearby
concepts, and use its OWN internal state to flag which ones land in SPARSE regions
of its knowledge manifold (= poor info).

Loop:  seed topic --> model proposes related concepts (common -> obscure)
       --> each proposal's request-encoding scored for sparsity vs a reference
           manifold of ~100 common concepts (dyf leaf-OOD + flat kNN-distance)
       --> rank; high-sparsity = "it sees this as sparse / poorly known"

Validation (judge-free): do sparse-flagged concepts behave like poor-knowledge ones?
  - hedge rate: "not sure / not familiar / I believe / might be ..." in a plain answer
  - inconsistency: dispersion of K resampled answers (MiniLM embeddings)
A calibrated model may meet thin knowledge with confabulation (high inconsistency) OR
abstention (high hedge, low inconsistency); either confirms a real knowledge gap. Only
sparse==dense behavior falsifies the signal.

  .venv/bin/python experiments/introspect_sparse.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct --layer 15
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from introspect_loop import IntrospectiveModel, IntrospectTool

REFERENCE_CONCEPTS = [
    # science
    "gravity", "photosynthesis", "DNA", "evolution", "atoms", "electricity",
    "the water cycle", "plate tectonics", "the immune system", "black holes",
    "vaccines", "the periodic table", "ecosystems", "magnetism", "the solar system",
    # history
    "World War II", "the Roman Empire", "the French Revolution", "the Cold War",
    "ancient Egypt", "the Renaissance", "the Industrial Revolution",
    "the American Civil War", "the printing press", "feudalism",
    # geography
    "the Amazon rainforest", "Mount Everest", "the Sahara desert", "the Nile river",
    "continents", "oceans", "climate zones", "volcanoes", "capital cities", "time zones",
    # arts
    "jazz music", "Renaissance painting", "Shakespeare", "classical music",
    "photography", "ballet", "the novel", "impressionism", "film", "poetry",
    # tech
    "the internet", "computers", "smartphones", "artificial intelligence",
    "electric cars", "social media", "encryption", "databases", "programming", "robotics",
    # everyday
    "cooking", "gardening", "exercise", "sleep", "nutrition", "money", "weather",
    "traffic", "recycling", "coffee",
    # math
    "algebra", "geometry", "statistics", "fractions", "probability", "calculus",
    "prime numbers", "logarithms",
    # body
    "the heart", "the brain", "lungs", "muscles", "bones", "digestion", "vision", "hearing",
]

SEEDS = ["machine learning", "the French Revolution", "human anatomy",
         "jazz music", "quantum physics", "ancient Rome"]

HEDGES = ("not sure", "i'm not sure", "not familiar", "i believe", "i think",
          "might be", "may be", "possibly", "perhaps", "i don't have", "as far as i know",
          "i'm not certain", "could be", "not entirely", "to my knowledge")


class SparseProbe:
    def __init__(self, model_id: str, layer: int):
        self.m = IntrospectiveModel(model_id, layer=layer)

    @torch.no_grad()
    def encode(self, concept: str) -> np.ndarray:
        # neutral framing, NO introspection system prompt; reference & query match
        ids = self.m._chat_ids([{"role": "user", "content": f"Tell me about {concept}."}])
        out = self.m.mdl(ids, output_hidden_states=True)
        return out.hidden_states[self.m.layer][0, -1].float().cpu().numpy()

    @torch.no_grad()
    def propose(self, seed: str, n: int = 12) -> list[str]:
        prompt = (f"List {n} specific concepts or subtopics related to {seed}, "
                  f"ranging from very common to highly obscure and specialized. "
                  f"Give only the names, one per line, no explanations.")
        ids = self.m._chat_ids([{"role": "user", "content": prompt}])
        out = self.m.mdl.generate(ids, max_new_tokens=220, do_sample=False,
                                  pad_token_id=self.m.tok.eos_token_id)
        txt = self.m.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        items = []
        for line in txt.splitlines():
            s = line.strip().lstrip("0123456789.-) *#").strip()
            if 2 < len(s) < 60 and s.lower() != seed.lower():
                items.append(s)
        return items[:n]

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled last-layer hidden state per text (self-contained, no extra dep)."""
        vecs = []
        for t in texts:
            ids = self.m.tok(t, return_tensors="pt", truncation=True,
                             max_length=64).input_ids.to(self.m.device)
            h = self.m.mdl(ids, output_hidden_states=True).hidden_states[-1][0]
            vecs.append(h.float().mean(0).cpu().numpy())
        return np.asarray(vecs)

    @torch.no_grad()
    def answer_samples(self, concept: str, k: int = 4):
        ids = self.m._chat_ids([{"role": "user",
                                 "content": f"What is {concept}? Answer in one sentence."}])
        outs = []
        for _ in range(k):
            o = self.m.mdl.generate(ids, max_new_tokens=45, do_sample=True,
                                    temperature=0.9, top_p=0.95,
                                    pad_token_id=self.m.tok.eos_token_id)
            outs.append(self.m.tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True))
        return outs


def run(model_id: str, layer: int):
    p = SparseProbe(model_id, layer)
    print(f"=== {model_id} layer={p.m.layer} :: self-directed sparsity probing ===")

    # build reference manifold
    Ref = np.stack([p.encode(c) for c in REFERENCE_CONCEPTS]).astype(np.float32)
    tool = IntrospectTool(layer=p.m.layer).fit(Ref, ["ref"] * len(Ref))

    def knn_dist(x, k=8):
        d = np.linalg.norm(Ref - x, axis=1)
        return float(np.sort(d)[:k].mean())

    scored = []  # (seed, concept, dyf_ood, knn_d)
    for seed in SEEDS:
        props = list(dict.fromkeys(p.propose(seed)))  # dedupe, keep order
        rows = []
        for c in props:
            x = p.encode(c).astype(np.float32)
            rows.append((c, tool.route(x).ood_score, knn_dist(x)))
        rows.sort(key=lambda r: -r[2])  # rank by kNN-distance (ood saturates)
        print(f"\n--- seed: {seed}  ({len(rows)} proposed, ranked by sparsity) ---")
        for c, ood, kd in rows:
            print(f"   knn_d={kd:5.1f}  ood={ood:.2f}   {c}")
        scored.extend((seed, c, ood, kd) for c, ood, kd in rows)

    # validation: top vs bottom sparsity quartile by kNN-distance
    scored.sort(key=lambda r: r[3])
    q = max(4, len(scored) // 4)
    dense, sparse = scored[:q], scored[-q:]
    print(f"\n=== VALIDATION: sparse (n={len(sparse)}) vs dense (n={len(dense)}) ===")
    embedder = p.embed_texts

    def behavior(group):
        out = []  # (knn_d, hedge, disp)
        for _, c, _, kd in group:
            ans = p.answer_samples(c, k=4)
            hedge = np.mean([any(h in a.lower() for h in HEDGES) for a in ans])
            E = embedder(ans)
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            iu = np.triu_indices(len(ans), 1)
            disp = float(1 - (E @ E.T)[iu].mean())
            out.append((kd, float(hedge), disp))
        return out

    bs, bd = behavior(sparse), behavior(dense)
    h_s, d_s = np.mean([x[1] for x in bs]), np.mean([x[2] for x in bs])
    h_d, d_d = np.mean([x[1] for x in bd]), np.mean([x[2] for x in bd])
    print(f"  hedge rate:        sparse={h_s:.2f}  dense={h_d:.2f}  (Δ={h_s-h_d:+.2f})")
    print(f"  answer dispersion: sparse={d_s:.2f}  dense={d_d:.2f}  (Δ={d_s-d_d:+.2f})")
    allb = bs + bd
    kd = [x[0] for x in allb]
    print(f"  Spearman(sparsity, hedge)      = {_spearman(kd, [x[1] for x in allb]):+.2f}")
    print(f"  Spearman(sparsity, dispersion) = {_spearman(kd, [x[2] for x in allb]):+.2f}")
    print("  (validates if sparsity predicts hedging and/or inconsistency > 0)")


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--layer", type=int, default=15)
    args = ap.parse_args()
    run(args.model, args.layer)
