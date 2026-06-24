"""Trace 10×10 multiplication tables through a tiny GPT.

Trains a 1.2M / 6L / 128d char-level GPT on tier-1 arithmetic
(+,-,* with operands 0-9) mixed with SVO filler. Then forwards
all 100 prompts "{a}*{b}=" for a,b in 0..9, grabs the =-token
hidden state at every layer, and renders a Plotly 3D trajectory
plot: Z=layer, color=product, one polyline per (a,b) pair.

Usage:
    python experiments/times_table_trace.py             # train + probe + viz
    python experiments/times_table_trace.py --no-train  # reuse checkpoint
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table"
CKPT = ROOT / "checkpoint.pt"
STATES = ROOT / "hidden_states.npz"
HTML = ROOT / "times_table_trace.html"

# ---------- vocab ----------

DIGITS = "0123456789"
OPS = "+-*="
NEG = "-"  # already in OPS
SUBJ = ["cat", "dog", "bird", "fish", "bee"]
VERB = ["sees", "likes", "finds", "hits", "chases"]
OBJ = ["man", "kid", "boy", "girl", "fox"]
ART = ["a", "the"]

CHARS = sorted(set(DIGITS + OPS + " \n" + "".join(SUBJ + VERB + OBJ + ART)))
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(CHARS)


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(ITOS[i] for i in ids)


# ---------- model ----------


@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head = C // self.n_head
        q = q.view(B, T, self.n_head, head).transpose(1, 2)
        k = k.view(B, T, self.n_head, head).transpose(1, 2)
        v = v.view(B, T, self.n_head, head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=False)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=False)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.head.weight  # tie
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()

    def forward(self, idx, targets=None, return_states: bool = False):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        states = [] if return_states else None
        for blk in self.blocks:
            x = blk(x)
            if return_states:
                states.append(x)
        x = self.ln_f(x)
        if targets is not None:
            logits = self.head(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-1)
            return logits, loss, states
        return self.head(x[:, [-1], :]), None, states


# ---------- data ----------


def arithmetic_sample(rng: random.Random) -> str:
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    op = rng.choice("+-*")
    if op == "+":
        c = a + b
    elif op == "-":
        c = a - b
    else:
        c = a * b
    return f"{a}{op}{b}={c}\n"


def svo_sample(rng: random.Random) -> str:
    return f"{rng.choice(ART)} {rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(ART)} {rng.choice(OBJ)}\n"


def build_corpus(n_samples: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        # 70% arithmetic, 30% SVO (the memory says "mixed with SVO filler")
        out.append(arithmetic_sample(rng) if rng.random() < 0.7 else svo_sample(rng))
    return "".join(out)


def make_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


# ---------- train ----------


def train(steps: int = 5000, batch_size: int = 64, lr: float = 3e-4, device: str = "cpu") -> GPT:
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M, vocab: {VOCAB_SIZE}")

    corpus = build_corpus(n_samples=20000)
    data = torch.tensor(encode(corpus), dtype=torch.long)
    print(f"corpus chars: {len(corpus):,}, tokens: {len(data):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(steps):
        x, y = make_batch(data, cfg.block_size, batch_size, device)
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # Quick eval: multiplication accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                prompt = f"{a}*{b}="
                pred = generate(model, prompt, max_new=3, device=device)
                ans = pred[len(prompt) :].split("\n", 1)[0]
                try:
                    if int(ans) == a * b:
                        correct += 1
                except ValueError:
                    pass
                total += 1
    print(f"multiplication accuracy: {correct}/{total} = {correct/total:.1%}")

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "stoi": STOI}, CKPT)
    print(f"saved checkpoint: {CKPT}")
    return model


def generate(model: GPT, prompt: str, max_new: int, device: str) -> str:
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _, _ = model(idx_cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())


def load(device: str = "cpu") -> GPT:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------- probe ----------


@torch.no_grad()
def collect_states(model: GPT, device: str = "cpu") -> dict:
    """Forward all 100 prompts a*b= and grab the =-token state at every layer."""
    model.eval()
    rows = []  # (a, b, product, layer, vec)
    for a in range(10):
        for b in range(10):
            prompt = f"{a}*{b}="
            idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            _, _, states = model(idx, return_states=True)
            eq_pos = len(prompt) - 1  # the '=' token
            for layer, h in enumerate(states):
                vec = h[0, eq_pos, :].cpu().numpy()
                rows.append((a, b, a * b, layer, vec))

    a_arr = np.array([r[0] for r in rows], dtype=np.int32)
    b_arr = np.array([r[1] for r in rows], dtype=np.int32)
    p_arr = np.array([r[2] for r in rows], dtype=np.int32)
    l_arr = np.array([r[3] for r in rows], dtype=np.int32)
    H = np.stack([r[4] for r in rows])  # (600, 128)
    np.savez(STATES, a=a_arr, b=b_arr, product=p_arr, layer=l_arr, H=H)
    print(f"saved hidden states: {STATES}  shape={H.shape}")
    return {"a": a_arr, "b": b_arr, "product": p_arr, "layer": l_arr, "H": H}


# ---------- viz ----------


def visualize(data: dict) -> None:
    import umap
    import plotly.graph_objects as go

    H = data["H"]
    print(f"UMAP fitting {H.shape} ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    Y = reducer.fit_transform(H)  # (600, 2)

    a_arr = data["a"]
    b_arr = data["b"]
    p_arr = data["product"]
    l_arr = data["layer"]
    n_layers = int(l_arr.max()) + 1

    # group by (a,b) → 6 layer-points each, in order
    fig = go.Figure()
    for a in range(10):
        for b in range(10):
            mask = (a_arr == a) & (b_arr == b)
            order = np.argsort(l_arr[mask])
            xs = Y[mask][order, 0]
            ys = Y[mask][order, 1]
            zs = l_arr[mask][order]
            product = a * b
            text = [f"{a}×{b}={product}, L{z}" for z in zs]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines+markers",
                    line=dict(
                        color=[product] * len(zs),
                        colorscale="Viridis",
                        cmin=0,
                        cmax=81,
                        width=3,
                    ),
                    marker=dict(
                        size=4,
                        color=[product] * len(zs),
                        colorscale="Viridis",
                        cmin=0,
                        cmax=81,
                        showscale=(a == 0 and b == 0),
                        colorbar=dict(title="a × b") if (a == 0 and b == 0) else None,
                    ),
                    hovertext=text,
                    hoverinfo="text",
                    showlegend=False,
                    name=f"{a}×{b}",
                )
            )

    fig.update_layout(
        title=f"Times-table trajectories through {n_layers} layers (NanoGPT 1.2M, joint UMAP)",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="Layer (=-token)",
            zaxis=dict(tickmode="linear", tick0=0, dtick=1),
        ),
        width=1100,
        height=850,
    )
    fig.write_html(HTML)
    print(f"saved viz: {HTML}")


# ---------- main ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-train", action="store_true", help="reuse checkpoint")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.no_train and CKPT.exists():
        print(f"loading checkpoint: {CKPT}")
        model = load(args.device)
    else:
        model = train(steps=args.steps, device=args.device)

    data = collect_states(model, args.device)
    visualize(data)


if __name__ == "__main__":
    main()
