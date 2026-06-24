"""Joint int+decimal multiplication trace through a tiny GPT.

Trains a 1.2M / 6L / 128d char-level GPT on a mix of:
  - integer arithmetic:  "{a}{op}{b}={c}"      a,b in 0..9
  - decimal arithmetic:  "0.{a}*0.{b}=0.{cc}"  a,b in 0..9, cc 2-digit
  - SVO filler

Then forwards all 100 integer + 100 decimal multiplication prompts
and grabs the =-token hidden state at every layer. Joint UMAP →
Plotly 3D plot: Z=layer, color=digit-product (so 3*7 and 0.3*0.7
share a color), marker shape int=circle / dec=diamond, line dash
int=solid / dec=dash. The question: do analogous (int, dec) pairs
share trajectories (shared circuit) or do they sit apart?

Probe-point note: we probe the =-token state for both prompt types.
For the integer prompt "3*7=" that state is committed to "21". For
"0.3*0.7=" the next token is trivially "0", but the model still
needs to encode the product to autoregressively continue ".21", so
the deep-layer state should encode it. If the decimal trajectories
look uninformative, switch to the post-"0." probe (`=0.` suffix).
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

ROOT = Path(__file__).parent / "data" / "nanogpt_times_table_floats"
CKPT = ROOT / "checkpoint.pt"
STATES = ROOT / "hidden_states.npz"
HTML = ROOT / "times_table_trace.html"

# ---------- vocab ----------

DIGITS = "0123456789"
OPS = "+-*=."
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
        self.wte.weight = self.head.weight
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


def int_arith(rng: random.Random) -> str:
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    op = rng.choice("+-*")
    if op == "+":
        c = a + b
    elif op == "-":
        c = a - b
    else:
        c = a * b
    return f"{a}{op}{b}={c}\n"


def dec_arith(rng: random.Random) -> str:
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    # only multiplication for decimals; product 0..81 → fixed 2-decimal "0.cc"
    c = a * b
    return f"0.{a}*0.{b}=0.{c:02d}\n"


def svo_sample(rng: random.Random) -> str:
    return f"{rng.choice(ART)} {rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(ART)} {rng.choice(OBJ)}\n"


def build_corpus(n_samples: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        r = rng.random()
        if r < 0.35:
            out.append(int_arith(rng))
        elif r < 0.70:
            out.append(dec_arith(rng))
        else:
            out.append(svo_sample(rng))
    return "".join(out)


def make_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
    return x, y


# ---------- train ----------


def train(steps: int, batch_size: int, lr: float, device: str) -> GPT:
    cfg = GPTConfig()
    model = GPT(cfg).to(device)
    print(f"params: {model.num_params() / 1e6:.2f}M, vocab: {VOCAB_SIZE}")

    corpus = build_corpus(n_samples=25000)
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
            print(f"step {step:5d}  loss {loss.item():.4f}", flush=True)

    # eval: int mult + dec mult accuracy
    model.eval()
    int_ok, dec_ok = 0, 0
    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                # int
                prompt = f"{a}*{b}="
                pred = generate(model, prompt, max_new=3, device=device)
                ans = pred[len(prompt) :].split("\n", 1)[0]
                try:
                    if int(ans) == a * b:
                        int_ok += 1
                except ValueError:
                    pass
                # dec
                prompt = f"0.{a}*0.{b}="
                pred = generate(model, prompt, max_new=5, device=device)
                ans = pred[len(prompt) :].split("\n", 1)[0]
                target = f"0.{a*b:02d}"
                if ans.startswith(target):
                    dec_ok += 1
    print(f"int  multiplication: {int_ok}/100 = {int_ok}%")
    print(f"dec  multiplication: {dec_ok}/100 = {dec_ok}%")

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


def load(device: str) -> GPT:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------- probe ----------


@torch.no_grad()
def collect_states(model: GPT, device: str) -> dict:
    """Forward 100 int and 100 dec prompts, grab =-token state at every layer."""
    model.eval()
    rows = []  # (kind, a, b, product, layer, vec)
    for kind, fmt in [("int", "{a}*{b}="), ("dec", "0.{a}*0.{b}=")]:
        for a in range(10):
            for b in range(10):
                prompt = fmt.format(a=a, b=b)
                idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
                _, _, states = model(idx, return_states=True)
                eq_pos = len(prompt) - 1  # the '=' token
                for layer, h in enumerate(states):
                    vec = h[0, eq_pos, :].cpu().numpy()
                    rows.append((kind, a, b, a * b, layer, vec))

    kind_arr = np.array([r[0] for r in rows])
    a_arr = np.array([r[1] for r in rows], dtype=np.int32)
    b_arr = np.array([r[2] for r in rows], dtype=np.int32)
    p_arr = np.array([r[3] for r in rows], dtype=np.int32)
    l_arr = np.array([r[4] for r in rows], dtype=np.int32)
    H = np.stack([r[5] for r in rows])  # (1200, 128)
    np.savez(STATES, kind=kind_arr, a=a_arr, b=b_arr, product=p_arr, layer=l_arr, H=H)
    print(f"saved hidden states: {STATES}  shape={H.shape}")
    return {"kind": kind_arr, "a": a_arr, "b": b_arr, "product": p_arr, "layer": l_arr, "H": H}


# ---------- viz ----------


def visualize(data: dict) -> None:
    import umap
    import plotly.graph_objects as go

    H = data["H"]
    print(f"UMAP fitting {H.shape} ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    Y = reducer.fit_transform(H)  # (1200, 2)

    kind_arr = data["kind"]
    a_arr = data["a"]
    b_arr = data["b"]
    l_arr = data["layer"]
    n_layers = int(l_arr.max()) + 1

    fig = go.Figure()
    first_int = True
    first_dec = True
    for kind in ("int", "dec"):
        symbol = "circle" if kind == "int" else "diamond"
        dash = "solid" if kind == "int" else "dash"
        label = "integer" if kind == "int" else "decimal"
        for a in range(10):
            for b in range(10):
                mask = (kind_arr == kind) & (a_arr == a) & (b_arr == b)
                order = np.argsort(l_arr[mask])
                xs = Y[mask][order, 0]
                ys = Y[mask][order, 1]
                zs = l_arr[mask][order]
                product = a * b
                pretty = f"{a}×{b}" if kind == "int" else f"0.{a}×0.{b}"
                text = [f"{pretty}={product if kind == 'int' else f'0.{product:02d}'}, L{z}" for z in zs]
                show_scale = (kind == "int" and a == 0 and b == 0)
                # legend entry once per kind
                show_legend = (kind == "int" and first_int) or (kind == "dec" and first_dec)
                if kind == "int" and first_int:
                    first_int = False
                if kind == "dec" and first_dec:
                    first_dec = False
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
                            dash=dash,
                        ),
                        marker=dict(
                            size=4,
                            symbol=symbol,
                            color=[product] * len(zs),
                            colorscale="Viridis",
                            cmin=0,
                            cmax=81,
                            showscale=show_scale,
                            colorbar=dict(title="a × b") if show_scale else None,
                        ),
                        hovertext=text,
                        hoverinfo="text",
                        showlegend=show_legend,
                        legendgroup=kind,
                        name=label,
                    )
                )

    fig.update_layout(
        title=(
            f"Int + decimal multiplication trajectories through {n_layers} layers "
            "(NanoGPT 1.2M, joint UMAP) — circle=int, diamond=decimal, color=digit-product"
        ),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="Layer (=-token)",
            zaxis=dict(tickmode="linear", tick0=0, dtick=1),
        ),
        width=1200,
        height=900,
    )
    fig.write_html(HTML)
    print(f"saved viz: {HTML}")


# ---------- main ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    if args.no_train and CKPT.exists():
        print(f"loading checkpoint: {CKPT}")
        model = load(args.device)
    else:
        model = train(args.steps, args.batch_size, args.lr, args.device)

    data = collect_states(model, args.device)
    visualize(data)


if __name__ == "__main__":
    main()
