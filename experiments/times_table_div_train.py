"""Train + capture for a 4-operator NanoGPT (+, −, ×, ÷).

Self-contained (doesn't share vocab with times_table_trace.py).
Division semantics: floor division a // b for b > 0.  Division-by-zero
is *not* in the training corpus — it's only seen at inference time as
an OOD test.

Outputs (written into the data directory of the published viz):
  - hidden_states_mul.npz  (replaces existing — captured with new model)
  - hidden_states_add.npz
  - hidden_states_sub.npz
  - hidden_states_div.npz  (new)
  - predicted_chars.json   (now covers all 4 operators)
  - checkpoint_div.pt
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Output to the published viz's data directory directly.
DATA_OUT = Path("/Users/jdonaldson/Projects/nanogpt-arithmetic-viz/data")
DATA_OUT.mkdir(parents=True, exist_ok=True)
CKPT_OUT = DATA_OUT / "checkpoint_div.pt"

# ─── Vocab — same as times_table_trace plus '/' ────────────────────

DIGITS = "0123456789"
OPS = "+-*/="
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


# ─── Model — same architecture as times_table_trace ─────────────────


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
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE),
                                    targets.view(-1), ignore_index=-1)
            return logits, loss, states
        return self.head(x[:, [-1], :]), None, states


# ─── Data ───────────────────────────────────────────────────────────


def arithmetic_sample(rng: random.Random) -> str:
    """Sample one arithmetic prompt — division has b > 0 (no div-by-zero
    in training).  Floor division for ÷."""
    op = rng.choice("+-*/")
    if op == "/":
        a = rng.randint(0, 9)
        b = rng.randint(1, 9)  # b > 0 only
        c = a // b
    else:
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        if op == "+":
            c = a + b
        elif op == "-":
            c = a - b
        else:  # *
            c = a * b
    return f"{a}{op}{b}={c}\n"


def svo_sample(rng: random.Random) -> str:
    return (f"{rng.choice(ART)} {rng.choice(SUBJ)} {rng.choice(VERB)} "
            f"{rng.choice(ART)} {rng.choice(OBJ)}\n")


def build_corpus(n_samples: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_samples):
        out.append(arithmetic_sample(rng) if rng.random() < 0.7
                   else svo_sample(rng))
    return "".join(out)


def make_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix]).to(device)
    return x, y


# ─── Train ─────────────────────────────────────────────────────────


def train(steps: int = 8000, batch_size: int = 64, lr: float = 3e-4,
          device: str = "cpu") -> GPT:
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
            print(f"  step {step:5d}  loss {loss.item():.4f}")

    model.eval()
    print("\nQuick accuracy check (each op, all 100 pairs):")
    for op_char in "+-*/":
        correct = 0
        valid = 0
        with torch.no_grad():
            for a in range(10):
                for b in range(10):
                    if op_char == "/" and b == 0:
                        continue  # skip div-by-0 in eval
                    prompt = f"\n{a}{op_char}{b}="
                    pred = generate(model, prompt, max_new=3, device=device)
                    ans = pred[len(prompt):].split("\n", 1)[0]
                    expected = (a + b if op_char == "+" else
                                a - b if op_char == "-" else
                                a * b if op_char == "*" else
                                a // b)
                    try:
                        if int(ans) == expected:
                            correct += 1
                    except ValueError:
                        pass
                    valid += 1
        print(f"  {op_char}: {correct}/{valid} = {correct/valid:.1%}")

    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(),
                "stoi": STOI}, CKPT_OUT)
    print(f"\nsaved checkpoint: {CKPT_OUT}")
    return model


def generate(model, prompt, max_new, device):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _, _ = model(idx_cond)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return decode(idx[0].tolist())


def load_or_train(device: str = "cpu") -> GPT:
    if CKPT_OUT.exists():
        print(f"Loading existing checkpoint: {CKPT_OUT}")
        ck = torch.load(CKPT_OUT, map_location=device)
        cfg = GPTConfig(**ck["cfg"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ck["state_dict"])
        return model
    return train(device=device)


# ─── Capture hidden states ─────────────────────────────────────────


def capture_states(model: GPT, op_char: str, op_name: str, device: str):
    """For each (a, b) in 0..9², capture hidden state at =-token at
    every layer.  For division, b=0 is INCLUDED (OOD test)."""
    model.eval()
    rows = {"a": [], "b": [], "value": [], "layer": [], "H": []}
    with torch.no_grad():
        for a in range(10):
            for b in range(10):
                prompt = f"\n{a}{op_char}{b}="
                ids = torch.tensor([encode(prompt)], dtype=torch.long,
                                   device=device)
                logits, _, states = model(ids, return_states=True)
                # =-token state at each layer (last position)
                for L, h in enumerate(states):
                    rows["a"].append(a)
                    rows["b"].append(b)
                    if op_char == "+":
                        v = a + b
                    elif op_char == "-":
                        v = a - b
                    elif op_char == "*":
                        v = a * b
                    else:  # /
                        v = a // b if b != 0 else -999  # marker for div-by-0
                    rows["value"].append(v)
                    rows["layer"].append(L)
                    rows["H"].append(h[0, -1, :].cpu().numpy())
    out = DATA_OUT / f"hidden_states_{op_name}.npz"
    if op_name == "mul":
        out = DATA_OUT / "hidden_states.npz"  # legacy filename
    np.savez(
        out,
        a=np.array(rows["a"]),
        b=np.array(rows["b"]),
        **({"product": np.array(rows["value"])} if op_char == "*"
           else {"sum": np.array(rows["value"])} if op_char == "+"
           else {"diff": np.array(rows["value"])} if op_char == "-"
           else {"quotient": np.array(rows["value"])}),
        layer=np.array(rows["layer"]),
        H=np.array(rows["H"]),
    )
    print(f"  → {out}  (shape H: {np.array(rows['H']).shape})")


def capture_predictions(model: GPT, device: str):
    """Predicted first char at =-token for each (a, b, op).  Division-
    by-zero predictions are included as an OOD test."""
    model.eval()
    out = {}
    for op_name, op_char in [("mul", "*"), ("add", "+"),
                              ("sub", "-"), ("div", "/")]:
        pred_chars = {}
        exp_chars = {}
        cnt_pred = {}
        cnt_exp = {}
        mismatches = []
        for a in range(10):
            for b in range(10):
                prompt = f"\n{a}{op_char}{b}="
                ids = torch.tensor([encode(prompt)], dtype=torch.long,
                                   device=device)
                with torch.no_grad():
                    logits, _, _ = model(ids)
                pred_id = int(logits[0, -1].argmax().item())
                pred_char = ITOS[pred_id]

                if op_char == "+":
                    expected = a + b
                elif op_char == "-":
                    expected = a - b
                elif op_char == "*":
                    expected = a * b
                else:  # /
                    expected = a // b if b != 0 else None
                exp_char = str(expected)[0] if expected is not None else "?"

                pred_chars[f"{a},{b}"] = pred_char
                exp_chars[f"{a},{b}"] = exp_char
                cnt_pred[pred_char] = cnt_pred.get(pred_char, 0) + 1
                cnt_exp[exp_char] = cnt_exp.get(exp_char, 0) + 1
                if pred_char != exp_char and expected is not None:
                    mismatches.append([a, b, exp_char, pred_char])
        out[op_name] = {
            "predicted": pred_chars,
            "expected": exp_chars,
            "pred_counts": cnt_pred,
            "exp_counts": cnt_exp,
            "mismatches": mismatches,
        }
        print(f"\n{op_name}:")
        print(f"  predicted counts: {cnt_pred}")
        if op_char == "/":
            # Highlight what the model emits for div-by-zero
            print(f"  division-by-zero predictions (OOD):")
            for a in range(10):
                print(f"    {a}/0 → {pred_chars[f'{a},0']!r}")

    (DATA_OUT / "predicted_chars.json").write_text(json.dumps(out, indent=2))
    print(f"\n  → {DATA_OUT / 'predicted_chars.json'}")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_or_train(device=device)

    print("\nCapturing hidden states...")
    for op_char, op_name in [("*", "mul"), ("+", "add"),
                              ("-", "sub"), ("/", "div")]:
        print(f"\n  capturing {op_name} ({op_char}):")
        capture_states(model, op_char, op_name, device=device)

    print("\nCapturing predictions (including div-by-zero OOD)...")
    capture_predictions(model, device=device)
    print("\nDone.")


if __name__ == "__main__":
    main()
