"""Gemma 3n E2B shape check — does AltUp leak through `output_hidden_states`?

Every probe and frame readout in turnstyle assumes ONE residual vector per layer
(`hidden_states[l]` -> (batch, seq, hidden)).  Gemma 3n has `altup_num_inputs: 4`,
so the stack carries four parallel prediction streams internally.  Before fitting
the frame family on this backbone we need to know which of these is true:

  (a) `hidden_states` is already the reduced/active stream -> shape (B, T, 2048).
      Frames work unchanged; `altup_active_idx` tells us which stream we read.
  (b) `hidden_states` carries the stacked streams -> an extra axis of size 4, or
      hidden 8192.  Every readout needs an explicit stream choice first.

Also inventories the PLE table (for the bridge experiment in
[[flash_resident_probe_library]]) and confirms the text sub-config numbers that
[[ple_flash_tiering]] recorded via WebFetch rather than a direct file read.

    uv run python experiments/gemma3n_shape_check.py

Runs one forward pass, no generation.  Must run OUTSIDE the sandbox or torch
misreports MPS as unavailable (bogus "macOS 14.0+" error).
"""

import torch
from transformers import AutoConfig, AutoTokenizer

MODEL = "google/gemma-3n-E2B-it"
PROMPT = "The enormous ancient stone bridge"


def banner(s):
    print(f"\n{'=' * 62}\n{s}\n{'=' * 62}")


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {dev}  torch: {torch.__version__}")

    banner("1. CONFIG — verify the numbers we recorded second-hand")
    cfg = AutoConfig.from_pretrained(MODEL)
    print(f"top-level class: {type(cfg).__name__}")
    text = getattr(cfg, "text_config", cfg)
    print(f"text config    : {type(text).__name__}")
    for k in (
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "intermediate_size",
        "hidden_size_per_layer_input",
        "vocab_size_per_layer_input",
        "altup_num_inputs",
        "altup_active_idx",
        "num_kv_shared_layers",
        "laurel_rank",
    ):
        print(f"  {k:32s} {getattr(text, k, '<absent>')}")

    n_layer = text.num_hidden_layers
    hidden = text.hidden_size
    ple_dim = getattr(text, "hidden_size_per_layer_input", None)
    ple_vocab = getattr(text, "vocab_size_per_layer_input", None)
    if ple_dim and ple_vocab:
        n = ple_vocab * n_layer * ple_dim
        print(f"\n  PLE table = {ple_vocab:,} x {n_layer} x {ple_dim} = {n / 1e9:.2f}B params")

    banner("2. LOAD — text-only, no vision tower")
    # Both AutoModelForCausalLM and Gemma3nForConditionalGeneration build the full
    # multimodal wrapper, which instantiates a TimmWrapperModel vision tower and dies
    # on a missing `timm`.  We only ever want text, so load the text-only class and
    # skip the vision/audio weights entirely (cheaper, and no extra dependency).
    from transformers import Gemma3nForCausalLM

    model = Gemma3nForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    print(f"loaded -> {type(model).__name__}")

    model = model.to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)

    banner("3. THE QUESTION — hidden_states shapes")
    ids = tok(PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)

    hs = out.hidden_states
    print(f"len(hidden_states) = {len(hs)}   (expect n_layer+1 = {n_layer + 1})")
    print(f"seq len            = {ids['input_ids'].shape[1]}")
    shapes = {tuple(h.shape) for h in hs}
    print(f"distinct shapes    = {shapes}")

    h0 = hs[0]
    print(f"\nhidden_states[0].shape  = {tuple(h0.shape)}  ndim={h0.ndim}")
    print(f"hidden_states[-1].shape = {tuple(hs[-1].shape)}")

    banner("VERDICT")
    n_alt = getattr(text, "altup_num_inputs", 1)
    if h0.ndim == 3 and h0.shape[-1] == hidden:
        print(f"(a) CLEAN — one {hidden}-dim residual per layer, {h0.ndim}D as turnstyle expects.")
        print(f"    AltUp's {n_alt} streams are reduced internally; readouts work unchanged.")
        print(f"    (active stream = altup_active_idx {getattr(text, 'altup_active_idx', '?')})")
    elif h0.ndim == 4:
        print(f"(b) LEAKS — 4D {tuple(h0.shape)}; AltUp streams are exposed on their own axis.")
        print(f"    Every probe/frame readout must select a stream before projecting.")
    elif h0.ndim == 3 and h0.shape[-1] == hidden * n_alt:
        print(f"(b) LEAKS — last dim {h0.shape[-1]} = hidden {hidden} x altup {n_alt}, streams concatenated.")
        print(f"    Slice to the active stream before projecting.")
    else:
        print(f"UNEXPECTED — shape {tuple(h0.shape)} vs hidden {hidden}, altup {n_alt}. Inspect by hand.")

    # A frame readout is a dot product against the LAST subword's residual
    # ([[color_semantic_frame]]).  With the stream axis FIRST, the correct slice is
    # h[stream, batch, -1] -- a 3D-style h[0, -1] silently yields (T, H) for stream 0,
    # i.e. the whole sequence, not the last token.
    import numpy as np

    active = getattr(text, "altup_active_idx", 0)
    banner(f"4. READOUT SANITY — last-subword vector, stream {active} (altup_active_idx)")
    vecs = [h[active, 0, -1].float().cpu() for h in hs]
    norms = np.array([float(v.norm()) for v in vecs])
    print(f"per-layer last-token norms: min {norms.min():.1f}  max {norms.max():.1f}")
    print(f"  L0 {norms[0]:.1f} | L1 {norms[1]:.1f} | mid {norms[len(norms) // 2]:.1f} | last {norms[-1]:.1f}")
    print(f"any NaN: {bool(np.isnan(norms).any())}   (bf16 deep-layer NaNs are a known trap)")
    print(f"\nbuffering one stream's last-token stack costs "
          f"{len(hs) * hidden * 2 / 1024:.0f}KB at bf16 "
          f"(cf. SmolLM2's 98KB in [[flash_resident_probe_library]]); "
          f"all {hs[0].shape[0]} streams = {len(hs) * hidden * 2 * hs[0].shape[0] / 1024:.0f}KB")

    # Does the stream choice actually matter for a frame readout?  If the streams are
    # near-identical, picking the wrong one is harmless; if they diverge, every probe
    # fit on this backbone has to pin the stream explicitly.
    banner("4b. STREAM DIVERGENCE — does the choice matter?")
    n_stream = hs[0].shape[0]
    print(f"cosine(stream {active}, stream j) at the last token, by layer:")
    print(f"  {'layer':>6}  " + "  ".join(f"s{j}" .rjust(7) for j in range(n_stream) if j != active))
    for li in (0, 1, len(hs) // 2, len(hs) - 2, len(hs) - 1):
        ref = hs[li][active, 0, -1].float()
        cos = []
        for j in range(n_stream):
            if j == active:
                continue
            other = hs[li][j, 0, -1].float()
            denom = (ref.norm() * other.norm()).clamp_min(1e-9)
            cos.append(float((ref @ other) / denom))
        print(f"  {li:>6}  " + "  ".join(f"{c:>7.3f}" for c in cos))

    banner("5. PLE TABLE — locate it for the bridge experiment")
    hits = [
        (n, tuple(p.shape), p.numel())
        for n, p in model.named_parameters()
        if "per_layer" in n or "altup" in n
    ]
    # Rank by size: the 2B lookup table is the thing we care about, and it drowns in
    # the ~240 tiny per-layer projections if we print in declaration order.
    print("largest per_layer/altup tensors:")
    for n, shape, numel in sorted(hits, key=lambda h: -h[2])[:6]:
        print(f"  {n:52s} {str(shape):24s} {numel / 1e6:8.1f}M")
    total = sum(h[2] for h in hits)
    allp = sum(p.numel() for p in model.parameters())
    print(f"\nper_layer/altup params: {total / 1e9:.2f}B of {allp / 1e9:.2f}B total "
          f"({100 * total / allp:.0f}%)")


if __name__ == "__main__":
    main()
