"""Why is native hyperbaton only 85%? Per-example breakdown: commit/abstain/wrong, the
abstain reason, and for failures the adjective subjectivity scores (is the 1D axis
mis-ordering specific category pairs?).
"""
from __future__ import annotations
import sys
sys.path.insert(0, "experiments")
from turnstyle.bbh import load_task
from turnstyle.hyperbaton import solve_hyperbaton, _adjectives, _OPT_RE, _TMPL


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turnstyle.profile import load_profile
    from turnstyle.semantic_frame import _word_vectors
    mid = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
    prof = load_profile(mdl)
    axis = prof.get_subjectivity()
    print(f"subjectivity axis @L{axis.layer}\n")

    ex = load_task("hyperbaton")
    commit = correct = 0
    abst = {"tie": 0, "not_perm": 0, "few_adj": 0, "other": 0}
    fails = []
    for e in ex:
        ans = solve_hyperbaton(e["input"], mdl, tok, dev, axis)
        tgt = e["target"].strip()
        if ans is None:
            opts = _OPT_RE.findall(e["input"])
            if len(opts) != 2:
                abst["other"] += 1
            else:
                ta, tb = opts[0][1], opts[1][1]
                aa = _adjectives(ta)
                if len(aa) < 2 or sorted(ta.split()) != sorted(tb.split()):
                    abst["not_perm" if len(aa) >= 2 else "few_adj"] += 1
                else:
                    abst["tie"] += 1                       # ia==ib
            fails.append(("ABSTAIN", e, None, tgt))
            continue
        commit += 1
        ok = ans.lower() == tgt.lower()
        correct += ok
        if not ok:
            fails.append(("WRONG", e, ans, tgt))

    n = len(ex)
    # end-to-end (abstains would fall to gen ~chance 0.5 on binary)
    print(f"committed {commit}/{n}  committed_acc={correct/commit*100:.1f}%  "
          f"(committed_correct={correct})")
    print(f"abstains: {abst}  -> these fall to zero-shot gen (~chance on a 2-option task)")
    print(f"\n=== failures (adjective subjectivity scores; correct = DECREASING) ===")
    for kind, e, ans, tgt in fails[:14]:
        opts = _OPT_RE.findall(e["input"])
        line = f"[{kind}] tgt={tgt} ans={ans}"
        print(line)
        for letter, txt in opts:
            adj = _adjectives(txt)
            if len(adj) >= 2:
                sc = [axis.project(v) for v in _word_vectors(mdl, tok, dev, adj, axis.layer, _TMPL)]
                seq = "  ".join(f"{a}:{s:+.2f}" for a, s in zip(adj, sc))
                print(f"    ({letter}) {seq}")


if __name__ == "__main__":
    main()
