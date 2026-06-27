"""Route-classification probe: given an MC prompt, pick WHICH recognition probe to
use (or abstain), so probe selection is automatic instead of a manual choice.

A last-token hidden-state classifier over the calibrated probe tasks (prototype:
100% 7-way CV @L11). Its max-softmax confidence is the COVERAGE GATE — route only
when confident, else fall through to the zero-shot floor / generation. This extends
`commitment = coverage` from the symbolic solvers (gate on parse coverage) to the
probes (gate on router confidence).

Serialized as linear params (scaler + multinomial coef/intercept), no pickled
sklearn — same convention as the other ModelProfile components.
"""
from __future__ import annotations

import numpy as np


class RouteProbe:
    def __init__(self, layer, classes, mean, std, coef, intercept, threshold=0.9):
        self.layer = int(layer)
        self.classes = list(classes)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.coef = np.asarray(coef, dtype=np.float32)        # [n_classes, H]
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.threshold = float(threshold)

    def _scores(self, h):
        z = (h - self.mean) / self.std
        logits = self.coef @ z + self.intercept
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def route(self, prompt, model, tokenizer, device):
        """Return (task_name or None, confidence). None when below threshold."""
        import torch
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        h = hs[self.layer][0, -1, :].float().cpu().numpy()
        p = self._scores(h)
        idx = int(p.argmax())
        conf = float(p[idx])
        return (self.classes[idx] if conf >= self.threshold else None), conf

    def to_dict(self):
        return {
            "layer": self.layer, "classes": self.classes, "threshold": self.threshold,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "coef": self.coef.tolist(), "intercept": self.intercept.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["layer"], d["classes"], d["mean"], d["std"],
                   d["coef"], d["intercept"], d.get("threshold", 0.9))


def calibrate_route_probe(model, tokenizer, device, tasks, n=60,
                          layer=None, threshold=0.9, verbose=False):
    """Fit a RouteProbe over `tasks` from last-token hidden states. Picks the best
    layer by 5-fold CV if `layer` is None. Returns (RouteProbe, cv_accuracy)."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from turnstyle.bbh import load_task

    def last_hidden(prompt):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        return torch.stack(hs, 0)[:, 0, -1, :].float().cpu().numpy()   # [L+1, H]

    X, y = [], []
    for ti, t in enumerate(tasks):
        for ex in load_task(t)[:n]:
            X.append(last_hidden(ex["input"])); y.append(ti)
        if verbose:
            print(f"  [route] collected {t}", flush=True)
    X = np.stack(X); y = np.array(y)
    nL = X.shape[1]

    def fit_layer(L):
        sc = StandardScaler().fit(X[:, L])
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[:, L]), y)
        return sc, clf

    if layer is None:
        best = (-1.0, 0)
        for L in range(nL):
            sc = StandardScaler().fit(X[:, L])
            acc = float(cross_val_score(LogisticRegression(max_iter=2000, C=0.5),
                                        sc.transform(X[:, L]), y, cv=5).mean())
            if acc > best[0]:
                best = (acc, L)
        cv_acc, layer = best
        if verbose:
            print(f"  [route] best layer L{layer} cv={cv_acc:.3f}", flush=True)
    else:
        cv_acc = float(cross_val_score(LogisticRegression(max_iter=2000, C=0.5),
                       StandardScaler().fit_transform(X[:, layer]), y, cv=5).mean())

    sc, clf = fit_layer(layer)
    probe = RouteProbe(layer=layer, classes=list(tasks), mean=sc.mean_, std=sc.scale_,
                       coef=clf.coef_, intercept=clf.intercept_, threshold=threshold)
    return probe, cv_acc
