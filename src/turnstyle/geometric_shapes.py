"""geometric_shapes — deterministic SVG-path shape classifier (graph-walking).

The path is modeled as an attributed graph: nodes = unique coordinates (coincident
points merged, which collapses the pen-up/pen-down "M start L end M end L next" style),
edges = segments typed line|arc. The shape is then a property of the graph + traversal:

  arc edges only          -> ellipse
  arc edges + line radii   -> sector
  open, 2 nodes            -> line
  closed cycle, N nodes    -> N-gon (3=triangle .. 10=decagon); N=4 -> trapezoid|kite
                              by geometry (parallel opposite sides vs adjacent-equal sides)

This is serialization-invariant: the answer depends on the graph, not on how the path
was drawn (pen lifts, point repetition, command order). Purely structural SVG parsing —
no natural-language synonyms, no keyword lists. The tokenizer doubles as the coverage
gate: parse success ⇒ it's a geometric_shapes prompt ⇒ commit; failure ⇒ abstain.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_DPATH = re.compile(r'd="([^"]+)"')
_OPT = re.compile(r"\(([A-Z])\)\s*([^\n]+)")
_TOKEN = re.compile(r"[A-Za-z]|-?\d+\.?\d*(?:[eE][-+]?\d+)?")

# SVG path command -> number of numeric args per coordinate group
_ARGC = {"M": 2, "L": 2, "A": 7, "Z": 0, "H": 1, "V": 1}
_NGON = {3: "triangle", 5: "pentagon", 6: "hexagon", 7: "heptagon",
         8: "octagon", 9: "nonagon", 10: "decagon"}


@dataclass
class PathGraph:
    points: list                 # unique nodes (coincident merged)
    edges: list                  # (u_idx, v_idx, kind) kind in {"line","arc"}
    seq: list                    # pen path in order (for cycle ordering)

    def edge_kinds(self) -> set:
        return {k for _, _, k in self.edges}


def tokenize_path(d: str):
    """Scan a `d` string into [(cmd, [args])], expanding implicit coordinate repeats
    (one command letter may carry many groups). Returns None on malformed / unsupported
    (bezier C/S/Q/T) input — the coverage gate."""
    toks = _TOKEN.findall(d)
    cmds, i = [], 0
    while i < len(toks):
        c = toks[i]
        if not c.isalpha():
            return None                      # number with no command
        i += 1
        n = _ARGC.get(c.upper())
        if n is None:
            return None                      # unsupported command (bezier, etc.)
        if n == 0:
            cmds.append((c, []))
            continue
        if i >= len(toks) or toks[i].isalpha():
            return None                      # command with no args
        while i < len(toks) and not toks[i].isalpha():   # implicit repeats
            grp = toks[i:i + n]
            if len(grp) < n:
                return None
            cmds.append((c, [float(x) for x in grp]))
            i += n
    return cmds or None


def _key(p):
    return (round(p[0], 1), round(p[1], 1))


def build_graph(cmds) -> PathGraph | None:
    """Walk the command stream into a PathGraph. Handles absolute + relative (lowercase)
    commands; M=pen move (no edge), L/H/V=line edge, A=arc edge, Z=close (line to start)."""
    pts, idx, edges, seq = [], {}, [], []
    cur = start = None

    def node(p):
        k = _key(p)
        if k not in idx:
            idx[k] = len(pts)
            pts.append(p)
        return idx[k]

    for c, a in cmds:
        rel, C = c.islower(), c.upper()
        if C == "M":
            p = (a[0], a[1])
            if rel and cur:
                p = (cur[0] + p[0], cur[1] + p[1])
            cur = start = p
            node(cur); seq.append(cur)
        elif C in ("L", "A", "H", "V"):
            if cur is None:
                return None                  # segment before any moveto: malformed
            if C == "H":
                p = ((cur[0] + a[0]) if rel else a[0], cur[1])
            elif C == "V":
                p = (cur[0], (cur[1] + a[0]) if rel else a[0])
            else:
                px, py = (a[-2], a[-1])
                p = (cur[0] + px, cur[1] + py) if rel else (px, py)
            if cur is None:
                return None
            edges.append((node(cur), node(p), "arc" if C == "A" else "line"))
            cur = p; seq.append(cur)
        elif C == "Z":
            if cur is not None and start is not None:
                edges.append((node(cur), node(start), "line"))
                cur = start; seq.append(cur)
    if not edges:
        return None
    return PathGraph(pts, edges, seq)


def _ordered_vertices(seq):
    """Pen path -> cycle-ordered unique vertices (drop consecutive + closing repeats)."""
    out = []
    for p in seq:
        if not out or _key(out[-1]) != _key(p):
            out.append(p)
    if len(out) > 1 and _key(out[0]) == _key(out[-1]):
        out.pop()
    return out


def _quad(v) -> str:
    """4-gon: kite (two pairs of ADJACENT equal sides) vs trapezoid (one pair of PARALLEL
    opposite sides)."""
    def sub(a, b): return (b[0] - a[0], b[1] - a[1])
    def ln(a, b): return math.hypot(b[0] - a[0], b[1] - a[1])
    def cross(a, b): return a[0] * b[1] - a[1] * b[0]

    s = [ln(v[i], v[(i + 1) % 4]) for i in range(4)]
    d = [sub(v[i], v[(i + 1) % 4]) for i in range(4)]

    def eq(a, b, rel=0.10): return abs(a - b) <= rel * max(a, b, 1e-9)

    def par(a, b, tol=0.07):
        na, nb = math.hypot(*a), math.hypot(*b)
        return na > 0 and nb > 0 and abs(cross(a, b)) / (na * nb) < tol

    kite = (eq(s[0], s[1]) and eq(s[2], s[3])) or (eq(s[1], s[2]) and eq(s[3], s[0]))
    trap = par(d[0], d[2]) or par(d[1], d[3])
    if kite and not trap:
        return "kite"
    if trap and not kite:
        return "trapezoid"
    return "trapezoid" if trap else "kite"     # tie/neither: parallel is the defining cue


def classify(graph: PathGraph) -> str | None:
    kinds = graph.edge_kinds()
    if "arc" in kinds:
        return "sector" if "line" in kinds else "ellipse"
    verts = _ordered_vertices(graph.seq)
    n = len(verts)
    if n == 2:
        return "line"
    if n == 4:
        return _quad(verts)
    return _NGON.get(n)


def classify_path(d: str) -> str | None:
    """SVG `d` string -> shape name, or None if unparseable/unsupported."""
    cmds = tokenize_path(d)
    if cmds is None:
        return None
    g = build_graph(cmds)
    return classify(g) if g is not None else None


def solve_geometric_shapes(prompt: str) -> str | None:
    """Return the option letter '(X)' whose text names the path's shape, or None
    (abstain) if there's no parseable path or the shape isn't among the options."""
    m = _DPATH.search(prompt)
    if not m:
        return None
    shape = classify_path(m.group(1))
    if shape is None:
        return None
    for letter, txt in _OPT.findall(prompt):
        if shape in txt.strip().lower():
            return f"({letter})"
    return None


__all__ = ["PathGraph", "tokenize_path", "build_graph", "classify", "classify_path",
           "solve_geometric_shapes"]
