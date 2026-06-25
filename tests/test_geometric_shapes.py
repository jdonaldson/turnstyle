"""No-model tests for the deterministic geometric_shapes solver (graph-walking).

Real `d`-strings from the BBH cache for the tricky shapes (arc-based + 4-gons), and
generated regular polygons for the vertex-count ladder. Also checks the parser-as-gate
(declines on non-path / malformed input) and dispatch routing to the GeometricShape ADT.
"""
import math

import pytest

from turnstyle.geometric_shapes import (
    classify_path,
    solve_geometric_shapes,
    tokenize_path,
)

# real d-strings observed in the cache (pen-up/pen-down style with repeated points)
REAL = {
    "triangle": "M 30.17,45.97 L 58.79,40.36 L 18.10,15.70 M 18.10,15.70 L 30.17,45.97",
    "trapezoid": "M 21.10,97.94 L 22.14,97.44 L 10.48,73.34 L 9.43,73.84 L 21.10,97.94",
    "kite": ("M 55.46,58.72 L 70.25,50.16 M 70.25,50.16 L 78.35,57.33 "
             "M 78.35,57.33 L 71.18,65.42 L 55.46,58.72"),
    "ellipse": ("M 22.34,17.53 A 19.21,19.21 220.48 1,0 51.57,42.47 "
                "A 19.21,19.21 220.48 1,0 22.34,17.53"),
    "sector": ("M 48.48,23.04 L 30.68,44.97 M 30.68,44.97 "
               "A 28.25,28.25 317.18 0,1 20.40,19.91 L 48.48,23.04"),
    "line": "M 10.00,10.00 L 50.00,60.00",
}


def ngon(n, r=30.0, cx=50.0, cy=50.0):
    """A closed regular n-gon path (M + n line segments back to start)."""
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
           for i in range(n)]
    segs = "".join(f" L {x:.2f},{y:.2f}" for x, y in pts[1:] + [pts[0]])
    return f"M {pts[0][0]:.2f},{pts[0][1]:.2f}{segs}"


def test_real_shapes_classify():
    for shape, d in REAL.items():
        assert classify_path(d) == shape, shape


def test_polygons_by_vertex_count():
    for n, name in [(5, "pentagon"), (6, "hexagon"), (7, "heptagon"), (8, "octagon")]:
        assert classify_path(ngon(n)) == name, name


def test_tokenizer_expands_implicit_repeats():
    cmds = tokenize_path("M 0,0 L 1,1 2,2 3,3")     # one L, three coord pairs
    assert sum(1 for c, _ in cmds if c == "L") == 3
    assert classify_path("M 0,0 L 1,1 2,2 0,0") == "triangle"   # 3 unique verts


def test_gate_declines_non_path():
    assert classify_path("M") is None              # command with no args
    assert classify_path("hello world") is None
    assert solve_geometric_shapes("What is 3 + 4 * 2?") is None


def test_solve_end_to_end_returns_option_letter():
    prompt = (f'A shape <path d="{ngon(6)}"/> draws a\n'
              "Options:\n(A) circle\n(B) hexagon\n(C) line")
    assert solve_geometric_shapes(prompt) == "(B)"


def test_solve_abstains_when_shape_not_in_options():
    prompt = (f'<path d="{ngon(7)}"/> draws a\n'
              "Options:\n(A) circle\n(B) square")   # heptagon not offered
    assert solve_geometric_shapes(prompt) is None


def test_dispatch_routes_to_geometric_shape():
    from turnstyle.dispatch import Ctx, GeometricShape, parse
    prompt = (f'<path d="{ngon(8)}"/> draws a\n'
              "Options:\n(A) circle\n(B) octagon\n(C) line")
    task = parse(prompt, Ctx())                     # no model needed
    assert isinstance(task, GeometricShape) and task.answer == "(B)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
