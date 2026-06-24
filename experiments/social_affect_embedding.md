# Monkey-Sphere Affect Embedding — a first-class experiment

**Status:** design / not yet built. Sits *on top of* turnstyle (consumes its
affect primitives; does not modify the core).
**Date:** 2026-06-21
**Lineage:** extends the EPA/`SemanticFrame` line (`osgood_epa_frame`,
`epa_modulated_embeddings`, `semantic_programming_direction`) into **relational,
social** affect. Inherits the relational-falsification caution from the
dyf×turnstyle arc.

---

## 0. One-line thesis

Scalar affect (valence/arousal, EPA/PAD) is **individuation-insufficient** for
*social* emotions. A social emotion is not a point in affect space — it is
**perceived affect-flux on an edge of an affect-weighted relational graph**.
Build an embedding for that, on top of turnstyle's per-entity affect reads.

---

## 1. Why scalar affect isn't enough (the motivating argument)

- EPA (Evaluation/Potency/Activity) ≈ PAD (Pleasure/Arousal/Dominance):
  Evaluation↔Valence, Activity↔Arousal, **Potency↔Dominance** (the "third"
  axis). Near-isomorphic; both are *scalar magnitude* spaces — every emotion is a
  point answering *how much*.
- **Jealousy vs. envy** occupy nearly identical coordinates (negative, aroused,
  low-control) yet are different emotions. What separates them is **relational
  structure**, not magnitude:
  - envy = *dyadic*: self wants what other has → self is **on** the affect edge.
  - jealousy = *triadic*: self perceives affect on the **beloved↔rival** edge →
    self is **off** the edge, observing a flow it has a stake in.
- The discriminator is **the appraiser's position relative to the flow**
  (on-edge vs observing-an-edge-between-others), and the *arity* of the relation.
  Neither is a scalar; you cannot bolt on a "sociality 0–1" axis and recover it.

## 2. The model — emotion as edge-flux

Relocate the emotion from *node* / *whole-graph* to a **specific edge**, and from
*static topology* to **flow**. The object is an **affect-weighted directed
graph** where the EPA/polarity reads *are the edge weights*:

```
self → beloved      stake      (self's valence toward beloved)
beloved → rival     threat     (perceived displaced flow)   ← jealousy lives here
beloved → self      reassurance
```

`jealousy(self) ≈ f( perceived(beloved→rival) relative to (beloved→self),
                     scaled by stake(self→beloved) )`

This is a **contested sink**: two sources (self, rival) compete for the beloved's
affective output; jealousy is the perception the output routes to the rival. The
emotion is a **flux on an edge**, not a state of a node.

Generative bonus (hypothesis): varying edge sign / direction / where self sits
may generate the whole **social-emotion family** — schadenfreude (self off-edge,
negative affect arriving at other, positive response), empathy (self mirrors
other's node), vicarious pride (self on a positive edge to a third party's gain).
Scope honesty: this is the form of the **social** emotions, NOT all emotion —
fear/disgust/plain-joy are node-local or self-object dyads and don't need it.

## 3. Spectral treatment of the social dimension — what works, what doesn't

Spectral methods are the canonical bridge between a **relational graph** and
**continuous geometry**, so they are the right instinct. But the result splits
cleanly:

| Sub-problem | Spectral? | Why |
|---|---|---|
| **Arity / motif** (dyad vs triad, contested-sink shape) | ✅ | graph spectra encode topology. envy≈`K₂` spectrum `{0,2}`; jealousy≈`K₃` spectrum `{0,3,3}` |
| **Flow / direction** (displaced, asymmetric transfer) | ✅ (with the right operator) | use a **directed transfer operator** (random-walk / Perron–Frobenius / Koopman; Chung directed Laplacian). Non-symmetric ⇒ **complex** spectra whose imaginary parts encode circulation/direction |
| **Role assignment** (which entity IS the rival) | ❌ | spectra are **permutation-invariant** by construction — this is exactly what they erase. (also: cospectral graphs exist ⇒ spectrum isn't a complete invariant) |

Key upgrade from the conversation: framing the emotion as **transfer** (flow)
rather than **topology** (static shape) is what makes it spectrally tractable —
it forces a *directed* operator whose spectrum can see displaced flow, which the
undirected Laplacian erases. But role *identity* still needs a separate symbolic
typing step.

→ **Pipeline:** local affect reads → edge weights on an externalized relation
graph → transfer-operator spectrum (the structural/flow read) → symbolic role
labels (who's who). This is the project's recurring **"probe local, rebuild
global symbolically"** pattern (cf. `probe_locality_symbolic_global`).

## 4. Capacity — the monkey sphere as a *rank* budget

Dunbar ("monkey sphere") is **layered** (~5 / 15 / 50 / 150 / 500 / 1500, ~3×).
The edge-flux model *predicts* layering, because emotional capacities have
different combinatorial order under a fixed budget `B` (N ≈ B^(1/k)):

| Capacity | tracks | cost | sphere | fit |
|---|---|---|---|---|
| recognition / dyadic affect | nodes | O(N) | ~1500 | budget |
| relational ("who relates to whom") | edges among others | O(N²) | √1500≈39 | band ~50 |
| triadic flux (jealousy-grade) | edges between pairs of others | O(N³) | ∛1500≈11 | clique/sympathy ~5–15 |

**Honest caveats:** (1) Dunbar numbers are empirically shaky (Lindenfors et al.
2021 reanalysis → CI ~5–500); the *layered ~3× structure* replicates better than
the integers. (2) fixed fungible `B` is a loose assumption. (3) brains **don't**
store O(N²) edges — they **sparsify** (factions/cliques, transitive shortcuts).

That third caveat is the payoff: real social graphs are **sparse + community-
structured** ⇒ **low effective rank** ⇒ spectrally compressible. So the limit is
a **rank/compressibility budget, not a headcount.** Dunbar's number is the count
you reach at *typical* graph compressibility. **Prediction:** people in highly
cliqued networks sustain larger active spheres than people in flat
everyone-different networks (same brain, different rank budget). The monkey-sphere
limit and the theory-of-mind depth limit (recursion O(N^d), caps ~2–3) are the
same budget hitting two axes — breadth N and depth d.

## 5. Architecture — egocentric vs allocentric (the load-bearing choice)

Same split spatial cognition makes (self-rooted place map vs viewpoint-free grid
map). Two versions are two different objects:

### V1 — single-ego (egocentric, rooted at self) — **buildable now**
- `M[j]` = ego→j affect coords — **a `SemanticFrame` projection per entity (HAVE IT).**
- `R[j,k]` = ego's perceived affect on the j→k edge — **the hard part** (a
  *relational* read; expect the dyf×turnstyle relational-falsification wall →
  externalize entities first, then attach edge-affect).
- ego's emotions are functions of `R` (see §2).

### V2 — all-inputs-per-entity (allocentric/sociocentric tensor)
- `T[i,j]` = i→j affect; `T[i,j,k]` = i's perception of j→k.
- **Catch:** "all inputs for each entity" mostly don't exist as data — other
  minds are *latent*. So V2 is either:
  - **(a) a generative theory-of-mind simulator** (channels inferred, not read), or
  - **(b) a multi-agent simulation** where each agent IS a V1 ego and channels
    are real-but-synthetic → emergent gossip cascades, jealousy contagion,
    faction crystallization. **(b) is the buildable form of "all inputs."**
- Compression is **tensor** spectral (HOSVD/Tucker): `T ≈ consensus_graph ⊕
  per-ego residuals`. **The residuals are where the social emotions live** — the
  gap between ego's view and consensus is the seed of jealousy/paranoia/betrayal.
  Affect concentrates in the **high-residual (low-agreement)** entries.
- Cost of V2 = tensor rank along the **perceiver** axis = how much people
  **disagree**. Shared reality → cheap, big sphere. Divergent private worlds →
  high rank, saturates fast. *Disagreement is the expensive thing.*

The likely real design wants **both**: an egocentric map (where emotions are
felt) + an allocentric consensus graph (cheap, shareable). **Your emotions are
the residual between them.**

## 6. Build plan on turnstyle

**Phase 0 — primitives audit (cheap).** Confirm what we can read:
- node-affect per entity: ✅ `SemanticFrame` / `polarity` / EPA frame.
- edge-affect `R[j,k]`: ⚠️ unknown — design a probe; expect to need
  externalize-then-attach (entity extraction → edge typing → affect read).

**Phase 1 — V1 single-ego embedding (minimal).**
1. entity extraction from a social scene (self, alters).
2. node-affect via `SemanticFrame`.
3. edge-affect `R` (start with a crude rule / LLM-extracted relation + EPA on the
   relation phrase; measure against the relational-falsification baseline).
4. build the affect-weighted directed graph; compute the transfer-operator
   spectrum (random-walk `P` / Chung directed Laplacian).
5. read off social emotions (§2) and compare to labels.

**Phase 2 — minimal multi-agent (V2b).** A handful of V1 agents exchange
observations; build the per-ego residual tensor; test whether residual magnitude
predicts who develops jealousy. Tests the full stack without needing `R` read
from a real model's activations.

## 7. Validation & falsification (cheap-first)

- **F1 (spectral signature exists):** define the 3-node affect-weighted transfer
  operator for jealousy / envy / schadenfreude scenarios; does the **subdominant
  eigenvector separate "self-on-edge" from "self-observing"**? If not, the flux
  story is metaphor, not mechanism. *(cheapest — pure synthetic graphs, no model.)*
- **F2 (arity is spectral):** jealousy vs envy sentence pairs → attention-graph
  Laplacian spectrum over named entities → does **effective rank track arity**?
- **F3 (residual = emotion):** in V2b, does affect concentrate in high-residual
  tensor entries (low cross-ego agreement)?
- **F4 (rank budget):** does cliqued-network *effective rank* predict sustainable
  sphere size better than raw N?

Run F1 first — it's a few synthetic 3-node graphs and an eigendecomposition; it
gates whether any of the rest is worth building.

## 8. Open questions / risks

- **Edge-affect readability** is the crux risk (relational falsification). If
  `R[j,k]` can't be read even after externalization, V1 leans entirely on
  symbolic relation extraction + per-relation EPA, and the "embedding" is
  graph-structured, not a flat vector.
- **Attention ≠ information flow** — F2's attention-graph may not carry the
  relation cleanly (superposition across heads).
- **Directed spectral is messy** (non-normal operators, complex spectra) — keep
  the operator choice principled (transfer/random-walk) rather than ad hoc.
- Scope discipline: this is the **social** emotion family, not a theory of all
  emotion. Don't oversell.

## 9. Connections

- turnstyle primitives: `semantic_frame.BipolarAxis` / `SemanticFrame` (node
  affect), `polarity`, the bundled EPA/Osgood findings.
- memories: `osgood_epa_frame`, `epa_modulated_embeddings`,
  `subjectivity_cross_lingual_ordering`, `semantic_programming_direction`
  (predicted boundary: scalar extends, relational needs structure),
  `probe_locality_symbolic_global` (the pipeline pattern),
  `feedback_manifold_viz_laplacian` (spectral viz precedent).
- caution: the dyf×turnstyle **relational falsification** — scalar/similarity
  geometry is the wrong shape for n-ary relations; that wall applies to `R`.
