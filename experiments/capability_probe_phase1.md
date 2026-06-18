# Phase 1 — Probe Training at L18

5-fold CV, LogReg with `class_weight='balanced'`.

| task | tier | p(succ) | acc | baseline | gain | AUC | Brier |
|---|---|---|---|---|---|---|---|
| `penguins_in_a_table` | sql | 0.91 | 0.923 | 0.909 | +0.014 | 0.920 | 0.068 |
| `penguins_in_a_table` | knowledge_poll | 0.08 | 0.853 | 0.916 | -0.063 | 0.841 | 0.118 |
| `penguins_in_a_table` | logit_poll | 0.23 | 0.713 | 0.769 | -0.056 | 0.639 | 0.241 |
| `penguins_in_a_table` | baseline | 0.23 | 0.755 | 0.769 | -0.014 | 0.684 | 0.200 |
| `tracking_shuffled_objects_three_objects` | sql | 0.00 | 1.000 | 1.000 | +0.000 | nan | 0.000 |
| `tracking_shuffled_objects_three_objects` | logit_poll | 0.30 | 0.603 | 0.700 | -0.097 | 0.583 | 0.314 |
| `tracking_shuffled_objects_three_objects` | baseline | 0.32 | 0.555 | 0.684 | -0.130 | 0.449 | 0.382 |
| `object_counting` | sql | 0.27 | 0.968 | 0.729 | +0.239 | 0.976 | 0.028 |
| `object_counting` | baseline | 0.06 | 0.911 | 0.935 | -0.024 | 0.656 | 0.073 |
| `navigate` | ir | 0.78 | 0.769 | 0.777 | -0.008 | 0.789 | 0.193 |
| `navigate` | baseline | 0.51 | 0.599 | 0.510 | +0.089 | 0.650 | 0.312 |
| `web_of_lies` | ir | 0.00 | 1.000 | 1.000 | +0.000 | nan | 0.000 |
| `web_of_lies` | baseline | 0.49 | 0.526 | 0.514 | +0.012 | 0.511 | 0.410 |