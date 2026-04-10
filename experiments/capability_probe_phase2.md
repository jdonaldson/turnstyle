# Phase 2 — Router Evaluation

Sequential (production order) vs probe-routed vs oracle. Router uses out-of-fold CV probabilities.

| task | n | sequential | routed | oracle | gap closed | avg tries |
|---|---|---|---|---|---|---|
| `penguins_in_a_table` | 143 | 0.944 | 0.881 | 0.958 | -450.0% | 1.05 |
| `tracking_shuffled_objects_three_objects` | 247 | 0.300 | 0.279 | 0.615 | -6.4% | 1.00 |
| `object_counting` | 247 | 0.271 | 0.320 | 0.336 | +75.0% | 1.00 |
| `navigate` | 247 | 0.798 | 0.753 | 0.862 | -68.7% | 1.03 |
| `web_of_lies` | 247 | 0.486 | 0.486 | 0.486 | +0.0% | 1.00 |