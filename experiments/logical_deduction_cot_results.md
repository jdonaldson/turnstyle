Loading HuggingFaceTB/SmolLM2-1.7B-Instruct on mps…
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/218 [00:00<?, ?it/s]Loading weights:   0%|          | 1/218 [00:00<00:34,  6.23it/s]Loading weights:  14%|█▍        | 30/218 [00:00<00:01, 129.80it/s]Loading weights:  22%|██▏       | 48/218 [00:00<00:01, 126.01it/s]Loading weights:  30%|███       | 66/218 [00:00<00:01, 126.24it/s]Loading weights:  39%|███▊      | 84/218 [00:00<00:01, 126.36it/s]Loading weights:  47%|████▋     | 102/218 [00:00<00:00, 131.82it/s]Loading weights:  55%|█████▌    | 120/218 [00:00<00:00, 132.09it/s]Loading weights:  63%|██████▎   | 138/218 [00:01<00:00, 134.96it/s]Loading weights:  72%|███████▏  | 156/218 [00:01<00:00, 127.61it/s]Loading weights:  80%|███████▉  | 174/218 [00:01<00:00, 122.35it/s]Loading weights:  88%|████████▊ | 192/218 [00:01<00:00, 128.49it/s]Loading weights:  96%|█████████▋| 210/218 [00:01<00:00, 130.36it/s]Loading weights: 100%|██████████| 218/218 [00:01<00:00, 129.67it/s]

──────────────────────────────────────────────────────────────────────
logical_deduction_three_objects  (n=50)
──────────────────────────────────────────────────────────────────────
  ex      T1        T2      T3  note
   0       ✗        ok      ok  T1=None  T2=ok  T3=ok
   1      ok        ok      ok  
   2       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
   3       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
   4       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
   5       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
   6       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
   7       ✗         ✗      ok  T1=None  T2=gt=3/cot=3  T3=ok
   8       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
   9       ✗        ok      ok  T1=None  T2=ok  T3=ok
  10       ✗        ok      ok  T1=None  T2=ok  T3=ok
  11       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  12       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  13       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  14      ok         ✗      ok  T1=ok  T2=gt=3/cot=3  T3=ok
  15       ✗        ok      ok  T1=None  T2=ok  T3=ok
  16       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  17       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  18       ✗        ok      ok  T1=None  T2=ok  T3=ok
  19      ok         ✗       ✗  T1=ok  T2=gt=3/cot=3  T3=(C)
  20       ✗        ok      ok  T1=None  T2=ok  T3=ok
  21       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  22       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  23       ✗         ✗      ok  T1=None  T2=gt=3/cot=3  T3=ok
  24       ✗         ✗      ok  T1=None  T2=gt=3/cot=3  T3=ok
  25       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
  26       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  27       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  28       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(C)
  29       ✗        ok      ok  T1=None  T2=ok  T3=ok
  30       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
  31       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
  32       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
  33      ok         ✗       ✗  T1=ok  T2=gt=3/cot=3  T3=(B)
  34       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  35       ✗        ok      ok  T1=None  T2=ok  T3=ok
  36       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  37       ✗        ok      ok  T1=None  T2=ok  T3=ok
  38       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(B)
  39       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  40       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  41       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  42       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  43       ✗         ✗       ✗  T1=None  T2=gt=3/cot=0  T3=fail
  44       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  45       ✗        ok      ok  T1=None  T2=ok  T3=ok
  46       ✗         ✗       ✗  T1=None  T2=gt=3/cot=3  T3=(A)
  47       ✗         ✗      ok  T1=None  T2=gt=3/cot=3  T3=ok
  48       ✗        ok      ok  T1=None  T2=ok  T3=ok
  49       ✗        ok      ok  T1=None  T2=ok  T3=ok

Test 1 (CoT answer):        4/50  (8%)
Test 2 (order extraction):  13/50  (26%)
Test 3 (intercept+correct): 18/50  (36%)
  order length mismatches: {'gt=3 cot=3': 31, 'gt=3 cot=0': 6}

──────────────────────────────────────────────────────────────────────
logical_deduction_five_objects  (n=50)
──────────────────────────────────────────────────────────────────────
  ex      T1        T2      T3  note
   0       ✗         ✗       ✗  T1=(E)  T2=gt=5/cot=5  T3=(E)
   1       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(A)
   2       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   3       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   4       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   5       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   6       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   7       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   8       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
   9       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  10       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  11       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(E)
  12       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  13       ✗         ✗      ok  T1=None  T2=gt=5/cot=5  T3=ok
  14       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  15       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(D)
  16       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  17       ✗         ✗      ok  T1=None  T2=gt=5/cot=5  T3=ok
  18       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  19      ok         ✗       ✗  T1=ok  T2=gt=5/cot=5  T3=(A)
  20       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(A)
  21       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  22       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(A)
  23       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  24       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  25       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  26      ok         ✗       ✗  T1=ok  T2=gt=5/cot=0  T3=fail
  27       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  28      ok         ✗       ✗  T1=ok  T2=gt=5/cot=0  T3=fail
  29       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  30       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(E)
  31       ✗         ✗      ok  T1=None  T2=gt=5/cot=5  T3=ok
  32       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(B)
  33       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  34       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  35       ✗         ✗      ok  T1=None  T2=gt=5/cot=5  T3=ok
  36       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  37       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  38       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  39       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(A)
  40       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  41       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(C)
  42       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(D)
  43       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  44       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(C)
  45       ✗         ✗      ok  T1=None  T2=gt=5/cot=5  T3=ok
  46       ✗         ✗       ✗  T1=None  T2=gt=5/cot=5  T3=(B)
  47       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  48       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail
  49       ✗         ✗       ✗  T1=None  T2=gt=5/cot=0  T3=fail

Test 1 (CoT answer):        3/50  (6%)
Test 2 (order extraction):  0/50  (0%)
Test 3 (intercept+correct): 5/50  (10%)
  order length mismatches: {'gt=5 cot=0': 31, 'gt=5 cot=5': 19}

──────────────────────────────────────────────────────────────────────
logical_deduction_seven_objects  (n=50)
──────────────────────────────────────────────────────────────────────
  ex      T1        T2      T3  note
   0       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   1       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   2       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(F)
   3       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   4       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   5       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(C)
   6      ok         ✗       ✗  T1=ok  T2=gt=7/cot=7  T3=(A)
   7       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   8       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
   9       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  10       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  11       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  12       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  13       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  14       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  15       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  16       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(G)
  17       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  18       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  19       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  20       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  21       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  22       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  23       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(D)
  24       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  25       ✗         ✗       ✗  T1=(C)  T2=gt=7/cot=0  T3=fail
  26       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  27       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(D)
  28       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  29       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  30       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  31       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(A)
  32       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  33       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  34       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  35       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  36       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  37       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  38       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  39       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  40       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  41       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  42       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  43       ✗         ✗       ✗  T1=(C)  T2=gt=7/cot=0  T3=fail
  44       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(G)
  45       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(G)
  46       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  47       ✗         ✗       ✗  T1=None  T2=gt=7/cot=7  T3=(C)
  48       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail
  49       ✗         ✗       ✗  T1=None  T2=gt=7/cot=0  T3=fail

Test 1 (CoT answer):        1/50  (2%)
Test 2 (order extraction):  0/50  (0%)
Test 3 (intercept+correct): 0/50  (0%)
  order length mismatches: {'gt=7 cot=0': 40, 'gt=7 cot=7': 10}
