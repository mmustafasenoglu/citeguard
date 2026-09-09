# External benchmark protocol

Raw external datasets are never committed.  Acquire them directly from their
rights holders and supply `--dataset-dir` locally.

| Benchmark | Purpose | Source | Labels | Split | Limitation |
| --- | --- | --- | --- | --- | --- |
| PAN-PC-11 | Document/source overlap detection | Webis/PAN | passage/source spans | deterministic document-level seed 20260909 when official partitions are unsuitable | external corpus required |
| Webis-CPC-11 | Paraphrase versus non-paraphrase characterization | Webis | pair labels | dataset-defined | not a plagiarism dataset |
| Reduction engineering | Rewrite safety regressions | CiteGuard-authored | deterministic safety labels | fixed | not external validation |

Threshold selection is development-only: maximize F1 subject to precision at
least 0.90.  The holdout is run only after selected thresholds are recorded.
PAN scores must state the dataset, split, sample size, and metric; they must
not be described as a generic "plagiarism accuracy".
