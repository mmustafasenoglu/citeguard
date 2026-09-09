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

## Turkish capability

- **STS Benchmark Turkish**: external machine translation of STS Benchmark;
  official train/dev/test files; semantic-similarity labels; license was not
  stated in the acquired repository, so raw data is not redistributed.
- **Turkish STS Dataset**: 81 human-rated pairs, CC BY-SA 4.0, used only as a
  small independent STS sanity check; raw data is not committed.
- **Turkish detection engineering**: 324 CiteGuard-authored cases across nine
  academic domains, split deterministically with seed 20260909. It measures
  regression behavior, not real-world plagiarism accuracy.
- **Turkish rewrite safety**: 210 deterministic cases exercising independent
  safety gates. It is an engineering suite, not population-level evidence.
