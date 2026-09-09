# External benchmark protocol

Raw external datasets are never committed.  Acquire them directly from their
rights holders and supply `--dataset-dir` locally.

| Benchmark | Purpose | Source | Labels | Split | Limitation |
| --- | --- | --- | --- | --- | --- |
| PAN-PC-11 | Document/source overlap detection | Webis/PAN | passage/source spans | deterministic document-level seed 20260909 when official partitions are unsuitable | external corpus required |
| Webis-CPC-11 | Paraphrase versus non-paraphrase characterization | Webis | pair labels | dataset-defined | not a plagiarism dataset |
| Reduction engineering | Rewrite safety regressions | CiteGuard-authored | deterministic safety labels | fixed | not external validation |
| Plagiarism engineering | Exact/lexical/attribution/localization smoke | CiteGuard-authored bilingual synthetic cases | source-aware labels | fixed | not external validation |

Threshold selection is development-only: maximize F1 subject to precision at
least 0.90.  The holdout is run only after selected thresholds are recorded.
PAN scores must state the dataset, split, sample size, and metric; they must
not be described as a generic "plagiarism accuracy".

Run the plagiarism engineering smoke set with:

```bash
python scripts/benchmarks/run_plagiarism_engineering.py \
  --output benchmarks/results/plagiarism_engineering.json
```

Exact and lexical precision/recall/F1, attribution precision/recall/F1, and passage localization
are reported independently. Optional semantic Recall@1/3/5 remains measured by the existing
multilingual semantic retrieval benchmark because downloading a model is not part of hermetic CI.

## Turkish capability

- **STS Benchmark Turkish**: external machine translation of STS Benchmark;
  official train/dev/test files; semantic-similarity labels; license was not
  stated in the acquired repository, so raw data is not redistributed.
- **Turkish STS Dataset**: 81 human-rated pairs, CC BY-SA 4.0, used only as a
  small independent STS sanity check; raw data is not committed.
- **Turkish detection engineering v2**: 840 CiteGuard-authored cases across nine
  academic domains, split by source/template group with seed 20260909. The locked
  168-case engineering holdout achieved precision 1.000, recall 0.875, and F1
  0.933 for textual overlap. It measures regression behavior, not real-world
  plagiarism accuracy.
- **Turkish rewrite safety v2**: 280 valid deterministic cases exercising independent
  safety gates, with 168 development, 56 validation, and 56 holdout cases. It is an
  engineering suite, not population-level evidence.
- **Turkish semantic retrieval**: the selected
  `intfloat/multilingual-e5-small` model (MIT) achieved Recall@1 0.717, Recall@3 0.950,
  Recall@5 0.983, and MRR 0.836 on 60 close-paraphrase queries. Semantic candidates
  never create textual spans or increase textual-overlap percentages.
- **NLI v2**: production retains `cross-encoder/nli-deberta-v3-base` (Apache-2.0);
  the evaluated MoritzLaurer alternatives are MIT-licensed and were not promoted
  because they exceeded English regression budgets.
