# Supported Turkish claims

On CiteGuard's 324-case Turkish engineering detection suite, the locked
65-case holdout achieved precision 1.000, recall 0.865, and F1 0.928 using the
development-selected textual threshold. This is internal regression evidence,
not real-world plagiarism accuracy.

On the validated 280-case Turkish engineering rewrite-safety suite, all constructed safe
candidates were accepted and all constructed unsafe candidates were rejected.
This deterministic result does not estimate population performance.

On Turkish detection benchmark v2, the locked 168-case group-isolated holdout
achieved precision 1.000, recall 0.875, and F1 0.933 for textual overlap. Its
close-paraphrase textual recall was 0.000. With the separately selected
`intfloat/multilingual-e5-small` embedding model, semantic candidate retrieval
on the 60 close-paraphrase queries achieved Recall@1 0.717, Recall@3 0.950,
Recall@5 0.983, and MRR 0.836. Semantic retrieval does not create textual spans
or increase the textual-overlap percentage.

The selected semantic model achieved Pearson 0.768 and Spearman 0.763 on the
official translated Turkish STS test split. On the 81-pair human-annotated
sanity set it achieved Pearson 0.822 and Spearman 0.808. These correlations
measure semantic capability, not plagiarism detection accuracy.

The current `cross-encoder/nli-deberta-v3-base` backend achieved accuracy
0.507, macro-F1 0.490, and contradiction recall 0.295 on the 9,824-case
SNLI-TR official test split. This is weak Turkish entailment evidence and does
not justify a positive production-capability claim or a model change by itself.

The multilingual `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` alternative improved
development accuracy to 0.730, macro-F1 to 0.725, and contradiction recall to
0.616. It was not promoted because external English regression evidence was not
part of this benchmark run.

No external Turkish plagiarism-performance claim is supported. STS results
measure correlation with semantic-similarity judgements, not plagiarism.
