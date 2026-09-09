# NLI v2 claims

- Four local NLI models were evaluated on the same official English SNLI and Turkish SNLI-TR
  development splits.
- The synthetic grouped rewrite suite is an engineering safety signal, not evidence that a model is
  safe for production.
- No architecture is promoted unless the official English budget and every safety gate pass.
- This run does not make a 1% production-safety claim; validation and holdout remain untouched after
  the external gate decision.
