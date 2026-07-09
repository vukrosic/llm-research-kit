# Contributing

This repo is open-source research, not a finished model package.

The best contributions make the claims easier to check. A good PR usually does one of these:

- verifies that the implementation matches the reference attention mechanism;
- adds a diagnostic that exposes routing behavior;
- runs a small ablation with a reproducible command and saved result;
- improves the paper or README so the research question is clearer;
- fixes a test or adds a missing correctness check.

## Research Standard

Every experimental PR should include:

- the exact command used;
- the device used;
- the seed and model config;
- the dataset or synthetic task;
- a JSON, CSV, table, or plot artifact;
- a short interpretation that says what changed and what did not.

Do not overclaim small smoke runs. If a result is preliminary, say so.

## Useful First Issues

Start with issues labeled `contributor-friendly`, `correctness`, `diagnostics`, or `paper`.

The most valuable early work is:

- implementation audit against the reference image;
- dense-oracle selected-block recall;
- long-context retrieval benchmark;
- local-plus-routed attention ablation;
- reproducible run manifest improvements.

## Running Tests

```bash
python -m unittest discover -s tests
```

## Building The Paper

```bash
docs/build_short_paper.sh
```

The compiled PDF is written to `sparse_attention_short_paper.pdf` at the repository root.
