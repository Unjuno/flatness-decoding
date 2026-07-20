# Data

Raw datasets and caches are not tracked by normal Git.

Use this directory for local data only, or track large data artifacts with DVC.

Recommended layout:

```text
data/
  raw/
  cache/
  processed/
```

The raw dataset itself should usually be downloaded from its original source rather than committed to this repository.
This repository does not redistribute the original WritingPrompts dataset or
its reference continuations. Tracked prompt files are reduced public subsets
used to document experiment inputs; users who need the original dataset or
reference-based metrics must obtain it separately and follow its terms.
