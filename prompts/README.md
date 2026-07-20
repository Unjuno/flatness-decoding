# Prompts

This directory stores small prompt subsets and prompt ID lists that are safe to track in Git.

Large raw datasets should not be committed directly. Use dataset download scripts or DVC-managed storage instead.

The tracked WritingPrompts files contain prompt text and provenance fields only;
reference story continuations are intentionally excluded from this public
archive. Obtain the source dataset separately when reproducing reference-based
metrics, subject to its terms.

Expected JSONL schema:

```json
{"id": "toy_001", "prompt": "Once upon a time"}
```
