# Outputs

Full generations, token-level logs, and raw metric dumps are not tracked by normal Git.

Recommended layout:

```text
outputs/
  generations/
  token_logs/
  raw_metrics/
```

Use DVC for versioning large artifacts when needed.

Do not store full logits or full probability vectors unless there is a specific analysis need; they become very large quickly.
