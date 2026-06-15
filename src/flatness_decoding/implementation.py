from __future__ import annotations

# Bump this string whenever an implementation-level optimization changes the
# exact generation code path in a way that should not be mixed with older
# output files during resume/skip.
IMPLEMENTATION_PROFILE = "safe_opt_full_prefix_microbatch_v1"

# Official experiment path. Cached lookahead is disabled because it did not
# pass full-prefix equivalence in the current GPT-2/Transformers stack.
USE_KV_CACHE_LOOKAHEAD = False
