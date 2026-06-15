from __future__ import annotations

# Bump this string whenever an implementation-level optimization changes the
# exact generation code path in a way that should not be mixed with older
# output files during resume/skip.
IMPLEMENTATION_PROFILE = "safe_opt_legacy_kv_cache_v1"
