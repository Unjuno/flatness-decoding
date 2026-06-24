#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_dir="${root_dir}/artifacts"
stamp="$(date +%Y%m%d_%H%M%S)"
archive_name="writingprompts_artifacts_${stamp}.tar.gz"
archive_path="${artifact_dir}/${archive_name}"
checksum_path="${archive_path}.sha256"

required_paths=(
  "outputs/generations/comparison_distilgpt2_writingprompts"
  "outputs/token_logs/comparison_distilgpt2_writingprompts"
  "outputs/generations/comparison_gpt2_writingprompts"
  "outputs/token_logs/comparison_gpt2_writingprompts"
  "results/summary"
  "results/figures"
  "results/analysis"
  "prompts/writingprompts_fixed_100.jsonl"
  "configs"
)

for rel in "${required_paths[@]}"; do
  if [[ ! -e "${root_dir}/${rel}" ]]; then
    printf 'missing required artifact path: %s\n' "${rel}" >&2
    exit 1
  fi
done

mkdir -p "${artifact_dir}"

tar -C "${root_dir}" -czf "${archive_path}" \
  outputs/generations/comparison_distilgpt2_writingprompts \
  outputs/token_logs/comparison_distilgpt2_writingprompts \
  outputs/generations/comparison_gpt2_writingprompts \
  outputs/token_logs/comparison_gpt2_writingprompts \
  results/summary \
  results/figures \
  results/analysis \
  prompts/writingprompts_fixed_100.jsonl \
  configs

if command -v sha256sum >/dev/null 2>&1; then
  checksum_cmd=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  checksum_cmd=(shasum -a 256)
else
  printf 'missing checksum tool: need sha256sum or shasum -a 256\n' >&2
  exit 1
fi

"${checksum_cmd[@]}" "${archive_path}" > "${checksum_path}"

archive_size="$(du -h "${archive_path}" | awk '{print $1}')"
printf 'archive: %s\n' "${archive_path}"
printf 'checksum: %s\n' "${checksum_path}"
printf 'size: %s\n' "${archive_size}"
printf 'archive created successfully; not committed automatically\n'
