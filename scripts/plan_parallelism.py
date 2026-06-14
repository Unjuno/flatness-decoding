from __future__ import annotations

import argparse
import json

from flatness_decoding.parallelism import detect_parallel_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", default="auto")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = detect_parallel_plan(
        requested_num_shards=args.num_shards,
        requested_workers=args.workers,
    )
    data = plan.to_dict()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print("Parallel execution plan")
    print(f"backend: {plan.backend}")
    print(f"cpu_count: {plan.cpu_count}")
    print(f"cuda_available: {plan.cuda_available}")
    print(f"cuda_device_count: {plan.cuda_device_count}")
    print(f"mps_available: {plan.mps_available}")
    print(f"recommended_workers: {plan.recommended_workers}")
    print(f"recommended_num_shards: {plan.recommended_num_shards}")
    print(f"recommended_devices: {plan.recommended_devices}")
    print(f"caution: {plan.caution}")


if __name__ == "__main__":
    main()
