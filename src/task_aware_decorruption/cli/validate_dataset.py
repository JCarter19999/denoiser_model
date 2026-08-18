import argparse
from collections import Counter

from task_aware_decorruption.data.acdc import discover_acdc_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    samples = discover_acdc_pairs(args.root, args.split)
    counts = Counter(sample.condition for sample in samples)
    print(f"samples={len(samples)}")
    for condition, count in sorted(counts.items()):
        print(f"{condition}={count}")


if __name__ == "__main__":
    main()
