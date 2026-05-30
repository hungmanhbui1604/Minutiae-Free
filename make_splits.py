import argparse
from data import create_recog_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="data/recog_splits.json")
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    splits = create_recog_splits(args.data_root, args.output, min_samples=args.min_samples, seed=args.seed)
    print(splits)
