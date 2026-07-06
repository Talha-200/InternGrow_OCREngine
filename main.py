"""
CLI entry point.

Usage:
    python main.py --epochs 15
"""
from src.train import parse_args, run_training

if __name__ == "__main__":
    args = parse_args()
    run_training(epochs=args.epochs)
