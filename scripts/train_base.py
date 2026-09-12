"""Minimal training script stub for a base model (placeholder, CPU-friendly)."""
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='vgg19 | densenet201 | mobilenet_v2')
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Training {args.model} with config: {cfg}")
    print("NOTE: This is a stub. Implement training loop in src/training or scripts/train_base.py")
