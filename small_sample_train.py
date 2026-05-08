"""
Stage 2 training: NCSN++ score-based prior on PROBES at 128*128.
Single-GPU run, ~32k optimization steps.

Usage:
    python small_sample_train.py --data_dir ./data/gals_gband_norm \
                                 --output_dir ./output/probes_diffusion_subset \
                                 --image_size 128 --batch_size 64 --epochs 1000 \
                                 --lr 2e-4 --nf 128 --ch_mult 1 2 2 2
"""

import os
import glob
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from score_models import ScoreModel, NCSNpp

# ----------------------------
# Data
# ----------------------------
def load_probes(path, n_subset=None, image_size=128, seed=21):
    """Load preprocessed PROBES g-band .npy files in [-1, 1]"""
    rng = np.random.RandomState(seed)
    files = sorted(glob.glob(str(Path(path) / "*.npy")))
    if not files:
        raise ValueError(f"No .npy files found in {path}")
    
    if n_subset and n_subset < len(files):
        files = [files[i] for i in rng.choice(len(files), size=n_subset, replace=False)]
    
    images = np.stack([np.load(f) for f in files]).astype(np.float32)
    if images.ndim == 4:
        images = images[:, 0]
    print(f'Loaded: {images.shape}, range=[{images.min():.4g}, {images.max():.4g}]')

    if images.shape[1] != image_size or images.shape[2] != image_size:
        t = torch.from_numpy(images).unsqueeze(1)
        t = F.interpolate(t, size=(image_size, image_size),
                          mode='bilinear', align_corners=False)
        images = t.squeeze(1).numpy()
        print(f'Resized to {image_size}x{image_size}')

    print(f'Final: {images.shape}, range=[{images.min():.4f}, {images.max():.4f}]')
    return images

class ProbesDataset(Dataset):
    """Tensors (N, 1, H, W) in [-1, 1], pre-moved to device"""
    def __init__(self, images, device=None):
        t = torch.from_numpy(images).float().unsqueeze(1)
        self.images = t.to(device) if device is not None else t

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
# ----------------------------
# Sigma_max heuristic
# ----------------------------

def estimate_sigma_max(dataset, n_pairs=5000, seed=21):
    flat = dataset.images.view(len(dataset), -1)
    rng = np.random.RandomState(seed)
    pairs = rng.randint(0, len(flat), (n_pairs, 2))
    sigma_max = max((flat[i] - flat[j]).norm().item() for i, j in pairs)
    return sigma_max

# ----------------------------
# Main (training loop)
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_subset', type=int, default=-1,
                        help='-1 for full dataset')
    parser.add_argument('--image_size', type=int, default=128)

    # Architecture
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 2])

    # SDE
    parser.add_argument('--sigma_min', type=float, default=1e-4)
    parser.add_argument('--sigma_max', type=float, default=-1,
                        help='-1 to auto-estimate from data')

    # Training
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=21)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    n_subset = None if args.n_subset == -1 else args.n_subset
    images = load_probes(args.data_dir,
                         n_subset=n_subset,
                         image_size=args.image_size)
    dataset = ProbesDataset(images, device=device)
    print(f'Dataset: {len(dataset)} images on {dataset.images.device}')

    # ---- Sigma_max ----
    if args.sigma_max < 0:
        sigma_max = estimate_sigma_max(dataset)
    else:
        sigma_max = args.sigma_max
    print(f'VE SDE: sigma_min={args.sigma_min:.1e}, sigma_max={sigma_max:.2f}')

    # ---- Model ----
    net = NCSNpp(
        channels = 1,
        nf = args.nf,
        ch_mult = args.ch_mult,
        dimensions = 2,
    )
    model = ScoreModel(
        model = net,
        sigma_min = args.sigma_min,
        sigma_max = sigma_max,
        device = device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'NCSN++: {n_params:,} parameters')

    # ---- Steps estimate ----
    steps_per_epoch = max(1, len(dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f'Plan: {args.epochs} epochs × {steps_per_epoch} steps = {total_steps:,} steps')

    # ---- Train ----
    print('\nTraining...')
    t0 = time.time()
    model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoints_directory=str(output_dir),
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f'\nTraining complete in {elapsed/3600:.2f} h')
    print(f'Checkpoints: {output_dir}')


if __name__ == '__main__':
    main()