"""
Stage 2 training: NCSN++ score-based prior on PROBES at 256*256.
4-GPU DistributedDataParallel run.

Launch with torchrun:
    torchrun --standalone --nproc_per_node=4 train_prior.py \
        --data_dir ./data/gals_gband_norm \
        --output_dir ./output/probes_diffusion_prior \
        --image_size 256 --batch_size 4 --epochs 2700 \
        --lr 2e-4 --nf 128 --ch_mult 1 1 2 2 2 2 2

Defaults follow Adam et al. 2022:
    * effective batch size = batch_size * world_size = 4 * 4 = 16
    * ~2700 epochs * (2059 // 16) ≈ 350k optimization steps
    * sigma_max = 263.4 (their estimate from PROBES pairwise distances)
    * sigma_min = 1e-4
Pass `--sigma_max -1` to recompute the estimate from the loaded data.
"""

import os
import glob
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset

from score_models import ScoreModel, NCSNpp


# ----------------------------
# Distributed helpers
# ----------------------------
def setup_distributed():
    """Initialize torch.distributed when launched via torchrun, else fall back to single-process."""
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ----------------------------
# Data
# ----------------------------
def load_probes(path, n_subset=None, image_size=256, seed=21, verbose=True):
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
    if verbose:
        print(f'Loaded: {images.shape}, range=[{images.min():.4g}, {images.max():.4g}]')

    if images.shape[1] != image_size or images.shape[2] != image_size:
        t = torch.from_numpy(images).unsqueeze(1)
        t = F.interpolate(t, size=(image_size, image_size),
                          mode='bilinear', align_corners=False)
        images = t.squeeze(1).numpy()
        if verbose:
            print(f'Resized to {image_size}x{image_size}')

    if verbose:
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
    parser.add_argument('--image_size', type=int, default=256)

    # Architecture
    parser.add_argument('--nf', type=int, default=128)
    parser.add_argument('--ch_mult', type=int, nargs='+',
                        default=[1, 1, 2, 2, 2, 2, 2],
                        help='Yang Song reference NCSN++ for 256x256 '
                             '(7 levels: 256->128->64->32->16->8->4)')

    # SDE
    parser.add_argument('--sigma_min', type=float, default=1e-4)
    parser.add_argument('--sigma_max', type=float, default=263.4,
                        help='set <0 to auto-estimate from data; '
                             '263.4 matches Adam et al. 2022 PROBES estimate')

    # Training
    parser.add_argument('--epochs', type=int, default=2700,
                        help='~350k steps at effective bs=16 on 2059 galaxies')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='per-GPU batch size; effective = batch_size * world_size '
                             '(paper uses total bs=16 on 4 GPUs)')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=21)

    # Optimization / regularization
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--warmup', type=int, default=0,
                        help='linear LR warmup in optimizer steps')
    parser.add_argument('--clip', type=float, default=0.0,
                        help='gradient-norm clip; 0 disables')

    # Wallclock / checkpointing
    parser.add_argument('--max_hours', type=float, default=float('inf'),
                        help='stop training after this many hours and save a checkpoint')
    parser.add_argument('--ckpt_every_steps', type=int, default=0,
                        help='checkpoint cadence in optimizer steps; '
                             'converted to epoch cadence using steps_per_epoch (0 = library default)')
    parser.add_argument('--log_every_steps', type=int, default=0,
                        help='accepted for compatibility; library logs per-epoch only. '
                             '>0 enables verbose=1 epoch logging')
    parser.add_argument('--keep_last_n', type=int, default=2,
                        help='maps to models_to_keep')

    args = parser.parse_args()

    # ---- Distributed setup ----
    local_rank, rank, world_size = setup_distributed()
    main_proc = is_main(rank)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    if main_proc:
        print(f'World size: {world_size} | rank: {rank} | local_rank: {local_rank}')
        print(f'Using device: {device}')
        if device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(local_rank)}')
            print(f'Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.2f} GB')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    if main_proc:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Data ----
    n_subset = None if args.n_subset == -1 else args.n_subset
    images = load_probes(args.data_dir,
                         n_subset=n_subset,
                         image_size=args.image_size,
                         verbose=main_proc)
    dataset = ProbesDataset(images, device=device)
    if main_proc:
        print(f'Dataset: {len(dataset)} images on {dataset.images.device}')

    # ---- Sigma_max ----
    if args.sigma_max < 0:
        sigma_max = estimate_sigma_max(dataset)
    else:
        sigma_max = args.sigma_max
    if main_proc:
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
    if main_proc:
        n_params = sum(p.numel() for p in model.parameters())
        print(f'NCSN++: {n_params:,} parameters')

    # ---- Steps estimate ----
    effective_bs = args.batch_size * world_size
    steps_per_epoch = max(1, len(dataset) // effective_bs)
    total_steps = steps_per_epoch * args.epochs
    if main_proc:
        print(f'Plan: {args.epochs} epochs × {steps_per_epoch} steps '
              f'= {total_steps:,} steps (per-GPU bs={args.batch_size}, effective bs={effective_bs})')

    # Library cadence is per-epoch; convert step-based flag if provided.
    if args.ckpt_every_steps > 0:
        checkpoints_every_epochs = max(1, round(args.ckpt_every_steps / steps_per_epoch))
    else:
        checkpoints_every_epochs = 10

    # ---- Train ----
    if main_proc:
        print('\nTraining...')
        print(f'Checkpoint every {checkpoints_every_epochs} epochs '
              f'(~{checkpoints_every_epochs * steps_per_epoch} steps), '
              f'keep last {args.keep_last_n}, max_time={args.max_hours} h')
    t0 = time.time()
    model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ema_decay=args.ema_decay,
        warmup=args.warmup,
        clip=args.clip,
        max_time=args.max_hours,
        checkpoints=checkpoints_every_epochs,
        models_to_keep=args.keep_last_n,
        checkpoints_directory=str(output_dir),
        seed=args.seed,
        verbose=1 if args.log_every_steps > 0 else 0,
    )
    elapsed = time.time() - t0
    if main_proc:
        print(f'\nTraining complete in {elapsed/3600:.2f} h')
        print(f'Checkpoints: {output_dir}')

    cleanup_distributed()


if __name__ == '__main__':
    main()
