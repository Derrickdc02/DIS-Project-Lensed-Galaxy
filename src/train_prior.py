"""
DDP training for the 256x256 PROBES score-based prior.

This script intentionally does not call ScoreModel.fit().  The score_models
library provides the NCSN++ architecture, VE SDE setup, and DSM loss through
ScoreModel.loss_fn(), while this file owns distributed data loading, optimizer
steps, EMA, checkpointing, and resume.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from score_models import NCSNpp, ScoreModel


def setup_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def distributed_any(flag, device):
    tensor = torch.tensor(int(flag), device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def load_probes(path, n_subset=None, image_size=256, seed=21, verbose=True):
    """Load preprocessed PROBES .npy files as arrays with shape (N, H, W)."""
    rng = np.random.RandomState(seed)
    files = sorted(glob.glob(str(Path(path) / "*.npy")))
    if not files:
        raise ValueError(f"No .npy files found in {path}")

    if n_subset is not None and n_subset > 0 and n_subset < len(files):
        subset = rng.choice(len(files), size=n_subset, replace=False)
        files = [files[i] for i in subset]

    images = np.stack([np.load(f) for f in files]).astype(np.float32)
    if images.ndim == 4:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"Expected images with shape (N, H, W), got {images.shape}")

    if images.shape[1] != image_size or images.shape[2] != image_size:
        t = torch.from_numpy(images).unsqueeze(1)
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        images = t.squeeze(1).numpy()
        if verbose:
            print(f"Resized images to {image_size}x{image_size}")

    if verbose:
        print(
            f"Loaded {len(images)} images, shape={images.shape}, "
            f"range=[{images.min():.4f}, {images.max():.4f}]"
        )
    return images


class ProbesDataset(Dataset):
    """CPU tensor dataset. Batches are moved to GPU inside the training loop."""

    def __init__(self, images):
        self.images = torch.from_numpy(images).float().unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def estimate_sigma_max(images, n_pairs=5000, seed=21):
    """Estimate max pairwise Euclidean distance on the normalized images."""
    flat = torch.from_numpy(images).float().view(len(images), -1)
    rng = np.random.RandomState(seed)
    pairs = rng.randint(0, len(flat), size=(n_pairs, 2))
    sigma_max = 0.0
    for i, j in pairs:
        sigma_max = max(sigma_max, (flat[i] - flat[j]).norm().item())
    return sigma_max


class EMA:
    def __init__(self, parameters, decay):
        params = [p for p in parameters if p.requires_grad]
        self.decay = decay
        self.shadow = [p.detach().clone() for p in params]
        self.backup = None

    @torch.no_grad()
    def update(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for shadow, param in zip(self.shadow, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def store(self, parameters):
        self.backup = [p.detach().clone() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def copy_to(self, parameters):
        params = [p for p in parameters if p.requires_grad]
        for param, shadow in zip(params, self.shadow):
            param.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters):
        if self.backup is None:
            return
        params = [p for p in parameters if p.requires_grad]
        for param, backup in zip(params, self.backup):
            param.copy_(backup)
        self.backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": [shadow.detach().cpu() for shadow in self.shadow],
        }

    def load_state_dict(self, state, parameters):
        params = [p for p in parameters if p.requires_grad]
        self.decay = float(state["decay"])
        self.shadow = [
            shadow.to(device=param.device, dtype=param.dtype)
            for shadow, param in zip(state["shadow"], params)
        ]


def cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(path, raw_net, optimizer, ema, epoch, step, args, score_model_hparams):
    current_model = cpu_state_dict(raw_net)

    ema.store(raw_net.parameters())
    ema.copy_to(raw_net.parameters())
    ema_model = cpu_state_dict(raw_net)
    ema.restore(raw_net.parameters())

    checkpoint = {
        "model": current_model,
        "ema_model": ema_model,
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "args": vars(args),
        "score_model_hyperparameters": score_model_hparams,
    }
    atomic_torch_save(checkpoint, path)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(path, raw_net, optimizer, ema, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_net.load_state_dict(strip_module_prefix(ckpt["model"]))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], raw_net.parameters())
        return int(ckpt.get("epoch", -1)) + 1, int(ckpt.get("step", 0))

    raw_net.load_state_dict(strip_module_prefix(ckpt))
    return 0, 0


def prune_old_checkpoints(output_dir, keep_last_n):
    if keep_last_n <= 0:
        return
    paths = sorted(Path(output_dir).glob("checkpoint_step_*.pt"))
    excess = len(paths) - keep_last_n
    for path in paths[: max(0, excess)]:
        path.unlink(missing_ok=True)


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_subset", type=int, default=-1, help="-1 for the full dataset")
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 1, 2, 2, 2, 2, 2])

    parser.add_argument("--sigma_min", type=float, default=1e-4)
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=263.4,
        help="Use 263.4 for Adam et al. reproduction; set <0 to estimate from data.",
    )
    parser.add_argument("--sigma_max_pairs", type=int, default=5000)

    parser.add_argument("--epochs", type=int, default=2700)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_every_steps", type=int, default=1000)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--keep_last_n", type=int, default=3)
    parser.add_argument("--max_hours", type=float, default=float("inf"))
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="'auto' loads output_dir/latest.pt if present; 'none' disables resume; otherwise pass a path.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    local_rank, rank, world_size = setup_distributed()
    main_proc = is_main(rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    if main_proc:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    barrier()

    n_subset = None if args.n_subset == -1 else args.n_subset
    images = load_probes(
        args.data_dir,
        n_subset=n_subset,
        image_size=args.image_size,
        seed=args.seed,
        verbose=main_proc,
    )

    if args.sigma_max < 0:
        sigma_tensor = torch.zeros((), device=device)
        if main_proc:
            sigma_tensor.fill_(estimate_sigma_max(images, args.sigma_max_pairs, args.seed))
        if dist.is_initialized():
            dist.broadcast(sigma_tensor, src=0)
        sigma_max = float(sigma_tensor.item())
    else:
        sigma_max = args.sigma_max

    dataset = ProbesDataset(images)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    raw_net = NCSNpp(
        channels=1,
        nf=args.nf,
        ch_mult=args.ch_mult,
        dimensions=2,
    ).to(device)
    score_model = ScoreModel(
        model=raw_net,
        sigma_min=args.sigma_min,
        sigma_max=sigma_max,
        device=device,
    )

    if dist.is_initialized():
        score_model.model = DDP(raw_net, device_ids=[local_rank], output_device=local_rank)

    train_net = score_model.model
    optimizer = torch.optim.Adam(train_net.parameters(), lr=args.lr)
    ema = EMA(raw_net.parameters(), decay=args.ema_decay)

    start_epoch = 0
    step = 0
    latest_ckpt = output_dir / "latest.pt"
    if args.resume == "auto" and latest_ckpt.exists():
        if main_proc:
            print(f"Resuming from {latest_ckpt}")
        start_epoch, step = load_checkpoint(latest_ckpt, raw_net, optimizer, ema, device)
    elif args.resume.lower() not in {"auto", "none", ""}:
        if main_proc:
            print(f"Resuming from {args.resume}")
        start_epoch, step = load_checkpoint(args.resume, raw_net, optimizer, ema, device)
    barrier()

    effective_bs = args.batch_size * world_size
    steps_per_epoch = len(loader)
    planned_steps = steps_per_epoch * args.epochs
    n_params = sum(param.numel() for param in raw_net.parameters())

    if main_proc:
        print(f"World size: {world_size} | rank: {rank} | local_rank: {local_rank}")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"Dataset: {len(dataset)} images")
        print(f"VE SDE: sigma_min={args.sigma_min:.1e}, sigma_max={sigma_max:.4f}")
        print(f"NCSN++: {n_params:,} parameters")
        print(
            f"Plan: {args.epochs} epochs x {steps_per_epoch} steps = {planned_steps:,} "
            f"steps (per-GPU bs={args.batch_size}, effective bs={effective_bs})"
        )
        print(f"Starting from epoch={start_epoch}, step={step}")

    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            sampler.set_epoch(epoch)
            train_net.train()

            for batch in loader:
                should_stop = (time.time() - t0) / 3600.0 > args.max_hours
                if distributed_any(should_stop, device):
                    if main_proc:
                        save_checkpoint(
                            latest_ckpt,
                            raw_net,
                            optimizer,
                            ema,
                            epoch,
                            step,
                            args,
                            score_model.hyperparameters,
                        )
                        print(f"Reached max_hours={args.max_hours}; saved {latest_ckpt}")
                    barrier()
                    return

                x = batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                loss = score_model.loss_fn(x)
                loss.backward()

                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(train_net.parameters(), max_norm=args.clip)

                if args.warmup > 0 and step < args.warmup:
                    lr_now = args.lr * min((step + 1) / args.warmup, 1.0)
                    for group in optimizer.param_groups:
                        group["lr"] = lr_now

                optimizer.step()
                ema.update(raw_net.parameters())

                loss_value = loss.detach().item()
                running_loss += loss_value
                running_count += 1
                step += 1

                if args.log_every_steps > 0 and step % args.log_every_steps == 0:
                    local_avg = running_loss / max(1, running_count)
                    global_avg = reduce_mean(local_avg, device)
                    running_loss = 0.0
                    running_count = 0
                    if main_proc:
                        lr = optimizer.param_groups[0]["lr"]
                        elapsed = (time.time() - t0) / 3600.0
                        print(
                            f"epoch={epoch} step={step} loss={global_avg:.4e} "
                            f"lr={lr:.3e} elapsed={elapsed:.2f}h"
                        )

                if (
                    main_proc
                    and args.ckpt_every_steps > 0
                    and step % args.ckpt_every_steps == 0
                ):
                    save_checkpoint(
                        latest_ckpt,
                        raw_net,
                        optimizer,
                        ema,
                        epoch,
                        step,
                        args,
                        score_model.hyperparameters,
                    )
                    save_checkpoint(
                        output_dir / f"checkpoint_step_{step:08d}.pt",
                        raw_net,
                        optimizer,
                        ema,
                        epoch,
                        step,
                        args,
                        score_model.hyperparameters,
                    )
                    prune_old_checkpoints(output_dir, args.keep_last_n)
                    print(f"Saved checkpoint at step {step}")

            barrier()

        if main_proc:
            save_checkpoint(
                latest_ckpt,
                raw_net,
                optimizer,
                ema,
                args.epochs - 1,
                step,
                args,
                score_model.hyperparameters,
            )
            print(f"Training complete. Saved {latest_ckpt}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
