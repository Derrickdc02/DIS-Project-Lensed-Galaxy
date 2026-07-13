"""Unconditional draws from the trained NCSN++ score prior, for prior validation.

Companion to src/sample.py (posterior sampling): here we draw x ~ p(x) with no
likelihood term, in resumable chunks, and save them for the PQMass prior check in
notebooks/PQMassPriorCheck.ipynb. Defaults: 1000 samples, chunks of 50, 4000 steps.
"""

import argparse
import time
from pathlib import Path

import torch

from sample import load_model, atomic_save


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the unconditional prior-sampling command-line parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=str, default="./outputs/probes_final/prior_check",
                   help="where to write prior samples; --ckpt resolves relative to it")
    p.add_argument("--ckpt", type=str, default="../latest.pt",
                   help="checkpoint filename inside output_dir (or an absolute path)")
    p.add_argument("--n_samples", type=int, default=1000, help="total unconditional draws")
    p.add_argument("--chunk", type=int, default=50, help="draws per model.sample call")
    p.add_argument("--steps", type=int, default=4000, help="Euler-Maruyama sampling steps")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=21)
    return p


def main():
    """Generate resumable unconditional prior draws."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.n_samples <= 0 or args.chunk <= 0:
        parser.error("--n_samples and --chunk must be positive")
    if args.n_samples % args.chunk:
        parser.error("--n_samples must be a multiple of --chunk")
    n_chunks = args.n_samples // args.chunk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = output_dir / args.ckpt
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(ckpt_path, device)

    # Each chunk c is seeded args.seed+c and checkpointed as it finishes, so a job
    # killed at walltime resumes by re-running: completed chunks are skipped.
    print(f"\nDrawing {args.n_samples} unconditional samples "
          f"({n_chunks} x {args.chunk}) at {args.steps} steps...")
    t0 = time.time()
    for c in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{c:03d}.pt"
        if chunk_path.exists():
            print(f"  chunk {c + 1}/{n_chunks} already on disk -> {chunk_path.name}, skipping")
            continue
        torch.manual_seed(args.seed + c)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + c)
        chunk = model.sample(
            shape=[args.chunk, 1, args.image_size, args.image_size],
            steps=args.steps,
        ).cpu()
        atomic_save(chunk, chunk_path)
        elapsed = time.time() - t0
        print(f"  chunk {c + 1}/{n_chunks} done -> {chunk_path.name} ({elapsed:.1f}s elapsed)")

    samples = torch.cat(
        [torch.load(chunks_dir / f"chunk_{c:03d}.pt", map_location="cpu")
         for c in range(n_chunks)],
        dim=0,
    )                                           # (N, 1, H, W)
    samples_path = output_dir / "prior_samples.pt"
    atomic_save(samples, samples_path)
    print(f"Saved {samples.shape[0]} samples -> {samples_path}")


if __name__ == "__main__":
    main()
