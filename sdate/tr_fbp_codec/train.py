"""Training entry point for tr_fbp_codec.

Single-process (CPU smoke test or 1 GPU):
    python -m sdate.tr_fbp_codec.train --max_steps 5000 --ckpt_dir /path/to/ckpts

Multi-GPU (DDP, one node -- e.g. a CSCS Clariden GH200 node, 4 GPUs):
    torchrun --nproc_per_node=4 -m sdate.tr_fbp_codec.train \\
      --max_steps 5000 --ckpt_dir /path/to/ckpts --device cuda

``--batch_size`` is PER-GPU (per-process) in both cases -- effective global
batch size under torchrun is ``batch_size * nproc_per_node``. torchrun sets
``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` in the environment; their presence is
what switches this script into DDP mode, no separate flag needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import CodecConfig, DataConfig, QuantConfig
from .data import TrFbpCodecDataset, collate
from .losses import ce_loss, nats_to_bits
from .model import FramePredictorResNet3D, ModelConfig, stack_inputs


def warmup_cosine_lr_lambda(warmup_steps: int, max_steps: int):
    """Linear warmup to peak lr, then cosine decay to 0 over the remaining steps.

    Plain constant-lr Adam plateaued within ~2000 steps and never improved
    over the remaining ~58000 (see run history 2026-09-09/10 on Clariden) --
    warmup avoids early instability at the larger DDP global batch size,
    cosine decay lets the tail of a short run actually settle instead of
    oscillating at a fixed step size.
    """
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def setup_distributed(device_arg: str):
    """Returns (rank, world_size, local_rank, device, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}"), True
    return 0, 1, 0, torch.device(device_arg), False


def build_datasets(data_cfg: DataConfig, codec_cfg: CodecConfig, quant_cfg: QuantConfig):
    train_ds = TrFbpCodecDataset(data_cfg, codec_cfg, quant_cfg, split="train")
    holdout_ds = None
    if data_cfg.holdout_frame_start is not None:
        holdout_ds = TrFbpCodecDataset(data_cfg, codec_cfg, quant_cfg, split="holdout")
    return train_ds, holdout_ds


@torch.no_grad()
def eval_holdout_bpp(model, loader, codec_cfg: CodecConfig, device, max_batches: int = 20) -> float:
    model.eval()
    total_nats, total_px = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        context = batch["context"].to(device)
        prior = batch.get("prior")
        prior = prior.to(device) if prior is not None else None
        target_q = batch["target_q"].to(device)
        x = stack_inputs(context, prior, n_levels=codec_cfg.n_classes)
        logits = model(x)
        bm, bs = codec_cfg.block_margin, codec_cfg.block_size
        if bs is not None:
            logits = logits[:, :, bm:bm + bs, bm:bm + bs]
        loss = ce_loss(logits, target_q)
        total_nats += loss.item() * target_q.numel()
        total_px += target_q.numel()
    model.train()
    return nats_to_bits(torch.tensor(total_nats / total_px)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--use_fbp_prior", type=int, default=None, choices=[0, 1])
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--n_blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR (after warmup)")
    p.add_argument(
        "--warmup_steps", type=int, default=None,
        help="LR linear-warmup steps; default 5%% of max_steps",
    )
    p.add_argument("--batch_size", type=int, default=32, help="PER-GPU batch size")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    rank, world_size, local_rank, device, is_ddp = setup_distributed(args.device)
    is_main = rank == 0

    data_cfg = DataConfig()
    codec_cfg = CodecConfig()
    if args.k is not None:
        codec_cfg.k = args.k
    if args.use_fbp_prior is not None:
        codec_cfg.use_fbp_prior = bool(args.use_fbp_prior)
    if args.block_size is not None:
        codec_cfg.block_size = args.block_size
    quant_cfg = QuantConfig(mode="truncate")

    train_ds, holdout_ds = build_datasets(data_cfg, codec_cfg, quant_cfg)

    train_sampler = None
    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    holdout_loader = None
    if holdout_ds is not None and is_main:
        # Held-out eval only runs on rank 0 (see README/DDP notes) -- other
        # ranks idle briefly at the barrier below rather than duplicating it.
        holdout_loader = DataLoader(
            holdout_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
            num_workers=max(args.num_workers // 2, 1),
        )

    depth_in = codec_cfg.k + (1 if codec_cfg.use_fbp_prior else 0)
    model_cfg = ModelConfig(
        depth_in=depth_in, n_classes=codec_cfg.n_classes,
        base_channels=args.base_channels, n_blocks=args.n_blocks,
    )
    model = FramePredictorResNet3D(model_cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(0.05 * args.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=warmup_cosine_lr_lambda(warmup_steps, args.max_steps)
    )

    ckpt_dir = Path(args.ckpt_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "config.json").write_text(json.dumps({
            "data_cfg": asdict(data_cfg), "codec_cfg": asdict(codec_cfg),
            "quant_cfg": asdict(quant_cfg), "model_cfg": asdict(model_cfg),
            "args": vars(args), "world_size": world_size, "warmup_steps": warmup_steps,
        }, indent=2))
        print(f"[train] device={device} world_size={world_size} depth_in={depth_in} "
              f"train_targets={len(train_ds.targets)} "
              f"holdout_targets={len(holdout_ds.targets) if holdout_ds else 0} "
              f"global_batch={args.batch_size * world_size}")

    def raw_model():
        return model.module if is_ddp else model

    step = 0
    t0 = time.time()
    history = []
    epoch = 0
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break
            context = batch["context"].to(device)
            prior = batch.get("prior")
            prior = prior.to(device) if prior is not None else None
            target_q = batch["target_q"].to(device)

            x = stack_inputs(context, prior, n_levels=codec_cfg.n_classes)
            logits = model(x)
            bm, bs = codec_cfg.block_margin, codec_cfg.block_size
            if bs is not None:
                logits = logits[:, :, bm:bm + bs, bm:bm + bs]
            loss = ce_loss(logits, target_q)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if is_main and step % args.log_every == 0:
                elapsed = time.time() - t0
                cur_lr = sched.get_last_lr()[0]
                print(f"[train] step={step} loss_bits={nats_to_bits(loss).item():.4f} "
                      f"lr={cur_lr:.6g} elapsed={elapsed:.1f}s")
                history.append({"step": step, "train_bpp": nats_to_bits(loss).item(), "lr": cur_lr})

            if holdout_loader is not None and step % args.eval_every == 0 and step > 0:
                ho_bpp = eval_holdout_bpp(raw_model(), holdout_loader, codec_cfg, device)
                print(f"[eval] step={step} holdout_bpp={ho_bpp:.4f}")
                if history:
                    history[-1]["holdout_bpp"] = ho_bpp
                torch.save(
                    {"model": raw_model().state_dict(), "model_cfg": asdict(model_cfg), "step": step},
                    ckpt_dir / f"ckpt_step{step}.pt",
                )
            if is_ddp:
                dist.barrier()  # other ranks wait for rank 0's eval/checkpoint above

            step += 1

    if is_main:
        torch.save(
            {"model": raw_model().state_dict(), "model_cfg": asdict(model_cfg), "step": step},
            ckpt_dir / "ckpt_final.pt",
        )
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[train] done. final checkpoint -> {ckpt_dir / 'ckpt_final.pt'}")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
