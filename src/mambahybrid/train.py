"""Training entry point.

    python -m mambahybrid.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import WOMDDataset, collate
from .losses import compute_losses
from .metrics import occlusion_reconstruction_error, trajectory_metrics
from .model import MambaHybrid


def lr_lambda(step: int, cfg: Config):
    if step < cfg.warmup_iters:
        return step / max(1, cfg.warmup_iters)
    prog = (step - cfg.warmup_iters) / max(1, cfg.max_steps - cfg.warmup_iters)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def move(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches: int = 50) -> dict:
    model.eval()
    agg: dict[str, list[float]] = {}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move(batch, device)
        out = model(batch)
        tm = trajectory_metrics(out, batch, cfg)
        re = occlusion_reconstruction_error(out, batch, cfg)
        for slc in ("all", "occluded", "clean"):
            for k in ("minADE", "minFDE", "MissRate"):
                v = tm[slc][k]
                if v == v:  # not nan
                    agg.setdefault(f"{slc}/{k}", []).append(v)
        for k in ("recon_ade_ft", "recon_ade_fupd"):
            if re[k] == re[k]:
                agg.setdefault(k, []).append(re[k])
    model.train()
    return {k: sum(v) / len(v) for k, v in agg.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config) if os.path.exists(args.config) else Config()
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")   # TF32 matmuls; we train in fp32
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.ckpt_dir, "config.yaml"))
    log_fh = open(os.path.join(cfg.ckpt_dir, "log.jsonl"), "a")

    run = None
    if cfg.wandb:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project, name=os.path.basename(cfg.ckpt_dir.rstrip("/")),
            config=dict(cfg.__dict__), resume="allow",
            id=hashlib.sha1(cfg.ckpt_dir.encode()).hexdigest()[:16],
        )

    pc = cfg.preprocess()
    train_ds = WOMDDataset(cfg.train_shards, pc, "train", cfg.cache_dir, cfg.seed)
    val_ds = WOMDDataset(cfg.val_shards, pc, "val", cfg.cache_dir, cfg.seed)
    print(f"train scenarios: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, collate_fn=collate, persistent_workers=cfg.num_workers > 0,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), collate_fn=collate,
    )

    model = MambaHybrid(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, cfg))
    # AMP measurably does nothing here (the step loop is CUDA-launch bound, not
    # flop bound) and collides with the fp32-forced selective scan under compile.
    use_amp = cfg.amp and device == "cuda" and not cfg.compile
    if cfg.amp and cfg.compile:
        print("note: amp disabled (compile is on; amp gives no speedup for this model)")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f} M")
    if run is not None:
        run.summary["n_params_M"] = n_params / 1e6

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed at step {step}")

    model.train()
    t0 = time.time()
    best = math.inf
    data_iter = iter(train_loader)
    while step < cfg.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = move(batch, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch)
            losses = compute_losses(out, batch, cfg, step=step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % cfg.log_every == 0:
            rec = {"step": step, "lr": sched.get_last_lr()[0],
                   "it_s": cfg.log_every / (time.time() - t0),
                   **{k: float(v.detach()) for k, v in losses.items()}}
            print(f"step {step:>7} | loss {rec['total']:.3f} | traj {rec['traj']:.3f} "
                  f"| recon_ft {rec['recon_ft']:.3f} recon_fupd {rec['recon_fupd']:.3f} "
                  f"| {rec['it_s']:.2f} it/s")
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            if run is not None:
                run.log({"train/lr": rec["lr"], "train/it_s": rec["it_s"],
                         **{f"train/{k}": v for k, v in rec.items()
                            if k not in ("step", "lr", "it_s")}}, step=step)
            t0 = time.time()

        if step % cfg.val_every == 0 or step == cfg.max_steps:
            metrics = evaluate(model, val_loader, cfg, device)
            metrics["step"] = step
            print(f"  [val] " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != "step"))
            log_fh.write(json.dumps({"val": metrics}) + "\n")
            log_fh.flush()
            if run is not None:
                run.log({f"val/{k}": v for k, v in metrics.items() if k != "step"}, step=step)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                        "cfg": cfg.__dict__}, os.path.join(cfg.ckpt_dir, "last.pt"))
            key = metrics.get("all/minADE", math.inf)
            if key < best:
                best = key
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg.__dict__},
                           os.path.join(cfg.ckpt_dir, "best.pt"))
            t0 = time.time()

    log_fh.close()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
