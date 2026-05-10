import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pressure.data.event_dataset import PSUEventOffsetDataset
from pressure.models.event_offset import EventOffsetRegressor
from pressure.util.util import load_config, split_chunk_paths

DATA_DIMS = {
    "BODY25": (24, 3),
    "BODY25_3D": (24, 4),
    "MOCAP": (17, 3),
    "MOCAP_3D": (17, 4),
    "MOCAP_MRK": (39, 4),
    "UNDERPRESSURE_POS": (23, 4),
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(cfg, subject, split, files):
    return PSUEventOffsetDataset(
        files,
        cfg=cfg,
        split=split,
        window_frames=cfg.event.window_frames,
        samples_per_event=getattr(cfg.event, "samples_per_event", 1),
        min_event_offset=cfg.event.min_event_offset,
        max_event_offset=cfg.event.max_event_offset,
        max_cached_chunks=getattr(cfg.event, "max_cached_chunks", 8),
        seed=int(cfg.default.seed) + int(subject) * 101 + {"train": 0, "val": 1, "test": 2}[split],
    )


def make_loaders(cfg, subject):
    train_files, val_files, test_files = split_chunk_paths(
        cfg.default.data_path,
        subject,
        train_val_split=cfg.training.train_val_split,
        shuffle=cfg.data.shuffle_data,
    )
    train_dataset = make_dataset(cfg, subject, "train", train_files)
    val_dataset = make_dataset(cfg, subject, "val", val_files)
    test_dataset = make_dataset(cfg, subject, "test", test_files)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.dataloader_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def make_model(cfg):
    num_joints, joint_dim = DATA_DIMS[cfg.data.data_type]
    num_channels = cfg.data.num_regions[0] * cfg.data.num_regions[1] * 2
    return EventOffsetRegressor(
        num_joints=num_joints,
        joint_dim=joint_dim,
        num_channels=num_channels,
        window_frames=cfg.event.window_frames,
        hidden_dim=cfg.network.hidden_dim,
        num_layers=cfg.network.num_layers,
        dropout=cfg.network.dropout,
    )


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    losses = []
    for batch in loader:
        joints = batch["joint"].to(device, non_blocking=True)
        offsets = batch["offset"].to(device, non_blocking=True)
        event_type = batch["event_type"].to(device, non_blocking=True)
        channel = batch["channel"].to(device, non_blocking=True)

        pred = model(joints, event_type, channel)
        loss = F.smooth_l1_loss(pred, offsets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    fps = float(cfg.event.fps)
    scale_frames = float(cfg.event.window_frames - 1)
    all_errors = []
    by_event = {0: [], 1: []}

    for batch in loader:
        joints = batch["joint"].to(device, non_blocking=True)
        offsets = batch["offset"].to(device, non_blocking=True)
        event_type = batch["event_type"].to(device, non_blocking=True)
        channel = batch["channel"].to(device, non_blocking=True)

        pred = model(joints, event_type, channel)
        errors_ms = (pred - offsets).abs() * scale_frames * 1000.0 / fps
        all_errors.extend(errors_ms.detach().cpu().numpy().tolist())

        event_np = event_type.detach().cpu().numpy()
        error_np = errors_ms.detach().cpu().numpy()
        for event_id in (0, 1):
            by_event[event_id].extend(error_np[event_np == event_id].tolist())

    def stats(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {"mae_ms": 0.0, "median_ms": 0.0, "count": 0}
        return {
            "mae_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "count": int(values.size),
        }

    return {
        "all": stats(all_errors),
        "onset": stats(by_event[0]),
        "departure": stats(by_event[1]),
    }


def train_subject(cfg, subject, out_dir):
    device = torch.device(cfg.default.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(cfg, subject)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.decay,
    )

    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        val_metrics = evaluate(model, val_loader, device, cfg)
        val_mae = val_metrics["all"]["mae_ms"]
        print(
            f"[Subject {subject} | Epoch {epoch:03d}] "
            f"train_loss={train_loss:.5f} val_mae={val_mae:.2f} ms"
        )

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.training.early_stop_patience:
                print(f"[Subject {subject}] Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, cfg)
    ckpt_path = out_dir / f"subject{subject}_event_offset.pt"
    torch.save({"model": model.state_dict(), "subject": subject, "metrics": test_metrics}, ckpt_path)
    print(
        f"[Subject {subject}] test all={test_metrics['all']['mae_ms']:.2f} ms | "
        f"onset={test_metrics['onset']['mae_ms']:.2f} ms | "
        f"departure={test_metrics['departure']['mae_ms']:.2f} ms"
    )
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train event-offset regression model.")
    parser.add_argument("--config", type=str, default="configs/event_offset_psu_50fps.yaml")
    parser.add_argument("--subject", type=int, default=None, help="Run one LOSO subject only.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.default.seed))

    out_dir = Path(cfg.default.results_path) / "event_offset" / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.subject is not None:
        subjects = [args.subject]
    elif cfg.default.subjects == "all":
        subjects = list(range(1, 11))
    else:
        subjects = list(cfg.default.subjects)

    rows = []
    for subject in subjects:
        metrics = train_subject(cfg, int(subject), out_dir)
        rows.append(
            {
                "subject": int(subject),
                "all_mae_ms": metrics["all"]["mae_ms"],
                "all_median_ms": metrics["all"]["median_ms"],
                "onset_mae_ms": metrics["onset"]["mae_ms"],
                "onset_median_ms": metrics["onset"]["median_ms"],
                "departure_mae_ms": metrics["departure"]["mae_ms"],
                "departure_median_ms": metrics["departure"]["median_ms"],
            }
        )

    csv_path = out_dir / "event_offset_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
        if key != "subject"
    }
    summary_path = out_dir / "event_offset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved results to {csv_path}")
    print(f"Saved summary to {summary_path}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
