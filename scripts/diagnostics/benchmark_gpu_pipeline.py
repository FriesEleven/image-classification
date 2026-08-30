"""Bounded GPU pipeline benchmark; no formal runs or test-set evaluation."""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from image_classification.config import load_config
from image_classification.data import build_dataloaders
from image_classification.models import build_model
from image_classification.paths import PROJECT_ROOT
from image_classification.training.cuda_graph import prepare_training_graph
from image_classification.training.engine import _train_epoch
from image_classification.training.evaluate import validate
from image_classification.training.provenance import runtime_provenance
from image_classification.utils import seed_everything


class TimedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.wait_seconds = 0.0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.wait_seconds = 0.0
        started = time.perf_counter()
        iterator = iter(self.loader)
        self.wait_seconds += time.perf_counter() - started
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.wait_seconds += time.perf_counter() - started
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/independent_leaky_middle.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0, help="0 preserves the installed runtime default")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    if not 2 <= args.epochs <= 10:
        raise ValueError("Benchmark is limited to 2–10 epochs")
    if args.threads:
        torch.set_num_threads(args.threads)
    config = replace(load_config(["--config", str(PROJECT_ROOT / args.config)]),
                     num_workers=args.workers, seed=args.seed, evaluate_test=False)
    if not config.cuda_graph:
        raise ValueError("This benchmark compares remaining bottlenecks after enabling CUDA Graph")
    seed_everything(config.seed)
    device = torch.device("cuda")
    model = build_model(config).to(device)
    capture = prepare_training_graph(model, config.batch_size, device, config.amp)
    loaders = build_dataloaders(config.dataset, config.batch_size, config.num_workers,
                               config.prefetch_factor, config.validation_size, config.seed)
    if args.cached:
        batches = []
        for batch in loaders.train:
            batches.append(batch)
            if len(batches) == 64:
                break
        train_loader = TimedLoader(batches)
    else:
        train_loader = TimedLoader(loaders.train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4,
                                  betas=(0.9, 0.999), **({"fused": True} if args.fused else {}))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.lr, epochs=200, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=100,
    )
    scaler = GradScaler("cuda", enabled=config.amp)
    criterion = torch.nn.CrossEntropyLoss()
    receipt = {"config": config.to_dict(), "fused": args.fused, "cached": args.cached,
               "torch_threads": torch.get_num_threads(), "runtime": runtime_provenance(),
               "graph_capture": capture, "test_evaluated": False, "epochs": [],
               "note": "200-epoch LR schedule, only first benchmark epochs executed; no checkpoint I/O"}
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        train_metrics = _train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, config, device)
        train_seconds = time.perf_counter() - started
        validation_metrics = None
        validation_seconds = 0.0
        if not args.cached:
            started = time.perf_counter()
            validation_metrics, *_ = validate(model, loaders.validation, criterion, device)
            validation_seconds = time.perf_counter() - started
        row = {"epoch": epoch + 1, "train_seconds": train_seconds,
               "loader_wait_seconds": train_loader.wait_seconds, "validation_seconds": validation_seconds,
               "total_seconds": train_seconds + validation_seconds, "train_metrics": train_metrics,
               "validation_metrics": validation_metrics, "amp_scale": scaler.get_scale(),
               "optimizer_updates": scheduler.last_epoch}
        receipt["epochs"].append(row)
        print(json.dumps(row), flush=True)
    receipt["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    receipt["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    receipt["warm_epoch_mean_seconds"] = sum(row["total_seconds"] for row in receipt["epochs"][1:]) / (args.epochs - 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
