"""A/B the frozen and optimized loops on identical cached training batches."""

import argparse
import importlib.util
import json
import sys
import time
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
from image_classification.training.provenance import file_sha256, runtime_provenance
from image_classification.utils import seed_everything


def load_reference(path):
    spec = importlib.util.spec_from_file_location("_frozen_training_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._train_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/independent_leaky_middle.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--device-cache", action="store_true", help="isolate loop overhead from host transfers")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--reference-no-cache", action="store_true",
                        help="compare graph with the same eager AMP-cache-disabled semantics")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    config = load_config(["--config", str(PROJECT_ROOT / args.config), "--cuda_graph", str(args.cuda_graph)])
    reference_path = PROJECT_ROOT / args.reference_root / "src/image_classification/training/engine.py"
    if args.reference_no_cache and not args.cuda_graph:
        raise ValueError("--reference-no-cache requires --cuda-graph")
    if args.reference_no_cache:
        reference_path = PROJECT_ROOT / "src/image_classification/training/engine.py"
        frozen = _train_epoch
    else:
        frozen = load_reference(reference_path)
    device = torch.device("cuda")
    seed_everything(config.seed)
    loaders = build_dataloaders(config.dataset, config.batch_size, config.num_workers,
                               config.prefetch_factor, config.validation_size, config.seed)
    batches = []
    for batch in loaders.train:
        batches.append(tuple(value.to(device) for value in batch) if args.device_cache else batch)
        if len(batches) == args.batches:
            break
    del loaders
    result = {"reference_engine_sha256": file_sha256(reference_path), "config": config.to_dict(),
              "reference_no_cache": args.reference_no_cache,
              "runtime": runtime_provenance(), "batches": len(batches), "device_cache": args.device_cache,
              "note": "Cached identical augmented train batches, no data-loading or checkpoint I/O timing; no test evaluation.",
              "trials": []}
    reference_state = None
    for label, function in (("legacy", frozen), ("optimized", _train_epoch),
                            ("optimized", _train_epoch), ("legacy", frozen)):
        seed_everything(config.seed)
        model = build_model(config).to(device)
        capture = prepare_training_graph(model, config.batch_size, device, config.amp) if (
            label == "optimized" and args.cuda_graph
        ) else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.lr, epochs=200, steps_per_epoch=len(batches),
            pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=100,
        )
        scaler = GradScaler("cuda", enabled=config.amp)
        times = []
        metrics = None
        for epoch in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            metrics = function(model, batches, torch.nn.CrossEntropyLoss(), optimizer, scheduler, scaler, config, device)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if reference_state is None:
            reference_state = state
        deltas = [(state[key].double() - reference_state[key].double()).abs().max().item() for key in state]
        trial = {"implementation": label, "epoch_seconds": times, "warm_seconds_mean": sum(times[1:]) / 2,
                 "graph_capture": capture,
                 "final_metrics": metrics, "scheduler_last_epoch": scheduler.last_epoch,
                 "amp_scale": scaler.get_scale(), "max_state_difference_from_first_legacy": max(deltas)}
        result["trials"].append(trial)
        print(json.dumps(trial), flush=True)
        del model, optimizer, scheduler, scaler, state
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
