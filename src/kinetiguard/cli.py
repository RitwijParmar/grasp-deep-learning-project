"""Command-line entry points for experiments, not a one-off notebook workflow."""

import argparse
import json
from pathlib import Path
from typing import List

import torch

from .data import (
    download_3bpa,
    load_frames,
    materialize_3bpa_splits,
    save_frames,
    synthetic_harmonic_frames,
)
from .evaluation import evaluate_conformal_coverage, evaluate_force_field
from .model import ModelConfig
from .training import TrainConfig, load_member, split_indices, train_ensemble
from .uncertainty import ConformalForceCalibrator, ensemble_forces


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def command_download_3bpa(args: argparse.Namespace) -> None:
    source = download_3bpa(Path(args.destination))
    outputs = materialize_3bpa_splits(source, Path(args.destination) / "processed")
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


def command_train(args: argparse.Namespace) -> None:
    frames = load_frames(Path(args.dataset))
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        force_weight=args.force_weight,
        energy_weight=args.energy_weight,
        patience=args.patience,
        seed=args.seed,
        device=_device(args.device),
    )
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        radial_basis=args.radial_basis,
        interactions=args.interactions,
        cutoff=args.cutoff,
    )
    paths = train_ensemble(frames, Path(args.output), model_config, config, members=args.members)
    print(json.dumps({"checkpoints": [str(path) for path in paths]}, indent=2))


def command_evaluate(args: argparse.Namespace) -> None:
    frames = load_frames(Path(args.dataset))
    checkpoints = sorted(Path(args.checkpoints).glob("member_*.pt"))
    if len(checkpoints) < 2:
        raise ValueError("expected at least two member_*.pt checkpoints")
    members = [load_member(path, args.device)[0] for path in checkpoints]
    for member in members:
        member.eval()
    frames = frames.to(torch.device(args.device))
    metrics = evaluate_force_field(
        members, frames.atomic_numbers, frames.positions, frames.energies, frames.forces
    ).as_dict()
    print(json.dumps(metrics, indent=2))


def command_smoke(args: argparse.Namespace) -> None:
    """Short end-to-end invariant/force/ensemble validation used on a GPU job."""
    device = _device(args.device)
    frames = synthetic_harmonic_frames(frames=args.frames, atoms=5, seed=args.seed)
    root = Path(args.output)
    dataset_path = root / "synthetic.pt"
    save_frames(frames, dataset_path)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=16,
        learning_rate=3e-3,
        force_weight=10.0,
        patience=max(4, args.epochs // 3),
        seed=args.seed,
        device=device,
    )
    model_config = ModelConfig(hidden_dim=32, radial_basis=8, interactions=2, cutoff=4.0)
    checkpoints = train_ensemble(frames, root / "ensemble", model_config, config, members=2)
    members = [load_member(path, device)[0] for path in checkpoints]
    device_frames = frames.to(torch.device(device))
    member_forces = ensemble_forces(members, device_frames.atomic_numbers, device_frames.positions)
    calibration_indices, test_indices = split_indices(len(device_frames), 0.25, args.seed + 31)
    calibrator = ConformalForceCalibrator(alpha=0.2).fit(
        member_forces[:, calibration_indices], device_frames.forces[calibration_indices]
    )
    metrics = evaluate_force_field(
        members,
        device_frames.atomic_numbers,
        device_frames.positions[test_indices],
        device_frames.energies[test_indices],
        device_frames.forces[test_indices],
    ).as_dict()
    coverage = evaluate_conformal_coverage(
        members,
        calibrator,
        device_frames.atomic_numbers,
        device_frames.positions[test_indices],
        device_frames.forces[test_indices],
    )
    # Exact invariance check for the first member under a fixed proper rotation.
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    base_energy = members[0](device_frames.atomic_numbers, device_frames.positions).detach()
    rotated_energy = members[0](
        device_frames.atomic_numbers, torch.matmul(device_frames.positions, rotation)
    ).detach()
    report = {
        "device": device,
        "frames": len(frames),
        "metrics": metrics,
        "conformal": coverage,
        "rotation_energy_max_abs_error": float((base_energy - rotated_energy).abs().max().item()),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg", description="KinetiGuard molecular-force-field research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download-3bpa", help="download and materialize official 3BPA splits")
    download.add_argument("--destination", default="data/3bpa")
    download.set_defaults(func=command_download_3bpa)
    train = subparsers.add_parser("train", help="train a conservative ensemble")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--members", type=int, default=3)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--force-weight", type=float, default=25.0)
    train.add_argument("--energy-weight", type=float, default=1.0)
    train.add_argument("--patience", type=int, default=25)
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--radial-basis", type=int, default=16)
    train.add_argument("--interactions", type=int, default=4)
    train.add_argument("--cutoff", type=float, default=5.0)
    train.set_defaults(func=command_train)
    evaluate = subparsers.add_parser("evaluate", help="evaluate an ensemble on one split")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoints", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=command_evaluate)
    smoke = subparsers.add_parser("smoke", help="run a synthetic end-to-end force-field check")
    smoke.add_argument("--output", default="artifacts/smoke")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--epochs", type=int, default=10)
    smoke.add_argument("--frames", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.set_defaults(func=command_smoke)
    return parser


def main(argv: List[str] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
