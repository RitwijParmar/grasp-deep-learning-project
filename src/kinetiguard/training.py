"""Deterministic, ensemble-aware training for energy-and-force supervision."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import EquivariantEnergyModel, ModelConfig
from .types import MolecularFrames


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    force_weight: float = 25.0
    energy_weight: float = 1.0
    validation_fraction: float = 0.15
    patience: int = 25
    seed: int = 11
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(count: int, validation_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if count < 8:
        raise ValueError("at least eight frames are needed for a train/validation split")
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(count, generator=generator)
    validation_count = max(2, int(round(count * validation_fraction)))
    return shuffled[validation_count:], shuffled[:validation_count]


def _normalization(frames: MolecularFrames, train_indices: torch.Tensor) -> Dict[str, float]:
    selected_energy = frames.energies[train_indices]
    selected_force = frames.forces[train_indices]
    return {
        "energy_mean": float(selected_energy.mean().item()),
        "energy_scale": float(selected_energy.std().clamp_min(1e-5).item()),
        "force_scale": float(selected_force.std().clamp_min(1e-5).item()),
    }


def _loader(frames: MolecularFrames, indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(frames.positions[indices], frames.energies[indices], frames.forces[indices])
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, drop_last=False)


def _batch_loss(
    model: EquivariantEnergyModel,
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    energies: torch.Tensor,
    forces: torch.Tensor,
    normalization: Dict[str, float],
    config: TrainConfig,
    create_graph: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predicted_energy, predicted_forces = model.energy_and_forces(
        atomic_numbers, positions, create_graph=create_graph
    )
    energy_error = (predicted_energy - energies) / normalization["energy_scale"]
    force_error = (predicted_forces - forces) / normalization["force_scale"]
    energy_loss = energy_error.square().mean()
    force_loss = force_error.square().mean()
    loss = config.energy_weight * energy_loss + config.force_weight * force_loss
    return loss, {
        "loss": float(loss.detach().item()),
        "energy_mae": float((predicted_energy - energies).abs().mean().detach().item()),
        "force_mae": float((predicted_forces - forces).abs().mean().detach().item()),
    }


def evaluate_model(
    model: EquivariantEnergyModel,
    frames: MolecularFrames,
    indices: torch.Tensor,
    normalization: Dict[str, float],
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(config.device)
    accumulator = {"loss": 0.0, "energy_mae": 0.0, "force_mae": 0.0, "count": 0}
    for positions, energies, forces in _loader(frames, indices, config.batch_size, shuffle=False):
        positions, energies, forces = positions.to(device), energies.to(device), forces.to(device)
        loss, batch_metrics = _batch_loss(
            model,
            frames.atomic_numbers.to(device),
            positions,
            energies,
            forces,
            normalization,
            config,
            create_graph=False,
        )
        batch_count = int(positions.shape[0])
        for key in ("loss", "energy_mae", "force_mae"):
            accumulator[key] += batch_metrics[key] * batch_count
        accumulator["count"] += batch_count
    return {key: accumulator[key] / accumulator["count"] for key in ("loss", "energy_mae", "force_mae")}


def train_member(
    frames: MolecularFrames,
    model_config: ModelConfig,
    train_config: TrainConfig,
    member_seed: int,
    train_indices: Optional[torch.Tensor] = None,
    validation_indices: Optional[torch.Tensor] = None,
) -> Tuple[EquivariantEnergyModel, Dict[str, object]]:
    """Train one bootstrap member and return its best validation checkpoint."""
    set_seed(member_seed)
    device = torch.device(train_config.device)
    if train_indices is None or validation_indices is None:
        train_indices, validation_indices = split_indices(
            len(frames), train_config.validation_fraction, member_seed
        )
    normalization = _normalization(frames, train_indices)
    model = EquivariantEnergyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_config.epochs, 1))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics: Optional[Dict[str, float]] = None
    wait = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        for positions, energies, forces in _loader(frames, train_indices, train_config.batch_size, shuffle=True):
            positions, energies, forces = positions.to(device), energies.to(device), forces.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _batch_loss(
                model,
                frames.atomic_numbers.to(device),
                positions,
                energies,
                forces,
                normalization,
                train_config,
                create_graph=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        scheduler.step()
        validation = evaluate_model(model, frames, validation_indices, normalization, train_config)
        validation["epoch"] = float(epoch)
        history.append(validation)
        if best_metrics is None or validation["loss"] < best_metrics["loss"]:
            best_metrics = validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_config.patience:
                break
    if best_state is None or best_metrics is None:  # pragma: no cover - defensive
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    report: Dict[str, object] = {
        "member_seed": member_seed,
        "normalization": normalization,
        "best_validation": best_metrics,
        "epochs_completed": len(history),
        "history": history,
        "train_size": int(train_indices.numel()),
        "validation_size": int(validation_indices.numel()),
    }
    return model, report


def save_member(
    path: Path,
    model: EquivariantEnergyModel,
    report: Dict[str, object],
    model_config: ModelConfig,
    train_config: TrainConfig,
    atomic_numbers: torch.Tensor,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": model_config.as_dict(),
            "train_config": asdict(train_config),
            "atomic_numbers": atomic_numbers.cpu(),
            "report": report,
        },
        path,
    )


def load_member(path: Path, device: str = "cpu") -> Tuple[EquivariantEnergyModel, Dict[str, object]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = EquivariantEnergyModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def train_ensemble(
    frames: MolecularFrames,
    output_dir: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    members: int = 3,
    train_indices: Optional[torch.Tensor] = None,
    validation_indices: Optional[torch.Tensor] = None,
) -> List[Path]:
    if members < 2:
        raise ValueError("at least two ensemble members are required for uncertainty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: List[Path] = []
    reports = []
    for member_index in range(members):
        seed = train_config.seed + 1009 * member_index
        model, report = train_member(
            frames, model_config, train_config, seed, train_indices, validation_indices
        )
        checkpoint_path = output_dir / "member_{:02d}.pt".format(member_index)
        save_member(checkpoint_path, model, report, model_config, train_config, frames.atomic_numbers)
        checkpoints.append(checkpoint_path)
        reports.append(report)
    manifest = {
        "members": [str(path.name) for path in checkpoints],
        "model_config": model_config.as_dict(),
        "train_config": asdict(train_config),
        "reports": reports,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return checkpoints
