"""Metrics that refuse to equate low force error with kinetic faithfulness."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from .model import EquivariantEnergyModel
from .uncertainty import ConformalForceCalibrator, ensemble_forces, force_rms


def _rank(values: torch.Tensor) -> torch.Tensor:
    return torch.argsort(torch.argsort(values)).float()


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = torch.sqrt(first.square().sum() * second.square().sum()).clamp_min(1e-12)
    return float((first * second).sum().div(denominator).item())


@dataclass
class ForceFieldMetrics:
    energy_mae: float
    force_mae: float
    force_rmse: float
    uncertainty_error_spearman: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_force_field(
    members: Sequence[EquivariantEnergyModel],
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    energies: torch.Tensor,
    forces: torch.Tensor,
) -> ForceFieldMetrics:
    member_energies: List[torch.Tensor] = []
    with torch.enable_grad():
        member_forces = ensemble_forces(members, atomic_numbers, positions)
        for member in members:
            member_energies.append(member(atomic_numbers, positions.detach().clone()).detach())
    predicted_energy = torch.stack(member_energies).mean(dim=0)
    predicted_force = member_forces.mean(dim=0)
    uncertainty = force_rms(member_forces.std(dim=0, unbiased=len(members) > 1))
    errors = force_rms(predicted_force - forces)
    return ForceFieldMetrics(
        energy_mae=float((predicted_energy - energies).abs().mean().item()),
        force_mae=float((predicted_force - forces).abs().mean().item()),
        force_rmse=float(torch.sqrt((predicted_force - forces).square().mean()).item()),
        uncertainty_error_spearman=_pearson(_rank(uncertainty), _rank(errors)),
    )


def evaluate_conformal_coverage(
    members: Sequence[EquivariantEnergyModel],
    calibrator: ConformalForceCalibrator,
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    reference_forces: torch.Tensor,
) -> Dict[str, float]:
    member_forces = ensemble_forces(members, atomic_numbers, positions)
    mean_force = member_forces.mean(dim=0)
    errors = force_rms(mean_force - reference_forces)
    bounds = calibrator.bound(member_forces)
    return {
        "target_coverage": 1.0 - calibrator.alpha,
        "empirical_coverage": float(errors.le(bounds).float().mean().item()),
        "mean_force_bound": float(bounds.mean().item()),
        "mean_force_error": float(errors.mean().item()),
    }


def temperature_ood_report(
    members: Sequence[EquivariantEnergyModel],
    atomic_numbers: torch.Tensor,
    split_frames: Mapping[str, object],
) -> Dict[str, Dict[str, float]]:
    """Evaluate named 3BPA splits without mixing 300K and high-temperature results."""
    report: Dict[str, Dict[str, float]] = {}
    for name, frames in split_frames.items():
        metrics = evaluate_force_field(
            members, atomic_numbers, frames.positions, frames.energies, frames.forces
        )
        report[name] = metrics.as_dict()
    return report


def barrier_metrics(
    reference_minimum_energies: torch.Tensor,
    reference_transition_energies: torch.Tensor,
    predicted_minimum_energies: torch.Tensor,
    predicted_transition_energies: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate barrier heights at matched KTN stationary points.

    This does not pretend that a stationary point has been recovered merely because
    its energy is accurate; callers must report topology reconstruction separately.
    """
    if not (
        reference_minimum_energies.shape
        == reference_transition_energies.shape
        == predicted_minimum_energies.shape
        == predicted_transition_energies.shape
    ):
        raise ValueError("all barrier tensors must have matching shapes")
    reference_barrier = reference_transition_energies - reference_minimum_energies
    predicted_barrier = predicted_transition_energies - predicted_minimum_energies
    return {
        "barrier_mae": float((reference_barrier - predicted_barrier).abs().mean().item()),
        "barrier_rmse": float(torch.sqrt((reference_barrier - predicted_barrier).square().mean()).item()),
        "barrier_rank_spearman": _pearson(_rank(reference_barrier), _rank(predicted_barrier)),
    }
