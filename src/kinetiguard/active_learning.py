"""Replay-oracle active learning with auditable, kinetic-risk acquisition."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch

from .model import EquivariantEnergyModel
from .types import MolecularFrames
from .uncertainty import ConformalForceCalibrator, KineticRiskScorer, select_risk_diverse


@dataclass
class AcquisitionResult:
    selected_indices: torch.Tensor
    risk: torch.Tensor
    conformal_bound: torch.Tensor
    min_curvature: torch.Tensor


def select_replay_oracle_labels(
    members: Sequence[EquivariantEnergyModel],
    calibrator: ConformalForceCalibrator,
    pool: MolecularFrames,
    count: int,
    scorer: KineticRiskScorer,
    diversity_weight: float = 0.25,
) -> AcquisitionResult:
    """Select labels from a pre-existing DFT pool; no fake online oracle is claimed."""
    risk, bound, curvature = scorer.score(
        members, calibrator, pool.atomic_numbers, pool.positions
    )
    selected = select_risk_diverse(risk, pool.positions, count, diversity_weight)
    return AcquisitionResult(selected, risk.detach(), bound.detach(), curvature.detach())


def write_acquisition_audit(
    result: AcquisitionResult, output: Path, round_index: int, pool_name: str
) -> None:
    """Persist every score, not just chosen examples, for post-hoc scientific audit."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = set(int(index) for index in result.selected_indices.cpu().tolist())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["round", "pool", "index", "selected", "kinetic_risk", "conformal_bound", "min_curvature"],
        )
        writer.writeheader()
        for index in range(result.risk.numel()):
            writer.writerow(
                {
                    "round": round_index,
                    "pool": pool_name,
                    "index": index,
                    "selected": index in selected,
                    "kinetic_risk": float(result.risk[index].item()),
                    "conformal_bound": float(result.conformal_bound[index].item()),
                    "min_curvature": float(result.min_curvature[index].item()),
                }
            )
