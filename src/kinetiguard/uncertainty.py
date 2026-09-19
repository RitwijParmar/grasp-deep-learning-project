"""Calibration and kinetic-risk acquisition for conservative force fields."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import torch

from .model import EquivariantEnergyModel


def force_rms(values: torch.Tensor) -> torch.Tensor:
    """Per-frame RMS over atom and Cartesian dimensions."""
    return torch.sqrt(values.square().mean(dim=(-1, -2)).clamp_min(1e-16))


@dataclass
class ConformalForceCalibrator:
    """Finite-sample conformal multiplier for ensemble force disagreement."""

    alpha: float = 0.1
    multiplier: float = 1.0
    epsilon: float = 1e-6
    calibration_size: int = 0

    def fit(self, member_forces: torch.Tensor, reference_forces: torch.Tensor) -> "ConformalForceCalibrator":
        if member_forces.ndim != 4:
            raise ValueError("member_forces must have shape [members, frames, atoms, 3]")
        if member_forces.shape[1:] != reference_forces.shape:
            raise ValueError("ensemble and reference force shapes do not match")
        mean_force = member_forces.mean(dim=0)
        residual = force_rms(mean_force - reference_forces)
        spread = force_rms(member_forces.std(dim=0, unbiased=member_forces.shape[0] > 1))
        nonconformity = residual / (spread + self.epsilon)
        n = int(nonconformity.numel())
        if n < 2:
            raise ValueError("conformal calibration needs at least two configurations")
        rank = min(n, int(torch.ceil(torch.tensor((n + 1) * (1.0 - self.alpha))).item()))
        self.multiplier = float(torch.kthvalue(nonconformity, rank).values.item())
        self.calibration_size = n
        return self

    def bound(self, member_forces: torch.Tensor) -> torch.Tensor:
        spread = force_rms(member_forces.std(dim=0, unbiased=member_forces.shape[0] > 1))
        return self.multiplier * (spread + self.epsilon)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ConformalBarrierCalibrator:
    """Conformal intervals for path-level barrier predictions.

    This is deliberately separate from force calibration: small pointwise force
    error does not imply a reliable energy difference between a minimum and a
    transition state.  The caller supplies matched barrier labels from a
    calibration network, never from the held-out network it will report.
    """

    alpha: float = 0.1
    multiplier: float = 1.0
    epsilon: float = 1e-6
    calibration_size: int = 0

    def fit(self, member_barriers: torch.Tensor, reference_barriers: torch.Tensor) -> "ConformalBarrierCalibrator":
        if member_barriers.ndim != 2:
            raise ValueError("member_barriers must have shape [members, edges]")
        if member_barriers.shape[1:] != reference_barriers.shape:
            raise ValueError("ensemble and reference barrier shapes do not match")
        mean = member_barriers.mean(dim=0)
        spread = member_barriers.std(dim=0, unbiased=member_barriers.shape[0] > 1)
        scores = (mean - reference_barriers).abs() / (spread + self.epsilon)
        n = int(scores.numel())
        if n < 2:
            raise ValueError("barrier calibration needs at least two network edges")
        rank = min(n, int(torch.ceil(torch.tensor((n + 1) * (1.0 - self.alpha))).item()))
        self.multiplier = float(torch.kthvalue(scores, rank).values.item())
        self.calibration_size = n
        return self

    def interval(self, member_barriers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = member_barriers.mean(dim=0)
        radius = self.multiplier * (
            member_barriers.std(dim=0, unbiased=member_barriers.shape[0] > 1) + self.epsilon
        )
        return mean - radius, mean + radius

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def certify_barrier_order(
    lower: torch.Tensor, upper: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return pairwise certificates and the indices that need another label.

    An edge's order is certified only when its interval does not overlap any
    other edge interval. This is conservative by design; un-certified edges are
    exactly the appropriate targets for a kinetic-topology acquisition policy.
    """
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("lower and upper must be matching one-dimensional tensors")
    overlap = (lower[:, None] <= upper[None, :]) & (upper[:, None] >= lower[None, :])
    overlap.fill_diagonal_(False)
    certified = ~overlap.any(dim=1)
    return certified, torch.nonzero(~certified, as_tuple=False).flatten()


@torch.no_grad()
def ensemble_forces(
    members: Iterable[EquivariantEnergyModel], atomic_numbers: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Evaluate force members while preserving coordinate gradients inside each model."""
    outputs: List[torch.Tensor] = []
    # `no_grad` cannot surround autograd-based force evaluation.  The decorator is
    # intentionally overridden in the local context below.
    with torch.enable_grad():
        for member in members:
            _, forces = member.energy_and_forces(atomic_numbers, positions.detach().clone(), create_graph=False)
            outputs.append(forces.detach())
    return torch.stack(outputs, dim=0)


class CurvatureProbe:
    """Low-rank Hessian probe for transition-region awareness.

    A negative directional curvature is not itself a transition state.  It is used
    only as a label-free acquisition signal, never as a scientific claim about a
    discovered saddle point.
    """

    def __init__(self, directions: int = 4, seed: int = 0) -> None:
        self.directions = int(directions)
        self.seed = int(seed)

    def min_rayleigh(
        self, model: EquivariantEnergyModel, atomic_numbers: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if self.directions < 1:
            raise ValueError("directions must be positive")
        generator = torch.Generator(device=positions.device)
        generator.manual_seed(self.seed)
        rays = []
        for _ in range(self.directions):
            x = positions.detach().clone().requires_grad_(True)
            vector = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            vector = vector - vector.mean(dim=1, keepdim=True)
            vector = vector / torch.linalg.vector_norm(vector.flatten(1), dim=1).view(-1, 1, 1).clamp_min(1e-8)
            energy = model(atomic_numbers, x)
            gradient = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
            hessian_vector = torch.autograd.grad((gradient * vector).sum(), x)[0]
            rays.append((hessian_vector * vector).sum(dim=(-1, -2)).detach())
        return torch.stack(rays, dim=0).min(dim=0).values


class KineticRiskScorer:
    """Acquire points where a calibrated model may perturb kinetic topology.

    ``risk = conformal_force_bound * (1 + beta * transition_weight)`` where the
    transition weight grows for negative Hessian directions.  This combines the
    usual committee disagreement with a label-free geometry signal and is the
    project\'s central, falsifiable acquisition hypothesis.
    """

    def __init__(self, beta: float = 1.0, curvature_scale: float = 1.0, directions: int = 4) -> None:
        if curvature_scale <= 0:
            raise ValueError("curvature_scale must be positive")
        self.beta = float(beta)
        self.curvature_scale = float(curvature_scale)
        self.probe = CurvatureProbe(directions=directions)

    def score(
        self,
        members: Iterable[EquivariantEnergyModel],
        calibrator: ConformalForceCalibrator,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        members = list(members)
        stacked_forces = ensemble_forces(members, atomic_numbers, positions)
        upper_bound = calibrator.bound(stacked_forces)
        mean_model = members[0]
        min_curvature = self.probe.min_rayleigh(mean_model, atomic_numbers, positions)
        transition_weight = torch.sigmoid(-min_curvature / self.curvature_scale)
        return upper_bound * (1.0 + self.beta * transition_weight), upper_bound, min_curvature


def pair_distance_descriptor(positions: torch.Tensor) -> torch.Tensor:
    """Rotation/translation invariant descriptor for greedy acquisition diversity."""
    atom_count = positions.shape[1]
    pair_distances = torch.cdist(positions, positions)
    upper = torch.triu_indices(atom_count, atom_count, offset=1, device=positions.device)
    return pair_distances[:, upper[0], upper[1]]


def select_risk_diverse(
    risk: torch.Tensor, positions: torch.Tensor, count: int, diversity_weight: float = 0.25
) -> torch.Tensor:
    """Greedy risk/diversity selection with an invariant pair-distance descriptor."""
    if count < 1 or count > risk.numel():
        raise ValueError("count must be in [1, number of candidates]")
    descriptor = pair_distance_descriptor(positions).float()
    descriptor = (descriptor - descriptor.mean(dim=0)) / descriptor.std(dim=0).clamp_min(1e-6)
    normalized_risk = (risk - risk.min()) / (risk.max() - risk.min()).clamp_min(1e-8)
    selected = [int(torch.argmax(normalized_risk).item())]
    available = torch.ones(risk.numel(), dtype=torch.bool, device=risk.device)
    available[selected[0]] = False
    while len(selected) < count:
        distance = torch.cdist(descriptor, descriptor[selected]).min(dim=1).values
        normalized_distance = distance / distance.max().clamp_min(1e-8)
        score = normalized_risk + diversity_weight * normalized_distance
        score[~available] = -torch.inf
        choice = int(torch.argmax(score).item())
        selected.append(choice)
        available[choice] = False
    return torch.tensor(selected, device=risk.device, dtype=torch.long)
