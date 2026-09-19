import torch

from kinetiguard.data import synthetic_harmonic_frames
from kinetiguard.model import EquivariantEnergyModel, ModelConfig
from kinetiguard.uncertainty import (
    ConformalBarrierCalibrator,
    ConformalForceCalibrator,
    certify_barrier_order,
    select_risk_diverse,
)


def test_conformal_bound_is_finite() -> None:
    torch.manual_seed(2)
    reference = torch.randn(12, 4, 3)
    members = reference.unsqueeze(0) + 0.1 * torch.randn(3, 12, 4, 3)
    calibrator = ConformalForceCalibrator(alpha=0.2).fit(members, reference)
    bounds = calibrator.bound(members)
    assert bounds.shape == (12,)
    assert calibrator.multiplier > 0
    assert torch.isfinite(bounds).all()


def test_diverse_selector_returns_unique_valid_indices() -> None:
    frames = synthetic_harmonic_frames(frames=12, atoms=4)
    risk = torch.linspace(0.0, 1.0, 12)
    selected = select_risk_diverse(risk, frames.positions, count=4)
    assert selected.shape == (4,)
    assert len(set(selected.tolist())) == 4
    assert selected.min() >= 0
    assert selected.max() < len(frames)


def test_barrier_intervals_abstain_on_overlap() -> None:
    reference = torch.tensor([1.0, 2.0, 4.0, 6.0])
    members = torch.stack([reference - 0.1, reference, reference + 0.1])
    calibrator = ConformalBarrierCalibrator(alpha=0.25).fit(members, reference)
    lower, upper = calibrator.interval(members)
    certified, requested = certify_barrier_order(lower, upper)
    assert certified.all()
    assert requested.numel() == 0
    overlap_lower = torch.tensor([0.0, 0.4])
    overlap_upper = torch.tensor([1.0, 1.3])
    certified, requested = certify_barrier_order(overlap_lower, overlap_upper)
    assert not certified.any()
    assert requested.tolist() == [0, 1]
