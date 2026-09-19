# KinetiGuard

**Uncertainty-calibrated active learning for kinetically faithful molecular force fields.**

KinetiGuard asks a harder question than “does a molecular model have a low
test MAE?”: *does its learned potential preserve conformers and transition
barriers well enough to support molecular dynamics?* It is a research-grade
3D scientific ML system built around conservative equivariant neural potentials,
conformal uncertainty, curvature-aware acquisition, and kinetic-landscape
evaluation.

> Status: research implementation. The repository contains a verified synthetic
> end-to-end baseline and a reproducible GPU campaign; it does **not** claim
> benchmark superiority before versioned 3BPA/Landscape17 experiments exist.

## Why this replaces the original GRASP project

The original course project used a 2D GIN to predict BBBP, ESOL and Tox21 from
SMILES. That is a useful introduction to molecular GNNs, but it has no 3D
physics, force supervision, uncertainty calibration, molecular dynamics, or
scientific evaluation beyond standard property prediction.

KinetiGuard is deliberately a different class of work:

| | Original GRASP | KinetiGuard |
| --- | --- | --- |
| Molecular representation | 2D SMILES graph | 3D atomic coordinates with scalar/vector equivariance |
| Target | static properties | energy and conservative forces |
| Core question | prediction accuracy | kinetic faithfulness under conformational shift |
| Data regime | i.i.d.-style property tasks | 3BPA temperature OOD, rMD17/MD22, Landscape17 KTN |
| Uncertainty | none | ensemble disagreement + finite-sample conformal bound |
| Label policy | fixed data | curvature-aware, risk-diverse replay-oracle acquisition |
| Reproducibility | notebook/app | tests, configs, run manifests, Vertex GPU launcher |

The historical course assets remain available under `legacy_grasp/`; they are
not the primary project surface.

## Method

For a configuration \(x\), each ensemble member predicts an invariant energy
\(E(x)\) and forces \(F(x)=-\nabla_xE(x)\). A held-out calibration set converts
ensemble force spread \(s(x)\) into a finite-sample conformal upper bound
\(b(x)=q_{1-\alpha}s(x)\). KinetiGuard then scores candidate DFT labels as:

\[
\operatorname{risk}(x)=b(x)\left[1+\beta\,\sigma\left(-\hat\lambda_{\min}(H_E(x))/\tau\right)\right].
\]

Here, \(\hat\lambda_{\min}\) is a low-rank Hessian probe. Negative curvature
is treated only as a label-free *transition-region signal*, not as a claim that
a point is a transition state. Selection then adds invariant pair-distance
diversity so the campaign does not repeatedly label one geometry.

The method is falsifiable: it must beat or match random, uncertainty-only and
curvature-only acquisition at equal label budgets—not merely show a lower force
error after seeing more data. See [the protocol](docs/research_plan.md).

The initial implementation intentionally does not oversell this as a new use of
conformal prediction or uncertainty-driven sampling; both already exist in the
literature. The defensible research contribution is a **kinetic-topology risk
controller** that certifies or abstains on barrier ordering at a fixed DFT
budget. The exact prior-art boundary and experiments required to establish that
claim are documented in the [novelty audit](docs/novelty_audit.md).

## Scientific benchmark suite

- **3BPA:** a flexible, drug-like molecule with 300 K training data and 600/1200
  K conformational OOD evaluation. Its complex dihedral surface makes standard
  force MAE insufficient. [Dataset details](https://docs.openqdc.io/main/API/datasets/3bpa.html)
  · [original force-field study](https://doi.org/10.1021/acs.jctc.1c00647)
- **rMD17 / MD22:** clean DFT molecular-dynamics labels and a scale check beyond
  tiny molecules. rMD17 specifically warns against leakage from correlated
  frames. [rMD17 record](https://archive.materialscloud.org/record/466)
  · [MD22 paper](https://arxiv.org/abs/2209.14865)
- **Landscape17:** matched minima, transition states and kinetic transition
  networks—so barrier error and topology can be evaluated independently of
  ordinary test MAE. [Dataset/workflow](https://github.com/VladCarare/mlp-landscapes)
  · [paper](https://doi.org/10.1038/s41524-025-01878-x)

The project does **not** use the noisy original MD17 split. It reports units,
data fingerprints, time-aware held-out partitions, and three seeds.

## Quickstart

```bash
python -m pip install -e ".[dev,chemistry]"
pytest

# Validates conservative force gradients, E(3) energy invariance, ensemble
# uncertainty and conformal calibration on a potential with an exact oracle.
kg smoke --device cpu --output artifacts/smoke
```

For the first public chemistry benchmark:

```bash
kg download-3bpa --destination data/3bpa
kg train \
  --dataset data/3bpa/processed/train_300K.pt \
  --output runs/3bpa/pilot --device cuda --members 3 --epochs 120
kg evaluate \
  --dataset data/3bpa/processed/test_600K.pt \
  --checkpoints runs/3bpa/pilot --device cuda
```

The full paper protocol, including required baselines and non-negotiable
reporting limits, is in [`docs/research_plan.md`](docs/research_plan.md).

## GPU execution on GCP

The Vertex launcher packages only the new research code, submits a bounded GPU
job, and uploads the JSON result to a specified existing bucket. It defaults to
a cheap synthetic CUDA validation before any dataset campaign:

```bash
python infra/vertex/submit.py \
  --project YOUR_PROJECT \
  --bucket YOUR_EXISTING_BUCKET \
  --mode smoke
```

Inspect the exact Vertex request without creating a job:

```bash
python infra/vertex/submit.py --bucket YOUR_EXISTING_BUCKET --dry-run
```

Use `--mode 3bpa` only after the smoke report confirms CUDA and the scientific
checks. The launcher never creates a VM, enables APIs, deletes buckets, or
pretends an interrupted job is a result.

If Vertex custom-training GPU quota is unavailable, the repository also has a
finite Cloud Run L4 job path. It builds the CUDA image, runs one smoke task,
uploads the JSON report, and exits:

```bash
BUCKET=YOUR_EXISTING_BUCKET bash infra/cloud_run/run_gpu_smoke.sh
```

It is deliberately a **job**, not a deployed inference service: no warm GPU is
left running after its report is written.

### Verified GPU smoke run

The initial bounded Cloud Run execution (`kinetiguard-gpu-smoke-54frb`) completed
on an NVIDIA L4 in `us-east4` in 1m16s. Its fixed-oracle report records a maximum
rotation-energy error of `8.94e-08` and 1.00 empirical coverage for a target 0.80
conformal bound. These validate the CUDA/invariance/calibration path only—not
chemical benchmark quality. See the immutable run summary in
[`reports/gpu_smoke_20260919.json`](reports/gpu_smoke_20260919.json).

## Repository map

```text
src/kinetiguard/
  model.py            conservative scalar/vector E(3)-equivariant potential
  uncertainty.py      ensemble/conformal bounds, Hessian probe, risk-diversity selection
  training.py         deterministic energy+force ensemble training
  active_learning.py  auditable replay-oracle selection logs
  evaluation.py       OOD, calibration and barrier metrics
  data.py             3BPA adapter and exact synthetic oracle
infra/vertex/         bounded Vertex AI GPU job submission
infra/cloud_run/      finite Cloud Run L4 smoke-job path
configs/              paper experiment and GPU smoke configurations
tests/                invariance and calibration tests
reports/              versioned, scope-limited run summaries
legacy_grasp/         preserved source material from the original course project
```

## What constitutes a publishable result

A credible conference submission will require more than a polished repository:

1. matched-label-budget learning curves for all three acquisition ablations;
2. 3BPA 300/600/1200 K force error **and** conformal coverage across three seeds;
3. Landscape17 barrier and topology measurements independent of the training
   frames;
4. a failure analysis where curvature risk hurts, not only where it helps;
5. frozen configs, commit hashes, dataset licenses/checksums and cloud costs.

That is the bar KinetiGuard is built to meet. Until then, it is an honest,
strong scientific ML engineering project—not an inflated “SOTA” claim.

## References

- [Allegro: local equivariant representations for atomistic dynamics](https://www.nature.com/articles/s41467-023-36329-y)
- [Uncertainty quantification in neural network potentials](https://www.nature.com/articles/s41524-023-01180-8)
- [Landscape17: global properties of the energy landscape](https://doi.org/10.1038/s41524-025-01878-x)
- [3BPA / ACE force-field benchmark](https://doi.org/10.1021/acs.jctc.1c00647)

## License

MIT. Benchmark datasets retain their own licenses and citation requirements.
