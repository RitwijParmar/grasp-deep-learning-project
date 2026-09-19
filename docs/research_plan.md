# KinetiGuard research plan

## Research question

Can a label-efficient force-field learner improve molecular kinetics by acquiring
configurations where both predictive error and local potential-energy-surface
curvature are risky? Conventional benchmarks emphasize energy/force MAE. That
is necessary but insufficient: a force field with small average error can still
invent a conformer, erase a transition barrier, or destabilize a trajectory.

## Falsifiable hypothesis

Kinetic-risk acquisition—conformal ensemble force bound multiplied by a
negative-curvature transition weight—will recover the same or better OOD force
accuracy and barrier-height accuracy with fewer 300 K labels than each ablation:

1. uniform random replay-oracle selection;
2. ensemble uncertainty only;
3. curvature only.

The method must not be described as successful unless all baselines use equal
label budgets, identical train/validation partitions, three seeds, and the
published selection logs.

## Protocol

| Stage | Dataset | What is measured | Why it matters |
| --- | --- | --- | --- |
| Controlled sanity | exact synthetic harmonic potential | energy conservation structure, force-gradient consistency, E(3) invariance | catches a mathematically invalid implementation |
| Flexible-molecule OOD | 3BPA train 300 K; test 300/600/1200 K | energy/force MAE, calibration coverage, risk-coverage curves | tests temperature/conformer shift |
| Kinetic landscape | Landscape17 matched minima and transition states | barrier MAE/RMSE, barrier rank correlation, topology reconstruction | evaluates the scientific quantity we care about |
| Scale stress | selected MD22 systems | wall time, memory, uncertainty calibration | prevents an elegant small-molecule-only result |

## Claims boundary

KinetiGuard is an experimental research system, not a chemical-design or
medical tool. The repository may claim only results that exist in a versioned
run manifest. In particular, barrier prediction at supplied stationary points
is not equivalent to reconstructing a complete kinetic transition network; the
latter must use an independent landscape exploration procedure.

## Compute budget

1. A Vertex T4 synthetic smoke run checks CUDA, gradients, invariance and
   report upload before using public chemistry data.
2. A 3BPA pilot trains 3 members for 120 epochs and verifies preprocessing.
3. The paper grid uses five-member ensembles, four acquisition rounds, four
   ablations, three seeds and a fixed stop budget. Spot VMs are appropriate for
   retriable members; every job writes configuration, data fingerprint and git
   commit to its run manifest.

The design intentionally separates the small validation run from the expensive
paper grid, so an interrupted cloud job never becomes a scientific result.
