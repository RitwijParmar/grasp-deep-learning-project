# Novelty audit: what KinetiGuard may and may not claim

This file exists to prevent a common scientific-ML failure mode: branding a
competent combination of known components as a new method.

## Prior art that rules out easy claims

| Existing result | Consequence for KinetiGuard |
| --- | --- |
| [Uncertainty-driven dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC10766548/) uses ensemble uncertainty to discover under-represented, transition-relevant configurations. | “Uncertainty-driven active learning finds transition states” is not novel. |
| [Flexible uncertainty calibration for MLIPs](https://doi.org/10.1038/s41524-026-02080-3) studies conformal prediction for force fields. | “We applied conformal prediction to an MLIP” is not novel. |
| [Landscape17](https://doi.org/10.1038/s41524-025-01878-x) makes complete kinetic transition networks available for six MD17 molecules. | “We evaluate barriers instead of only force MAE” is valuable but not, by itself, a method contribution. |
| Active-learning NEB and reactive-MLIP work already target transition-state pathways. | “We train an MLIP for barriers” is not novel. |

## Contribution worth testing

The publishable hypothesis is narrower and stronger:

> A **kinetic-topology risk controller** should acquire frames that reduce
> calibrated uncertainty in *barrier ordering and network connectivity*, not
> merely pointwise force error. It should use a transition-network proxy to
> allocate an equal DFT budget across kinetically central pathways, with a
> finite-sample conformal abstention rule when barrier order cannot be certified.

This creates a testable output unavailable from pointwise uncertainty alone:
for every transition-network edge, either (a) certify the barrier ordering to a
chosen error level or (b) abstain and request the most informative path frame.
The primary result is a **kinetic risk–coverage curve**: fraction of transition
edges certified versus barrier-order error among non-abstained edges, at fixed
label budget.

## Required implementation extensions before a submission

1. Import Landscape17 kinetic transition networks without training on their
   final stationary-point labels.
2. Add path-level conformal calibration for barrier residuals, separately from
   force-level calibration.
3. Define a kinetic-centrality allocation rule and prove its data flow never
   consults held-out transition labels during selection.
4. Compare against uncertainty-driven dynamics, ensemble-only selection,
   pretrained-representation acquisition, random, and a path/NEB baseline.
5. Report calibration under trajectory correlation and under the temperature
   shift; ordinary exchangeability does not automatically hold for MD frames.

## Research claims policy

Until those experiments are complete, the honest claim is:

> KinetiGuard is an open, conservative MLIP research scaffold that makes
> kinetic-faithfulness and calibration first-class evaluation targets.

It is not appropriate to claim novelty, state-of-the-art performance, reliable
transition-state discovery, or publication acceptance from the implementation
alone.
