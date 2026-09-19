# KinetiGuard demo walkthrough

**Target length:** 2–3 minutes. Use the in-page “Take the 90-second tour”
button while speaking; each marker intentionally lands on the thing being
explained. Keep the tone conversational—this is a walkthrough, not a pitch.

## Opening — home panel

“Most molecular ML demos show a prediction and say it is accurate. I wanted to
work on the part that matters once you actually simulate a molecule: whether the
model preserves the landscape between the stable shapes. This is KinetiGuard.
It is a 3D force-field research system, and its narrow goal is to test kinetic
faithfulness rather than just optimize a single average error number.”

**Marker 1: hero.** Point to the energy landscape. “These two valleys are
conformers. The bump between them is a transition barrier. A model can look good
on ordinary force MAE and still move that bump enough to change a simulated
rate.”

## The gap — marker 2

“That is why the project is not another SMILES-property predictor. The question
is: when a molecule walks around, is the learned landscape still useful? We test
that on a flexible drug-like molecule under hotter, out-of-distribution
conditions and then on a kinetic transition network.”

## The method — marker 3

“The model predicts an energy first, and gets forces by differentiating the
energy. That gives the force field one consistent source of truth. Then I use a
small ensemble. Where its members disagree, I have a signal that the model may
need another expensive quantum-chemistry label.”

“The unusual part is that I do not treat every uncertain frame as equal. The
acquisition score also looks for negative local curvature—a cheap signal that we
might be near a transition-relevant part of the surface. It is a prioritization
signal, not me pretending we discovered a transition state.”

## Interactive lab — marker 4

“Here is a transparent toy version. When I move toward the barrier, the ensemble
bound increases and the curvature-based transition weight increases. The label
priority rises only when both are present. That makes the decision inspectable
instead of being a black box.”

## Evidence ledger — marker 5

“I keep the evidence separate from the plan. The green card is already verified:
the actual L4 run checked CUDA, conservative force gradients, rotation
invariance, and the conformal-calibration path. The orange card is the real 3BPA
pilot. Until it finishes, it says ‘not yet’—no invented chemistry result. The
Landscape17 card is the higher bar: barrier ordering and network connectivity,
not just force error.”

## Close — marker 6

“The project is built to make a defensible scientific claim, not to make a flashy
but empty dashboard. The next result has to beat equal-budget random,
uncertainty-only, and curvature-only acquisition across fixed seeds. If it does
not, the failure analysis becomes part of the result.”

“That is KinetiGuard: molecular dynamics with receipts.”
