const state = { step: 0, active: false };
const steps = [
  { selector: '[data-tour="hero"]', kicker: 'Start here', title: 'The research question', text: 'Most molecular models are judged on snapshots. This project asks whether the whole energy landscape stays useful when a molecule moves.', place: 'right' },
  { selector: '[data-tour="problem"]', kicker: 'Why it matters', title: 'Rates depend on barriers', text: 'A tiny average force error can still move a transition barrier enough to change a simulated rate. That is why ordinary MAE is necessary, but not sufficient.', place: 'bottom' },
  { selector: '[data-tour="model"]', kicker: 'Model', title: 'Energy first, forces second', text: 'KinetiGuard predicts an energy and differentiates it to get force. The force field is conservative by construction.', place: 'bottom' },
  { selector: '[data-tour="lab"]', kicker: 'Interactive lab', title: 'What gets a new label?', text: 'Try the slider. The system prioritizes a configuration only when it sees uncertainty and a signal that the local landscape might be transition-relevant.', place: 'top' },
  { selector: '[data-tour="ledger"]', kicker: 'Evidence', title: 'The ledger separates fact from plan', text: 'The L4 mathematical check is real. The 3BPA and Landscape17 results are displayed as in-progress or planned until their runs complete.', place: 'top' },
  { selector: '[data-tour="claims"]', kicker: 'Claim boundary', title: 'This is how scientific demos stay credible', text: 'The demo never turns an infrastructure check into a chemistry result. When the experiment finishes, the ledger gets the numbers and the exact run artifact.', place: 'top' },
];

const layer = document.getElementById('tourLayer');
const marker = document.getElementById('tourMarker');
const card = document.getElementById('tourCard');
const number = document.getElementById('markerNumber');
const kicker = document.getElementById('tourKicker');
const title = document.getElementById('tourTitle');
const text = document.getElementById('tourText');
const progress = document.getElementById('tourProgress');

function positionTour() {
  const current = steps[state.step];
  const target = document.querySelector(current.selector);
  if (!target) return;
  const rect = target.getBoundingClientRect();
  const markerTop = Math.max(20, rect.top + Math.min(rect.height * .25, 105));
  const markerLeft = Math.min(window.innerWidth - 30, Math.max(30, rect.left + Math.min(rect.width * .18, 120)));
  marker.style.top = `${markerTop - 19}px`;
  marker.style.left = `${markerLeft - 19}px`;
  const cardWidth = Math.min(340, window.innerWidth - 42);
  const roomRight = window.innerWidth - markerLeft;
  let left = roomRight > cardWidth + 75 ? markerLeft + 58 : markerLeft - cardWidth - 36;
  left = Math.max(21, Math.min(window.innerWidth - cardWidth - 21, left));
  let top = markerTop - 25;
  if (current.place === 'bottom') top = markerTop + 45;
  if (current.place === 'top') top = markerTop - 240;
  top = Math.max(20, Math.min(window.innerHeight - 255, top));
  card.style.left = `${left}px`;
  card.style.top = `${top}px`;
}

function renderTour() {
  const current = steps[state.step];
  number.textContent = state.step + 1;
  kicker.textContent = current.kicker;
  title.textContent = current.title;
  text.textContent = current.text;
  progress.textContent = `${state.step + 1} / ${steps.length}`;
  document.getElementById('tourBack').style.visibility = state.step ? 'visible' : 'hidden';
  document.getElementById('tourNext').textContent = state.step === steps.length - 1 ? 'Finish' : 'Next';
  document.querySelector(current.selector).scrollIntoView({ behavior: 'smooth', block: 'center' });
  window.setTimeout(positionTour, 380);
}

function startTour() { state.step = 0; state.active = true; layer.classList.add('active'); renderTour(); }
function endTour() { state.active = false; layer.classList.remove('active'); }
document.getElementById('startTour').addEventListener('click', startTour);
document.getElementById('heroTour').addEventListener('click', startTour);
document.getElementById('footerTour').addEventListener('click', startTour);
document.getElementById('tourNext').addEventListener('click', () => { if (state.step === steps.length - 1) endTour(); else { state.step += 1; renderTour(); } });
document.getElementById('tourBack').addEventListener('click', () => { if (state.step) { state.step -= 1; renderTour(); } });
window.addEventListener('resize', () => state.active && positionTour());

const slider = document.getElementById('pathSlider');
function updatePath() {
  const x = Number(slider.value) / 100;
  const nearBarrier = Math.exp(-Math.pow((x - .5) / .23, 2));
  const bound = .07 + .15 * nearBarrier + .025 * Math.sin(x * 16);
  const weight = .12 + .84 * nearBarrier;
  const risk = bound * (1 + weight);
  document.getElementById('pathDot').style.left = `${slider.value}%`;
  document.getElementById('boundValue').textContent = bound.toFixed(2);
  document.getElementById('weightValue').textContent = weight.toFixed(2);
  document.getElementById('riskValue').textContent = risk.toFixed(2);
  const readout = x < .3 ? 'Near conformer A: the model is stable enough that this frame is a lower acquisition priority.' : x > .7 ? 'Near conformer B: uncertainty is modest and the local landscape looks basin-like.' : 'At the barrier: the system pays more attention to uncertainty that could change the transition.';
  document.getElementById('pathReadout').textContent = readout;
  document.getElementById('priorityText').textContent = risk > .27 ? 'high — add to candidate queue' : risk > .16 ? 'medium — monitor' : 'low — do not spend a label yet';
}
slider.addEventListener('input', updatePath); updatePath();

fetch('status.json').then(r => r.ok ? r.json() : null).then(status => {
  if (!status) return;
  document.getElementById('navStatus').textContent = status.nav || 'research run active';
  document.getElementById('pilotState').textContent = status.state || 'running';
  document.getElementById('pilotDetail').textContent = status.detail || document.getElementById('pilotDetail').textContent;
  document.getElementById('pilotClaim').textContent = status.claim || 'not yet';
}).catch(() => {});
