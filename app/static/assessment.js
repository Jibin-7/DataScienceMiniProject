const form = document.querySelector('#risk-form');
const result = document.querySelector('#result');
const feedbackPanel = document.querySelector('#feedback-panel');
const feedbackStatus = document.querySelector('#feedback-status');
const feedbackSend = document.querySelector('#feedback-send');
const feedbackComment = document.querySelector('#feedback-comment');
const ratingButtons = [...document.querySelectorAll('.feedback-rating')];
let selectedRating = 0;

function setFeedbackEnabled(enabled) {
  ratingButtons.forEach((button) => { button.disabled = !enabled; });
  feedbackSend.disabled = !enabled;
  feedbackComment.disabled = !enabled;
}

function resetFeedback() {
  selectedRating = 0;
  ratingButtons.forEach((button) => {
    button.setAttribute('aria-pressed', 'false');
    button.classList.remove('primary');
  });
  feedbackComment.value = '';
  feedbackStatus.hidden = true;
  feedbackStatus.textContent = '';
  setFeedbackEnabled(true);
}

ratingButtons.forEach((button) => button.addEventListener('click', () => {
  selectedRating = Number(button.dataset.rating);
  ratingButtons.forEach((peer) => {
    const active = peer === button;
    peer.setAttribute('aria-pressed', String(active));
    peer.classList.toggle('primary', active);
  });
}));

async function submitFeedback() {
  if (!selectedRating) {
    feedbackStatus.hidden = false;
    feedbackStatus.textContent = 'Please choose a rating from 1 to 5 first.';
    return;
  }
  setFeedbackEnabled(false);
  feedbackStatus.hidden = false;
  feedbackStatus.textContent = 'Sending your feedback…';
  try {
    const response = await fetch('/api/v1/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ disease: form.dataset.disease, rating: selectedRating, comment: feedbackComment.value.trim() || null }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(' ') : (data.detail || 'Feedback unavailable.'));
    feedbackStatus.textContent = data.message || 'Thank you for your feedback!';
  } catch (error) {
    setFeedbackEnabled(true);
    feedbackStatus.textContent = error.message;
  }
}

feedbackSend?.addEventListener('click', submitFeedback);
feedbackComment?.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); submitFeedback(); } });

form?.addEventListener('submit', async (event) => {
  event.preventDefault();
  feedbackPanel.hidden = true;
  resetFeedback();
  const payload = Object.fromEntries(new FormData(form));
  const fields = [...form.querySelectorAll('input')];
  let invalid = false;
  fields.forEach((field) => {
    const value = Number(field.value);
    const valid = Number.isFinite(value) && value >= Number(field.min) && value <= Number(field.max);
    field.toggleAttribute('aria-invalid', !valid);
    invalid ||= !valid;
  });
  if (invalid) { result.hidden = false; result.className = 'result error'; result.textContent = 'Please correct the highlighted values before continuing.'; return; }
  const button = form.querySelector('button'); button.disabled = true; button.textContent = 'Assessing…';
  try {
    const response = await fetch(`/api/v1/predict/${form.dataset.disease}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(d => d.msg).join(' ')
        : (data.detail || 'Assessment unavailable.');
      throw new Error(detail);
    }
    const recommendation = data.positive_probability >= data.threshold
      ? 'We recommend scheduling a consultation with a certified healthcare provider for comprehensive clinical evaluation.'
      : 'Maintain regular routine health checkups and a balanced lifestyle.';
    result.hidden = false; result.className = 'result';
    result.innerHTML = `<p class="eyebrow">Preliminary result</p><h2>${data.risk_label}</h2><div class="score"><strong>${data.positive_probability}%</strong><span>model-estimated probability of the dataset’s positive class</span></div><p>The top model factors globally were ${data.top_global_factors.map(f => `<b>${f.feature}</b>`).join(', ')}. This is an explanation of overall model behavior, not an individual clinical explanation.</p><div class="compact-notice"><p class="eyebrow">Recommendation</p><p>${recommendation}</p><p class="result-disclaimer">For educational, preliminary assessment only. This is not a diagnosis.</p></div><p class="result-disclaimer">${data.disclaimer}</p>`;
    feedbackPanel.hidden = false;
  } catch (error) { result.hidden = false; result.className = 'result error'; result.textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Assess preliminary risk <span aria-hidden="true">→</span>'; }
});
