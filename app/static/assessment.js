const form = document.querySelector('#risk-form');
const result = document.querySelector('#result');
form?.addEventListener('submit', async (event) => {
  event.preventDefault();
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
    if (!response.ok) throw new Error(data.detail || 'Assessment unavailable.');
    result.hidden = false; result.className = 'result';
    result.innerHTML = `<p class="eyebrow">Preliminary result</p><h2>${data.risk_label}</h2><div class="score"><strong>${data.positive_probability}%</strong><span>model-estimated probability of the dataset’s positive class</span></div><p>The top model factors globally were ${data.top_global_factors.map(f => `<b>${f.feature}</b>`).join(', ')}. This is an explanation of overall model behavior, not an individual clinical explanation.</p><p class="result-disclaimer">${data.disclaimer}</p>`;
  } catch (error) { result.hidden = false; result.className = 'result error'; result.textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Assess preliminary risk <span aria-hidden="true">→</span>'; }
});
