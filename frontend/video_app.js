const videoInput = document.getElementById('videoInput');
const preview = document.getElementById('preview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultBox = document.getElementById('result');
const timeline = document.getElementById('timeline');
let selectedFile = null;

videoInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  selectedFile = f;
  preview.src = URL.createObjectURL(f);
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    alert('Please select a video file first');
    return;
  }
  analyzeBtn.disabled = true;
  resultBox.innerText = 'Analyzing... this may take a while.';
  timeline.innerHTML = '';

  const fd = new FormData();
  fd.append('video', selectedFile);
  try {
    const res = await fetch('/api/v1/video-authenticity', {
      method: 'POST',
      body: fd,
    });
    if (!res.ok) {
      const txt = await res.text();
      resultBox.innerText = `Error: ${txt}`;
      return;
    }
    const j = await res.json();
    renderResult(j);
  } catch (err) {
    resultBox.innerText = 'Error: ' + err;
  } finally {
    analyzeBtn.disabled = false;
  }
});

function renderResult(payload) {
  const label = payload.label || 'Unknown';
  const conf = Math.round((payload.confidence || 0) * 100);
  resultBox.innerHTML = `<strong>${label}</strong> — ${conf}%`;
  resultBox.className = 'result-box ' + (label.toLowerCase().includes('fake') ? 'auth-ai' : 'auth-original');

  // timeline: show suspicious frames as red markers
  timeline.innerHTML = '';
  const frames = payload.frame_scores || [];
  const suspicious = new Set(payload.suspicious_frames || []);
  frames.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'timeline-item';
    el.title = `Frame ${i} score ${s.toFixed(3)}`;
    if (suspicious.has(i)) el.style.background = 'red';
    timeline.appendChild(el);
  });
}
