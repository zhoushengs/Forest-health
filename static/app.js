const $ = (id) => document.getElementById(id);
let file = null, original = null, result = null, ready = false, busy = false;
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function updateButton() { $('detect').disabled = !file || !ready || busy; }
function clearResult() {
  result = null;
  ['result-tab', 'download-image', 'download-json'].forEach(id => $(id).disabled = true);
  $('count').textContent = '0 targets'; $('rows').replaceChildren(); $('table-wrap').hidden = true;
  $('summary').textContent = 'Run detection to explore target classes, confidence scores and coordinates.';
}
function showImage(annotated) {
  $('preview').src = annotated && result ? result.annotated_image : original;
  $('preview').alt = annotated ? 'Annotated detection results' : 'Uploaded image preview';
  ['original-tab', 'result-tab'].forEach((id, i) => {
    const active = Boolean(i) === annotated;
    $(id).classList.toggle('active', active); $(id).setAttribute('aria-pressed', String(active));
  });
}
function selectFile(selected) {
  if (!selected || busy) return;
  if (!/\.(jpe?g|png|webp)$/i.test(selected.name)) { message('Please choose a JPG, PNG or WebP image', true); return; }
  if (selected.size > 15 * 1024 * 1024) { message('Image size must not exceed 15 MB', true); return; }
  if (original) URL.revokeObjectURL(original);
  file = selected; original = URL.createObjectURL(file); clearResult();
  $('filename').textContent = file.name; $('empty').hidden = true; $('preview').hidden = false;
  $('image-info').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  showImage(false); message('Image ready. Click Run detection to continue.'); updateButton();
}
$('file').addEventListener('change', event => selectFile(event.target.files[0]));
$('preview').addEventListener('error', () => {
  file = null; clearResult(); $('preview').hidden = true; $('empty').hidden = false;
  message('Unable to preview this image. Please choose a valid image.', true); $('file').value = ''; updateButton();
});
for (const type of ['dragenter', 'dragover', 'dragleave', 'drop']) {
  $('dropzone').addEventListener(type, event => {
    event.preventDefault(); $('dropzone').classList.toggle('drag', type === 'dragenter' || type === 'dragover');
    if (type === 'drop') selectFile(event.dataTransfer.files[0]);
  });
}
for (const id of ['confidence', 'iou']) {
  $(id).addEventListener('input', () => {
    $(id + '-value').value = Number($(id).value).toFixed(2);
    if (result) message('Settings updated. Run detection again to apply.');
  });
}
$('original-tab').addEventListener('click', () => { if (original) showImage(false); });
$('result-tab').addEventListener('click', () => { if (result) showImage(true); });
async function checkHealth() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw new Error();
    const data = await response.json(); ready = true;
    $('service').textContent = 'Model ready'; $('service').className = 'status ready';
    $('model-name').textContent = data.model;
    $('model-detail').textContent = `${data.input_size.join(' × ')} · ${data.provider === 'CPUExecutionProvider' ? 'CPU inference' : data.provider}`;
  } catch {
    ready = false; $('service').textContent = 'Model offline'; $('service').className = 'status error';
    $('model-detail').textContent = 'Waiting for the service…';
  }
  updateButton();
}
$('detect').addEventListener('click', async () => {
  if (!file || busy) return;
  busy = true; clearResult(); showImage(false); updateButton(); $('loading').hidden = false;
  ['file', 'confidence', 'iou', 'original-tab'].forEach(id => $(id).disabled = true);
  message('Running detection. Please wait…');
  const data = new FormData(); data.append('file', file); data.append('confidence', $('confidence').value); data.append('iou', $('iou').value);
  try {
    const response = await fetch('/api/detect', { method: 'POST', body: data, signal: AbortSignal.timeout(120000) });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Invalid request parameters');
    result = body; showImage(true);
    ['result-tab', 'download-image', 'download-json'].forEach(id => $(id).disabled = false);
    $('count').textContent = `${body.detections.length} ${body.detections.length === 1 ? 'target' : 'targets'}`;
    const counts = {};
    body.detections.forEach((detection, i) => {
      counts[detection.class_name] = (counts[detection.class_name] || 0) + 1;
      const row = document.createElement('tr');
      const values = [String(i + 1).padStart(2, '0'), detection.class_name, `${(detection.confidence * 100).toFixed(1)}%`, detection.box.join(', ')];
      values.forEach((value, j) => {
        const cell = document.createElement('td');
        if (j === 1) { const badge = document.createElement('span'); badge.className = `class-badge ${value === 'sicktree' ? 'sick' : ''}`; badge.textContent = value; cell.append(badge); }
        else cell.textContent = value;
        row.append(cell);
      });
      $('rows').append(row);
    });
    $('table-wrap').hidden = !body.detections.length;
    const categories = Object.entries(counts).map(([name, count]) => `${name}: ${count}`).join(' · ');
    $('summary').textContent = `${categories || 'No sicktree targets detected at this threshold. Try lowering the confidence threshold.'}${categories ? ' · ' : ' '}Inference ${body.inference_ms} ms · Processing ${body.total_ms} ms`;
    $('image-info').textContent = `${body.image_size.join(' × ')} px · Confidence ${body.confidence_threshold} / IoU ${body.iou_threshold}`;
    message('Detection complete. View or download your results.');
  } catch (error) {
    message(error.name === 'TimeoutError' ? 'Request timed out. Please try again later.' : error.message === 'Failed to fetch' ? 'Connection failed. Check that the service is running.' : error.message, true);
  } finally {
    busy = false; $('loading').hidden = true;
    ['file', 'confidence', 'iou', 'original-tab'].forEach(id => $(id).disabled = false);
    updateButton();
  }
});
function download(href, name) { const link = document.createElement('a'); link.href = href; link.download = name; link.click(); }
$('download-image').addEventListener('click', () => { if (result) download(result.annotated_image, `${file.name.replace(/\.[^.]+$/, '')}_detected.jpg`); });
$('download-json').addEventListener('click', () => {
  if (!result) return;
  const { annotated_image, ...data } = result;
  const url = URL.createObjectURL(new Blob([JSON.stringify({ filename: file.name, ...data }, null, 2)], { type: 'application/json' }));
  download(url, `${file.name.replace(/\.[^.]+$/, '')}_detections.json`); setTimeout(() => URL.revokeObjectURL(url), 2000);
});
checkHealth(); setInterval(checkHealth, 15000);
