const video = document.querySelector('#camera');
const overlay = document.querySelector('#overlay');
const processCanvas = document.querySelector('#processingCanvas');
const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const heartRateEl = document.querySelector('#heartRate');
const confidenceText = document.querySelector('#confidenceText');
const qualityEl = document.querySelector('#quality');
const durationEl = document.querySelector('#duration');
const fpsEl = document.querySelector('#fps');
const roiModeEl = document.querySelector('#roiMode');
const statusBadge = document.querySelector('#statusBadge');

const SAMPLE_SECONDS = 30;
const MIN_ANALYSIS_SECONDS = 10;
const MIN_BPM = 45;
const MAX_BPM = 180;
const ROI_GRID = 10;
const MAX_SAMPLES = 900;

const state = {
  stream: null,
  running: false,
  detector: null,
  lastFace: null,
  samples: [],
  raf: 0,
  lastEstimate: null,
  lastFrameTime: 0,
};

function setStatus(text, tone = 'ok') {
  statusBadge.textContent = text;
  statusBadge.style.color = tone === 'bad' ? 'var(--bad)' : tone === 'warn' ? 'var(--warn)' : 'var(--accent)';
}

async function start() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, min: 24 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    state.detector = 'FaceDetector' in window ? new FaceDetector({ fastMode: true, maxDetectedFaces: 1 }) : null;
    state.running = true;
    state.samples = [];
    state.lastEstimate = null;
    startButton.disabled = true;
    stopButton.disabled = false;
    setStatus('正在采集', 'ok');
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    setStatus('摄像头不可用', 'bad');
    confidenceText.textContent = '请确认浏览器权限，并使用 HTTPS 或 localhost 打开页面。';
  }
}

function stop() {
  state.running = false;
  cancelAnimationFrame(state.raf);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus('已停止', 'warn');
}

async function detectFace(width, height) {
  if (!state.detector) return fallbackRoi(width, height);
  try {
    const faces = await state.detector.detect(video);
    if (faces.length) {
      const box = faces[0].boundingBox;
      state.lastFace = { x: box.x, y: box.y, width: box.width, height: box.height, mode: 'FaceDetector' };
    }
  } catch {
    state.detector = null;
  }
  return state.lastFace ?? fallbackRoi(width, height);
}

function fallbackRoi(width, height) {
  return { x: width * 0.28, y: height * 0.16, width: width * 0.44, height: height * 0.50, mode: '中心人脸先验' };
}

async function processFrame(now) {
  if (!state.running || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    state.raf = requestAnimationFrame(processFrame);
    return;
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    state.raf = requestAnimationFrame(processFrame);
    return;
  }

  [processCanvas.width, overlay.width] = [width, width];
  [processCanvas.height, overlay.height] = [height, height];
  const ctx = processCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, width, height);
  const roi = await detectFace(width, height);
  const sample = extractRobustRgb(ctx, roi, now);

  if (sample && (!state.lastFrameTime || now - state.lastFrameTime > 24)) {
    state.samples.push(sample);
    state.lastFrameTime = now;
    trimSamples();
    estimateHeartRate();
  }

  drawOverlay(roi, sample?.quality ?? 0);
  updateReadouts();
  state.raf = requestAnimationFrame(processFrame);
}

function extractRobustRgb(ctx, roi, time) {
  const x0 = Math.max(0, Math.floor(roi.x + roi.width * 0.18));
  const y0 = Math.max(0, Math.floor(roi.y + roi.height * 0.16));
  const w = Math.min(ctx.canvas.width - x0, Math.floor(roi.width * 0.64));
  const h = Math.min(ctx.canvas.height - y0, Math.floor(roi.height * 0.45));
  if (w < 40 || h < 30) return null;

  const image = ctx.getImageData(x0, y0, w, h).data;
  const cells = [];
  for (let gy = 0; gy < ROI_GRID; gy++) {
    for (let gx = 0; gx < ROI_GRID; gx++) {
      cells.push(averageCell(image, w, h, gx, gy));
    }
  }
  const skinCells = cells.filter(isLikelySkin).sort((a, b) => b.score - a.score);
  const stableCells = skinCells.slice(0, Math.max(8, Math.floor(skinCells.length * 0.55)));
  if (stableCells.length < 8) return null;

  const mean = stableCells.reduce((acc, cell) => [acc[0] + cell.r, acc[1] + cell.g, acc[2] + cell.b], [0, 0, 0]).map((value) => value / stableCells.length);
  const brightness = (mean[0] + mean[1] + mean[2]) / 3;
  const quality = Math.max(0, Math.min(1, stableCells.length / cells.length)) * (brightness > 35 && brightness < 235 ? 1 : 0.55);
  return { time: time / 1000, r: mean[0], g: mean[1], b: mean[2], quality };
}

function averageCell(data, width, height, gx, gy) {
  const x1 = Math.floor((gx * width) / ROI_GRID);
  const x2 = Math.floor(((gx + 1) * width) / ROI_GRID);
  const y1 = Math.floor((gy * height) / ROI_GRID);
  const y2 = Math.floor(((gy + 1) * height) / ROI_GRID);
  let r = 0, g = 0, b = 0, n = 0;
  for (let y = y1; y < y2; y += 2) {
    for (let x = x1; x < x2; x += 2) {
      const i = (y * width + x) * 4;
      r += data[i]; g += data[i + 1]; b += data[i + 2]; n++;
    }
  }
  r /= n; g /= n; b /= n;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const score = (max - min) / Math.max(1, max) + Math.min(1, (r + g + b) / 360);
  return { r, g, b, score };
}

function isLikelySkin({ r, g, b }) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return r > 35 && g > 25 && b > 18 && max - min > 8 && r > b * 0.9 && g > b * 0.75;
}

function trimSamples() {
  const newest = state.samples.at(-1)?.time ?? 0;
  state.samples = state.samples.filter((sample) => newest - sample.time <= SAMPLE_SECONDS).slice(-MAX_SAMPLES);
}

function estimateHeartRate() {
  if (state.samples.length < 120) return;
  const duration = state.samples.at(-1).time - state.samples[0].time;
  if (duration < MIN_ANALYSIS_SECONDS) return;

  const fps = (state.samples.length - 1) / duration;
  const uniform = resample(state.samples, fps);
  const pulse = posProjection(uniform);
  const detrended = movingAverageRemove(pulse, Math.max(3, Math.round(fps * 1.2)));
  const spectrum = goertzelBand(detrended, fps, MIN_BPM / 60, MAX_BPM / 60, 0.015);
  const best = spectrum.reduce((a, b) => (b.power > a.power ? b : a), spectrum[0]);
  const totalPower = spectrum.reduce((sum, bin) => sum + bin.power, 0) || 1;
  const confidence = Math.min(1, best.power / totalPower * 7) * averageQuality();

  if (confidence > 0.22) {
    const bpm = best.frequency * 60;
    state.lastEstimate = state.lastEstimate ? state.lastEstimate * 0.72 + bpm * 0.28 : bpm;
    heartRateEl.textContent = Math.round(state.lastEstimate);
    confidenceText.textContent = confidence > 0.55 ? '高置信度' : '中等置信度，请保持稳定';
  } else {
    confidenceText.textContent = '信号较弱，建议改善光照或减少运动';
  }
}

function resample(samples, fps) {
  const step = 1 / Math.max(20, Math.min(30, fps));
  const output = [];
  for (let t = samples[0].time; t <= samples.at(-1).time; t += step) {
    const right = samples.findIndex((sample) => sample.time >= t);
    const a = samples[Math.max(0, right - 1)];
    const b = samples[Math.max(0, right)];
    const mix = b.time === a.time ? 0 : (t - a.time) / (b.time - a.time);
    output.push(['r', 'g', 'b'].map((key) => a[key] + (b[key] - a[key]) * mix));
  }
  return output;
}

function posProjection(rgb) {
  const mean = [0, 1, 2].map((i) => rgb.reduce((sum, row) => sum + row[i], 0) / rgb.length || 1);
  const normalized = rgb.map((row) => row.map((value, i) => value / mean[i] - 1));
  const x = normalized.map(([r, g, b]) => g - b);
  const y = normalized.map(([r, g, b]) => -2 * r + g + b);
  const alpha = std(x) / (std(y) || 1);
  return x.map((value, i) => value + alpha * y[i]);
}

function movingAverageRemove(signal, window) {
  return signal.map((value, index) => {
    const start = Math.max(0, index - window);
    const end = Math.min(signal.length, index + window);
    const avg = signal.slice(start, end).reduce((sum, item) => sum + item, 0) / (end - start);
    return value - avg;
  });
}

function goertzelBand(signal, fps, minHz, maxHz, stepHz) {
  const bins = [];
  for (let frequency = minHz; frequency <= maxHz; frequency += stepHz) {
    let real = 0, imag = 0;
    signal.forEach((value, n) => {
      const angle = (2 * Math.PI * frequency * n) / fps;
      const window = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (signal.length - 1));
      real += value * window * Math.cos(angle);
      imag -= value * window * Math.sin(angle);
    });
    bins.push({ frequency, power: real * real + imag * imag });
  }
  return bins;
}

function std(values) {
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
}

function averageQuality() {
  return state.samples.reduce((sum, sample) => sum + sample.quality, 0) / Math.max(1, state.samples.length);
}

function drawOverlay(roi, quality) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.strokeStyle = quality > 0.45 ? '#5eead4' : '#fbbf24';
  ctx.lineWidth = 4;
  ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
  ctx.fillStyle = 'rgba(94, 234, 212, 0.12)';
  ctx.fillRect(roi.x + roi.width * 0.18, roi.y + roi.height * 0.16, roi.width * 0.64, roi.height * 0.45);
}

function updateReadouts() {
  const duration = state.samples.length ? state.samples.at(-1).time - state.samples[0].time : 0;
  const fps = duration > 0 ? (state.samples.length - 1) / duration : 0;
  const quality = averageQuality();
  durationEl.textContent = `${Math.round(duration)}s`;
  fpsEl.textContent = fps ? `${fps.toFixed(1)} fps` : '-- fps';
  qualityEl.textContent = quality > 0.6 ? '优秀' : quality > 0.35 ? '可用' : '偏弱';
  roiModeEl.textContent = state.lastFace?.mode ?? '中心人脸先验';
}

startButton.addEventListener('click', start);
stopButton.addEventListener('click', stop);
window.addEventListener('beforeunload', stop);
