/**
 * Main thread: controls, rendering, and nothing else.
 *
 * Every sample of signal processing happens on the audio thread. This file
 * receives one analysis frame per chirp (~12 Hz), stashes it, and paints at
 * whatever rate the display runs — so a slow repaint can stretch a canvas
 * frame but can never cost us a chirp.
 */

import './style.css';
import { SonarSession, type DeviceReport } from './audio';
import { DEFAULT_CONFIG, type SonarConfig, type SonarFrameMessage } from './dsp/types';
import { rangeResolutionMeters, samplesToMeters } from './dsp/chirp';
import { SpectrogramView } from './ui/spectrogram';
import { RangePlotView } from './ui/rangeplot';
import { CwPlotView } from './ui/cwplot';

const session = new SonarSession();
let config: SonarConfig = { ...DEFAULT_CONFIG };
let latestFrame: SonarFrameMessage | null = null;
let deviceReport: DeviceReport | null = null;
let trackedBin: number | null = null;
let lastGestureAt = 0;
let lastGestureLabel = '—';
let framesSeen = 0;
let lastFrameTime = 0;
let frameRate = 0;

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
};

const toggleButton = el<HTMLButtonElement>('toggle');
const statusLabel = el<HTMLSpanElement>('status');
const warningsBox = el<HTMLDivElement>('warnings');
const audibilityNote = el<HTMLParagraphElement>('audibility');
const diagnosticsBox = el<HTMLDivElement>('diagnostics');
const cwPanel = el<HTMLElement>('cw-panel');

const spectrogram = new SpectrogramView(el<HTMLCanvasElement>('spec-canvas'));
const rangePlot = new RangePlotView(el<HTMLCanvasElement>('range-canvas'));
const cwPlot = new CwPlotView(el<HTMLCanvasElement>('cw-canvas'));

// ---------------------------------------------------------------- controls --

interface SliderSpec {
  key: string;
  name: string;
  min: number;
  max: number;
  step: number;
  get: () => number;
  set: (v: number) => void;
  format: (v: number) => string;
  hint?: string;
  visible?: () => boolean;
}

/**
 * Bandwidth is exposed rather than the sweep end frequency, because bandwidth
 * is the number that means something: range resolution is c/(2B) and nothing
 * else in the system depends on where the sweep ends.
 */
const sliders: SliderSpec[] = [
  {
    key: 'f0',
    name: 'Start frequency',
    min: 14000,
    max: 21000,
    step: 250,
    get: () => config.f0,
    set: (v) => {
      const bandwidth = config.f1 - config.f0;
      config.f0 = v;
      config.f1 = v + bandwidth;
    },
    format: (v) => `${(v / 1000).toFixed(2)} kHz`,
    hint: 'Lower is easier for the speaker, but more audible.',
  },
  {
    key: 'bandwidth',
    name: 'Bandwidth',
    min: 1000,
    max: 6000,
    step: 250,
    get: () => config.f1 - config.f0,
    set: (v) => {
      config.f1 = config.f0 + v;
    },
    format: (v) => `${(v / 1000).toFixed(2)} kHz  →  ${(rangeResolutionMeters(v) * 100).toFixed(1)} cm`,
    hint: 'Range resolution is c / (2 × bandwidth).',
  },
  {
    key: 'chirpDurationSec',
    name: 'Chirp duration',
    min: 0.004,
    max: 0.04,
    step: 0.001,
    get: () => config.chirpDurationSec,
    set: (v) => {
      config.chirpDurationSec = v;
    },
    format: (v) => `${(v * 1000).toFixed(0)} ms`,
    hint: 'Longer chirps carry more energy, so more range.',
  },
  {
    key: 'periodSamples',
    name: 'Repeat period',
    min: 11,
    max: 14,
    step: 1,
    get: () => Math.log2(config.periodSamples),
    set: (v) => {
      config.periodSamples = 2 ** Math.round(v);
    },
    format: (v) => {
      const samples = 2 ** Math.round(v);
      const sr = session.sampleRate || 48000;
      return `${samples} samples · ${((samples / sr) * 1000).toFixed(0)} ms · ${(sr / samples).toFixed(1)} fps`;
    },
    hint: 'A power of two, so correlation can wrap cleanly.',
  },
  {
    key: 'cwFrequency',
    name: 'CW carrier',
    min: 15000,
    max: 22000,
    step: 100,
    get: () => config.cwFrequency,
    set: (v) => {
      config.cwFrequency = v;
    },
    format: (v) => `${(v / 1000).toFixed(2)} kHz`,
    visible: () => config.mode === 'cw',
  },
  {
    key: 'maxRangeMeters',
    name: 'Display range',
    min: 0.5,
    max: 6,
    step: 0.25,
    get: () => config.maxRangeMeters,
    set: (v) => {
      config.maxRangeMeters = v;
    },
    format: (v) => `${v.toFixed(2)} m`,
  },
  {
    key: 'backgroundTimeConstantSec',
    name: 'Room memory',
    min: 0.5,
    max: 15,
    step: 0.5,
    get: () => config.backgroundTimeConstantSec,
    set: (v) => {
      config.backgroundTimeConstantSec = v;
    },
    format: (v) => `${v.toFixed(1)} s`,
    hint: 'How fast a stationary object fades into the background.',
  },
  {
    key: 'presenceThreshold',
    name: 'Sensitivity',
    min: 2,
    max: 20,
    step: 0.5,
    get: () => config.presenceThreshold,
    set: (v) => {
      config.presenceThreshold = v;
    },
    format: (v) => `${v.toFixed(1)} × noise`,
    hint: 'Lower detects more, and more false alarms.',
  },
  {
    key: 'guardBins',
    name: 'Guard zone',
    min: 2,
    max: 80,
    step: 1,
    get: () => config.guardBins,
    set: (v) => {
      config.guardBins = Math.round(v);
    },
    format: (v) => {
      const sr = session.sampleRate || 48000;
      return `${Math.round(v)} bins · ${(samplesToMeters(v, sr) * 100).toFixed(1)} cm`;
    },
    hint: 'Blind zone around the speaker-to-mic leakage.',
  },
  {
    key: 'volume',
    name: 'Transmit volume',
    min: 0,
    max: 1,
    step: 0.02,
    get: () => volume,
    set: (v) => {
      volume = v;
      session.setVolume(v);
    },
    format: (v) => `${Math.round(v * 100)}%`,
  },
];

let volume = 0.5;
const sliderNodes = new Map<string, { wrapper: HTMLElement; input: HTMLInputElement; value: HTMLElement }>();

function buildControls(): void {
  const container = el<HTMLDivElement>('controls');
  container.innerHTML = '';

  // Mode selector.
  const modeWrap = document.createElement('div');
  modeWrap.className = 'control';
  modeWrap.innerHTML = `
    <div class="control-top">
      <span class="control-name">Mode</span>
    </div>
    <select id="mode-select">
      <option value="fmcw">FMCW chirp — ranging + gestures</option>
      <option value="cw">Continuous tone — pure Doppler</option>
    </select>
    <span class="control-hint">FMCW measures distance; CW measures speed directly.</span>
  `;
  container.appendChild(modeWrap);
  const modeSelect = modeWrap.querySelector('select') as HTMLSelectElement;
  modeSelect.value = config.mode;
  modeSelect.addEventListener('change', () => {
    config.mode = modeSelect.value as SonarConfig['mode'];
    applyConfig();
    refreshControlVisibility();
  });

  for (const spec of sliders) {
    const wrapper = document.createElement('div');
    wrapper.className = 'control';
    wrapper.innerHTML = `
      <div class="control-top">
        <span class="control-name"></span>
        <span class="control-value"></span>
      </div>
      <input type="range" />
      ${spec.hint ? `<span class="control-hint">${spec.hint}</span>` : ''}
    `;
    const nameNode = wrapper.querySelector('.control-name') as HTMLElement;
    const valueNode = wrapper.querySelector('.control-value') as HTMLElement;
    const input = wrapper.querySelector('input') as HTMLInputElement;
    nameNode.textContent = spec.name;
    input.min = String(spec.min);
    input.max = String(spec.max);
    input.step = String(spec.step);
    input.value = String(spec.get());
    valueNode.textContent = spec.format(spec.get());

    input.addEventListener('input', () => {
      spec.set(Number(input.value));
      valueNode.textContent = spec.format(spec.get());
      if (spec.key !== 'volume') applyConfig();
      updateAudibilityNote();
      syncSliderValues();
    });

    container.appendChild(wrapper);
    sliderNodes.set(spec.key, { wrapper, input, value: valueNode });
  }

  // Freeze background.
  const freezeWrap = document.createElement('div');
  freezeWrap.className = 'control checkbox';
  freezeWrap.innerHTML = `
    <input type="checkbox" id="freeze-bg" />
    <label for="freeze-bg" class="control-name">Freeze room model</label>
  `;
  container.appendChild(freezeWrap);
  const freeze = freezeWrap.querySelector('input') as HTMLInputElement;
  freeze.addEventListener('change', () => {
    config.freezeBackground = freeze.checked;
    applyConfig();
  });

  refreshControlVisibility();
  updateAudibilityNote();
}

function refreshControlVisibility(): void {
  for (const spec of sliders) {
    const node = sliderNodes.get(spec.key);
    if (!node) continue;
    node.wrapper.hidden = spec.visible ? !spec.visible() : false;
  }
  cwPanel.hidden = config.mode !== 'cw';
}

/** Keep displayed values in step when one control changes another. */
function syncSliderValues(): void {
  for (const spec of sliders) {
    const node = sliderNodes.get(spec.key);
    if (!node) continue;
    if (document.activeElement !== node.input) node.input.value = String(spec.get());
    node.value.textContent = spec.format(spec.get());
  }
}

function applyConfig(): void {
  config = session.updateConfig(config);
  rangePlot.setMaxRange(config.maxRangeMeters);
  syncSliderValues();
}

function updateAudibilityNote(): void {
  const low = config.mode === 'cw' ? config.cwFrequency : Math.min(config.f0, config.f1);
  const parts: string[] = [];
  if (low < 17000) {
    parts.push(
      `At ${(low / 1000).toFixed(1)} kHz this is clearly audible to most people. Keep the volume low.`,
    );
  } else if (low < 19000) {
    parts.push(
      `At ${(low / 1000).toFixed(1)} kHz many people under about 25 will hear this as a faint whine.`,
    );
  } else {
    parts.push(`At ${(low / 1000).toFixed(1)} kHz most adults will not hear this.`);
  }
  parts.push(
    'Dogs, cats and rodents hear far above 20 kHz and will hear it at any setting — if a pet is in the room, keep sessions short and the volume down.',
  );
  audibilityNote.textContent = parts.join(' ');
}

// ---------------------------------------------------------------- warnings --

function renderWarnings(report: DeviceReport | null): void {
  const items: { level: string; title: string; body: string }[] = [];

  if (report) {
    if (!report.constraintsHonoured) {
      items.push({
        level: 'bad',
        title: 'The browser is still processing the microphone',
        body: `${report.unhonoured.join(', ')} could not be turned off. Echo cancellation removes exactly the reflected copy of our own chirp that this depends on, so expect a flat, empty range profile. Try Chrome, or a different input device.`,
      });
    }
    if (report.likelyBluetooth) {
      items.push({
        level: 'bad',
        title: 'This looks like a Bluetooth audio path',
        body: `Bluetooth codecs low-pass well below 20 kHz and add tens of milliseconds of wandering latency. Switch to the built-in speaker and microphone. (Input: ${report.inputLabel || 'unknown'})`,
      });
    }
    if (report.sampleRateAdjusted || report.sampleRate !== 48000) {
      items.push({
        level: 'info',
        title: `Running at ${(report.sampleRate / 1000).toFixed(1)} kHz`,
        body:
          report.sampleRate < 48000
            ? 'The sweep has been moved down to stay below Nyquist, which costs some bandwidth and therefore some range resolution.'
            : 'Sample rate differs from the requested 48 kHz; the band was checked against Nyquist.',
      });
    }
  }

  if (items.length === 0) {
    warningsBox.hidden = true;
    warningsBox.innerHTML = '';
    return;
  }
  warningsBox.hidden = false;
  warningsBox.innerHTML = items
    .map(
      (i) =>
        `<div class="warning ${i.level}"><b>${escapeHtml(i.title)}</b>${escapeHtml(i.body)}</div>`,
    )
    .join('');
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] as string,
  );
}

// ------------------------------------------------------------- diagnostics --

function renderDiagnostics(): void {
  if (!deviceReport) return;
  const r = deviceReport;
  const d = latestFrame?.diagnostics;
  const sr = r.sampleRate;

  const rows: [string, string, string?][] = [
    ['Sample rate', `${sr.toLocaleString()} Hz`, sr >= 48000 ? 'ok' : 'warn'],
    [
      'Echo cancellation',
      r.echoCancellation === false ? 'off' : r.echoCancellation === true ? 'ON — breaks sonar' : 'unknown',
      r.echoCancellation === false ? 'ok' : r.echoCancellation === true ? 'bad' : 'warn',
    ],
    [
      'Noise suppression',
      r.noiseSuppression === false ? 'off' : r.noiseSuppression === true ? 'ON' : 'unknown',
      r.noiseSuppression === false ? 'ok' : r.noiseSuppression === true ? 'bad' : 'warn',
    ],
    [
      'Auto gain control',
      r.autoGainControl === false ? 'off' : r.autoGainControl === true ? 'ON' : 'unknown',
      r.autoGainControl === false ? 'ok' : r.autoGainControl === true ? 'bad' : 'warn',
    ],
    ['Input device', r.inputLabel || 'unnamed'],
    ['Graph latency', `${((r.baseLatency + r.outputLatency) * 1000).toFixed(1)} ms`],
    ['Update rate', frameRate > 0 ? `${frameRate.toFixed(1)} fps` : '—'],
  ];

  if (d) {
    const bin = samplesToMeters(1, sr);
    rows.push(
      [
        'Direct-path delay',
        `${d.directPathIndex.toFixed(1)} samples · ${((d.directPathIndex / sr) * 1000).toFixed(2)} ms`,
      ],
      ['Range bin size', `${(bin * 100).toFixed(2)} cm`],
      ['In-band mic level', `${d.inBandLevelDb.toFixed(1)} dBFS`, d.inBandLevelDb > -70 ? 'ok' : 'warn'],
      ['Residual noise floor', `${d.noiseFloorDb.toFixed(1)} dB below direct path`],
      [
        'Spectral tilt across sweep',
        `${d.spectralTiltDb >= 0 ? '+' : ''}${d.spectralTiltDb.toFixed(1)} dB`,
        d.spectralTiltDb > -20 ? 'ok' : 'warn',
      ],
      [
        'Usable bandwidth',
        d.usableBandwidthHz > 0
          ? `${(d.usableBandwidthHz / 1000).toFixed(2)} kHz → ${(rangeResolutionMeters(d.usableBandwidthHz) * 100).toFixed(1)} cm`
          : 'measuring…',
        d.usableBandwidthHz > (config.f1 - config.f0) * 0.6 ? 'ok' : 'warn',
      ],
    );
  }

  diagnosticsBox.innerHTML = rows
    .map(
      ([k, v, cls]) =>
        `<div class="diag-row"><span class="diag-key">${escapeHtml(k)}</span><span class="diag-val ${cls ?? ''}">${escapeHtml(v)}</span></div>`,
    )
    .join('');
}

// ------------------------------------------------------------------ render --

function setCard(id: string, valueId: string, subId: string, value: string, sub: string, cls: string): void {
  const card = el<HTMLElement>(id);
  card.classList.remove('active-good', 'active-violet', 'active-accent');
  if (cls) card.classList.add(cls);
  el<HTMLElement>(valueId).textContent = value;
  el<HTMLElement>(subId).textContent = sub;
}

function renderReadouts(): void {
  const f = latestFrame;
  if (!f) {
    setCard('card-presence', 'presence-value', 'presence-sub', '—', 'not running', '');
    return;
  }

  setCard(
    'card-presence',
    'presence-value',
    'presence-sub',
    f.presence ? 'PRESENCE' : 'CLEAR',
    f.presence ? 'something is moving' : 'nothing moving',
    f.presence ? 'active-good' : '',
  );

  const v = f.velocity;
  const moving = Math.abs(v) > 0.05 && f.motionEnergy > 3;
  setCard(
    'card-direction',
    'direction-value',
    'direction-sub',
    moving ? `${v > 0 ? '↓' : '↑'} ${Math.abs(v).toFixed(2)}` : '—',
    moving ? (v > 0 ? 'approaching, m/s' : 'receding, m/s') : 'approach / recede',
    moving ? 'active-accent' : '',
  );

  if (f.gesture !== 'none') {
    lastGestureLabel = f.gesture === 'swipeToward' ? 'SWIPE IN' : 'SWIPE OUT';
    lastGestureAt = performance.now();
  }
  const gestureFresh = performance.now() - lastGestureAt < 1600;
  setCard(
    'card-gesture',
    'gesture-value',
    'gesture-sub',
    gestureFresh ? lastGestureLabel : '—',
    gestureFresh ? 'just now' : 'swipe toward or away',
    gestureFresh ? 'active-violet' : '',
  );

  const b = f.breathing;
  const breathValue = b.bpm === null ? '—' : `${b.bpm.toFixed(1)}`;
  const breathSub =
    b.bpm === null
      ? b.fill < 0.35
        ? `collecting… ${Math.round(b.fill * 100)}%`
        : 'no steady rhythm'
      : `bpm at ${(b.trackedRangeMeters * 100).toFixed(0)} cm · conf ${b.confidence.toFixed(1)}`;
  setCard(
    'card-breathing',
    'breathing-value',
    'breathing-sub',
    breathValue,
    breathSub,
    b.bpm !== null ? 'active-accent' : '',
  );
}

function draw(): void {
  const f = latestFrame;
  if (f) {
    spectrogram.render();
    spectrogram.markBand(config.f0, config.f1);

    if (config.mode === 'cw') {
      cwPlot.render(f.cwSpectrum, f.cwHzPerBin, f.velocity);
    }
    rangePlot.render({
      raw: f.rawProfile,
      subtracted: f.profile,
      metersPerBin: f.metersPerBin,
      targets: f.targets,
      guardBins: config.guardBins,
      trackedBin,
    });
    renderReadouts();
  }
  requestAnimationFrame(draw);
}

// ------------------------------------------------------------------ wiring --

session.onFrame = (frame) => {
  latestFrame = frame;
  framesSeen++;
  const now = performance.now();
  if (lastFrameTime > 0) {
    const dt = (now - lastFrameTime) / 1000;
    frameRate = frameRate === 0 ? 1 / dt : frameRate * 0.9 + (1 / dt) * 0.1;
  }
  lastFrameTime = now;

  spectrogram.push(frame.spectrogram, frame.spectrogramBins, frame.spectrogramHzPerBin);

  if (framesSeen % 6 === 0) renderDiagnostics();
  statusLabel.textContent = `Running · ${framesSeen} chirps · ${frameRate.toFixed(1)} fps`;
};

toggleButton.addEventListener('click', async () => {
  if (session.running) {
    await session.stop();
    toggleButton.textContent = 'Start';
    toggleButton.dataset.running = 'false';
    statusLabel.textContent = 'Stopped';
    latestFrame = null;
    return;
  }

  toggleButton.disabled = true;
  statusLabel.textContent = 'Requesting microphone…';
  try {
    const report = await session.start(config);
    deviceReport = report;
    config = session.currentConfig;
    spectrogram.clear();
    spectrogram.setBand(0, report.sampleRate / 2);
    rangePlot.setMaxRange(config.maxRangeMeters);
    framesSeen = 0;
    lastFrameTime = 0;
    frameRate = 0;
    renderWarnings(report);
    renderDiagnostics();
    syncSliderValues();
    toggleButton.textContent = 'Stop';
    toggleButton.dataset.running = 'true';
    statusLabel.textContent = 'Running';
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    statusLabel.textContent = 'Could not start';
    warningsBox.hidden = false;
    warningsBox.innerHTML = `<div class="warning bad"><b>Could not start</b>${escapeHtml(message)}</div>`;
    await session.stop().catch(() => {});
  } finally {
    toggleButton.disabled = false;
  }
});

el<HTMLButtonElement>('reset-bg').addEventListener('click', () => {
  session.resetBackground();
  statusLabel.textContent = 'Relearning room…';
});

el<HTMLButtonElement>('clear-track').addEventListener('click', () => {
  trackedBin = null;
  session.setTrackedBin(null);
});

// Click the range plot to pin the bin used for phase tracking.
el<HTMLCanvasElement>('range-canvas').addEventListener('click', (event) => {
  const canvas = event.currentTarget as HTMLCanvasElement;
  const rect = canvas.getBoundingClientRect();
  const xCss = event.clientX - rect.left;
  const x = (xCss / rect.width) * canvas.width;
  const padL = 46;
  const padR = 12;
  const plotW = canvas.width - padL - padR;
  const frac = (x - padL) / plotW;
  if (frac < 0 || frac > 1 || !latestFrame) return;
  const binsShown = Math.min(
    latestFrame.rawProfile.length,
    Math.ceil(config.maxRangeMeters / latestFrame.metersPerBin),
  );
  trackedBin = Math.round(frac * (binsShown - 1));
  session.setTrackedBin(trackedBin);
});

buildControls();
renderWarnings(null);
requestAnimationFrame(draw);
