/**
 * The sonar processing chain.
 *
 * This class is deliberately free of any Web Audio dependency: it takes blocks
 * of microphone samples and produces analysis frames. That makes the entire
 * chain testable in Node against synthetic echoes, which is the only practical
 * way to know the DSP is right — on real hardware every bug looks like "bad
 * SNR".
 *
 * ---------------------------------------------------------------------------
 * Why velocity is measured two different ways
 * ---------------------------------------------------------------------------
 * The obvious approach — differencing the correlation-peak phase between
 * consecutive chirps — aliases almost immediately. One chirp period is 85 ms,
 * and phase wraps once the target moves half a wavelength (~4 mm at 20 kHz),
 * so the unambiguous velocity is
 *
 *     v_max = c / (4 * f_c * T) = 343 / (4 * 20000 * 0.0853) ~= 0.05 m/s
 *
 * A hand swipe is 20x faster than that, so inter-chirp phase is useless for
 * gestures. It is, however, exactly the right tool for breathing, where the
 * chest surface moves at a few mm/s.
 *
 * So the chain measures motion at two scales:
 *   - macro (gestures, presence): how far the moving-target energy *migrates
 *     between range bins* from one chirp to the next. Unambiguous to several
 *     m/s, and robust because it does not care about phase at all.
 *   - micro (breathing): unwrapped phase of one tracked range bin, which
 *     resolves sub-millimetre displacement.
 *
 * CW mode adds a third, independent measurement: a pure tone with true Doppler
 * sidebands, where approach and recede are separated by the sign of the
 * frequency shift.
 */

import { BiquadCascade } from './biquad';
import { FFT } from './fft';
import { Correlator, argMax, parabolicPeakOffset } from './xcorr';
import {
  SPEED_OF_SOUND,
  generateTransmitPeriod,
  generateTone,
  metersToSamples,
  samplesToMeters,
} from './chirp';
import { blackmanHarris, hann } from './window';
import { PhaseUnwrapper, detrend } from './unwrap';
import type {
  BreathingEstimate,
  DetectedTarget,
  GestureKind,
  SonarConfig,
  SonarDiagnostics,
} from './types';

/** Spectrogram analysis size. Short enough that a 15 ms chirp reads as a
 *  diagonal stripe rather than a smear: the 5.3 ms window sees ~1.4 kHz of a
 *  4 kHz sweep, and the 2.7 ms hop puts ~6 columns across each chirp. */
const SPEC_FFT = 256;
const SPEC_HOP = 128;

/** Frames of phase history kept for breathing analysis (~44 s at 11.7 Hz). */
const BREATH_HISTORY = 512;
/** Zero-padded FFT length for the breathing periodogram. */
const BREATH_FFT = 1024;
const BREATH_MIN_HZ = 0.1;
const BREATH_MAX_HZ = 0.7;

/** Frames of velocity history used for gesture classification (~1 s). */
const GESTURE_HISTORY = 14;

/**
 * Floor on the estimated noise level, relative to the direct-path peak.
 *
 * Detection works on the ratio of peak to noise, which misbehaves when the
 * noise estimate approaches zero: in a very quiet room a residual of 1e-6
 * divided by a noise floor of 1e-7 is a "10-sigma detection" of nothing at
 * all. Real returns sit 25-40 dB below the direct path and real rooms are far
 * noisier than this, so at -74 dB this floor never binds in practice — it just
 * stops the ratio running away.
 */
const MIN_NOISE_FLOOR = 2e-4;

export interface EngineFrame {
  frameIndex: number;
  time: number;
  /** Background-subtracted profile, normalised to the direct path. */
  profile: Float32Array;
  /** Raw profile, normalised to the direct path. */
  rawProfile: Float32Array;
  analysisBins: number;
  metersPerBin: number;
  presence: boolean;
  velocity: number;
  motionEnergy: number;
  targets: DetectedTarget[];
  gesture: GestureKind;
  breathing: BreathingEstimate;
  diagnostics: SonarDiagnostics;
  cwSpectrum: Float32Array | null;
  cwBins: number;
  cwHzPerBin: number;
  /** Spectrogram columns produced since the previous frame. */
  spectrogram: Uint8Array;
  spectrogramColumns: number;
  spectrogramBins: number;
  spectrogramHzPerBin: number;
}

export class SonarEngine {
  readonly sampleRate: number;
  private config: SonarConfig;

  // --- transmit ---------------------------------------------------------
  private txBuffer!: Float32Array;
  private actualCwFrequency = 0;

  // --- receive plumbing -------------------------------------------------
  private filter!: BiquadCascade;
  private frameBuffer!: Float32Array;
  private frameFill = 0;

  // --- FMCW correlation -------------------------------------------------
  private correlator!: Correlator;
  private corrRe!: Float64Array;
  private corrIm!: Float64Array;
  private corrMag!: Float64Array;

  // --- range profiles ---------------------------------------------------
  private analysisBins = 0;
  private profileMag!: Float32Array;
  private profileRe!: Float32Array;
  private profileIm!: Float32Array;
  private background!: Float32Array;
  private subtracted!: Float32Array;
  private prevSubtracted!: Float32Array;
  private backgroundReady = false;
  private backgroundFrames = 0;

  // --- CW ---------------------------------------------------------------
  private cwFft!: FFT;
  private cwWindow!: Float32Array;
  private cwRe!: Float64Array;
  private cwIm!: Float64Array;
  private cwSpectrum!: Float32Array;
  private cwBins = 0;

  // --- spectrogram ------------------------------------------------------
  private specFft = new FFT(SPEC_FFT);
  private specWindow = hann(SPEC_FFT, true);
  private specRe = new Float64Array(SPEC_FFT);
  private specIm = new Float64Array(SPEC_FFT);
  private specRing = new Float32Array(SPEC_FFT);
  private specRingFill = 0;
  private specBins = SPEC_FFT / 2 + 1;
  private specOut!: Uint8Array;
  private specColumns = 0;
  private specAverage = new Float32Array(SPEC_FFT / 2 + 1);
  private specAverageReady = false;

  // --- tracking state ---------------------------------------------------
  private frameIndex = 0;
  private unwrapper = new PhaseUnwrapper();
  private phaseHistory = new Float64Array(BREATH_HISTORY);
  private phaseFill = 0;
  private phaseWrite = 0;
  private breathFft = new FFT(BREATH_FFT);
  private breathRe = new Float64Array(BREATH_FFT);
  private breathIm = new Float64Array(BREATH_FFT);
  private breathScratch = new Float64Array(BREATH_HISTORY);
  private breathWindow = hann(BREATH_HISTORY, true);
  private breathResult: BreathingEstimate = {
    bpm: null,
    confidence: 0,
    fill: 0,
    trackedRangeMeters: 0,
  };
  private trackedBin = -1;
  private trackedBinLocked = false;
  private longTermProfile!: Float32Array;

  private velocityHistory = new Float32Array(GESTURE_HISTORY);
  private motionHistory = new Float32Array(GESTURE_HISTORY);
  private gestureWrite = 0;
  private gestureRefractory = 0;

  /** Preallocated scratch for the profile-to-profile correlation. */
  private maxLagBins = 0;
  private lagScores!: Float64Array;
  /** Preallocated Hann window for a partially-filled breathing buffer. */
  private breathPartialWindow = new Float64Array(BREATH_HISTORY);
  private breathPartialLength = -1;

  private presenceState = false;
  private presenceCounter = 0;

  private velocityLag = 0;

  constructor(sampleRate: number, config: SonarConfig) {
    this.sampleRate = sampleRate;
    this.config = { ...config };
    this.rebuild();
  }

  /** The waveform the transmitter should loop, exactly one period long. */
  get transmitBuffer(): Float32Array {
    return this.txBuffer;
  }

  /** Actual CW carrier after snapping it to a whole number of cycles. */
  get cwCarrierFrequency(): number {
    return this.actualCwFrequency;
  }

  get framePeriodSeconds(): number {
    return this.config.periodSamples / this.sampleRate;
  }

  get metersPerBin(): number {
    return samplesToMeters(1, this.sampleRate);
  }

  getConfig(): SonarConfig {
    return { ...this.config };
  }

  /**
   * Apply new settings. Anything that changes the transmitted waveform or the
   * buffer geometry forces a rebuild; the caller must then re-upload
   * `transmitBuffer` to the transmitter.
   */
  setConfig(next: SonarConfig): { transmitChanged: boolean } {
    const prev = this.config;
    const transmitChanged =
      prev.periodSamples !== next.periodSamples ||
      prev.f0 !== next.f0 ||
      prev.f1 !== next.f1 ||
      prev.chirpDurationSec !== next.chirpDurationSec ||
      prev.taperSec !== next.taperSec ||
      prev.mode !== next.mode ||
      prev.cwFrequency !== next.cwFrequency;
    const geometryChanged =
      transmitChanged || prev.maxRangeMeters !== next.maxRangeMeters;

    this.config = { ...next };
    if (geometryChanged) this.rebuild();
    return { transmitChanged };
  }

  resetBackground(): void {
    this.background.fill(0);
    this.backgroundReady = false;
    this.backgroundFrames = 0;
    this.prevSubtracted.fill(0);
  }

  /** Pin phase tracking to a specific range bin, or null to auto-select. */
  setTrackedBin(bin: number | null): void {
    if (bin === null) {
      this.trackedBinLocked = false;
      this.trackedBin = -1;
    } else {
      this.trackedBinLocked = true;
      this.trackedBin = Math.min(Math.max(0, Math.floor(bin)), this.analysisBins - 1);
    }
    this.unwrapper.reset();
    this.phaseFill = 0;
    this.phaseWrite = 0;
    this.breathResult = { bpm: null, confidence: 0, fill: 0, trackedRangeMeters: 0 };
  }

  private rebuild(): void {
    const cfg = this.config;
    const sr = this.sampleRate;
    const n = cfg.periodSamples;

    if ((n & (n - 1)) !== 0) {
      throw new Error(`periodSamples must be a power of two, got ${n}`);
    }

    // Transmit waveform.
    if (cfg.mode === 'cw') {
      const tone = generateTone({
        sampleRate: sr,
        frequency: cfg.cwFrequency,
        periodSamples: n,
        amplitude: 0.8,
      });
      this.txBuffer = tone.samples;
      this.actualCwFrequency = tone.actualFrequency;
    } else {
      this.txBuffer = generateTransmitPeriod({
        sampleRate: sr,
        f0: cfg.f0,
        f1: cfg.f1,
        durationSec: cfg.chirpDurationSec,
        taperSec: cfg.taperSec,
        amplitude: 0.9,
        periodSamples: n,
      });
      this.actualCwFrequency = cfg.cwFrequency;
    }

    // Front-end high-pass. A high-pass rather than a band-pass so the top of
    // the sweep stays flat: at 44.1 kHz the sweep already runs close to
    // Nyquist and a band-pass would tilt it further.
    const bandLow = cfg.mode === 'cw' ? cfg.cwFrequency : Math.min(cfg.f0, cfg.f1);
    const cutoff = Math.max(500, Math.min(bandLow - 2000, sr * 0.45));
    this.filter = BiquadCascade.highpass(sr, cutoff, 3);

    this.frameBuffer = new Float32Array(n);
    this.frameFill = 0;

    // Correlate against the *filtered* template, not the raw one.
    //
    // A 6th-order high-pass with its corner at 16 kHz has 12 samples of group
    // delay at 18 kHz and 39 at 22 kHz. That dispersion stretches the sweep
    // unevenly, so a matched filter built from the unfiltered chirp is no
    // longer matched: the compressed pulse grows a long asymmetric tail and
    // its peak lands several bins late — centimetres of range error, and a
    // direct-path skirt that swamps nearby targets. Pushing the template
    // through an identical filter restores the match and compensates the
    // dispersion exactly.
    //
    // The template is filtered twice and only the second pass kept, because
    // the transmitted signal loops forever: the second pass starts from the
    // filter state the first one left behind, which is the periodic steady
    // state the microphone actually sees.
    const templateFilter = BiquadCascade.highpass(sr, cutoff, 3);
    const warmup = new Float32Array(n);
    const template = new Float32Array(n);
    templateFilter.process(this.txBuffer, warmup);
    templateFilter.process(this.txBuffer, template);

    if (cfg.mode !== 'cw') {
      applyReceiveTaper(template, Math.round(cfg.chirpDurationSec * sr));
    }

    this.correlator = new Correlator(n, template, { analytic: true });
    this.corrRe = new Float64Array(n);
    this.corrIm = new Float64Array(n);
    this.corrMag = new Float64Array(n);

    const maxBins = Math.min(
      n >> 1,
      Math.max(32, Math.ceil(metersToSamples(cfg.maxRangeMeters, sr))),
    );
    this.analysisBins = maxBins;
    this.profileMag = new Float32Array(maxBins);
    this.profileRe = new Float32Array(maxBins);
    this.profileIm = new Float32Array(maxBins);
    this.background = new Float32Array(maxBins);
    this.subtracted = new Float32Array(maxBins);
    this.prevSubtracted = new Float32Array(maxBins);
    this.longTermProfile = new Float32Array(maxBins);
    this.backgroundReady = false;
    this.backgroundFrames = 0;

    // CW analysis.
    this.cwFft = new FFT(n);
    this.cwWindow = blackmanHarris(n, true);
    this.cwRe = new Float64Array(n);
    this.cwIm = new Float64Array(n);
    this.cwBins = 129; // +/-64 bins around the carrier
    this.cwSpectrum = new Float32Array(this.cwBins);

    // Spectrogram output: one flush per frame.
    const columnsPerFrame = Math.ceil(n / SPEC_HOP) + 2;
    this.specOut = new Uint8Array(columnsPerFrame * this.specBins);
    this.specColumns = 0;
    this.specRingFill = 0;
    this.specRing.fill(0);

    // +/-3 m/s of radial motion is well beyond any hand gesture, and bounds
    // the profile-to-profile search.
    const dt = n / sr;
    const guard = Math.max(1, Math.min(cfg.guardBins, maxBins - 2));
    this.maxLagBins = Math.max(
      4,
      Math.min(maxBins - guard - 2, Math.ceil(metersToSamples(3 * dt, sr))),
    );
    this.lagScores = new Float64Array(2 * this.maxLagBins + 1);
    this.breathPartialLength = -1;

    this.unwrapper.reset();
    this.phaseFill = 0;
    this.phaseWrite = 0;
    if (!this.trackedBinLocked) this.trackedBin = -1;
    this.velocityHistory.fill(0);
    this.motionHistory.fill(0);
    this.presenceState = false;
    this.presenceCounter = 0;
  }

  /**
   * Feed microphone samples. Calls `onFrame` once per completed transmit
   * period. The callback receives a view onto internal buffers — copy anything
   * that needs to outlive the call.
   */
  push(input: Float32Array, count: number, onFrame: (frame: EngineFrame) => void): void {
    const buf = this.frameBuffer;
    const n = buf.length;
    for (let i = 0; i < count; i++) {
      const x = this.filter.processSample(input[i]);
      buf[this.frameFill++] = x;

      // Spectrogram runs on its own shorter hop, independent of the frame.
      this.specRing[this.specRingFill++] = x;
      if (this.specRingFill === SPEC_FFT) {
        this.emitSpectrogramColumn();
        // Slide by the hop: keep the newest (SPEC_FFT - SPEC_HOP) samples.
        this.specRing.copyWithin(0, SPEC_HOP, SPEC_FFT);
        this.specRingFill = SPEC_FFT - SPEC_HOP;
      }

      if (this.frameFill === n) {
        this.frameFill = 0;
        onFrame(this.processFrame());
        this.specColumns = 0;
      }
    }
  }

  private emitSpectrogramColumn(): void {
    const re = this.specRe;
    const im = this.specIm;
    const win = this.specWindow;
    for (let i = 0; i < SPEC_FFT; i++) {
      re[i] = this.specRing[i] * win[i];
      im[i] = 0;
    }
    this.specFft.forward(re, im);

    const bins = this.specBins;
    const base = this.specColumns * bins;
    if (base + bins > this.specOut.length) return; // ring overflowed; drop
    const norm = 2 / SPEC_FFT;
    const avgAlpha = 0.02;
    for (let k = 0; k < bins; k++) {
      const mag = Math.hypot(re[k], im[k]) * norm;
      const db = 20 * Math.log10(mag + 1e-12);
      // Map -110..-10 dBFS onto 0..255 for transport.
      const v = ((db + 110) / 100) * 255;
      this.specOut[base + k] = v < 0 ? 0 : v > 255 ? 255 : v | 0;

      if (this.specAverageReady) {
        this.specAverage[k] += avgAlpha * (db - this.specAverage[k]);
      } else {
        this.specAverage[k] = db;
      }
    }
    this.specAverageReady = true;
    this.specColumns++;
  }

  private processFrame(): EngineFrame {
    const cfg = this.config;
    const dt = this.framePeriodSeconds;
    this.frameIndex++;
    const time = this.frameIndex * dt;

    const inBandLevelDb = 20 * Math.log10(rms(this.frameBuffer) + 1e-12);

    if (cfg.mode === 'cw') {
      return this.processCwFrame(time, inBandLevelDb);
    }
    return this.processFmcwFrame(time, dt, inBandLevelDb);
  }

  // -----------------------------------------------------------------------
  // FMCW
  // -----------------------------------------------------------------------

  private processFmcwFrame(time: number, dt: number, inBandLevelDb: number): EngineFrame {
    const cfg = this.config;
    const n = cfg.periodSamples;
    const bins = this.analysisBins;

    this.correlator.correlate(this.frameBuffer, this.corrRe, this.corrIm);
    for (let i = 0; i < n; i++) {
      this.corrMag[i] = Math.hypot(this.corrRe[i], this.corrIm[i]);
    }

    // --- Direct-path self-calibration ------------------------------------
    // The loudest thing in the room is always the speaker talking straight
    // into the microphone a few centimetres away. Where that lands in the
    // period depends on the browser's output+input latency, which is tens of
    // milliseconds, varies per device, and drifts. So we never assume it: we
    // find it every single frame and call it range zero. Everything else is
    // measured from there, which is what makes the system work with no
    // per-device calibration step at all.
    const directIdx = argMax(this.corrMag, 0, n);
    const directLevel = this.corrMag[directIdx];
    const directOffset = parabolicPeakOffset(this.corrMag, directIdx);
    const inv = directLevel > 0 ? 1 / directLevel : 0;

    for (let j = 0; j < bins; j++) {
      const idx = (directIdx + j) % n;
      this.profileMag[j] = this.corrMag[idx] * inv;
      this.profileRe[j] = this.corrRe[idx] * inv;
      this.profileIm[j] = this.corrIm[idx] * inv;
    }

    // --- Background subtraction ------------------------------------------
    // Walls, the table and the laptop's own chassis all return constant
    // echoes that dwarf a hand. An exponential moving average of the profile
    // converges to everything that is not moving; subtracting it leaves only
    // what changed.
    const alpha = cfg.freezeBackground
      ? 0
      : 1 - Math.exp(-dt / Math.max(0.05, cfg.backgroundTimeConstantSec));
    const bg = this.background;
    if (!this.backgroundReady) {
      bg.set(this.profileMag);
      this.backgroundFrames++;
      // Give the average a few frames before trusting any detection.
      if (this.backgroundFrames > 3) this.backgroundReady = true;
    } else {
      for (let j = 0; j < bins; j++) {
        bg[j] += alpha * (this.profileMag[j] - bg[j]);
      }
    }
    for (let j = 0; j < bins; j++) {
      this.subtracted[j] = this.profileMag[j] - bg[j];
    }

    // Long-term (very slow) profile used only to pick a bin for phase
    // tracking: it must survive a person sitting perfectly still, so it
    // cannot come from the background-subtracted signal.
    const ltAlpha = 1 - Math.exp(-dt / 10);
    for (let j = 0; j < bins; j++) {
      this.longTermProfile[j] += ltAlpha * (this.profileMag[j] - this.longTermProfile[j]);
    }

    const guard = Math.max(1, Math.min(cfg.guardBins, bins - 2));
    const noiseFloor = Math.max(
      robustNoiseFloor(this.subtracted, guard, bins),
      MIN_NOISE_FLOOR,
    );

    // --- Presence ---------------------------------------------------------
    let peakVal = 0;
    for (let j = guard; j < bins; j++) {
      const v = this.subtracted[j];
      if (v > peakVal) peakVal = v;
    }
    const snr = noiseFloor > 0 ? peakVal / noiseFloor : 0;
    const enterThreshold = cfg.presenceThreshold;
    const exitThreshold = cfg.presenceThreshold * 0.6;
    const wants = this.presenceState ? snr > exitThreshold : snr > enterThreshold;
    if (wants === this.presenceState) {
      this.presenceCounter = 0;
    } else if (++this.presenceCounter >= 2) {
      // Two consecutive frames (~170 ms) to flip, comfortably inside the
      // one-second requirement while rejecting single-frame noise.
      this.presenceState = wants;
      this.presenceCounter = 0;
    }
    const presence = this.backgroundReady && this.presenceState;

    // --- Macro velocity from range migration ------------------------------
    const { velocity, quality, lagBins } = this.estimateMigrationVelocity(guard, bins, dt);
    this.velocityLag = lagBins;

    let motionEnergy = 0;
    for (let j = guard; j < bins; j++) {
      motionEnergy += Math.abs(this.subtracted[j] - this.prevSubtracted[j]);
    }
    motionEnergy /= Math.max(1, bins - guard);
    const motionRelative = noiseFloor > 0 ? motionEnergy / noiseFloor : 0;

    this.prevSubtracted.set(this.subtracted);

    // --- Targets ----------------------------------------------------------
    const targets = this.findTargets(guard, bins, noiseFloor, velocity);

    // --- Gestures ---------------------------------------------------------
    const gesture = this.classifyGesture(velocity, motionRelative, quality);

    // --- Breathing --------------------------------------------------------
    const breathing = this.updateBreathing(guard, bins);

    // --- Diagnostics ------------------------------------------------------
    const diagnostics: SonarDiagnostics = {
      inBandLevelDb,
      noiseFloorDb: 20 * Math.log10(noiseFloor + 1e-12),
      directPathIndex: directIdx + directOffset,
      directPathLevel: directLevel,
      ...this.measureBandQuality(),
    };

    return {
      frameIndex: this.frameIndex,
      time,
      profile: this.subtracted,
      rawProfile: this.profileMag,
      analysisBins: bins,
      metersPerBin: this.metersPerBin,
      presence,
      velocity,
      motionEnergy: motionRelative,
      targets,
      gesture,
      breathing,
      diagnostics,
      cwSpectrum: null,
      cwBins: 0,
      cwHzPerBin: 0,
      spectrogram: this.specOut,
      spectrogramColumns: this.specColumns,
      spectrogramBins: this.specBins,
      spectrogramHzPerBin: this.sampleRate / SPEC_FFT,
    };
  }

  /**
   * How far did the moving-target energy slide between bins since the last
   * chirp? Correlating the current subtracted profile against the previous one
   * over a bounded lag range gives that directly, in bins, with no phase
   * ambiguity whatsoever. Positive lag means the energy moved to longer range.
   */
  private estimateMigrationVelocity(
    guard: number,
    bins: number,
    dt: number,
  ): { velocity: number; quality: number; lagBins: number } {
    const maxLag = this.maxLagBins;
    const cur = this.subtracted;
    const prev = this.prevSubtracted;

    let energyCur = 0;
    let energyPrev = 0;
    for (let j = guard; j < bins; j++) {
      const a = cur[j] > 0 ? cur[j] : 0;
      const b = prev[j] > 0 ? prev[j] : 0;
      energyCur += a * a;
      energyPrev += b * b;
    }
    if (energyCur <= 0 || energyPrev <= 0) {
      return { velocity: 0, quality: 0, lagBins: 0 };
    }
    const norm = 1 / Math.sqrt(energyCur * energyPrev);

    let bestLag = 0;
    let bestScore = -Infinity;
    const scores = this.lagScores;
    for (let lag = -maxLag; lag <= maxLag; lag++) {
      let s = 0;
      const from = Math.max(guard, guard - lag);
      const to = Math.min(bins, bins - lag);
      for (let j = from; j < to; j++) {
        const a = cur[j] > 0 ? cur[j] : 0;
        const b = prev[j - lag] > 0 ? prev[j - lag] : 0;
        s += a * b;
      }
      s *= norm;
      scores[lag + maxLag] = s;
      if (s > bestScore) {
        bestScore = s;
        bestLag = lag;
      }
    }

    // Sub-bin refinement on the correlation-of-profiles peak.
    let refined = bestLag;
    const li = bestLag + maxLag;
    if (li > 0 && li < scores.length - 1) {
      const denom = scores[li - 1] - 2 * scores[li] + scores[li + 1];
      if (denom !== 0) {
        const off = (0.5 * (scores[li - 1] - scores[li + 1])) / denom;
        if (Math.abs(off) <= 1) refined = bestLag + off;
      }
    }

    // Negative sign: a target moving to a *shorter* range is approaching, and
    // we report approach as positive.
    const velocity = (-refined * this.metersPerBin) / dt;
    return { velocity, quality: bestScore, lagBins: refined };
  }

  private findTargets(
    guard: number,
    bins: number,
    noiseFloor: number,
    velocity: number,
  ): DetectedTarget[] {
    const cfg = this.config;
    const threshold = noiseFloor * cfg.presenceThreshold;
    const sub = this.subtracted;
    const found: DetectedTarget[] = [];
    for (let j = guard + 1; j < bins - 1; j++) {
      const v = sub[j];
      if (v < threshold) continue;
      if (v < sub[j - 1] || v < sub[j + 1]) continue;
      const offset = parabolicPeakOffset(sub, j);
      found.push({
        rangeMeters: (j + offset) * this.metersPerBin,
        strength: v,
        snr: noiseFloor > 0 ? v / noiseFloor : 0,
        velocity,
      });
      if (found.length >= 8) break;
    }
    found.sort((a, b) => b.strength - a.strength);
    return found.slice(0, 5);
  }

  private classifyGesture(velocity: number, motion: number, quality: number): GestureKind {
    const idx = this.gestureWrite % GESTURE_HISTORY;
    this.velocityHistory[idx] = quality > 0.15 ? velocity : 0;
    this.motionHistory[idx] = motion;
    this.gestureWrite++;

    if (this.gestureRefractory > 0) {
      this.gestureRefractory--;
      return 'none';
    }
    if (this.gestureWrite < GESTURE_HISTORY) return 'none';

    // A swipe is a burst of motion whose radial direction is consistent.
    //
    // The motion gate sits at 3x the residual noise floor for a reason: two
    // consecutive frames of pure noise differ by about sqrt(2) times the noise
    // level, so anything below ~2 fires constantly in an empty room. Measured
    // on synthetic data, an idle room peaks at 1.7 and a hand swipe sits
    // around 6, so 3 separates them with margin on both sides.
    let movingFrames = 0;
    let positive = 0;
    let negative = 0;
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < GESTURE_HISTORY; i++) {
      const m = this.motionHistory[i];
      const v = this.velocityHistory[i];
      if (m > 3 && Math.abs(v) > 0.08) {
        movingFrames++;
        sum += v;
        if (v > 0) positive++;
        else negative++;
        if (Math.abs(v) > Math.abs(peak)) peak = v;
      }
    }
    if (movingFrames < 3) return 'none';
    const agreement = Math.max(positive, negative) / movingFrames;
    if (agreement < 0.7) return 'none';
    const mean = sum / movingFrames;
    if (Math.abs(mean) < 0.12) return 'none';

    // One second of refractory so a single swipe reports once, not fourteen
    // times as it slides through the history window.
    this.gestureRefractory = GESTURE_HISTORY;
    return mean > 0 ? 'swipeToward' : 'swipeAway';
  }

  /**
   * Track the carrier phase of one range bin and look for a slow oscillation.
   *
   * Phase is used here, not magnitude, because a still person's chest moves
   * only a few millimetres: far too little to move between range bins (each is
   * 3.6 mm of one-way range, and the compressed pulse is 12 bins wide), but a
   * full turn of 20 kHz phase every 8.6 mm.
   */
  private updateBreathing(guard: number, bins: number): BreathingEstimate {
    if (!this.trackedBinLocked) {
      // Re-evaluate occasionally; switching bins resets the unwrapper, so we
      // only move when another bin is clearly better.
      if (this.trackedBin < guard || this.frameIndex % 64 === 0) {
        const candidate = argMax(this.longTermProfile, guard, bins);
        if (this.trackedBin < guard) {
          this.trackedBin = candidate;
          this.unwrapper.reset();
          this.phaseFill = 0;
          this.phaseWrite = 0;
        } else if (
          candidate !== this.trackedBin &&
          this.longTermProfile[candidate] > 1.5 * this.longTermProfile[this.trackedBin]
        ) {
          this.trackedBin = candidate;
          this.unwrapper.reset();
          this.phaseFill = 0;
          this.phaseWrite = 0;
        }
      }
    }
    const bin = Math.min(Math.max(this.trackedBin, 0), bins - 1);

    const phase = Math.atan2(this.profileIm[bin], this.profileRe[bin]);
    const continuous = this.unwrapper.next(phase);
    this.phaseHistory[this.phaseWrite] = continuous;
    this.phaseWrite = (this.phaseWrite + 1) % BREATH_HISTORY;
    if (this.phaseFill < BREATH_HISTORY) this.phaseFill++;

    const fill = this.phaseFill / BREATH_HISTORY;
    this.breathResult.fill = fill;
    this.breathResult.trackedRangeMeters = bin * this.metersPerBin;

    // Need ~15 s before a 0.2 Hz periodicity means anything, and there is no
    // point recomputing the periodogram every 85 ms.
    const minSamples = Math.ceil(15 / this.framePeriodSeconds);
    if (this.phaseFill >= minSamples && this.frameIndex % 8 === 0) {
      this.computeBreathing();
    }
    return this.breathResult;
  }

  private computeBreathing(): void {
    const count = this.phaseFill;
    const scratch = this.breathScratch;
    // Unroll the ring buffer oldest-first.
    const start = (this.phaseWrite - count + BREATH_HISTORY) % BREATH_HISTORY;
    for (let i = 0; i < count; i++) {
      scratch[i] = this.phaseHistory[(start + i) % BREATH_HISTORY];
    }
    const view = scratch.subarray(0, count);
    detrend(view, view);

    const re = this.breathRe;
    const im = this.breathIm;
    re.fill(0);
    im.fill(0);
    // Window with a Hann of the right length for however much we have. The
    // partial-length window is cached because the buffer only grows.
    let win: Float32Array | Float64Array;
    if (count === BREATH_HISTORY) {
      win = this.breathWindow;
    } else {
      if (this.breathPartialLength !== count) {
        for (let i = 0; i < count; i++) {
          this.breathPartialWindow[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / count));
        }
        this.breathPartialLength = count;
      }
      win = this.breathPartialWindow;
    }
    for (let i = 0; i < count; i++) re[i] = view[i] * win[i];
    this.breathFft.forward(re, im);

    const hzPerBin = 1 / (this.framePeriodSeconds * BREATH_FFT);
    const lo = Math.max(1, Math.floor(BREATH_MIN_HZ / hzPerBin));
    const hi = Math.min(BREATH_FFT >> 1, Math.ceil(BREATH_MAX_HZ / hzPerBin));
    if (hi <= lo) {
      this.breathResult.bpm = null;
      this.breathResult.confidence = 0;
      return;
    }

    let best = lo;
    let bestVal = -Infinity;
    let sum = 0;
    for (let k = lo; k <= hi; k++) {
      const m = Math.hypot(re[k], im[k]);
      sum += m;
      if (m > bestVal) {
        bestVal = m;
        best = k;
      }
    }
    const mean = sum / (hi - lo + 1);
    const confidence = mean > 0 ? bestVal / mean : 0;

    // Interpolate the peak so the reported rate is not quantised to the bin
    // grid (which would be +/-1.4 bpm here).
    let refined = best;
    if (best > lo && best < hi) {
      const yl = Math.hypot(re[best - 1], im[best - 1]);
      const y0 = bestVal;
      const yr = Math.hypot(re[best + 1], im[best + 1]);
      const denom = yl - 2 * y0 + yr;
      if (denom !== 0) {
        const off = (0.5 * (yl - yr)) / denom;
        if (Math.abs(off) <= 1) refined = best + off;
      }
    }

    // A flat spectrum means we are looking at noise, not a rhythm.
    this.breathResult.confidence = confidence;
    this.breathResult.bpm = confidence >= 3 ? refined * hzPerBin * 60 : null;
  }

  /**
   * Estimate how much of the requested sweep the hardware actually delivers.
   * Consumer speakers roll off hard above ~18 kHz, so the effective bandwidth
   * — and therefore the true range resolution — is often much less than what
   * the user asked for. Better to measure and say so than to quote a
   * resolution we do not have.
   */
  private measureBandQuality(): { spectralTiltDb: number; usableBandwidthHz: number } {
    const cfg = this.config;
    if (!this.specAverageReady) return { spectralTiltDb: 0, usableBandwidthHz: 0 };
    const hzPerBin = this.sampleRate / SPEC_FFT;
    const lo = Math.min(cfg.f0, cfg.f1);
    const hi = Math.max(cfg.f0, cfg.f1);
    const loBin = Math.max(1, Math.round(lo / hzPerBin));
    const hiBin = Math.min(this.specBins - 1, Math.round(hi / hzPerBin));
    if (hiBin <= loBin) return { spectralTiltDb: 0, usableBandwidthHz: 0 };

    let maxDb = -Infinity;
    for (let k = loBin; k <= hiBin; k++) {
      if (this.specAverage[k] > maxDb) maxDb = this.specAverage[k];
    }
    // Level in the bottom and top tenth of the band.
    const span = hiBin - loBin;
    const lowEdge = average(this.specAverage, loBin, loBin + Math.max(1, span * 0.1));
    const highEdge = average(this.specAverage, hiBin - Math.max(1, span * 0.1), hiBin + 1);
    const spectralTiltDb = highEdge - lowEdge;

    // Usable = contiguous region within 15 dB of the in-band peak.
    const cutoff = maxDb - 15;
    let firstGood = -1;
    let lastGood = -1;
    for (let k = loBin; k <= hiBin; k++) {
      if (this.specAverage[k] >= cutoff) {
        if (firstGood < 0) firstGood = k;
        lastGood = k;
      }
    }
    const usableBandwidthHz =
      firstGood >= 0 && lastGood > firstGood ? (lastGood - firstGood) * hzPerBin : 0;
    return { spectralTiltDb, usableBandwidthHz };
  }

  // -----------------------------------------------------------------------
  // CW Doppler
  // -----------------------------------------------------------------------

  /**
   * Continuous-tone mode. With a pure carrier there is no range information at
   * all, but the Doppler shift is unambiguous and direct: a reflector moving
   * towards the microphone returns f*(1 + 2v/c), so approach lands entirely in
   * the upper sideband and recede entirely in the lower one. At 20 kHz, 1 m/s
   * is a 117 Hz shift — ten bins here, trivially separable.
   *
   * The catch is the carrier itself, which arrives ~60 dB stronger than any
   * reflection. That is why this path uses a Blackman-Harris window: with a
   * Hann window the carrier's own spectral leakage would fill both sidebands
   * symmetrically and wash out the asymmetry we are measuring.
   */
  private processCwFrame(time: number, inBandLevelDb: number): EngineFrame {
    const n = this.config.periodSamples;
    const re = this.cwRe;
    const im = this.cwIm;
    const win = this.cwWindow;
    for (let i = 0; i < n; i++) {
      re[i] = this.frameBuffer[i] * win[i];
      im[i] = 0;
    }
    this.cwFft.forward(re, im);

    const hzPerBin = this.sampleRate / n;
    const carrierBin = Math.round(this.actualCwFrequency / hzPerBin);
    const half = (this.cwBins - 1) >> 1;

    for (let i = 0; i < this.cwBins; i++) {
      const k = carrierBin - half + i;
      if (k < 0 || k >= n >> 1) {
        this.cwSpectrum[i] = -140;
        continue;
      }
      const mag = Math.hypot(re[k], im[k]) * (2 / n);
      this.cwSpectrum[i] = 20 * Math.log10(mag + 1e-12);
    }

    // Skip +/-2 bins of carrier and integrate 500 Hz of each sideband.
    const skip = 3;
    const width = Math.min(half, Math.round(500 / hzPerBin));
    let upper = 0;
    let lower = 0;
    let upperMoment = 0;
    let lowerMoment = 0;
    for (let d = skip; d <= width; d++) {
      const ku = carrierBin + d;
      const kl = carrierBin - d;
      const mu = ku < n >> 1 ? Math.hypot(re[ku], im[ku]) : 0;
      const ml = kl >= 0 ? Math.hypot(re[kl], im[kl]) : 0;
      upper += mu * mu;
      lower += ml * ml;
      upperMoment += d * mu * mu;
      lowerMoment += d * ml * ml;
    }
    const total = upper + lower;
    const asymmetry = total > 0 ? (upper - lower) / total : 0;

    // Energy-weighted mean shift, signed.
    const meanShiftBins =
      total > 0 ? (upperMoment - lowerMoment) / total : 0;
    const shiftHz = meanShiftBins * hzPerBin;
    const velocity = (shiftHz * SPEED_OF_SOUND) / (2 * this.actualCwFrequency);

    // Carrier level as the reference for "is anything moving at all".
    const carrierMag = Math.hypot(re[carrierBin], im[carrierBin]);
    const sidebandRatio = carrierMag > 0 ? Math.sqrt(total) / carrierMag : 0;
    const presence = sidebandRatio > 0.02;
    const motionEnergy = sidebandRatio * 100;

    const gesture = this.classifyGesture(velocity, motionEnergy, Math.abs(asymmetry));

    const diagnostics: SonarDiagnostics = {
      inBandLevelDb,
      noiseFloorDb: 20 * Math.log10(sidebandRatio + 1e-12),
      directPathIndex: 0,
      directPathLevel: carrierMag,
      ...this.measureBandQuality(),
    };

    this.profileMag.fill(0);
    this.subtracted.fill(0);

    return {
      frameIndex: this.frameIndex,
      time,
      profile: this.subtracted,
      rawProfile: this.profileMag,
      analysisBins: this.analysisBins,
      metersPerBin: this.metersPerBin,
      presence,
      velocity,
      motionEnergy,
      targets: [],
      gesture,
      breathing: this.breathResult,
      diagnostics,
      cwSpectrum: this.cwSpectrum,
      cwBins: this.cwBins,
      cwHzPerBin: hzPerBin,
      spectrogram: this.specOut,
      spectrogramColumns: this.specColumns,
      spectrogramBins: this.specBins,
      spectrogramHzPerBin: this.sampleRate / SPEC_FFT,
    };
  }

  /** Exposed for tests: the lag, in bins, behind the last velocity estimate. */
  get lastVelocityLagBins(): number {
    return this.velocityLag;
  }
}

/**
 * Amplitude-taper the receive template across the sweep — a deliberately
 * *mismatched* filter.
 *
 * A flat LFM sweep has a near-rectangular spectrum, and the Fourier transform
 * of a rectangle is a sinc: the compressed pulse carries range sidelobes only
 * 13 dB down that then decay as 1/range. Measured on this chirp, the direct
 * path is still 23 dB down a full metre away — and since speaker-to-mic
 * leakage is 20 dB stronger than any real target, those sidelobes do not just
 * add noise, they bias the peak of every nearby target by centimetres.
 *
 * Because an LFM sweep maps time linearly onto frequency, an amplitude taper
 * in time *is* a taper in frequency. A Hann envelope drops the near sidelobes
 * to about -32 dB and the far ones far below that, at the cost of widening the
 * main lobe by ~1.6x (4.3 cm of resolution becomes ~7 cm) and about 1.8 dB of
 * SNR.
 *
 * The taper is applied only on receive. The transmitted chirp stays flat so it
 * keeps every bit of the energy a quiet ultrasonic speaker can give us — the
 * sidelobe problem is entirely a property of the filter, not the waveform.
 */
export function applyReceiveTaper(template: Float32Array, chirpSamples: number): void {
  // Reach a little past the nominal chirp so the front-end filter's dispersed
  // tail is inside the window rather than being chopped off by it.
  const span = Math.min(template.length, chirpSamples + 64);
  for (let i = 0; i < span; i++) {
    template[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (span - 1)));
  }
  for (let i = span; i < template.length; i++) template[i] = 0;
}

function rms(buf: Float32Array): number {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / Math.max(1, buf.length));
}

function average(buf: Float32Array, from: number, to: number): number {
  const a = Math.max(0, Math.floor(from));
  const b = Math.min(buf.length, Math.ceil(to));
  if (b <= a) return 0;
  let s = 0;
  for (let i = a; i < b; i++) s += buf[i];
  return s / (b - a);
}

/**
 * Noise floor of the residual profile.
 *
 * A plain mean of |residual| is dragged upwards by whatever target is present,
 * which raises the detection threshold exactly when a target appears — the
 * opposite of what we want. One trimming pass, discarding everything above 3x
 * the first estimate, removes the target's own contribution.
 */
export function robustNoiseFloor(data: Float32Array, from: number, to: number): number {
  let sum = 0;
  let count = 0;
  for (let i = from; i < to; i++) {
    sum += Math.abs(data[i]);
    count++;
  }
  if (count === 0) return 0;
  const first = sum / count;
  const limit = first * 3;
  let sum2 = 0;
  let count2 = 0;
  for (let i = from; i < to; i++) {
    const v = Math.abs(data[i]);
    if (v <= limit) {
      sum2 += v;
      count2++;
    }
  }
  return count2 > 0 ? sum2 / count2 : first;
}
