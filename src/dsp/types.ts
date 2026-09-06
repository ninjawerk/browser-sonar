/**
 * Types shared between the main thread and the AudioWorklet.
 *
 * A note on transport: the brief allows SharedArrayBuffer when the page is
 * cross-origin isolated. We deliberately do not use it. Cross-origin isolation
 * needs COOP/COEP response headers, and GitHub Pages cannot set them — so the
 * SAB path would be dead code on the one deployment target we actually have.
 * `postMessage` with transferables costs a few microseconds per chirp at our
 * 11.7 Hz message rate, which is nowhere near a bottleneck.
 */

export type SonarMode = 'fmcw' | 'cw';

export interface SonarConfig {
  /** Transmit period in samples. Must be a power of two (circular correlation). */
  periodSamples: number;
  /** Sweep start frequency, Hz. */
  f0: number;
  /** Sweep end frequency, Hz. */
  f1: number;
  /** Chirp duration, seconds. */
  chirpDurationSec: number;
  /** Hann rise/fall time on each end of the chirp, seconds. */
  taperSec: number;
  /** 'fmcw' for ranging, 'cw' for continuous-tone Doppler. */
  mode: SonarMode;
  /** Carrier for CW mode, Hz. */
  cwFrequency: number;
  /** Furthest one-way range analysed and displayed, metres. */
  maxRangeMeters: number;
  /** Time constant of the background (clutter) estimate, seconds. */
  backgroundTimeConstantSec: number;
  /**
   * Bins immediately after the direct path to ignore.
   *
   * The compressed direct-path pulse is not a spike: with the tapered receive
   * filter its main lobe is ~9 bins wide at -3 dB and, measured on real
   * hardware, is still only 7 dB down at bin 12 and 19 dB down at bin 18. It
   * does not fall past 29 dB — below a typical hand return — until about bin
   * 24. Anything inside that is leakage, not a target, so the default clears
   * the main lobe and accepts a blind zone of roughly 8 cm.
   */
  guardBins: number;
  /** Presence threshold, in multiples of the residual noise floor. */
  presenceThreshold: number;
  /** Freeze the background estimate (used while calibrating an empty room). */
  freezeBackground: boolean;
}

export const DEFAULT_CONFIG: SonarConfig = {
  periodSamples: 4096,
  f0: 18000,
  f1: 22000,
  chirpDurationSec: 0.015,
  taperSec: 0.001,
  mode: 'fmcw',
  cwFrequency: 20000,
  maxRangeMeters: 4,
  backgroundTimeConstantSec: 3,
  guardBins: 22,
  presenceThreshold: 6,
  freezeBackground: false,
};

export interface DetectedTarget {
  /** One-way range in metres, relative to the direct path. */
  rangeMeters: number;
  /** Peak height above the background, normalised to the direct-path peak. */
  strength: number;
  /** Signal-to-noise ratio of this peak, linear. */
  snr: number;
  /** Positive = approaching, m/s. */
  velocity: number;
}

export type GestureKind = 'none' | 'swipeToward' | 'swipeAway';

export interface BreathingEstimate {
  /** Breaths per minute, or null if no reliable period was found. */
  bpm: number | null;
  /** Peak-to-mean ratio of the dominant spectral peak. Higher is better. */
  confidence: number;
  /** How much of the analysis buffer is filled, 0..1. */
  fill: number;
  /** Range bin being tracked, metres. */
  trackedRangeMeters: number;
}

/** Diagnostics for the device compatibility panel. */
export interface SonarDiagnostics {
  /** RMS of the microphone signal inside the chirp band, dBFS. */
  inBandLevelDb: number;
  /** Residual noise after background subtraction, relative to direct path, dB. */
  noiseFloorDb: number;
  /** Direct-path peak position, in samples of round-trip delay. */
  directPathIndex: number;
  /** Direct-path peak height, linear, before normalisation. */
  directPathLevel: number;
  /**
   * Received energy at the top of the sweep minus energy at the bottom, dB.
   * Very negative means the speaker cannot reach the top of the band and the
   * usable bandwidth (and hence range resolution) is worse than requested.
   */
  spectralTiltDb: number;
  /** Bandwidth we estimate is actually usable, Hz. */
  usableBandwidthHz: number;
}

/** One analysis frame, emitted once per chirp period. */
export interface SonarFrameMessage {
  type: 'frame';
  frameIndex: number;
  /** Seconds since processing started. */
  time: number;
  /** Background-subtracted range profile, normalised to the direct path. */
  profile: Float32Array;
  /** The raw (unsubtracted) range profile, same indexing. */
  rawProfile: Float32Array;
  /** Metres of one-way range per profile bin. */
  metersPerBin: number;
  presence: boolean;
  /** Aggregate radial velocity, positive = approaching, m/s. */
  velocity: number;
  /** Frame-to-frame motion energy, arbitrary units. */
  motionEnergy: number;
  targets: DetectedTarget[];
  gesture: GestureKind;
  breathing: BreathingEstimate;
  diagnostics: SonarDiagnostics;
  /** CW mode only: Doppler spectrum around the carrier, dB. */
  cwSpectrum: Float32Array | null;
  /** CW mode only: Hz per bin of `cwSpectrum`. */
  cwHzPerBin: number;
  /** Spectrogram columns accumulated since the last frame, dB mapped to 0..255. */
  spectrogram: Uint8Array;
  /** Number of frequency bins per spectrogram column. */
  spectrogramBins: number;
  /** Hz per spectrogram bin. */
  spectrogramHzPerBin: number;
}

export type WorkletToMain = SonarFrameMessage | { type: 'ready'; sampleRate: number };

export type MainToWorklet =
  | { type: 'config'; config: SonarConfig }
  | { type: 'resetBackground' }
  | { type: 'setTrackedBin'; bin: number | null };
