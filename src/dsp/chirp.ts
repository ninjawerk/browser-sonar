/**
 * Chirp (FMCW sweep) synthesis.
 *
 * A linear frequency sweep is what buys us range resolution without needing a
 * loud, short pulse. Sweeping bandwidth B gives a compressed peak of width
 * ~1/B after matched filtering, so range resolution is c / (2B) regardless of
 * how long the chirp is, while the *energy* (and therefore SNR) grows with
 * duration. That is the whole trick that makes near-ultrasonic sonar work
 * through a laptop speaker that is 30 dB down at 20 kHz.
 */

export const SPEED_OF_SOUND = 343; // m/s at ~20 C, dry air

export interface ChirpParams {
  sampleRate: number;
  /** Sweep start frequency in Hz. */
  f0: number;
  /** Sweep end frequency in Hz. */
  f1: number;
  /** Sweep duration in seconds. */
  durationSec: number;
  /** Length of the Hann rise/fall applied to each end, in seconds. */
  taperSec?: number;
  /** Peak amplitude before tapering. */
  amplitude?: number;
}

/**
 * One bare chirp, `round(durationSec * sampleRate)` samples long.
 *
 * Phase is the integral of the instantaneous frequency:
 *   f(t) = f0 + k*t,  k = (f1 - f0) / T
 *   phi(t) = 2*pi * (f0*t + k*t^2/2)
 */
export function generateChirp(params: ChirpParams): Float32Array {
  const { sampleRate, f0, f1, durationSec } = params;
  const amplitude = params.amplitude ?? 1;
  const taperSec = params.taperSec ?? 0.001;

  const n = Math.max(1, Math.round(durationSec * sampleRate));
  const out = new Float32Array(n);
  const T = n / sampleRate;
  const k = (f1 - f0) / T;

  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const phase = 2 * Math.PI * (f0 * t + 0.5 * k * t * t);
    out[i] = amplitude * Math.sin(phase);
  }

  applyChirpTaper(out, Math.round(taperSec * sampleRate));
  return out;
}

/** Hann rise/fall on the first and last `taperSamples`, unity in between. */
export function applyChirpTaper(buf: Float32Array, taperSamples: number): void {
  const n = buf.length;
  const t = Math.min(Math.max(0, Math.floor(taperSamples)), n >> 1);
  if (t <= 0) return;
  for (let i = 0; i < t; i++) {
    const g = 0.5 * (1 - Math.cos((Math.PI * i) / t));
    buf[i] *= g;
    buf[n - 1 - i] *= g;
  }
}

/**
 * A full transmit period: the chirp followed by silence, exactly
 * `periodSamples` long so an AudioBufferSourceNode can loop it sample-exactly.
 *
 * Keeping the period a power of two lets the receiver use a *circular*
 * correlation over exactly one period, which sidesteps every edge effect and
 * makes the result independent of where in the period we happen to start
 * listening. That independence is what lets the direct-path calibration work.
 */
export function generateTransmitPeriod(
  params: ChirpParams & { periodSamples: number },
): Float32Array {
  const chirp = generateChirp(params);
  if (chirp.length > params.periodSamples) {
    throw new Error(
      `chirp (${chirp.length} samples) does not fit in period (${params.periodSamples})`,
    );
  }
  const out = new Float32Array(params.periodSamples);
  out.set(chirp, 0);
  return out;
}

/**
 * Continuous tone for CW Doppler mode, frequency-snapped so that an integer
 * number of cycles fits in `periodSamples`. Without the snap the loop point
 * produces a phase discontinuity every period — an audible tick and a smear
 * across the whole spectrum, which defeats the point of CW analysis.
 */
export function generateTone(opts: {
  sampleRate: number;
  frequency: number;
  periodSamples: number;
  amplitude?: number;
}): { samples: Float32Array; actualFrequency: number } {
  const { sampleRate, frequency, periodSamples } = opts;
  const amplitude = opts.amplitude ?? 1;
  const cycles = Math.max(1, Math.round((frequency * periodSamples) / sampleRate));
  const actualFrequency = (cycles * sampleRate) / periodSamples;
  const samples = new Float32Array(periodSamples);
  for (let i = 0; i < periodSamples; i++) {
    samples[i] = amplitude * Math.sin((2 * Math.PI * cycles * i) / periodSamples);
  }
  return { samples, actualFrequency };
}

/** Instantaneous frequency of the sweep at time `t` seconds. */
export function instantaneousFrequency(params: ChirpParams, t: number): number {
  const T = Math.round(params.durationSec * params.sampleRate) / params.sampleRate;
  const k = (params.f1 - params.f0) / T;
  return params.f0 + k * t;
}

/** Theoretical range resolution in metres for a sweep of bandwidth `bandwidthHz`. */
export function rangeResolutionMeters(bandwidthHz: number): number {
  return SPEED_OF_SOUND / (2 * bandwidthHz);
}

/** One-way distance in metres corresponding to a round-trip delay of `samples`. */
export function samplesToMeters(samples: number, sampleRate: number): number {
  return (samples / sampleRate) * SPEED_OF_SOUND * 0.5;
}

/** Inverse of `samplesToMeters`. */
export function metersToSamples(meters: number, sampleRate: number): number {
  return (2 * meters * sampleRate) / SPEED_OF_SOUND;
}
