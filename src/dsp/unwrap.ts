/**
 * Phase unwrapping.
 *
 * Correlation-peak phase is only ever known modulo 2*pi. Breathing shows up as
 * a slow sinusoid in phase, but a chest moving a few millimetres at 20 kHz
 * (wavelength ~17 mm) can easily push the phase across the +/-pi boundary, so
 * the raw phase sequence is full of 2*pi jumps that have to be removed before
 * anything sinusoidal is visible.
 */

import type { FloatArray } from './fft';

const TWO_PI = 2 * Math.PI;

/** Wrap an angle to (-pi, pi]. */
export function wrapToPi(angle: number): number {
  let a = angle % TWO_PI;
  if (a > Math.PI) a -= TWO_PI;
  else if (a <= -Math.PI) a += TWO_PI;
  return a;
}

/**
 * Unwrap a whole phase sequence. Returns a new array unless `out` is given.
 * Jumps larger than pi between consecutive samples are assumed to be wraps.
 */
export function unwrap(phases: FloatArray, out?: Float64Array): Float64Array {
  const n = phases.length;
  const result = out ?? new Float64Array(n);
  if (n === 0) return result;
  result[0] = phases[0];
  let offset = 0;
  for (let i = 1; i < n; i++) {
    const delta = phases[i] - phases[i - 1];
    offset += -TWO_PI * Math.round(delta / TWO_PI);
    result[i] = phases[i] + offset;
  }
  return result;
}

/**
 * Streaming unwrapper: feed one wrapped phase per chirp, get a continuous
 * phase out. Keeps no history beyond the previous sample, so it is safe in the
 * worklet hot path.
 */
export class PhaseUnwrapper {
  private prevWrapped = 0;
  private accumulated = 0;
  private started = false;

  reset(): void {
    this.prevWrapped = 0;
    this.accumulated = 0;
    this.started = false;
  }

  /** @param wrapped phase in radians, any range; interpreted modulo 2*pi. */
  next(wrapped: number): number {
    if (!this.started) {
      this.started = true;
      this.prevWrapped = wrapped;
      this.accumulated = wrapped;
      return this.accumulated;
    }
    const delta = wrapToPi(wrapped - this.prevWrapped);
    this.accumulated += delta;
    this.prevWrapped = wrapped;
    return this.accumulated;
  }

  get value(): number {
    return this.accumulated;
  }
}

/**
 * Remove a least-squares linear trend in place.
 *
 * The unwrapped phase of a real target drifts, partly from genuine slow motion
 * and partly from clock offset between the speaker and microphone. Breathing
 * is the oscillation on top of that ramp, so we subtract the ramp before
 * looking for a period.
 */
export function detrend(data: Float64Array, out?: Float64Array): Float64Array {
  const n = data.length;
  const result = out ?? new Float64Array(n);
  if (n === 0) return result;
  if (n === 1) {
    result[0] = 0;
    return result;
  }
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i];
    sumXY += i * data[i];
    sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  const slope = denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  for (let i = 0; i < n; i++) {
    result[i] = data[i] - (slope * i + intercept);
  }
  return result;
}
