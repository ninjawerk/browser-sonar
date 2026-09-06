/**
 * Matched filtering (pulse compression) by FFT cross-correlation.
 *
 * We correlate against the *analytic* form of the transmitted chirp — the
 * template with its negative frequencies removed. The correlation output is
 * then complex, and that buys two things at once:
 *
 *   - |r[n]| is the smooth envelope of the compressed pulse, so peak-finding
 *     is not confused by the carrier oscillating at 20 kHz inside the peak;
 *   - arg(r[n]) is the carrier phase at that delay, which is what phase
 *     tracking (and therefore breathing detection) needs. A displacement of
 *     one wavelength — about 17 mm at 20 kHz — is a full turn of phase, so
 *     phase resolves motion roughly a hundred times finer than the range bins
 *     do.
 *
 * The correlation is circular over exactly one transmit period. Because the
 * transmitter loops the same period forever, circular correlation of any
 * contiguous period-length window of the microphone signal is exact: there is
 * no windowing loss and no edge effect, and the answer does not depend on
 * where in the period the window happens to start. Wrap-around aliases delays
 * longer than one period, which at 85 ms is ~14 m of round trip — far beyond
 * anything a laptop speaker can illuminate.
 */

import { FFT, type FloatArray } from './fft';

export interface CorrelatorOptions {
  /** Use the analytic (complex) template. Default true. */
  analytic?: boolean;
}

export class Correlator {
  readonly size: number;
  private readonly fft: FFT;
  /** conj(T[k]) of the analytic template, precomputed. */
  private readonly tRe: Float64Array;
  private readonly tIm: Float64Array;
  /** Scratch spectra, reused every frame so the hot path allocates nothing. */
  private readonly workRe: Float64Array;
  private readonly workIm: Float64Array;

  constructor(size: number, template: FloatArray, options: CorrelatorOptions = {}) {
    const analytic = options.analytic ?? true;
    if (template.length > size) {
      throw new Error(`template (${template.length}) longer than FFT size (${size})`);
    }
    this.size = size;
    this.fft = new FFT(size);
    this.tRe = new Float64Array(size);
    this.tIm = new Float64Array(size);
    this.workRe = new Float64Array(size);
    this.workIm = new Float64Array(size);

    this.tRe.set(template);
    this.fft.forward(this.tRe, this.tIm);

    if (analytic) {
      // Zero the negative frequencies and double the positive ones. DC and
      // Nyquist are their own mirror image and stay as they are.
      const half = size >> 1;
      for (let k = 1; k < half; k++) {
        this.tRe[k] *= 2;
        this.tIm[k] *= 2;
      }
      for (let k = half + 1; k < size; k++) {
        this.tRe[k] = 0;
        this.tIm[k] = 0;
      }
    }

    // Store the conjugate: correlation is X[k] * conj(T[k]).
    for (let k = 0; k < size; k++) {
      this.tIm[k] = -this.tIm[k];
    }
  }

  /**
   * Circular cross-correlation of `frame` against the template.
   *
   * `frame.length` must equal `size`. Results are written into `outRe`/`outIm`,
   * which must also be `size` long. `frame` is not modified.
   */
  correlate(frame: FloatArray, outRe: FloatArray, outIm: FloatArray): void {
    const n = this.size;
    if (frame.length !== n) {
      throw new Error(`frame length ${frame.length} != correlator size ${n}`);
    }
    const wr = this.workRe;
    const wi = this.workIm;
    wr.set(frame);
    wi.fill(0);
    this.fft.forward(wr, wi);

    const tr = this.tRe;
    const ti = this.tIm;
    for (let k = 0; k < n; k++) {
      const ar = wr[k];
      const ai = wi[k];
      wr[k] = ar * tr[k] - ai * ti[k];
      wi[k] = ar * ti[k] + ai * tr[k];
    }

    this.fft.inverse(wr, wi);

    for (let i = 0; i < n; i++) {
      outRe[i] = wr[i];
      outIm[i] = wi[i];
    }
  }
}

/** |z| for each element of a complex array, into `out`. */
export function complexMagnitude(re: FloatArray, im: FloatArray, out: FloatArray, count = out.length): void {
  for (let i = 0; i < count; i++) {
    out[i] = Math.hypot(re[i], im[i]);
  }
}

/**
 * Direct (naive) linear cross-correlation, r[lag] = sum_n a[n+lag] * b[n].
 * O(n*m) — reference implementation for the tests only.
 */
export function crossCorrelateDirect(a: FloatArray, b: FloatArray): Float64Array {
  const lags = a.length - b.length + 1;
  const out = new Float64Array(Math.max(0, lags));
  for (let lag = 0; lag < lags; lag++) {
    let sum = 0;
    for (let n = 0; n < b.length; n++) sum += a[n + lag] * b[n];
    out[lag] = sum;
  }
  return out;
}

/**
 * Sub-sample peak location by fitting a parabola through the peak and its two
 * neighbours. Returns the offset from `index` in samples, in (-0.5, 0.5).
 *
 * Worth doing: our bins are ~3.6 mm of range each, and interpolation typically
 * recovers a further factor of a few, which is the difference between a
 * jittery reading and a stable one.
 */
export function parabolicPeakOffset(mag: FloatArray, index: number): number {
  if (index <= 0 || index >= mag.length - 1) return 0;
  const yl = mag[index - 1];
  const y0 = mag[index];
  const yr = mag[index + 1];
  const denom = yl - 2 * y0 + yr;
  if (denom === 0) return 0;
  const offset = (0.5 * (yl - yr)) / denom;
  return Number.isFinite(offset) && Math.abs(offset) <= 1 ? offset : 0;
}

/** Index of the largest value in `mag[from..to)`. */
export function argMax(mag: FloatArray, from = 0, to = mag.length): number {
  let best = from;
  let bestVal = -Infinity;
  for (let i = from; i < to; i++) {
    if (mag[i] > bestVal) {
      bestVal = mag[i];
      best = i;
    }
  }
  return best;
}
