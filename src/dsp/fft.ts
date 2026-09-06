/**
 * Iterative radix-2 Cooley–Tukey FFT.
 *
 * Written from scratch (no libraries) because demonstrating the primitives is
 * the point of this project. Twiddle factors and the bit-reversal permutation
 * are precomputed in the constructor so that `forward`/`inverse` allocate
 * nothing — they run in the AudioWorklet hot path.
 */

export type FloatArray = Float32Array | Float64Array;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export class FFT {
  readonly size: number;
  /** log2(size) */
  private readonly levels: number;
  /** cos(-2*pi*k/N) for k in [0, N/2) */
  private readonly cosTable: Float64Array;
  /** sin(-2*pi*k/N) for k in [0, N/2) */
  private readonly sinTable: Float64Array;
  private readonly reverse: Uint32Array;

  constructor(size: number) {
    if (!isPowerOfTwo(size)) {
      throw new Error(`FFT size must be a power of two, got ${size}`);
    }
    this.size = size;
    this.levels = Math.round(Math.log2(size));

    const half = size >> 1;
    this.cosTable = new Float64Array(half);
    this.sinTable = new Float64Array(half);
    for (let k = 0; k < half; k++) {
      const angle = (-2 * Math.PI * k) / size;
      this.cosTable[k] = Math.cos(angle);
      this.sinTable[k] = Math.sin(angle);
    }

    this.reverse = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < this.levels; b++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
      }
      this.reverse[i] = r;
    }
  }

  /** In-place forward DFT. Unnormalised (matches the textbook definition). */
  forward(re: FloatArray, im: FloatArray): void {
    this.transform(re, im);
  }

  /** In-place inverse DFT, scaled by 1/N so that inverse(forward(x)) === x. */
  inverse(re: FloatArray, im: FloatArray): void {
    // conj -> forward -> conj -> scale
    this.transform(im, re);
    const n = this.size;
    const scale = 1 / n;
    for (let i = 0; i < n; i++) {
      re[i] *= scale;
      im[i] *= scale;
    }
  }

  private transform(re: FloatArray, im: FloatArray): void {
    const n = this.size;
    const rev = this.reverse;

    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }

    const cos = this.cosTable;
    const sin = this.sinTable;
    for (let span = 2; span <= n; span <<= 1) {
      const halfSpan = span >> 1;
      const step = n / span;
      for (let start = 0; start < n; start += span) {
        for (let k = 0, idx = start; k < halfSpan; k++, idx++) {
          const twiddle = k * step;
          const wr = cos[twiddle];
          const wi = sin[twiddle];
          const j = idx + halfSpan;
          const tr = re[j] * wr - im[j] * wi;
          const ti = re[j] * wi + im[j] * wr;
          re[j] = re[idx] - tr;
          im[j] = im[idx] - ti;
          re[idx] += tr;
          im[idx] += ti;
        }
      }
    }
  }
}

/**
 * Magnitude spectrum of the first `size/2 + 1` bins, written into `out`.
 * `out.length` must be at least size/2 + 1.
 */
export function magnitudeSpectrum(re: FloatArray, im: FloatArray, out: FloatArray): void {
  const bins = out.length;
  for (let i = 0; i < bins; i++) {
    out[i] = Math.hypot(re[i], im[i]);
  }
}

/**
 * Reference DFT. O(n^2), used only by tests to check `FFT` against the
 * definition — never call this from the audio path.
 */
export function naiveDFT(re: FloatArray, im: FloatArray): { re: Float64Array; im: Float64Array } {
  const n = re.length;
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      sr += re[t] * c - im[t] * s;
      si += re[t] * s + im[t] * c;
    }
    outRe[k] = sr;
    outIm[k] = si;
  }
  return { re: outRe, im: outIm };
}
