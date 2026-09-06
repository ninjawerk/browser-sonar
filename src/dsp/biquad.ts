/**
 * Biquad filters (RBJ audio EQ cookbook forms), used to reject everything
 * outside the chirp band before correlation.
 *
 * Why bother when the matched filter already rejects out-of-band energy?
 * Because the correlation gain is finite and the room is loud: speech and
 * music can sit 60 dB above our 20 kHz reflections, and their correlation
 * sidelobes are not zero. A cheap high-order high-pass costs a few
 * multiply-adds per sample and removes the problem entirely.
 */

export type BiquadKind = 'lowpass' | 'highpass' | 'bandpass';

export class Biquad {
  private b0 = 1;
  private b1 = 0;
  private b2 = 0;
  private a1 = 0;
  private a2 = 0;
  // Direct form II transposed state.
  private z1 = 0;
  private z2 = 0;

  static bandpass(sampleRate: number, freq: number, q: number): Biquad {
    const f = new Biquad();
    f.setCoefficients('bandpass', sampleRate, freq, q);
    return f;
  }

  static highpass(sampleRate: number, freq: number, q: number): Biquad {
    const f = new Biquad();
    f.setCoefficients('highpass', sampleRate, freq, q);
    return f;
  }

  static lowpass(sampleRate: number, freq: number, q: number): Biquad {
    const f = new Biquad();
    f.setCoefficients('lowpass', sampleRate, freq, q);
    return f;
  }

  setCoefficients(kind: BiquadKind, sampleRate: number, freq: number, q: number): void {
    // Keep the corner strictly inside the unit circle; 20 kHz at a 44.1 kHz
    // sample rate is already at 0.9 of Nyquist and the design degenerates if
    // we let it reach 1.
    const nyquist = sampleRate * 0.5;
    const f = Math.min(Math.max(freq, 1), nyquist * 0.995);
    const w0 = (2 * Math.PI * f) / sampleRate;
    const cosW0 = Math.cos(w0);
    const sinW0 = Math.sin(w0);
    const alpha = sinW0 / (2 * Math.max(q, 1e-4));

    let b0: number;
    let b1: number;
    let b2: number;
    switch (kind) {
      case 'lowpass':
        b0 = (1 - cosW0) / 2;
        b1 = 1 - cosW0;
        b2 = (1 - cosW0) / 2;
        break;
      case 'highpass':
        b0 = (1 + cosW0) / 2;
        b1 = -(1 + cosW0);
        b2 = (1 + cosW0) / 2;
        break;
      case 'bandpass':
        // Constant 0 dB peak gain form.
        b0 = alpha;
        b1 = 0;
        b2 = -alpha;
        break;
    }
    const a0 = 1 + alpha;
    const a1 = -2 * cosW0;
    const a2 = 1 - alpha;

    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
    this.reset();
  }

  reset(): void {
    this.z1 = 0;
    this.z2 = 0;
  }

  processSample(x: number): number {
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }

  /** Filter `input` into `output` (may be the same array). */
  process(input: Float32Array, output: Float32Array, count = input.length): void {
    for (let i = 0; i < count; i++) output[i] = this.processSample(input[i]);
  }

  /** Magnitude response at `freq`, for tests and for plotting. */
  magnitudeAt(sampleRate: number, freq: number): number {
    const w = (2 * Math.PI * freq) / sampleRate;
    const cw = Math.cos(w);
    const sw = Math.sin(w);
    const c2w = Math.cos(2 * w);
    const s2w = Math.sin(2 * w);
    const numRe = this.b0 + this.b1 * cw + this.b2 * c2w;
    const numIm = -(this.b1 * sw + this.b2 * s2w);
    const denRe = 1 + this.a1 * cw + this.a2 * c2w;
    const denIm = -(this.a1 * sw + this.a2 * s2w);
    return Math.hypot(numRe, numIm) / Math.hypot(denRe, denIm);
  }
}

/** A cascade of identical-order sections, for a steeper skirt. */
export class BiquadCascade {
  private readonly sections: Biquad[];

  constructor(sections: Biquad[]) {
    this.sections = sections;
  }

  /**
   * High-pass at `cutoff`, `order/2` Butterworth-ish sections.
   *
   * This is the default front end: below the chirp band there is nothing we
   * want, and a high-pass (rather than a band-pass) keeps the passband flat
   * right up to Nyquist so the top of the sweep is not attenuated.
   */
  static highpass(sampleRate: number, cutoff: number, sections = 3): BiquadCascade {
    const qs = butterworthQs(sections * 2);
    return new BiquadCascade(qs.map((q) => Biquad.highpass(sampleRate, cutoff, q)));
  }

  static bandpass(sampleRate: number, center: number, q: number, sections = 2): BiquadCascade {
    const list: Biquad[] = [];
    for (let i = 0; i < sections; i++) list.push(Biquad.bandpass(sampleRate, center, q));
    return new BiquadCascade(list);
  }

  reset(): void {
    for (const s of this.sections) s.reset();
  }

  processSample(x: number): number {
    let y = x;
    for (let i = 0; i < this.sections.length; i++) y = this.sections[i].processSample(y);
    return y;
  }

  process(input: Float32Array, output: Float32Array, count = input.length): void {
    for (let i = 0; i < count; i++) output[i] = this.processSample(input[i]);
  }

  magnitudeAt(sampleRate: number, freq: number): number {
    let m = 1;
    for (const s of this.sections) m *= s.magnitudeAt(sampleRate, freq);
    return m;
  }
}

/** Q values for the sections of a Butterworth filter of the given order. */
export function butterworthQs(order: number): number[] {
  const pairs = Math.floor(order / 2);
  const qs: number[] = [];
  for (let k = 0; k < pairs; k++) {
    const theta = (Math.PI * (2 * k + 1)) / (2 * order);
    qs.push(1 / (2 * Math.sin(theta)));
  }
  return qs;
}
