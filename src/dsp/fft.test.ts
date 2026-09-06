import { describe, expect, it } from 'vitest';
import { FFT, naiveDFT } from './fft';

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe('FFT', () => {
  it('rejects non-power-of-two sizes', () => {
    expect(() => new FFT(100)).toThrow(/power of two/);
    expect(() => new FFT(0)).toThrow();
  });

  it('matches a naive DFT on random input', () => {
    const n = 256;
    const fft = new FFT(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = Math.sin(i * 0.7) + 0.3 * Math.cos(i * 2.1);
      im[i] = 0.1 * Math.sin(i * 0.3);
    }
    const expected = naiveDFT(re, im);
    fft.forward(re, im);
    expect(maxAbsDiff(re, expected.re)).toBeLessThan(1e-9);
    expect(maxAbsDiff(im, expected.im)).toBeLessThan(1e-9);
  });

  it('puts a pure tone in exactly one bin', () => {
    const n = 512;
    const bin = 40;
    const fft = new FFT(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * bin * i) / n);
    fft.forward(re, im);

    const mag = Array.from({ length: n }, (_, k) => Math.hypot(re[k], im[k]));
    // A real cosine at an exact bin gives n/2 at +bin and -bin, nothing else.
    expect(mag[bin]).toBeCloseTo(n / 2, 6);
    expect(mag[n - bin]).toBeCloseTo(n / 2, 6);
    for (let k = 0; k < n; k++) {
      if (k === bin || k === n - bin) continue;
      expect(mag[k]).toBeLessThan(1e-9);
    }
  });

  it('round-trips forward then inverse', () => {
    const n = 1024;
    const fft = new FFT(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const origRe = new Float64Array(n);
    const origIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      re[i] = origRe[i] = Math.random() * 2 - 1;
      im[i] = origIm[i] = Math.random() * 2 - 1;
    }
    fft.forward(re, im);
    fft.inverse(re, im);
    expect(maxAbsDiff(re, origRe)).toBeLessThan(1e-12);
    expect(maxAbsDiff(im, origIm)).toBeLessThan(1e-12);
  });

  it('is linear', () => {
    const n = 128;
    const fft = new FFT(n);
    const a = new Float64Array(n);
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = Math.sin(i * 0.11);
      b[i] = Math.cos(i * 0.37);
    }
    const sum = new Float64Array(n);
    for (let i = 0; i < n; i++) sum[i] = a[i] + 2 * b[i];

    const aIm = new Float64Array(n);
    const bIm = new Float64Array(n);
    const sumIm = new Float64Array(n);
    fft.forward(a, aIm);
    fft.forward(b, bIm);
    fft.forward(sum, sumIm);

    for (let k = 0; k < n; k++) {
      expect(sum[k]).toBeCloseTo(a[k] + 2 * b[k], 9);
      expect(sumIm[k]).toBeCloseTo(aIm[k] + 2 * bIm[k], 9);
    }
  });

  it("satisfies Parseval's theorem", () => {
    const n = 256;
    const fft = new FFT(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    let timeEnergy = 0;
    for (let i = 0; i < n; i++) {
      re[i] = Math.random() * 2 - 1;
      timeEnergy += re[i] * re[i];
    }
    fft.forward(re, im);
    let freqEnergy = 0;
    for (let k = 0; k < n; k++) freqEnergy += re[k] * re[k] + im[k] * im[k];
    expect(freqEnergy / n).toBeCloseTo(timeEnergy, 8);
  });

  it('works with Float32Array input', () => {
    const n = 64;
    const fft = new FFT(n);
    const re = new Float32Array(n);
    const im = new Float32Array(n);
    re[0] = 1; // impulse -> flat spectrum
    fft.forward(re, im);
    for (let k = 0; k < n; k++) {
      expect(Math.hypot(re[k], im[k])).toBeCloseTo(1, 5);
    }
  });
});
