import { describe, expect, it } from 'vitest';
import {
  applyWindow,
  blackmanHarris,
  coherentGain,
  equivalentNoiseBandwidth,
  hamming,
  hann,
  makeWindow,
  taperEdges,
} from './window';

describe('hann', () => {
  it('is zero at both ends in symmetric form', () => {
    const w = hann(64);
    expect(w[0]).toBeCloseTo(0, 12);
    expect(w[63]).toBeCloseTo(0, 12);
  });

  it('peaks at 1 in the middle', () => {
    const w = hann(65);
    expect(w[32]).toBeCloseTo(1, 12);
  });

  it('is symmetric', () => {
    const w = hann(32);
    for (let i = 0; i < 16; i++) expect(w[i]).toBeCloseTo(w[31 - i], 12);
  });

  it('periodic form is not zero at the last sample', () => {
    const w = hann(64, true);
    expect(w[0]).toBeCloseTo(0, 12);
    expect(w[63]).toBeGreaterThan(0);
    // Periodic Hann has a mean of exactly 0.5 — that is what makes it seamless
    // when the DFT wraps.
    expect(coherentGain(w)).toBeCloseTo(0.5, 6);
  });

  it('handles the degenerate single-sample case', () => {
    expect(Array.from(hann(1))).toEqual([1]);
    expect(Array.from(hamming(1))).toEqual([1]);
    expect(Array.from(blackmanHarris(1))).toEqual([1]);
  });
});

describe('window properties', () => {
  it('has the textbook equivalent noise bandwidths', () => {
    // Reference values: Hann 1.50 bins, Hamming 1.36, Blackman-Harris 2.00.
    expect(equivalentNoiseBandwidth(hann(4096, true))).toBeCloseTo(1.5, 2);
    expect(equivalentNoiseBandwidth(hamming(4096, true))).toBeCloseTo(1.36, 2);
    expect(equivalentNoiseBandwidth(blackmanHarris(4096, true))).toBeCloseTo(2.0, 2);
  });

  it('blackmanHarris suppresses near sidelobes far better than hann', () => {
    // A half-bin-offset tone is the worst case for leakage. What matters for
    // CW Doppler is the *near* sidelobes: the reflections we want sit only a
    // few bins from a carrier that is ~60 dB louder, so leakage close to the
    // carrier is what buries them. (Far from the carrier Hann eventually wins,
    // because its sidelobes keep falling at 18 dB/octave while Blackman-Harris
    // flattens out — but by then both are far below anything that matters.)
    const n = 1024;
    const carrier = 100.5;
    const leakNear = (win: Float32Array) => {
      let worst = -Infinity;
      const re = new Float64Array(n);
      for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * carrier * i) / n) * win[i];
      for (let k = 105; k <= 115; k++) {
        let sr = 0;
        let si = 0;
        for (let t = 0; t < n; t++) {
          const a = (-2 * Math.PI * k * t) / n;
          sr += re[t] * Math.cos(a);
          si += re[t] * Math.sin(a);
        }
        worst = Math.max(worst, Math.hypot(sr, si));
      }
      return 20 * Math.log10(worst / (n / 2));
    };
    const hannLeak = leakNear(hann(n, true));
    const bhLeak = leakNear(blackmanHarris(n, true));
    expect(hannLeak).toBeLessThan(-30);
    expect(bhLeak).toBeLessThan(hannLeak - 20);
  });
});

describe('makeWindow', () => {
  it('dispatches to each kind', () => {
    expect(Array.from(makeWindow('hann', 8))).toEqual(Array.from(hann(8)));
    expect(Array.from(makeWindow('hamming', 8))).toEqual(Array.from(hamming(8)));
    expect(Array.from(makeWindow('blackmanHarris', 8))).toEqual(
      Array.from(blackmanHarris(8)),
    );
  });
});

describe('applyWindow', () => {
  it('multiplies in place', () => {
    const buf = new Float32Array([1, 1, 1, 1]);
    applyWindow(buf, new Float32Array([0, 0.5, 0.5, 0]));
    expect(Array.from(buf)).toEqual([0, 0.5, 0.5, 0]);
  });

  it('rejects a length mismatch', () => {
    expect(() => applyWindow(new Float32Array(4), new Float32Array(3))).toThrow();
  });
});

describe('taperEdges', () => {
  it('leaves the middle at unity and ramps the ends', () => {
    const buf = new Float32Array(100).fill(1);
    taperEdges(buf, 10);
    expect(buf[0]).toBeCloseTo(0, 12);
    expect(buf[99]).toBeCloseTo(0, 12);
    expect(buf[50]).toBeCloseTo(1, 12);
    // Monotonic rise over the taper region.
    for (let i = 1; i < 10; i++) expect(buf[i]).toBeGreaterThan(buf[i - 1]);
  });

  it('is a no-op for a zero-length taper', () => {
    const buf = new Float32Array(10).fill(1);
    taperEdges(buf, 0);
    expect(Array.from(buf)).toEqual(new Array(10).fill(1));
  });

  it('clamps a taper longer than half the buffer', () => {
    const buf = new Float32Array(10).fill(1);
    expect(() => taperEdges(buf, 999)).not.toThrow();
    expect(buf[0]).toBeCloseTo(0, 12);
  });
});
