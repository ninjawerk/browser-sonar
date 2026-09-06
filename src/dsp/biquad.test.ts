import { describe, expect, it } from 'vitest';
import { Biquad, BiquadCascade, butterworthQs } from './biquad';

const SR = 48000;

/**
 * Measured gain at a frequency, by running a steady tone through the filter.
 *
 * Uses RMS rather than the peak sample: at 8 kHz against a 48 kHz sample rate
 * the tone has exactly six samples per cycle, none of which land on the crest,
 * so a peak measurement would read 13% low for reasons that have nothing to do
 * with the filter.
 */
function measureGain(
  filter: { processSample(x: number): number; reset(): void },
  freq: number,
): number {
  filter.reset();
  const n = 16384;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const y = filter.processSample(Math.sin((2 * Math.PI * freq * i) / SR));
    // Ignore the transient at the start.
    if (i > n / 2) {
      sum += y * y;
      count++;
    }
  }
  // A unit-amplitude sine has an RMS of 1/sqrt(2).
  return Math.sqrt(sum / count) * Math.SQRT2;
}

describe('butterworthQs', () => {
  it('gives the textbook Q values', () => {
    expect(butterworthQs(2)[0]).toBeCloseTo(0.7071, 4);
    // Returned highest-Q-first, which is the pole order the formula produces.
    expect(butterworthQs(4).sort((a, b) => a - b)).toEqual([
      expect.closeTo(0.5412, 4),
      expect.closeTo(1.3066, 4),
    ]);
    const q6 = butterworthQs(6).sort((a, b) => a - b);
    expect(q6.length).toBe(3);
    expect(q6[0]).toBeCloseTo(0.5176, 4);
    expect(q6[1]).toBeCloseTo(0.7071, 4);
    expect(q6[2]).toBeCloseTo(1.9319, 4);
  });

  it('gives no sections for a first-order request', () => {
    expect(butterworthQs(1)).toEqual([]);
  });
});

describe('Biquad highpass', () => {
  it('is -3 dB at the cutoff', () => {
    const f = Biquad.highpass(SR, 4000, Math.SQRT1_2);
    expect(f.magnitudeAt(SR, 4000)).toBeCloseTo(Math.SQRT1_2, 3);
  });

  it('passes above the cutoff and rejects below it', () => {
    const f = Biquad.highpass(SR, 16000, Math.SQRT1_2);
    expect(f.magnitudeAt(SR, 20000)).toBeGreaterThan(0.9);
    expect(f.magnitudeAt(SR, 1000)).toBeLessThan(0.01);
    expect(f.magnitudeAt(SR, 100)).toBeLessThan(0.0001);
  });

  it('agrees with the measured response', () => {
    const f = Biquad.highpass(SR, 16000, Math.SQRT1_2);
    for (const freq of [2000, 8000, 20000]) {
      const predicted = f.magnitudeAt(SR, freq);
      expect(measureGain(f, freq)).toBeCloseTo(predicted, 2);
    }
  });
});

describe('Biquad bandpass', () => {
  it('peaks at the centre frequency with unity gain', () => {
    const f = Biquad.bandpass(SR, 20000, 4);
    expect(f.magnitudeAt(SR, 20000)).toBeCloseTo(1, 3);
    expect(f.magnitudeAt(SR, 20000)).toBeGreaterThan(f.magnitudeAt(SR, 14000));
    expect(f.magnitudeAt(SR, 20000)).toBeGreaterThan(f.magnitudeAt(SR, 23000));
  });

  it('has the bandwidth implied by Q well below Nyquist', () => {
    // The -3 dB points sit at fc +/- fc/(2Q) in the analog prototype. The
    // bilinear transform warps that badly near Nyquist, so this property is
    // checked at 2 kHz, where warping is negligible.
    const fc = 2000;
    const q = 4;
    const f = Biquad.bandpass(SR, fc, q);
    const bw = fc / q;
    expect(f.magnitudeAt(SR, fc + bw / 2)).toBeCloseTo(Math.SQRT1_2, 1);
    expect(f.magnitudeAt(SR, fc - bw / 2)).toBeCloseTo(Math.SQRT1_2, 1);
  });

  it('still behaves sanely at 20 kHz, where the transform warps hard', () => {
    const f = Biquad.bandpass(SR, 20000, 4);
    // Frequency warping widens the digital response, but the filter must stay
    // a bandpass: unity at the centre, falling on both sides, bounded.
    expect(f.magnitudeAt(SR, 20000)).toBeCloseTo(1, 3);
    expect(f.magnitudeAt(SR, 12000)).toBeLessThan(0.8);
    expect(f.magnitudeAt(SR, 23900)).toBeLessThan(1);
  });
});

describe('Biquad lowpass', () => {
  it('passes DC and rejects high frequencies', () => {
    const f = Biquad.lowpass(SR, 1000, Math.SQRT1_2);
    expect(f.magnitudeAt(SR, 0)).toBeCloseTo(1, 6);
    expect(f.magnitudeAt(SR, 10000)).toBeLessThan(0.02);
  });
});

describe('Biquad stability', () => {
  it('stays bounded near Nyquist', () => {
    // 22 kHz at a 44.1 kHz sample rate is above Nyquist; the design must clamp
    // rather than blow up. Devices that hand us 44.1 kHz really do hit this.
    const f = Biquad.highpass(44100, 22000, Math.SQRT1_2);
    let y = 0;
    for (let i = 0; i < 10000; i++) y = f.processSample(Math.random() * 2 - 1);
    expect(Number.isFinite(y)).toBe(true);
    expect(Math.abs(y)).toBeLessThan(100);
  });

  it('decays to zero after an impulse', () => {
    const f = Biquad.highpass(SR, 16000, Math.SQRT1_2);
    f.processSample(1);
    let last = 0;
    for (let i = 0; i < 5000; i++) last = f.processSample(0);
    expect(Math.abs(last)).toBeLessThan(1e-9);
  });

  it('resets its state', () => {
    const f = Biquad.highpass(SR, 16000, Math.SQRT1_2);
    for (let i = 0; i < 100; i++) f.processSample(1);
    f.reset();
    expect(f.processSample(0)).toBe(0);
  });
});

describe('BiquadCascade', () => {
  it('rolls off far more steeply than one section', () => {
    const one = Biquad.highpass(SR, 16000, Math.SQRT1_2);
    const three = BiquadCascade.highpass(SR, 16000, 3);
    const at4k = 20 * Math.log10(three.magnitudeAt(SR, 4000) / one.magnitudeAt(SR, 4000));
    // A 6th-order high-pass beats a 2nd-order one by ~24 dB per octave of
    // separation; two octaves below cutoff that is a very large margin.
    expect(at4k).toBeLessThan(-40);
  });

  it('keeps the chirp band essentially flat', () => {
    const c = BiquadCascade.highpass(SR, 16000, 3);
    for (const f of [18000, 20000, 22000]) {
      expect(c.magnitudeAt(SR, f)).toBeGreaterThan(0.85);
      expect(c.magnitudeAt(SR, f)).toBeLessThan(1.15);
    }
  });

  it('rejects speech-band energy that would pollute the correlation', () => {
    const c = BiquadCascade.highpass(SR, 16000, 3);
    expect(20 * Math.log10(c.magnitudeAt(SR, 1000))).toBeLessThan(-90);
    expect(20 * Math.log10(c.magnitudeAt(SR, 4000))).toBeLessThan(-50);
  });

  it('processes buffers and matches sample-at-a-time output', () => {
    const a = BiquadCascade.highpass(SR, 16000, 2);
    const b = BiquadCascade.highpass(SR, 16000, 2);
    const input = Float32Array.from({ length: 256 }, () => Math.random() * 2 - 1);
    const out = new Float32Array(256);
    a.process(input, out);
    for (let i = 0; i < input.length; i++) {
      expect(b.processSample(input[i])).toBeCloseTo(out[i], 6);
    }
  });

  it('can filter in place', () => {
    const c = BiquadCascade.highpass(SR, 16000, 2);
    const buf = Float32Array.from({ length: 64 }, (_, i) => Math.sin(i));
    expect(() => c.process(buf, buf)).not.toThrow();
  });
});
