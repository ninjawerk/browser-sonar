import { describe, expect, it } from 'vitest';
import { Correlator, argMax, complexMagnitude, crossCorrelateDirect, parabolicPeakOffset } from './xcorr';
import { generateTransmitPeriod } from './chirp';

const SR = 48000;

/** Wrap an angle to (-pi, pi] so phase differences can be compared. */
function wrap(a: number): number {
  let x = a % (2 * Math.PI);
  if (x > Math.PI) x -= 2 * Math.PI;
  if (x <= -Math.PI) x += 2 * Math.PI;
  return x;
}

function makeTemplate(periodSamples: number): Float32Array {
  return generateTransmitPeriod({
    sampleRate: SR,
    f0: 18000,
    f1: 22000,
    durationSec: 0.015,
    taperSec: 0.001,
    periodSamples,
  });
}

/** rx[i] = sum of delayed, scaled copies of the looping transmit period. */
function synthesise(
  tx: Float32Array,
  echoes: { delay: number; amplitude: number }[],
): Float32Array {
  const n = tx.length;
  const out = new Float32Array(n);
  for (const e of echoes) {
    for (let i = 0; i < n; i++) {
      out[i] += e.amplitude * tx[(((i - e.delay) % n) + n) % n];
    }
  }
  return out;
}

describe('Correlator', () => {
  it('rejects a template longer than the FFT', () => {
    expect(() => new Correlator(64, new Float32Array(128))).toThrow(/longer than/);
  });

  it('rejects a mismatched frame length', () => {
    const c = new Correlator(1024, new Float32Array(1024));
    expect(() => c.correlate(new Float32Array(512), new Float64Array(1024), new Float64Array(1024))).toThrow();
  });

  it('puts the peak at the echo delay', () => {
    const n = 4096;
    const tx = makeTemplate(n);
    const c = new Correlator(n, tx);
    const delay = 733;
    const rx = synthesise(tx, [{ delay, amplitude: 1 }]);

    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const mag = new Float64Array(n);
    c.correlate(rx, re, im);
    complexMagnitude(re, im, mag);

    expect(argMax(mag)).toBe(delay);
  });

  it('resolves two echoes separated by more than c/(2B)', () => {
    const n = 4096;
    const tx = makeTemplate(n);
    const c = new Correlator(n, tx);
    // 4 kHz of bandwidth gives ~4.3 cm resolution = ~12 samples. 40 samples
    // apart should be unambiguous.
    const rx = synthesise(tx, [
      { delay: 500, amplitude: 1 },
      { delay: 540, amplitude: 0.8 },
    ]);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const mag = new Float64Array(n);
    c.correlate(rx, re, im);
    complexMagnitude(re, im, mag);

    // Overlapping compressed pulses interfere slightly, so allow a sample of
    // pull on each apparent peak — 1 sample is 3.6 mm of range.
    expect(Math.abs(argMax(mag, 480, 520) - 500)).toBeLessThanOrEqual(2);
    expect(Math.abs(argMax(mag, 521, 560) - 540)).toBeLessThanOrEqual(2);
    // Both peaks must stand clear of the valley between them.
    const valley = Math.min(...Array.from(mag.subarray(510, 531)));
    expect(mag[500]).toBeGreaterThan(valley * 2);
    expect(mag[540]).toBeGreaterThan(valley * 2);
  });

  it('is circular, so the answer does not depend on where the window starts', () => {
    // This is the property the direct-path calibration relies on: the receiver
    // has no idea where in the transmit period it started listening.
    const n = 4096;
    const tx = makeTemplate(n);
    const c = new Correlator(n, tx);
    const rx = synthesise(tx, [{ delay: 200, amplitude: 1 }]);

    const rotate = (buf: Float32Array, by: number) => {
      const out = new Float32Array(buf.length);
      for (let i = 0; i < buf.length; i++) out[i] = buf[(i + by) % buf.length];
      return out;
    };

    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const mag = new Float64Array(n);

    c.correlate(rx, re, im);
    complexMagnitude(re, im, mag);
    const peakA = argMax(mag);
    const heightA = mag[peakA];

    const shifted = rotate(rx, 1234);
    c.correlate(shifted, re, im);
    complexMagnitude(re, im, mag);
    const peakB = argMax(mag);

    // The peak moves with the rotation but keeps the same height: no edge
    // effects, no windowing loss.
    expect(peakB).toBe(((peakA - 1234) % n + n) % n);
    expect(mag[peakB]).toBeCloseTo(heightA, 3);
  });

  it('gives a smooth envelope in analytic mode but an oscillating one otherwise', () => {
    const n = 2048;
    const tx = makeTemplate(n);
    const rx = synthesise(tx, [{ delay: 400, amplitude: 1 }]);

    const measure = (analytic: boolean) => {
      const c = new Correlator(n, tx, { analytic });
      const re = new Float64Array(n);
      const im = new Float64Array(n);
      const mag = new Float64Array(n);
      c.correlate(rx, re, im);
      complexMagnitude(re, im, mag);
      // Count sign changes of the derivative near the peak.
      let flips = 0;
      for (let i = 391; i < 410; i++) {
        const d1 = mag[i] - mag[i - 1];
        const d2 = mag[i + 1] - mag[i];
        if (d1 * d2 < 0) flips++;
      }
      return flips;
    };

    // The real-template envelope rectifies the 20 kHz carrier and wobbles;
    // the analytic one is a single smooth lobe.
    expect(measure(true)).toBeLessThan(measure(false));
    expect(measure(true)).toBeLessThanOrEqual(1);
  });

  it('recovers sub-sample delay as carrier phase, which is what breathing needs', () => {
    const n = 4096;
    const tx = makeTemplate(n);
    const c = new Correlator(n, tx);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    const mag = new Float64Array(n);

    const chirpSamples = Math.round(0.015 * SR);
    const sweepRate = 4000 / (chirpSamples / SR);
    const taper = Math.round(0.001 * SR);

    /** The transmitted chirp evaluated at a fractional sample position. */
    const chirpAt = (x: number): number => {
      if (x < 0 || x >= chirpSamples) return 0;
      const t = x / SR;
      let v = Math.sin(2 * Math.PI * (18000 * t + 0.5 * sweepRate * t * t));
      if (x < taper) v *= 0.5 * (1 - Math.cos((Math.PI * x) / taper));
      else if (x > chirpSamples - 1 - taper) {
        const d = Math.max(0, chirpSamples - 1 - x);
        v *= 0.5 * (1 - Math.cos((Math.PI * d) / taper));
      }
      return v;
    };

    const measure = (delaySamples: number) => {
      const rx = new Float32Array(n);
      for (let i = 0; i < n; i++) rx[i] = chirpAt(i - delaySamples);
      c.correlate(rx, re, im);
      complexMagnitude(re, im, mag);
      const peak = argMax(mag);
      return { peak, phase: Math.atan2(im[peak], re[peak]) };
    };

    // A quarter-sample shift does not move the peak bin at all...
    const a = measure(600);
    const b = measure(600.25);
    expect(a.peak).toBe(600);
    expect(b.peak).toBe(600);

    // ...but it is plainly visible in the phase. At a 20 kHz centre frequency
    // a quarter sample is 2*pi*20000*0.25/48000 = 0.65 rad. This is the whole
    // basis of breathing detection: displacement far below one range bin still
    // shows up, because phase resolves ~100x finer than the bin grid.
    const delta = Math.abs(wrap(a.phase - b.phase));
    expect(delta).toBeGreaterThan(0.4);
    expect(delta).toBeLessThan(0.9);

    // An integer-sample shift, by contrast, moves the bin and leaves the peak
    // phase essentially unchanged — the information has moved into the index.
    const cInt = measure(601);
    expect(cInt.peak).toBe(601);
    expect(Math.abs(wrap(cInt.phase - a.phase))).toBeLessThan(0.05);
  });
});

describe('crossCorrelateDirect', () => {
  it('finds a known lag', () => {
    const b = new Float64Array([1, 2, 3]);
    const a = new Float64Array([0, 0, 1, 2, 3, 0]);
    const r = crossCorrelateDirect(a, b);
    expect(argMax(r)).toBe(2);
  });

  it('agrees with the FFT correlator', () => {
    const n = 4096;
    const tx = makeTemplate(n);
    const rx = synthesise(tx, [{ delay: 300, amplitude: 1 }]);

    const c = new Correlator(n, tx, { analytic: false });
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    c.correlate(rx, re, im);

    // The naive linear correlation against the bare chirp agrees with the
    // circular FFT result wherever the wrap does not reach.
    const direct = crossCorrelateDirect(
      Float64Array.from(rx),
      Float64Array.from(tx.subarray(0, 720)),
    );
    const scale = direct[300];
    expect(scale).toBeGreaterThan(0);
    for (const lag of [250, 300, 350]) {
      expect(re[lag] / scale).toBeCloseTo(direct[lag] / scale, 6);
    }
  });
});

describe('parabolicPeakOffset', () => {
  it('is zero for a symmetric peak', () => {
    expect(parabolicPeakOffset(new Float64Array([1, 2, 1]), 1)).toBeCloseTo(0, 12);
  });

  it('leans towards the taller neighbour', () => {
    expect(parabolicPeakOffset(new Float64Array([1, 2, 1.5]), 1)).toBeGreaterThan(0);
    expect(parabolicPeakOffset(new Float64Array([1.5, 2, 1]), 1)).toBeLessThan(0);
  });

  it('recovers a known sub-sample shift of a parabola', () => {
    // y = -(x - 0.25)^2 sampled at -1, 0, 1
    const f = (x: number) => -((x - 0.25) ** 2);
    const off = parabolicPeakOffset(new Float64Array([f(-1), f(0), f(1)]), 1);
    expect(off).toBeCloseTo(0.25, 9);
  });

  it('returns zero at the array edges', () => {
    const m = new Float64Array([3, 2, 1]);
    expect(parabolicPeakOffset(m, 0)).toBe(0);
    expect(parabolicPeakOffset(m, 2)).toBe(0);
  });

  it('returns zero for a flat region rather than dividing by zero', () => {
    expect(parabolicPeakOffset(new Float64Array([1, 1, 1]), 1)).toBe(0);
  });
});

describe('argMax', () => {
  it('honours the search range', () => {
    const m = new Float64Array([5, 1, 9, 1, 7]);
    expect(argMax(m)).toBe(2);
    expect(argMax(m, 3)).toBe(4);
    expect(argMax(m, 0, 2)).toBe(0);
  });
});
