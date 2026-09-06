import { describe, expect, it } from 'vitest';
import { PhaseUnwrapper, detrend, unwrap, wrapToPi } from './unwrap';

describe('wrapToPi', () => {
  it('leaves in-range angles alone', () => {
    expect(wrapToPi(0)).toBeCloseTo(0, 12);
    expect(wrapToPi(1)).toBeCloseTo(1, 12);
    expect(wrapToPi(-1)).toBeCloseTo(-1, 12);
  });

  it('wraps beyond +/-pi', () => {
    expect(wrapToPi(Math.PI + 0.5)).toBeCloseTo(-Math.PI + 0.5, 12);
    expect(wrapToPi(-Math.PI - 0.5)).toBeCloseTo(Math.PI - 0.5, 12);
    expect(wrapToPi(7 * Math.PI)).toBeCloseTo(Math.PI, 12);
  });

  it('always lands in (-pi, pi]', () => {
    for (let i = -50; i <= 50; i++) {
      const v = wrapToPi(i * 0.7);
      expect(v).toBeGreaterThan(-Math.PI - 1e-12);
      expect(v).toBeLessThanOrEqual(Math.PI + 1e-12);
    }
  });
});

describe('unwrap', () => {
  it('removes 2*pi jumps from a ramp', () => {
    const trueRamp = Array.from({ length: 200 }, (_, i) => i * 0.4);
    const wrapped = Float64Array.from(trueRamp.map(wrapToPi));
    const out = unwrap(wrapped);
    for (let i = 0; i < trueRamp.length; i++) {
      // Unwrapping recovers the ramp up to a constant multiple of 2*pi.
      expect(out[i] - out[0]).toBeCloseTo(trueRamp[i] - trueRamp[0], 9);
    }
  });

  it('recovers a sinusoid that crosses the wrap boundary many times', () => {
    const n = 400;
    const truth = Array.from({ length: n }, (_, i) => 12 * Math.sin((2 * Math.PI * i) / 80));
    const wrapped = Float64Array.from(truth.map(wrapToPi));
    const out = unwrap(wrapped);
    for (let i = 0; i < n; i++) {
      expect(out[i] - out[0]).toBeCloseTo(truth[i] - truth[0], 9);
    }
  });

  it('handles empty and single-element inputs', () => {
    expect(unwrap(new Float64Array(0)).length).toBe(0);
    expect(Array.from(unwrap(new Float64Array([1.5])))).toEqual([1.5]);
  });

  it('writes into a provided output array', () => {
    const out = new Float64Array(3);
    const result = unwrap(new Float64Array([0, 1, 2]), out);
    expect(result).toBe(out);
  });
});

describe('PhaseUnwrapper', () => {
  it('matches the batch unwrapper sample by sample', () => {
    const truth = Array.from({ length: 300 }, (_, i) => 5 * Math.sin(i / 17) + i * 0.05);
    const wrapped = Float64Array.from(truth.map(wrapToPi));
    const batch = unwrap(wrapped);
    const streaming = new PhaseUnwrapper();
    for (let i = 0; i < wrapped.length; i++) {
      expect(streaming.next(wrapped[i])).toBeCloseTo(batch[i], 9);
    }
    expect(streaming.value).toBeCloseTo(batch[batch.length - 1], 9);
  });

  it('resets cleanly', () => {
    const u = new PhaseUnwrapper();
    u.next(0);
    u.next(3);
    u.next(-3);
    u.reset();
    expect(u.next(1.25)).toBeCloseTo(1.25, 12);
  });

  it('cannot follow a jump larger than pi, as expected', () => {
    // This is a real limitation, not a bug: a step of more than half a
    // wavelength between chirps is genuinely ambiguous. It is why breathing
    // uses phase and gestures do not.
    const u = new PhaseUnwrapper();
    u.next(0);
    const stepped = u.next(wrapToPi(4)); // true motion of 4 rad
    expect(stepped).not.toBeCloseTo(4, 1);
  });
});

describe('detrend', () => {
  it('removes an exact linear ramp', () => {
    const data = Float64Array.from({ length: 50 }, (_, i) => 3 + 0.7 * i);
    const out = detrend(data);
    for (let i = 0; i < data.length; i++) expect(out[i]).toBeCloseTo(0, 9);
  });

  it('preserves an oscillation riding on a ramp', () => {
    // This is the operation breathing detection depends on: the tracked bin's
    // phase drifts steadily (clock offset, slow posture change) with the
    // respiration signal riding on top, and only the drift should go.
    //
    // A least-squares line fitted to a ramp-plus-sinusoid is never exactly the
    // ramp — even over a whole number of periods, the sum of i*sin(2*pi*i/P)
    // is not zero — so the check is that the oscillation survives intact in
    // shape, not that every sample matches to the last decimal.
    const n = 240;
    const pure = Array.from({ length: n }, (_, i) => 2 * Math.sin((2 * Math.PI * i) / 40));
    const data = Float64Array.from({ length: n }, (_, i) => 10 + 0.3 * i + pure[i]);
    const out = detrend(data);

    // The constant and the ramp are gone.
    const mean = out.reduce((s, v) => s + v, 0) / n;
    expect(mean).toBeCloseTo(0, 9);

    // And what is left is the oscillation, essentially unchanged.
    let num = 0;
    let da = 0;
    let db = 0;
    for (let i = 0; i < n; i++) {
      num += out[i] * pure[i];
      da += out[i] * out[i];
      db += pure[i] * pure[i];
    }
    expect(num / Math.sqrt(da * db)).toBeGreaterThan(0.99);
  });

  it('handles degenerate lengths', () => {
    expect(detrend(new Float64Array(0)).length).toBe(0);
    expect(Array.from(detrend(new Float64Array([5])))).toEqual([0]);
  });

  it('can write in place', () => {
    const data = Float64Array.from([1, 2, 3, 4]);
    detrend(data, data);
    for (const v of data) expect(v).toBeCloseTo(0, 9);
  });
});
