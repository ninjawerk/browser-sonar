import { describe, expect, it } from 'vitest';
import {
  SPEED_OF_SOUND,
  applyChirpTaper,
  generateChirp,
  generateTone,
  generateTransmitPeriod,
  instantaneousFrequency,
  metersToSamples,
  rangeResolutionMeters,
  samplesToMeters,
} from './chirp';
import { FFT } from './fft';

const SR = 48000;
const PARAMS = {
  sampleRate: SR,
  f0: 18000,
  f1: 22000,
  durationSec: 0.015,
  taperSec: 0.001,
};

/** Dominant frequency of a short slice, by zero-crossing rate. */
function zeroCrossingFrequency(buf: Float32Array, from: number, to: number): number {
  let crossings = 0;
  for (let i = from + 1; i < to; i++) {
    if (buf[i - 1] <= 0 && buf[i] > 0) crossings++;
  }
  return (crossings * SR) / (to - from);
}

describe('generateChirp', () => {
  it('has the requested length', () => {
    expect(generateChirp(PARAMS).length).toBe(Math.round(0.015 * SR));
  });

  it('sweeps upward from f0 to f1', () => {
    const c = generateChirp({ ...PARAMS, taperSec: 0 });
    const n = c.length;
    const early = zeroCrossingFrequency(c, 20, 20 + 240);
    const late = zeroCrossingFrequency(c, n - 260, n - 20);
    expect(early).toBeGreaterThan(17000);
    expect(early).toBeLessThan(19500);
    expect(late).toBeGreaterThan(20500);
    expect(late).toBeLessThan(23000);
    expect(late).toBeGreaterThan(early);
  });

  it('concentrates its energy inside the swept band', () => {
    const n = 2048;
    const c = generateChirp(PARAMS);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    re.set(c.subarray(0, Math.min(c.length, n)));
    new FFT(n).forward(re, im);

    const hzPerBin = SR / n;
    let inBand = 0;
    let outOfBand = 0;
    for (let k = 1; k < n / 2; k++) {
      const p = re[k] * re[k] + im[k] * im[k];
      const f = k * hzPerBin;
      if (f >= 17500 && f <= 22500) inBand += p;
      else outOfBand += p;
    }
    // Tapered edges keep out-of-band splatter far below the sweep itself.
    expect(inBand / outOfBand).toBeGreaterThan(50);
  });

  it('tapers to silence at both ends to avoid an audible click', () => {
    const c = generateChirp(PARAMS);
    expect(Math.abs(c[0])).toBeLessThan(1e-6);
    expect(Math.abs(c[c.length - 1])).toBeLessThan(1e-6);
    // Middle still reaches near full amplitude.
    let peakMid = 0;
    for (let i = 300; i < 420; i++) peakMid = Math.max(peakMid, Math.abs(c[i]));
    expect(peakMid).toBeGreaterThan(0.9);
  });

  it('scales with amplitude', () => {
    const quiet = generateChirp({ ...PARAMS, amplitude: 0.25 });
    let peak = 0;
    for (const v of quiet) peak = Math.max(peak, Math.abs(v));
    expect(peak).toBeGreaterThan(0.2);
    expect(peak).toBeLessThanOrEqual(0.25 + 1e-6);
  });

  it('supports a downward sweep', () => {
    const c = generateChirp({ ...PARAMS, f0: 22000, f1: 18000, taperSec: 0 });
    const early = zeroCrossingFrequency(c, 20, 260);
    const late = zeroCrossingFrequency(c, c.length - 260, c.length - 20);
    expect(late).toBeLessThan(early);
  });
});

describe('applyChirpTaper', () => {
  it('is a no-op when the taper is zero', () => {
    const b = new Float32Array([1, 1, 1, 1]);
    applyChirpTaper(b, 0);
    expect(Array.from(b)).toEqual([1, 1, 1, 1]);
  });

  it('never reads past the ends', () => {
    const b = new Float32Array(4).fill(1);
    expect(() => applyChirpTaper(b, 100)).not.toThrow();
  });
});

describe('generateTransmitPeriod', () => {
  it('is exactly one period long with the chirp at the start', () => {
    const p = generateTransmitPeriod({ ...PARAMS, periodSamples: 4096 });
    expect(p.length).toBe(4096);
    let energyEarly = 0;
    let energyLate = 0;
    for (let i = 0; i < 720; i++) energyEarly += p[i] * p[i];
    for (let i = 720; i < 4096; i++) energyLate += p[i] * p[i];
    expect(energyEarly).toBeGreaterThan(0);
    expect(energyLate).toBe(0);
  });

  it('refuses a chirp that does not fit', () => {
    expect(() =>
      generateTransmitPeriod({ ...PARAMS, durationSec: 0.2, periodSamples: 4096 }),
    ).toThrow(/does not fit/);
  });

  it('loops without a discontinuity', () => {
    const p = generateTransmitPeriod({ ...PARAMS, periodSamples: 4096 });
    // The wrap point joins silence to silence.
    expect(Math.abs(p[4095] - p[0])).toBeLessThan(1e-9);
  });
});

describe('generateTone', () => {
  it('snaps to a whole number of cycles so the loop is seamless', () => {
    const { samples, actualFrequency } = generateTone({
      sampleRate: SR,
      frequency: 20000,
      periodSamples: 4096,
    });
    const cycles = (actualFrequency * 4096) / SR;
    expect(cycles).toBeCloseTo(Math.round(cycles), 9);
    // The snap moves the carrier by less than one bin.
    expect(Math.abs(actualFrequency - 20000)).toBeLessThan(SR / 4096);

    // Continuity across the loop point: predict the next sample and compare.
    const w = (2 * Math.PI * Math.round(cycles)) / 4096;
    const predicted = Math.sin(w * 4096);
    expect(Math.abs(predicted - samples[0])).toBeLessThan(1e-9);
  });

  it('produces a single spectral line', () => {
    const n = 4096;
    const { samples, actualFrequency } = generateTone({
      sampleRate: SR,
      frequency: 20000,
      periodSamples: n,
    });
    const re = Float64Array.from(samples);
    const im = new Float64Array(n);
    new FFT(n).forward(re, im);
    const carrierBin = Math.round((actualFrequency * n) / SR);
    const carrierMag = Math.hypot(re[carrierBin], im[carrierBin]);
    for (let k = 1; k < n / 2; k++) {
      if (Math.abs(k - carrierBin) <= 1) continue;
      expect(Math.hypot(re[k], im[k])).toBeLessThan(carrierMag * 1e-6);
    }
  });
});

describe('range arithmetic', () => {
  it('matches the c/(2B) resolution formula', () => {
    expect(rangeResolutionMeters(4000)).toBeCloseTo(SPEED_OF_SOUND / 8000, 9);
    // 4 kHz of sweep is about 4.3 cm.
    expect(rangeResolutionMeters(4000)).toBeGreaterThan(0.042);
    expect(rangeResolutionMeters(4000)).toBeLessThan(0.044);
  });

  it('converts samples to one-way metres and back', () => {
    const m = samplesToMeters(280, SR);
    expect(m).toBeCloseTo((280 / SR) * SPEED_OF_SOUND * 0.5, 9);
    expect(metersToSamples(m, SR)).toBeCloseTo(280, 6);
  });

  it('gives a bin size of about 3.6 mm at 48 kHz', () => {
    expect(samplesToMeters(1, SR)).toBeCloseTo(0.003573, 5);
  });
});

describe('instantaneousFrequency', () => {
  it('interpolates linearly across the sweep', () => {
    expect(instantaneousFrequency(PARAMS, 0)).toBeCloseTo(18000, 6);
    expect(instantaneousFrequency(PARAMS, 0.0075)).toBeCloseTo(20000, 0);
    expect(instantaneousFrequency(PARAMS, 0.015)).toBeCloseTo(22000, 0);
  });
});
