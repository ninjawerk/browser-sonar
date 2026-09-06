/**
 * Window functions.
 *
 * Two conventions matter here and they are easy to confuse:
 *  - "symmetric" windows (denominator N-1) are for filter design and for
 *    tapering a finite waveform such as our chirp;
 *  - "periodic" windows (denominator N) are for spectral analysis, because
 *    they make the window seamless when the DFT wraps around.
 */

import type { FloatArray } from './fft';

export type WindowKind = 'hann' | 'hamming' | 'blackmanHarris';

export function hann(n: number, periodic = false): Float32Array {
  const w = new Float32Array(n);
  const denom = periodic ? n : n - 1;
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / denom));
  }
  return w;
}

export function hamming(n: number, periodic = false): Float32Array {
  const w = new Float32Array(n);
  const denom = periodic ? n : n - 1;
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / denom);
  }
  return w;
}

/**
 * 4-term Blackman–Harris. ~-92 dB sidelobes, which is what we want for the
 * CW Doppler mode: the transmitted tone is enormously stronger than the
 * reflections, so spectral leakage from the carrier would otherwise bury the
 * sidebands we are trying to measure.
 */
export function blackmanHarris(n: number, periodic = false): Float32Array {
  const w = new Float32Array(n);
  const denom = periodic ? n : n - 1;
  const a0 = 0.35875;
  const a1 = 0.48829;
  const a2 = 0.14128;
  const a3 = 0.01168;
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < n; i++) {
    const x = (2 * Math.PI * i) / denom;
    w[i] = a0 - a1 * Math.cos(x) + a2 * Math.cos(2 * x) - a3 * Math.cos(3 * x);
  }
  return w;
}

export function makeWindow(kind: WindowKind, n: number, periodic = false): Float32Array {
  switch (kind) {
    case 'hann':
      return hann(n, periodic);
    case 'hamming':
      return hamming(n, periodic);
    case 'blackmanHarris':
      return blackmanHarris(n, periodic);
  }
}

/** Multiply `buf` by `win` in place. Lengths must match. */
export function applyWindow(buf: FloatArray, win: FloatArray): void {
  if (buf.length !== win.length) {
    throw new Error(`window length ${win.length} does not match buffer length ${buf.length}`);
  }
  for (let i = 0; i < buf.length; i++) buf[i] *= win[i];
}

/**
 * Taper only the first and last `taperSamples` of `buf` with the rising and
 * falling halves of a Hann window, leaving the middle at unity gain.
 *
 * This is how the chirp avoids an audible click: a rectangular gate on a
 * 18 kHz tone splatters energy across the whole spectrum, including the range
 * people can actually hear.
 */
export function taperEdges(buf: FloatArray, taperSamples: number): void {
  const n = buf.length;
  const t = Math.min(Math.floor(taperSamples), n >> 1);
  if (t <= 0) return;
  for (let i = 0; i < t; i++) {
    const g = 0.5 * (1 - Math.cos((Math.PI * i) / t));
    buf[i] *= g;
    buf[n - 1 - i] *= g;
  }
}

/** Coherent gain (mean) of a window — the DC scaling it applies. */
export function coherentGain(win: FloatArray): number {
  let s = 0;
  for (let i = 0; i < win.length; i++) s += win[i];
  return s / win.length;
}

/** Noise power bandwidth of a window, in bins. */
export function equivalentNoiseBandwidth(win: FloatArray): number {
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < win.length; i++) {
    s1 += win[i];
    s2 += win[i] * win[i];
  }
  return (win.length * s2) / (s1 * s1);
}
