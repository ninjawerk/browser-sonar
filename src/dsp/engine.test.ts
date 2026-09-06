import { describe, expect, it } from 'vitest';
import { SonarEngine, robustNoiseFloor, type EngineFrame } from './engine';
import { DEFAULT_CONFIG, type SonarConfig } from './types';
import { metersToSamples, samplesToMeters } from './chirp';

const SR = 48000;
const CFG: SonarConfig = { ...DEFAULT_CONFIG };
const METERS_PER_BIN = samplesToMeters(1, SR);

interface Echo {
  /** Round-trip delay in samples, as a function of absolute simulation time. */
  delay: (timeSec: number) => number;
  amplitude: number;
}

/**
 * A synthetic room.
 *
 * Echoes are evaluated analytically rather than by indexing into the transmit
 * buffer, so a target can sit at a fractional sample delay and move
 * continuously. That matters: sub-sample delay is exactly what breathing
 * detection lives on, and array indexing would quantise it away.
 */
class RoomSim {
  private t = 0;
  private readonly chirpSamples: number;
  private readonly sweepRate: number;
  private readonly taperSamples: number;
  private readonly cwCycles: number;

  constructor(
    private readonly cfg: SonarConfig,
    private readonly sampleRate = SR,
  ) {
    this.chirpSamples = Math.round(cfg.chirpDurationSec * sampleRate);
    this.sweepRate = (cfg.f1 - cfg.f0) / (this.chirpSamples / sampleRate);
    this.taperSamples = Math.round(cfg.taperSec * sampleRate);
    // Mirror the engine's cycle-snapping so the simulated tone loops the same
    // way the transmitted one does.
    this.cwCycles = Math.round((cfg.cwFrequency * cfg.periodSamples) / sampleRate);
  }

  /** Seconds of audio rendered so far. Motion is defined relative to this. */
  get now(): number {
    return this.t / this.sampleRate;
  }

  /** The transmitted waveform at fractional position `p` within a period. */
  private txAt(p: number): number {
    const n = this.cfg.periodSamples;
    let x = p % n;
    if (x < 0) x += n;

    if (this.cfg.mode === 'cw') {
      return Math.sin((2 * Math.PI * this.cwCycles * x) / n);
    }

    if (x >= this.chirpSamples) return 0;
    const t = x / this.sampleRate;
    let v = Math.sin(2 * Math.PI * (this.cfg.f0 * t + 0.5 * this.sweepRate * t * t));
    const tap = this.taperSamples;
    if (tap > 0) {
      if (x < tap) v *= 0.5 * (1 - Math.cos((Math.PI * x) / tap));
      else if (x > this.chirpSamples - 1 - tap) {
        const d = this.chirpSamples - 1 - x;
        v *= 0.5 * (1 - Math.cos((Math.PI * Math.max(0, d)) / tap));
      }
    }
    return v;
  }

  render(count: number, echoes: Echo[], noiseAmplitude = 0): Float32Array {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const timeSec = this.t / this.sampleRate;
      let v = 0;
      for (const e of echoes) v += e.amplitude * this.txAt(this.t - e.delay(timeSec));
      if (noiseAmplitude > 0) {
        v += noiseAmplitude * (Math.random() * 2 - 1);
      }
      out[i] = v;
      this.t++;
    }
    return out;
  }
}

interface RunResult {
  frames: EngineFrame[];
  last: EngineFrame;
}

/** Feed the engine in 128-sample render quanta, exactly as the worklet does. */
function run(
  engine: SonarEngine,
  sim: RoomSim,
  seconds: number,
  echoes: Echo[],
  noise = 0,
  collect: (frame: EngineFrame) => void = () => {},
): RunResult {
  const quanta = Math.floor((seconds * SR) / 128);
  const frames: EngineFrame[] = [];
  for (let q = 0; q < quanta; q++) {
    const chunk = sim.render(128, echoes, noise);
    engine.push(chunk, chunk.length, (frame) => {
      // Frames expose internal buffers; snapshot what the assertions need.
      const copy: EngineFrame = {
        ...frame,
        profile: new Float32Array(frame.profile.subarray(0, frame.analysisBins)),
        rawProfile: new Float32Array(frame.rawProfile.subarray(0, frame.analysisBins)),
        targets: frame.targets.map((t) => ({ ...t })),
        breathing: { ...frame.breathing },
        diagnostics: { ...frame.diagnostics },
        spectrogram: new Uint8Array(0),
      };
      frames.push(copy);
      collect(copy);
    });
  }
  return { frames, last: frames[frames.length - 1] };
}

// A plausible browser output+input latency, which is what puts the direct path
// at an arbitrary, unknown position in the period.
const LATENCY = 1731;
const direct: Echo = { delay: () => LATENCY, amplitude: 1 };
/** A little room noise; a perfectly silent room is not a realistic test. */
const NOISE = 0.002;

/** A target at a fixed range. */
function still(meters: number, amplitude = 0.08): Echo {
  return { delay: () => LATENCY + metersToSamples(meters, SR), amplitude };
}

/**
 * A target moving at constant radial speed, positive = approaching.
 * `t0` must be the simulator's clock at the moment the motion starts.
 */
function moving(startMeters: number, speed: number, t0: number, amplitude = 0.09): Echo {
  return {
    delay: (t) => LATENCY + metersToSamples(startMeters - speed * (t - t0), SR),
    amplitude,
  };
}

describe('SonarEngine construction', () => {
  it('produces a transmit buffer of exactly one period', () => {
    const e = new SonarEngine(SR, CFG);
    expect(e.transmitBuffer.length).toBe(CFG.periodSamples);
  });

  it('rejects a non-power-of-two period', () => {
    expect(() => new SonarEngine(SR, { ...CFG, periodSamples: 3000 })).toThrow(
      /power of two/,
    );
  });

  it('reports the frame rate and bin size', () => {
    const e = new SonarEngine(SR, CFG);
    expect(e.framePeriodSeconds).toBeCloseTo(4096 / 48000, 9);
    expect(e.metersPerBin).toBeCloseTo(0.003573, 5);
  });

  it('switches waveform when the mode changes', () => {
    const e = new SonarEngine(SR, CFG);
    const { transmitChanged } = e.setConfig({ ...CFG, mode: 'cw' });
    expect(transmitChanged).toBe(true);
    const tx = e.transmitBuffer;
    let lateEnergy = 0;
    for (let i = 2000; i < tx.length; i++) lateEnergy += tx[i] * tx[i];
    expect(lateEnergy).toBeGreaterThan(0);
  });

  it('does not rebuild for changes that only affect detection', () => {
    const e = new SonarEngine(SR, CFG);
    const { transmitChanged } = e.setConfig({ ...CFG, presenceThreshold: 9 });
    expect(transmitChanged).toBe(false);
    expect(e.getConfig().presenceThreshold).toBe(9);
  });
});

describe('direct-path calibration', () => {
  it('finds the leakage peak wherever browser latency happens to put it', () => {
    for (const latency of [0, 137, 1731, 4000]) {
      const engine = new SonarEngine(SR, CFG);
      const sim = new RoomSim(CFG);
      const { last } = run(engine, sim, 1, [{ delay: () => latency, amplitude: 1 }]);
      expect(Math.abs(last.diagnostics.directPathIndex - latency)).toBeLessThan(1);
    }
  });

  it('measures range relative to the direct path, not absolute time', () => {
    // Same geometry, wildly different latency: the reported range must be
    // identical. This is the property that removes per-device calibration.
    const targetMeters = 0.7;
    const ranges: number[] = [];
    for (const latency of [500, 2600]) {
      const engine = new SonarEngine(SR, CFG);
      const sim = new RoomSim(CFG);
      const lead: Echo = { delay: () => latency, amplitude: 1 };
      const target: Echo = {
        delay: () => latency + metersToSamples(targetMeters, SR),
        amplitude: 0.08,
      };
      run(engine, sim, 4, [lead], NOISE);
      const { last } = run(engine, sim, 1.5, [lead, target], NOISE);
      expect(last.targets.length).toBeGreaterThan(0);
      ranges.push(last.targets[0].rangeMeters);
    }
    expect(Math.abs(ranges[0] - ranges[1])).toBeLessThan(0.005);
    expect(Math.abs(ranges[0] - targetMeters)).toBeLessThan(0.02);
  });
});

describe('ranging', () => {
  it('places a single target well inside the 5 cm accuracy target', () => {
    for (const meters of [0.3, 0.6, 1.2, 2.0]) {
      const engine = new SonarEngine(SR, CFG);
      const sim = new RoomSim(CFG);
      run(engine, sim, 4, [direct], NOISE);
      const { last } = run(engine, sim, 1.2, [direct, still(meters)], NOISE);
      expect(last.targets.length).toBeGreaterThan(0);
      expect(Math.abs(last.targets[0].rangeMeters - meters)).toBeLessThan(0.02);
    }
  });

  it('resolves two hands 30 cm apart as two separate peaks', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], NOISE);
    const { last } = run(
      engine,
      sim,
      1.2,
      [direct, still(0.3, 0.08), still(0.6, 0.07)],
      NOISE,
    );

    const ranges = last.targets.map((t) => t.rangeMeters);
    expect(ranges.some((r) => Math.abs(r - 0.3) < 0.03)).toBe(true);
    expect(ranges.some((r) => Math.abs(r - 0.6) < 0.03)).toBe(true);
  });

  it('survives broadband noise at ten times the nominal level', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], 0.02);
    const { last } = run(engine, sim, 1.5, [direct, still(0.5, 0.06)], 0.02);
    expect(last.targets.length).toBeGreaterThan(0);
    expect(Math.abs(last.targets[0].rangeMeters - 0.5)).toBeLessThan(0.05);
  });

  it('keeps the direct-path sidelobes from masking a nearby target', () => {
    // The tapered receive filter exists for this case: leakage is ~20 dB
    // stronger than the target and only 30 cm away in range.
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    const { last } = run(engine, sim, 1.5, [direct], NOISE);
    const peak = last.rawProfile[0];
    // Sidelobes a metre out must be far below a plausible target return.
    for (const bin of [150, 300, 600]) {
      expect(last.rawProfile[bin] / peak).toBeLessThan(0.02);
    }
  });
});

describe('background subtraction and presence', () => {
  const clutter = [direct, still(0.75, 0.05), still(1.6, 0.03)];

  it('reports no presence in a static room', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 6, clutter, NOISE);
    const { frames } = run(engine, sim, 3, clutter, NOISE);
    expect(frames.filter((f) => f.presence).length).toBe(0);
  });

  it('flips to presence within a second of a target appearing', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 6, clutter, NOISE);
    const { frames } = run(engine, sim, 2, [...clutter, still(0.5)], NOISE);

    const firstPresent = frames.findIndex((f) => f.presence);
    expect(firstPresent).toBeGreaterThanOrEqual(0);
    expect(frames[firstPresent].time - frames[0].time).toBeLessThan(1);
  });

  it('lets static clutter fade back into the background', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 5, [direct], NOISE);
    const withObject = [direct, still(0.7)];

    // A newly-placed object registers as a target at first...
    const early = run(engine, sim, 1, withObject, NOISE);
    expect(early.frames.some((f) => f.presence)).toBe(true);

    // ...and stops being one once it has sat still for several time constants.
    const late = run(engine, sim, 20, withObject, NOISE);
    expect(late.frames.slice(-15).some((f) => f.presence)).toBe(false);
  });

  it('freezes the background on request', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 5, [direct], NOISE);
    engine.setConfig({ ...CFG, freezeBackground: true });

    const { frames } = run(engine, sim, 12, [direct, still(0.7)], NOISE);
    // With the background frozen the object never fades out.
    const tail = frames.slice(-15);
    expect(tail.filter((f) => f.presence).length).toBeGreaterThan(tail.length * 0.8);
  });

  it('re-learns the room after resetBackground', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    const withObject = [direct, still(0.7)];
    run(engine, sim, 5, [direct], NOISE);
    run(engine, sim, 1, withObject, NOISE);

    engine.resetBackground();
    const { frames } = run(engine, sim, 3, withObject, NOISE);
    // After a reset the object is part of the room again.
    expect(frames.slice(-10).some((f) => f.presence)).toBe(false);
  });
});

describe('velocity from range migration', () => {
  const meanVelocity = (frames: EngineFrame[]) => {
    const movingFrames = frames.filter((f) => f.motionEnergy > 3);
    expect(movingFrames.length).toBeGreaterThan(3);
    return movingFrames.reduce((s, f) => s + f.velocity, 0) / movingFrames.length;
  };

  it('reports approach as positive and recede as negative', () => {
    for (const speed of [0.6, 1.0]) {
      for (const sign of [1, -1]) {
        const engine = new SonarEngine(SR, CFG);
        const sim = new RoomSim(CFG);
        run(engine, sim, 4, [direct], NOISE);
        const t0 = sim.now;
        const start = sign > 0 ? 1.6 : 0.35;
        const { frames } = run(
          engine,
          sim,
          1.0,
          [direct, moving(start, sign * speed, t0)],
          NOISE,
        );
        const mean = meanVelocity(frames);
        expect(Math.sign(mean)).toBe(sign);
        expect(Math.abs(mean)).toBeGreaterThan(speed * 0.4);
        expect(Math.abs(mean)).toBeLessThan(speed * 1.8);
      }
    }
  });

  it('reads close to zero for a stationary target', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], NOISE);
    const { frames } = run(engine, sim, 2, [direct, still(0.8)], NOISE);
    const mean = frames.reduce((s, f) => s + f.velocity, 0) / frames.length;
    expect(Math.abs(mean)).toBeLessThan(0.15);
  });

  it('does not alias at speeds where inter-chirp phase would', () => {
    // Phase-based Doppler wraps above ~0.05 m/s here. Range migration must
    // still be right at 2 m/s, forty times past that limit.
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], NOISE);
    const t0 = sim.now;
    const { frames } = run(engine, sim, 0.9, [direct, moving(2.4, 2.0, t0)], NOISE);
    expect(meanVelocity(frames)).toBeGreaterThan(1.0);
  });
});

describe('gestures', () => {
  const swipe = (sign: number) => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], NOISE);
    const t0 = sim.now;
    const start = sign > 0 ? 1.3 : 0.35;
    const { frames } = run(engine, sim, 1.4, [direct, moving(start, sign * 0.9, t0)], NOISE);
    return frames.map((f) => f.gesture);
  };

  it('classifies a swipe toward the device', () => {
    const gestures = swipe(1);
    expect(gestures).toContain('swipeToward');
    expect(gestures).not.toContain('swipeAway');
  });

  it('classifies a swipe away from the device', () => {
    const gestures = swipe(-1);
    expect(gestures).toContain('swipeAway');
    expect(gestures).not.toContain('swipeToward');
  });

  it('fires once per swipe, not once per frame', () => {
    expect(swipe(1).filter((g) => g !== 'none').length).toBe(1);
  });

  it('stays silent in an empty room', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], 0.01);
    const { frames } = run(engine, sim, 6, [direct], 0.01);
    expect(frames.every((f) => f.gesture === 'none')).toBe(true);
  });

  it('stays silent for a target that is present but still', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    run(engine, sim, 4, [direct], NOISE);
    const { frames } = run(engine, sim, 4, [direct, still(0.6)], NOISE);
    expect(frames.every((f) => f.gesture === 'none')).toBe(true);
  });
});

describe('breathing', () => {
  it(
    'recovers a known respiration rate from sub-millimetre chest motion',
    { timeout: 180000 },
    () => {
      const engine = new SonarEngine(SR, CFG);
      const sim = new RoomSim(CFG);
      const rangeMeters = 0.5;
      const baseBins = metersToSamples(rangeMeters, SR);

      // 15 breaths per minute, 5 mm of chest displacement. That is far less
      // than one range bin (3.6 mm one-way per bin, and the compressed pulse
      // is ~18 bins wide), so only phase can see it.
      const bpm = 15;
      const fb = bpm / 60;
      const swingBins = metersToSamples(0.005, SR);
      engine.setTrackedBin(Math.round(baseBins));

      const t0 = sim.now;
      const chest: Echo = {
        delay: (t) =>
          LATENCY + baseBins + swingBins * Math.sin(2 * Math.PI * fb * (t - t0)),
        amplitude: 0.08,
      };
      const { last } = run(engine, sim, 45, [direct, chest], NOISE);

      expect(last.breathing.bpm).not.toBeNull();
      expect(Math.abs((last.breathing.bpm as number) - bpm)).toBeLessThan(2);
      expect(last.breathing.confidence).toBeGreaterThan(3);
      expect(last.breathing.trackedRangeMeters).toBeCloseTo(rangeMeters, 1);
    },
  );

  it('tracks a faster rate too', { timeout: 180000 }, () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    const baseBins = metersToSamples(0.5, SR);
    const bpm = 24;
    const swingBins = metersToSamples(0.004, SR);
    engine.setTrackedBin(Math.round(baseBins));
    const t0 = sim.now;
    const { last } = run(
      engine,
      sim,
      45,
      [
        direct,
        {
          delay: (t) =>
            LATENCY +
            baseBins +
            swingBins * Math.sin((2 * Math.PI * bpm * (t - t0)) / 60),
          amplitude: 0.08,
        },
      ],
      NOISE,
    );
    expect(last.breathing.bpm).not.toBeNull();
    expect(Math.abs((last.breathing.bpm as number) - bpm)).toBeLessThan(2);
  });

  it('reports no confident rate for a perfectly still target', { timeout: 180000 }, () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    const baseBins = metersToSamples(0.5, SR);
    engine.setTrackedBin(Math.round(baseBins));
    const { last } = run(engine, sim, 40, [direct, still(0.5)], 0.004);
    // Either no peak at all, or one with poor confidence — never a confident
    // wrong answer.
    if (last.breathing.bpm !== null) {
      expect(last.breathing.confidence).toBeLessThan(8);
    }
  });

  it('lets the tracked bin be pinned and released', () => {
    const engine = new SonarEngine(SR, CFG);
    engine.setTrackedBin(140);
    const sim = new RoomSim(CFG);
    const { last } = run(engine, sim, 2, [direct], NOISE);
    expect(last.breathing.trackedRangeMeters).toBeCloseTo(140 * METERS_PER_BIN, 6);
    expect(() => engine.setTrackedBin(null)).not.toThrow();
  });
});

describe('CW Doppler mode', () => {
  const cwCfg: SonarConfig = { ...CFG, mode: 'cw' };
  const cwDirect: Echo = { delay: () => LATENCY, amplitude: 1 };

  it('separates approach from recede by the sign of the shift', () => {
    for (const sign of [1, -1]) {
      const engine = new SonarEngine(SR, cwCfg);
      const sim = new RoomSim(cwCfg);
      run(engine, sim, 1, [cwDirect], NOISE);
      const t0 = sim.now;
      const { frames } = run(
        engine,
        sim,
        2,
        [cwDirect, moving(1.0, sign * 0.8, t0, 0.06)],
        NOISE,
      );
      const active = frames.filter((f) => Math.abs(f.velocity) > 0.05);
      expect(active.length).toBeGreaterThan(3);
      const mean = active.reduce((s, f) => s + f.velocity, 0) / active.length;
      expect(Math.sign(mean)).toBe(sign);
      // Within 30% of truth is plenty for a direction indicator.
      expect(Math.abs(mean)).toBeGreaterThan(0.8 * 0.5);
      expect(Math.abs(mean)).toBeLessThan(0.8 * 1.5);
    }
  });

  it('emits a spectrum in which the carrier dominates', () => {
    const engine = new SonarEngine(SR, cwCfg);
    const sim = new RoomSim(cwCfg);
    const { last } = run(engine, sim, 1, [cwDirect]);
    expect(last.cwSpectrum).not.toBeNull();
    expect(last.cwBins).toBe(129);
    const spec = last.cwSpectrum as Float32Array;
    const centre = (last.cwBins - 1) >> 1;
    for (let i = 0; i < spec.length; i++) {
      if (Math.abs(i - centre) > 4) expect(spec[i]).toBeLessThan(spec[centre]);
    }
  });

  it('snaps the carrier to a whole number of cycles per period', () => {
    const engine = new SonarEngine(SR, cwCfg);
    const cycles = (engine.cwCarrierFrequency * cwCfg.periodSamples) / SR;
    expect(cycles).toBeCloseTo(Math.round(cycles), 9);
  });

  it('is quiet when nothing moves', () => {
    const engine = new SonarEngine(SR, cwCfg);
    const sim = new RoomSim(cwCfg);
    run(engine, sim, 1, [cwDirect, still(0.8)], NOISE);
    const { frames } = run(engine, sim, 2, [cwDirect, still(0.8)], NOISE);
    const mean =
      frames.reduce((s, f) => s + Math.abs(f.velocity), 0) / frames.length;
    expect(mean).toBeLessThan(0.1);
  });
});

describe('diagnostics', () => {
  it('measures spectral tilt and usable bandwidth', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    const { last } = run(engine, sim, 3, [direct]);
    // A perfect simulated speaker is flat, so tilt is small and the usable
    // bandwidth is close to the full sweep.
    expect(Math.abs(last.diagnostics.spectralTiltDb)).toBeLessThan(12);
    expect(last.diagnostics.usableBandwidthHz).toBeGreaterThan(3000);
  });

  it('reports a sensible in-band level', () => {
    const loudEngine = new SonarEngine(SR, CFG);
    const loud = run(loudEngine, new RoomSim(CFG), 2, [direct]);
    const quietEngine = new SonarEngine(SR, CFG);
    const quiet = run(quietEngine, new RoomSim(CFG), 2, [
      { delay: () => LATENCY, amplitude: 0.1 },
    ]);
    expect(loud.last.diagnostics.inBandLevelDb).toBeGreaterThan(
      quiet.last.diagnostics.inBandLevelDb + 15,
    );
  });

  it('produces spectrogram columns every frame', () => {
    const engine = new SonarEngine(SR, CFG);
    const sim = new RoomSim(CFG);
    let columns = 0;
    let bins = 0;
    run(engine, sim, 1, [direct], 0, (f) => {
      columns = f.spectrogramColumns;
      bins = f.spectrogramBins;
    });
    // One column per 128-sample hop across a 4096-sample period.
    expect(columns).toBe(32);
    expect(bins).toBe(129);
  });
});

describe('sample rate fallback', () => {
  it('works at 44.1 kHz, where the sweep runs close to Nyquist', () => {
    const sr = 44100;
    const engine = new SonarEngine(sr, CFG);
    const sim = new RoomSim(CFG, sr);
    const lead: Echo = { delay: () => 900, amplitude: 1 };
    const target: Echo = {
      delay: () => 900 + metersToSamples(0.6, sr),
      amplitude: 0.09,
    };
    const quanta = Math.floor((5 * sr) / 128);
    let last: EngineFrame | null = null;
    for (let q = 0; q < quanta; q++) {
      const withTarget = q > quanta * 0.75;
      const chunk = sim.render(128, withTarget ? [lead, target] : [lead], NOISE);
      engine.push(chunk, chunk.length, (f) => {
        last = {
          ...f,
          targets: f.targets.map((t) => ({ ...t })),
        } as EngineFrame;
      });
    }
    expect(last).not.toBeNull();
    const frame = last as unknown as EngineFrame;
    expect(frame.targets.length).toBeGreaterThan(0);
    expect(Math.abs(frame.targets[0].rangeMeters - 0.6)).toBeLessThan(0.05);
  });
});

describe('robustNoiseFloor', () => {
  it('ignores a single large outlier', () => {
    const data = new Float32Array(100).fill(1);
    data[50] = 500;
    expect(robustNoiseFloor(data, 0, 100)).toBeCloseTo(1, 2);
  });

  it('tracks the level of the bulk of the data', () => {
    const data = new Float32Array(200);
    for (let i = 0; i < 200; i++) data[i] = (i % 2 === 0 ? 1 : -1) * 0.4;
    expect(robustNoiseFloor(data, 0, 200)).toBeCloseTo(0.4, 6);
  });

  it('returns zero for an empty range', () => {
    expect(robustNoiseFloor(new Float32Array(10), 5, 5)).toBe(0);
  });
});
