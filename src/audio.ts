/**
 * Web Audio plumbing: microphone capture, the worklet, and the transmitter.
 *
 * The awkward part of browser sonar is not the DSP, it is persuading the
 * browser to hand over an unprocessed microphone signal. Echo cancellation is
 * designed to remove exactly the thing we are trying to measure — a copy of
 * our own output arriving back through the room — so with it enabled the
 * signal simply is not there. We ask for it to be off and then verify, because
 * asking is not the same as getting.
 */

import workletUrl from './dsp/sonar-worklet.ts?worker&url';
import { DEFAULT_CONFIG, type SonarConfig, type SonarFrameMessage } from './dsp/types';

export interface DeviceReport {
  sampleRate: number;
  /** What the browser actually gave us, per MediaStreamTrack.getSettings(). */
  echoCancellation?: boolean;
  noiseSuppression?: boolean;
  autoGainControl?: boolean;
  /** True when all three processing constraints were honoured. */
  constraintsHonoured: boolean;
  /** Which of the three the browser refused to turn off. */
  unhonoured: string[];
  inputLabel: string;
  outputLabel: string;
  /** Heuristic: a Bluetooth path cannot carry 20 kHz and adds huge latency. */
  likelyBluetooth: boolean;
  /** Base latency reported by the AudioContext, seconds. */
  baseLatency: number;
  outputLatency: number;
  /** True if we asked for 48 kHz and got something else. */
  sampleRateAdjusted: boolean;
}

export interface TransmitInfo {
  cwFrequency: number;
  framePeriodSeconds: number;
}

const BLUETOOTH_HINTS = [
  'bluetooth',
  'airpods',
  'buds',
  'headset',
  'hands-free',
  'handsfree',
  'hfp',
  'a2dp',
  'wireless',
  'beats',
  'wh-1000',
  'soundcore',
  'jabra',
];

function looksBluetooth(label: string): boolean {
  const l = label.toLowerCase();
  return BLUETOOTH_HINTS.some((h) => l.includes(h));
}

/**
 * Keep the requested band inside what the sample rate can actually carry.
 *
 * A device that gives us 44.1 kHz has a Nyquist limit of 22.05 kHz, so the
 * default 18–22 kHz sweep would run right up against it, where the anti-alias
 * filter is already rolling off steeply. Backing the top of the sweep down to
 * ~0.47 of the sample rate costs bandwidth (and therefore range resolution),
 * but the alternative is a sweep whose upper half never makes it through.
 */
export function clampConfigToSampleRate(
  config: SonarConfig,
  sampleRate: number,
): { config: SonarConfig; adjusted: boolean } {
  const ceiling = sampleRate * 0.47;
  const next = { ...config };
  let adjusted = false;
  if (next.f1 > ceiling) {
    next.f1 = Math.round(ceiling);
    adjusted = true;
  }
  if (next.f0 >= next.f1 - 500) {
    next.f0 = Math.max(1000, next.f1 - 500);
    adjusted = true;
  }
  if (next.cwFrequency > ceiling) {
    next.cwFrequency = Math.round(ceiling);
    adjusted = true;
  }
  return { config: next, adjusted };
}

export class SonarSession {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;
  private txGain: GainNode | null = null;
  private txSource: AudioBufferSourceNode | null = null;
  private config: SonarConfig = { ...DEFAULT_CONFIG };
  private volume = 0.5;

  onFrame: ((frame: SonarFrameMessage) => void) | null = null;
  onTransmit: ((info: TransmitInfo) => void) | null = null;

  get running(): boolean {
    return this.context !== null;
  }

  get sampleRate(): number {
    return this.context?.sampleRate ?? 0;
  }

  get currentConfig(): SonarConfig {
    return { ...this.config };
  }

  async start(requested: SonarConfig): Promise<DeviceReport> {
    if (this.context) throw new Error('already running');

    // Ask for 48 kHz, but do not insist: some devices only offer 44.1 kHz and
    // failing outright would be worse than adapting.
    let context: AudioContext;
    let sampleRateAdjusted = false;
    try {
      context = new AudioContext({ sampleRate: 48000, latencyHint: 'interactive' });
      if (context.sampleRate !== 48000) sampleRateAdjusted = true;
    } catch {
      context = new AudioContext({ latencyHint: 'interactive' });
      sampleRateAdjusted = true;
    }
    this.context = context;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        // Chrome-specific twins of the above; harmless elsewhere.
        // @ts-expect-error non-standard but widely honoured
        googEchoCancellation: false,
        googAutoGainControl: false,
        googNoiseSuppression: false,
        googHighpassFilter: false,
        channelCount: 1,
      },
      video: false,
    });
    this.stream = stream;

    const track = stream.getAudioTracks()[0];
    const settings = track.getSettings();
    const unhonoured: string[] = [];
    if (settings.echoCancellation === true) unhonoured.push('echoCancellation');
    if (settings.noiseSuppression === true) unhonoured.push('noiseSuppression');
    if (settings.autoGainControl === true) unhonoured.push('autoGainControl');

    let outputLabel = '';
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const out = devices.find((d) => d.kind === 'audiooutput' && d.deviceId === 'default');
      outputLabel = out?.label ?? '';
    } catch {
      // enumerateDevices can reject in some privacy configurations; the label
      // is only used for a warning, so carry on without it.
    }

    const { config } = clampConfigToSampleRate(requested, context.sampleRate);
    this.config = config;

    await context.audioWorklet.addModule(workletUrl);
    if (context.state === 'suspended') await context.resume();

    const node = new AudioWorkletNode(context, 'sonar-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { config },
    });
    this.node = node;

    node.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'frame') {
        this.onFrame?.(msg as SonarFrameMessage);
      } else if (msg.type === 'transmit') {
        this.installTransmitBuffer(msg.buffer as Float32Array);
        this.onTransmit?.({
          cwFrequency: msg.cwFrequency,
          framePeriodSeconds: msg.framePeriodSeconds,
        });
      }
    };

    this.source = context.createMediaStreamSource(stream);
    this.source.connect(node);

    // An AudioWorkletNode is only pulled if something downstream wants its
    // output. We do not want to hear it, so it goes to a silent gain node.
    this.sink = context.createGain();
    this.sink.gain.value = 0;
    node.connect(this.sink);
    this.sink.connect(context.destination);

    this.txGain = context.createGain();
    this.txGain.gain.value = this.volume;
    this.txGain.connect(context.destination);

    const inputLabel = track.label;
    return {
      sampleRate: context.sampleRate,
      echoCancellation: settings.echoCancellation,
      noiseSuppression: settings.noiseSuppression,
      autoGainControl: settings.autoGainControl,
      constraintsHonoured: unhonoured.length === 0,
      unhonoured,
      inputLabel,
      outputLabel,
      likelyBluetooth:
        looksBluetooth(inputLabel) ||
        looksBluetooth(outputLabel) ||
        context.sampleRate <= 24000,
      baseLatency: context.baseLatency ?? 0,
      outputLatency: context.outputLatency ?? 0,
      sampleRateAdjusted,
    };
  }

  /**
   * Loop the transmit period through an AudioBufferSourceNode.
   *
   * The alternative is to synthesise inside the worklet's `process()`. Both
   * work; this way was chosen because an AudioBufferSourceNode loop is
   * sample-exact and maintained by the audio engine itself, so the transmitted
   * period stays rigidly periodic no matter what the worklet is doing. The
   * cost is that the transmitter and receiver are separate graph nodes with an
   * unknown relative offset — which costs us nothing, because the direct-path
   * calibration was always going to measure that offset anyway.
   */
  private installTransmitBuffer(samples: Float32Array): void {
    const context = this.context;
    if (!context || !this.txGain) return;

    this.txSource?.stop();
    this.txSource?.disconnect();

    const buffer = context.createBuffer(1, samples.length, context.sampleRate);
    buffer.getChannelData(0).set(samples);
    const src = context.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    src.connect(this.txGain);
    src.start();
    this.txSource = src;
  }

  setVolume(value: number): void {
    this.volume = value;
    if (this.txGain && this.context) {
      this.txGain.gain.setTargetAtTime(value, this.context.currentTime, 0.02);
    }
  }

  updateConfig(config: SonarConfig): SonarConfig {
    if (!this.context || !this.node) {
      this.config = config;
      return config;
    }
    const { config: clamped } = clampConfigToSampleRate(config, this.context.sampleRate);
    this.config = clamped;
    this.node.port.postMessage({ type: 'config', config: clamped });
    return clamped;
  }

  resetBackground(): void {
    this.node?.port.postMessage({ type: 'resetBackground' });
  }

  setTrackedBin(bin: number | null): void {
    this.node?.port.postMessage({ type: 'setTrackedBin', bin });
  }

  async stop(): Promise<void> {
    this.txSource?.stop();
    this.txSource?.disconnect();
    this.txSource = null;
    this.txGain?.disconnect();
    this.txGain = null;
    this.source?.disconnect();
    this.source = null;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    this.sink?.disconnect();
    this.sink = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.context) {
      await this.context.close();
      this.context = null;
    }
  }
}
