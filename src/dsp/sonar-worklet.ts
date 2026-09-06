/**
 * The AudioWorklet half of the sonar.
 *
 * Everything expensive happens here, on the audio thread, so that a busy main
 * thread (rendering canvases at 60 Hz) can never cause a dropped chirp. The
 * processor is a thin shell: it hands 128-sample render quanta to SonarEngine
 * and serialises whatever comes back.
 *
 * Note the buffering. AudioWorklet delivers exactly 128 samples per call —
 * 2.7 ms — which is far too short to correlate against a 15 ms chirp. The
 * engine accumulates quanta internally and only runs an analysis once a full
 * transmit period has arrived, roughly every 32 calls.
 */

import { SonarEngine } from './engine';
import { DEFAULT_CONFIG, type MainToWorklet, type SonarConfig } from './types';

class SonarProcessor extends AudioWorkletProcessor {
  private engine: SonarEngine;
  private running = true;

  constructor(options?: { processorOptions?: { config?: SonarConfig } }) {
    super();
    const config = options?.processorOptions?.config ?? DEFAULT_CONFIG;
    this.engine = new SonarEngine(sampleRate, config);

    this.port.onmessage = (event: MessageEvent<MainToWorklet>) => {
      const msg = event.data;
      switch (msg.type) {
        case 'config': {
          const { transmitChanged } = this.engine.setConfig(msg.config);
          if (transmitChanged) this.publishTransmit();
          break;
        }
        case 'resetBackground':
          this.engine.resetBackground();
          break;
        case 'setTrackedBin':
          this.engine.setTrackedBin(msg.bin);
          break;
      }
    };

    this.port.postMessage({ type: 'ready', sampleRate });
    this.publishTransmit();
  }

  /**
   * Send the transmit waveform to the main thread rather than letting it
   * generate its own. The correlation template and the transmitted signal must
   * be bit-identical; regenerating them independently on two threads is an
   * invitation for them to drift apart after a refactor.
   */
  private publishTransmit(): void {
    const buffer = new Float32Array(this.engine.transmitBuffer);
    this.port.postMessage(
      {
        type: 'transmit',
        buffer,
        cwFrequency: this.engine.cwCarrierFrequency,
        framePeriodSeconds: this.engine.framePeriodSeconds,
      },
      [buffer.buffer],
    );
  }

  process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0) return this.running;
    const channel = input[0];
    if (!channel || channel.length === 0) return this.running;

    this.engine.push(channel, channel.length, (frame) => {
      // One allocation set per chirp (~12 Hz), not per quantum. The arrays are
      // transferred, so there is no copy on the receiving side either.
      const profile = new Float32Array(frame.profile.subarray(0, frame.analysisBins));
      const rawProfile = new Float32Array(frame.rawProfile.subarray(0, frame.analysisBins));
      const specLength = frame.spectrogramColumns * frame.spectrogramBins;
      const spectrogram = new Uint8Array(frame.spectrogram.subarray(0, specLength));
      const cwSpectrum = frame.cwSpectrum ? new Float32Array(frame.cwSpectrum) : null;

      const transfer: ArrayBuffer[] = [
        profile.buffer,
        rawProfile.buffer,
        spectrogram.buffer,
      ];
      if (cwSpectrum) transfer.push(cwSpectrum.buffer);

      this.port.postMessage(
        {
          type: 'frame',
          frameIndex: frame.frameIndex,
          time: frame.time,
          profile,
          rawProfile,
          metersPerBin: frame.metersPerBin,
          presence: frame.presence,
          velocity: frame.velocity,
          motionEnergy: frame.motionEnergy,
          targets: frame.targets,
          gesture: frame.gesture,
          breathing: frame.breathing,
          diagnostics: frame.diagnostics,
          cwSpectrum,
          cwHzPerBin: frame.cwHzPerBin,
          spectrogram,
          spectrogramBins: frame.spectrogramBins,
          spectrogramHzPerBin: frame.spectrogramHzPerBin,
        },
        transfer,
      );
    });

    return this.running;
  }
}

registerProcessor('sonar-processor', SonarProcessor as never);
