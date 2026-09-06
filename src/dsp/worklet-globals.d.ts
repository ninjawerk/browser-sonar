/**
 * Minimal ambient declarations for AudioWorkletGlobalScope.
 *
 * TypeScript's DOM lib describes the main-thread half of Web Audio but not the
 * worklet scope, so these are declared by hand rather than pulling in a
 * dependency for five symbols.
 */

declare const sampleRate: number;
declare const currentTime: number;
declare const currentFrame: number;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: unknown);
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: never) => AudioWorkletProcessor,
): void;
