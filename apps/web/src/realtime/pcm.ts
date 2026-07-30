function encodePcm16(samples: readonly number[]): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  samples.forEach((rawSample, index) => {
    const sample = Math.max(-1, Math.min(1, rawSample));
    const pcm =
      sample < 0
        ? Math.round(sample * 0x8000)
        : Math.round(sample * 0x7fff);
    view.setInt16(index * 2, pcm, true);
  });
  return buffer;
}

export class Pcm16Resampler {
  private readonly sourceStep: number;
  private sourcePosition = 0;
  private pendingSource = new Float32Array();
  private pendingOutput: number[] = [];

  constructor(
    sourceSampleRate: number,
    targetSampleRate = 16_000,
    private readonly frameSamples = 320,
  ) {
    if (
      !Number.isFinite(sourceSampleRate) ||
      !Number.isFinite(targetSampleRate) ||
      sourceSampleRate <= 0 ||
      targetSampleRate <= 0 ||
      !Number.isInteger(frameSamples) ||
      frameSamples <= 0
    ) {
      throw new Error("Invalid PCM resampler configuration");
    }
    this.sourceStep = sourceSampleRate / targetSampleRate;
  }

  push(input: Float32Array): ArrayBuffer[] {
    if (input.length === 0) return [];
    const source = new Float32Array(
      this.pendingSource.length + input.length,
    );
    source.set(this.pendingSource);
    source.set(input, this.pendingSource.length);

    while (this.canInterpolate(source)) {
      const leftIndex = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - leftIndex;
      const left = source[leftIndex];
      const right =
        fraction === 0 ? left : source[leftIndex + 1];
      this.pendingOutput.push(left + (right - left) * fraction);
      this.sourcePosition += this.sourceStep;
    }

    const consumed = Math.min(
      Math.floor(this.sourcePosition),
      source.length,
    );
    this.pendingSource = source.slice(consumed);
    this.sourcePosition -= consumed;

    const frames: ArrayBuffer[] = [];
    while (this.pendingOutput.length >= this.frameSamples) {
      frames.push(
        encodePcm16(this.pendingOutput.splice(0, this.frameSamples)),
      );
    }
    return frames;
  }

  reset(): void {
    this.sourcePosition = 0;
    this.pendingSource = new Float32Array();
    this.pendingOutput = [];
  }

  private canInterpolate(source: Float32Array): boolean {
    const leftIndex = Math.floor(this.sourcePosition);
    if (leftIndex >= source.length) return false;
    const fraction = this.sourcePosition - leftIndex;
    return fraction === 0 || leftIndex + 1 < source.length;
  }
}

export function decodePcm16(
  bytes: ArrayBuffer,
): Float32Array<ArrayBuffer> {
  if (bytes.byteLength % 2 !== 0) {
    throw new Error("PCM16 byte length must be even");
  }
  const view = new DataView(bytes);
  const samples = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    const pcm = view.getInt16(index * 2, true);
    samples[index] = pcm < 0 ? pcm / 0x8000 : pcm / 0x7fff;
  }
  return samples;
}
