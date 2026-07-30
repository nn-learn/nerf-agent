import { describe, expect, it } from "vitest";

import { decodePcm16, Pcm16Resampler } from "./pcm";

function decodeFrames(frames: ArrayBuffer[]): number[] {
  return frames.flatMap((frame) => Array.from(new Int16Array(frame)));
}

describe("Pcm16Resampler", () => {
  it("resamples 48 kHz input into exact 20 ms PCM frames", () => {
    const resampler = new Pcm16Resampler(48_000, 16_000, 320);
    const frames = resampler.push(new Float32Array(960).fill(0.5));

    expect(frames).toHaveLength(1);
    expect(frames[0].byteLength).toBe(640);
    expect(new Int16Array(frames[0])[0]).toBe(16_384);
  });

  it("clips out-of-range samples before signed PCM conversion", () => {
    const resampler = new Pcm16Resampler(16_000, 16_000, 2);
    const [frame] = resampler.push(new Float32Array([-2, 2]));

    expect(Array.from(new Int16Array(frame))).toEqual([-32_768, 32_767]);
  });

  it("retains uneven source batches across pushes", () => {
    const resampler = new Pcm16Resampler(48_000, 16_000, 320);

    expect(resampler.push(new Float32Array(257).fill(0.25))).toEqual([]);
    expect(resampler.push(new Float32Array(703).fill(0.25))).toHaveLength(1);
  });

  it("resamples 44.1 kHz chunks without losing order across boundaries", () => {
    const resampler = new Pcm16Resampler(44_100, 16_000, 160);
    const ramp = Float32Array.from(
      { length: 882 },
      (_, index) => -0.9 + (index / 881) * 1.8,
    );

    const first = resampler.push(ramp.slice(0, 317));
    const rest = resampler.push(ramp.slice(317));
    const samples = decodeFrames([...first, ...rest]);

    expect(samples).toHaveLength(320);
    for (let index = 1; index < samples.length; index += 1) {
      expect(samples[index]).toBeGreaterThanOrEqual(samples[index - 1]);
    }
  });

  it("clears retained samples on reset", () => {
    const resampler = new Pcm16Resampler(48_000, 16_000, 320);
    resampler.push(new Float32Array(480).fill(0.75));

    resampler.reset();

    expect(resampler.push(new Float32Array(480).fill(0.75))).toEqual([]);
  });
});

describe("decodePcm16", () => {
  it("decodes little-endian signed PCM into normalized floats", () => {
    const bytes = new ArrayBuffer(6);
    const view = new DataView(bytes);
    view.setInt16(0, -32_768, true);
    view.setInt16(2, 0, true);
    view.setInt16(4, 32_767, true);

    expect(Array.from(decodePcm16(bytes))).toEqual([-1, 0, 1]);
  });

  it("rejects odd byte lengths", () => {
    expect(() => decodePcm16(new ArrayBuffer(3))).toThrow(
      "PCM16 byte length must be even",
    );
  });
});
