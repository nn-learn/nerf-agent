import { describe, expect, it, vi } from "vitest";

import { MicrophoneCapture } from "./audioCapture";

class FakeNode {
  connections: unknown[] = [];
  disconnected = false;

  connect(node: unknown): unknown {
    this.connections.push(node);
    return node;
  }

  disconnect(): void {
    this.disconnected = true;
  }
}

class FakeWorkletNode extends FakeNode {
  port = { onmessage: null as ((event: MessageEvent<Float32Array>) => void) | null };
}

it("captures constrained mono audio, frames it, and releases every resource", async () => {
  const track = { stop: vi.fn() };
  const stream = {
    getTracks: () => [track],
  } as unknown as MediaStream;
  const getUserMedia = vi.fn(async () => stream);
  const source = new FakeNode();
  const worklet = new FakeWorkletNode();
  const gain = Object.assign(new FakeNode(), { gain: { value: 1 } });
  const addModule = vi.fn(async () => undefined);
  const close = vi.fn(async () => undefined);
  const context = {
    sampleRate: 48_000,
    audioWorklet: { addModule },
    destination: new FakeNode(),
    createMediaStreamSource: vi.fn(() => source),
    createGain: vi.fn(() => gain),
    close,
  } as unknown as AudioContext;
  const capture = new MicrophoneCapture({
    contextFactory: () => context,
    getUserMedia,
    workletFactory: () => worklet as unknown as AudioWorkletNode,
  });
  const onFrame = vi.fn();

  await capture.start(onFrame);
  expect(getUserMedia).toHaveBeenCalledWith({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
  expect(addModule).toHaveBeenCalledWith("/pcm-capture-processor.js");
  expect(gain.gain.value).toBe(0);
  worklet.port.onmessage?.(
    new MessageEvent("message", {
      data: new Float32Array(960).fill(0.5),
    }),
  );
  expect(onFrame).toHaveBeenCalledTimes(1);
  expect(onFrame.mock.calls[0][0]).toBeInstanceOf(ArrayBuffer);
  expect((onFrame.mock.calls[0][0] as ArrayBuffer).byteLength).toBe(640);

  await capture.stop();
  expect(source.disconnected).toBe(true);
  expect(worklet.disconnected).toBe(true);
  expect(gain.disconnected).toBe(true);
  expect(track.stop).toHaveBeenCalledOnce();
  expect(close).toHaveBeenCalledOnce();
});

describe("MicrophoneCapture startup cleanup", () => {
  it("shares one acquisition across concurrent start calls", async () => {
    let resolveStream!: (stream: MediaStream) => void;
    const streamPromise = new Promise<MediaStream>((resolve) => {
      resolveStream = resolve;
    });
    const stop = vi.fn();
    const getUserMedia = vi.fn(() => streamPromise);
    const source = new FakeNode();
    const worklet = new FakeWorkletNode();
    const gain = Object.assign(new FakeNode(), { gain: { value: 1 } });
    const context = {
      sampleRate: 48_000,
      audioWorklet: { addModule: vi.fn(async () => undefined) },
      destination: new FakeNode(),
      createMediaStreamSource: vi.fn(() => source),
      createGain: vi.fn(() => gain),
      close: vi.fn(async () => undefined),
    } as unknown as AudioContext;
    const capture = new MicrophoneCapture({
      contextFactory: () => context,
      getUserMedia,
      workletFactory: () => worklet as unknown as AudioWorkletNode,
    });

    const first = capture.start(vi.fn());
    const second = capture.start(vi.fn());
    expect(getUserMedia).toHaveBeenCalledTimes(1);
    resolveStream({
      getTracks: () => [{ stop }],
    } as unknown as MediaStream);

    await Promise.all([first, second]);
    await capture.stop();
    expect(stop).toHaveBeenCalledOnce();
    await capture.stop();
    expect(stop).toHaveBeenCalledOnce();
  });

  it("stops acquired media if worklet setup fails", async () => {
    const stop = vi.fn();
    const context = {
      sampleRate: 48_000,
      audioWorklet: {
        addModule: vi.fn(async () => {
          throw new Error("module failed");
        }),
      },
      close: vi.fn(async () => undefined),
    } as unknown as AudioContext;
    const capture = new MicrophoneCapture({
      contextFactory: () => context,
      getUserMedia: async () =>
        ({
          getTracks: () => [{ stop }],
        }) as unknown as MediaStream,
    });

    await expect(capture.start(vi.fn())).rejects.toThrow("module failed");
    expect(stop).toHaveBeenCalledOnce();
    expect(context.close).toHaveBeenCalledOnce();
  });
});
