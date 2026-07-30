import { expect, it, vi } from "vitest";

import { PcmPlaybackQueue } from "./audioPlayback";

class FakeSource {
  buffer: AudioBuffer | null = null;
  onended: (() => void) | null = null;
  start = vi.fn();
  stop = vi.fn();
  connect = vi.fn();
  disconnect = vi.fn();
}

it("schedules PCM in order and reports playback lifecycle", () => {
  const sources: FakeSource[] = [];
  const context = {
    currentTime: 4,
    destination: {},
    createBuffer: vi.fn(() => ({
      duration: 0.02,
      copyToChannel: vi.fn(),
    })),
    createBufferSource: vi.fn(() => {
      const source = new FakeSource();
      sources.push(source);
      return source;
    }),
    close: vi.fn(async () => undefined),
  } as unknown as AudioContext;
  const onPlaybackStarted = vi.fn();
  const onPlaybackIdle = vi.fn();
  const queue = new PcmPlaybackQueue(
    { onPlaybackStarted, onPlaybackIdle },
    () => context,
  );

  queue.enqueue(new ArrayBuffer(640));
  queue.enqueue(new ArrayBuffer(640));

  expect(sources[0].start).toHaveBeenCalledWith(4);
  expect(sources[1].start).toHaveBeenCalledWith(4.02);
  expect(onPlaybackStarted).toHaveBeenCalledOnce();
  sources[0].onended?.();
  expect(sources[0].disconnect).toHaveBeenCalledOnce();
  expect(onPlaybackIdle).not.toHaveBeenCalled();
  sources[1].onended?.();
  expect(sources[1].disconnect).toHaveBeenCalledOnce();
  expect(onPlaybackIdle).toHaveBeenCalledOnce();
});

it("cancels active sources and isolates stale onended callbacks by generation", () => {
  const sources: FakeSource[] = [];
  const context = {
    currentTime: 7,
    destination: {},
    createBuffer: vi.fn(() => ({
      duration: 0.02,
      copyToChannel: vi.fn(),
    })),
    createBufferSource: vi.fn(() => {
      const source = new FakeSource();
      sources.push(source);
      return source;
    }),
    close: vi.fn(async () => undefined),
  } as unknown as AudioContext;
  const onPlaybackIdle = vi.fn();
  const queue = new PcmPlaybackQueue(
    { onPlaybackStarted: vi.fn(), onPlaybackIdle },
    () => context,
  );

  queue.enqueue(new ArrayBuffer(640));
  const staleOnEnded = sources[0].onended;
  queue.cancel();
  queue.enqueue(new ArrayBuffer(640));
  staleOnEnded?.();

  expect(sources[0].stop).toHaveBeenCalledOnce();
  expect(sources[1].start).toHaveBeenCalledWith(7);
  expect(onPlaybackIdle).toHaveBeenCalledOnce();
  sources[1].onended?.();
  expect(onPlaybackIdle).toHaveBeenCalledTimes(2);
});
