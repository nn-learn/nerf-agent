import { decodePcm16 } from "./pcm";

export interface PlaybackCallbacks {
  onPlaybackStarted: () => void;
  onPlaybackIdle: () => void;
}

export class PcmPlaybackQueue {
  private context?: AudioContext;
  private nextStartTime = 0;
  private generation = 0;
  private readonly activeSources = new Set<AudioBufferSourceNode>();

  constructor(
    private readonly callbacks: PlaybackCallbacks,
    private readonly contextFactory: () => AudioContext = () =>
      new AudioContext(),
  ) {}

  enqueue(bytes: ArrayBuffer): void {
    const samples = decodePcm16(bytes);
    if (samples.length === 0) return;
    const context = this.context ?? this.contextFactory();
    this.context = context;
    const buffer = context.createBuffer(1, samples.length, 16_000);
    buffer.copyToChannel(samples, 0);

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startsPlayback = this.activeSources.size === 0;
    const generation = this.generation;
    const startTime = Math.max(
      context.currentTime,
      this.nextStartTime,
    );
    this.nextStartTime = startTime + buffer.duration;
    this.activeSources.add(source);
    source.onended = () => {
      if (generation !== this.generation) return;
      this.activeSources.delete(source);
      source.disconnect();
      if (this.activeSources.size === 0) {
        this.nextStartTime = context.currentTime;
        this.callbacks.onPlaybackIdle();
      }
    };
    source.start(startTime);
    if (startsPlayback) {
      this.callbacks.onPlaybackStarted();
    }
  }

  cancel(): void {
    this.generation += 1;
    const hadPlayback = this.activeSources.size > 0;
    for (const source of this.activeSources) {
      try {
        source.stop();
      } catch {
        // A source may have ended immediately before cancellation.
      }
      source.disconnect();
    }
    this.activeSources.clear();
    this.nextStartTime = this.context?.currentTime ?? 0;
    if (hadPlayback) {
      this.callbacks.onPlaybackIdle();
    }
  }

  async close(): Promise<void> {
    this.cancel();
    const context = this.context;
    this.context = undefined;
    this.nextStartTime = 0;
    if (context && context.state !== "closed") {
      await context.close();
    }
  }
}
