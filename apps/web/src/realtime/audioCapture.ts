import { Pcm16Resampler } from "./pcm";

export interface MicrophoneCaptureDependencies {
  contextFactory: () => AudioContext;
  getUserMedia: (
    constraints: MediaStreamConstraints,
  ) => Promise<MediaStream>;
  workletFactory: (
    context: AudioContext,
    name: string,
  ) => AudioWorkletNode;
}

const defaultDependencies: MicrophoneCaptureDependencies = {
  contextFactory: () => new AudioContext(),
  getUserMedia: (constraints) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      return Promise.reject(
        new Error("Microphone capture is unavailable"),
      );
    }
    return navigator.mediaDevices.getUserMedia(constraints);
  },
  workletFactory: (context, name) =>
    new AudioWorkletNode(context, name, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    }),
};

export class MicrophoneCapture {
  private readonly dependencies: MicrophoneCaptureDependencies;
  private context?: AudioContext;
  private stream?: MediaStream;
  private source?: MediaStreamAudioSourceNode;
  private worklet?: AudioWorkletNode;
  private silentGain?: GainNode;
  private resampler?: Pcm16Resampler;
  private startPromise?: Promise<void>;
  private generation = 0;

  constructor(
    dependencies: Partial<MicrophoneCaptureDependencies> = {},
  ) {
    this.dependencies = { ...defaultDependencies, ...dependencies };
  }

  start(onFrame: (frame: ArrayBuffer) => void): Promise<void> {
    if (this.context) return Promise.resolve();
    if (this.startPromise) return this.startPromise;
    const generation = ++this.generation;
    const startup = this.startOwned(onFrame, generation).finally(() => {
      if (this.startPromise === startup) {
        this.startPromise = undefined;
      }
    });
    this.startPromise = startup;
    return startup;
  }

  async stop(): Promise<void> {
    this.generation += 1;
    const startup = this.startPromise;
    this.startPromise = undefined;
    const resources = {
      context: this.context,
      stream: this.stream,
      source: this.source,
      worklet: this.worklet,
      silentGain: this.silentGain,
      resampler: this.resampler,
    };
    this.context = undefined;
    this.stream = undefined;
    this.source = undefined;
    this.worklet = undefined;
    this.silentGain = undefined;
    this.resampler = undefined;

    if (startup) {
      try {
        await startup;
      } catch {
        // Startup owns and releases any partially acquired resources.
      }
    }
    await this.release(resources);
  }

  private async startOwned(
    onFrame: (frame: ArrayBuffer) => void,
    generation: number,
  ): Promise<void> {
    let stream: MediaStream | undefined;
    let context: AudioContext | undefined;
    let source: MediaStreamAudioSourceNode | undefined;
    let worklet: AudioWorkletNode | undefined;
    let silentGain: GainNode | undefined;
    let resampler: Pcm16Resampler | undefined;
    try {
      stream = await this.dependencies.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      if (generation !== this.generation) {
        await this.release({ stream });
        return;
      }

      context = this.dependencies.contextFactory();
      resampler = new Pcm16Resampler(
        context.sampleRate,
        16_000,
        320,
      );
      await context.audioWorklet.addModule(
        "/pcm-capture-processor.js",
      );
      if (generation !== this.generation) {
        await this.release({ context, stream, resampler });
        return;
      }
      source = context.createMediaStreamSource(stream);
      worklet = this.dependencies.workletFactory(
        context,
        "pcm-capture-processor",
      );
      silentGain = context.createGain();
      silentGain.gain.value = 0;
      worklet.port.onmessage = (
        event: MessageEvent<Float32Array>,
      ) => {
        if (
          generation !== this.generation ||
          !(event.data instanceof Float32Array)
        ) {
          return;
        }
        for (const frame of resampler?.push(event.data) ?? []) {
          onFrame(frame);
        }
      };
      source.connect(worklet);
      worklet.connect(silentGain);
      silentGain.connect(context.destination);
      this.context = context;
      this.stream = stream;
      this.source = source;
      this.worklet = worklet;
      this.silentGain = silentGain;
      this.resampler = resampler;
    } catch (error) {
      await this.release({
        context,
        stream,
        source,
        worklet,
        silentGain,
        resampler,
      });
      throw error;
    }
  }

  private async release(resources: {
    context?: AudioContext;
    stream?: MediaStream;
    source?: MediaStreamAudioSourceNode;
    worklet?: AudioWorkletNode;
    silentGain?: GainNode;
    resampler?: Pcm16Resampler;
  }): Promise<void> {
    if (resources.worklet) {
      resources.worklet.port.onmessage = null;
    }
    resources.source?.disconnect();
    resources.worklet?.disconnect();
    resources.silentGain?.disconnect();
    resources.stream?.getTracks().forEach((track) => track.stop());
    resources.resampler?.reset();
    const context = resources.context;
    if (context && context.state !== "closed") {
      await context.close();
    }
  }
}
