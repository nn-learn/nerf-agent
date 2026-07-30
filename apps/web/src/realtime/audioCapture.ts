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
  private generation = 0;

  constructor(
    dependencies: Partial<MicrophoneCaptureDependencies> = {},
  ) {
    this.dependencies = { ...defaultDependencies, ...dependencies };
  }

  async start(onFrame: (frame: ArrayBuffer) => void): Promise<void> {
    if (this.context) return;
    const generation = ++this.generation;
    try {
      this.stream = await this.dependencies.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      if (generation !== this.generation) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = undefined;
        return;
      }

      this.context = this.dependencies.contextFactory();
      this.resampler = new Pcm16Resampler(
        this.context.sampleRate,
        16_000,
        320,
      );
      await this.context.audioWorklet.addModule(
        "/pcm-capture-processor.js",
      );
      this.source = this.context.createMediaStreamSource(this.stream);
      this.worklet = this.dependencies.workletFactory(
        this.context,
        "pcm-capture-processor",
      );
      this.silentGain = this.context.createGain();
      this.silentGain.gain.value = 0;
      this.worklet.port.onmessage = (
        event: MessageEvent<Float32Array>,
      ) => {
        if (
          generation !== this.generation ||
          !(event.data instanceof Float32Array)
        ) {
          return;
        }
        for (const frame of this.resampler?.push(event.data) ?? []) {
          onFrame(frame);
        }
      };
      this.source.connect(this.worklet);
      this.worklet.connect(this.silentGain);
      this.silentGain.connect(this.context.destination);
    } catch (error) {
      await this.stop();
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.generation += 1;
    if (this.worklet) {
      this.worklet.port.onmessage = null;
    }
    this.source?.disconnect();
    this.worklet?.disconnect();
    this.silentGain?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.resampler?.reset();

    const context = this.context;
    this.context = undefined;
    this.stream = undefined;
    this.source = undefined;
    this.worklet = undefined;
    this.silentGain = undefined;
    this.resampler = undefined;
    if (context && context.state !== "closed") {
      await context.close();
    }
  }
}
