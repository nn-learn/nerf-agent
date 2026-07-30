import {
  type AudioStartMessage,
  parseServerMessage,
  type ServerMessage,
} from "./protocol";

export interface RealtimeClientHandlers {
  onMessage?: (message: ServerMessage) => void;
  onAudio?: (
    frame: ArrayBuffer,
    stream: AudioStartMessage,
  ) => void;
  onError?: (error: Error) => void;
  onClose?: (event: CloseEvent) => void;
}

export class LocalRealtimeClient {
  private socket?: WebSocket;
  private connectionPromise?: Promise<void>;
  private resolveConnection?: () => void;
  private rejectConnection?: (error: Error) => void;
  private readonly handlers = new Set<RealtimeClientHandlers>();
  private readonly retiredTurnIds = new Set<string>();
  private activeStream?: AudioStartMessage;
  private ready = false;
  private captureActive = false;
  private closed = false;

  constructor(
    private readonly url: string,
    private readonly accessToken: string,
    private readonly socketFactory: (url: string) => WebSocket = (
      url,
    ) => new WebSocket(url),
  ) {}

  get isReady(): boolean {
    return this.ready;
  }

  subscribe(handlers: RealtimeClientHandlers): () => void {
    this.handlers.add(handlers);
    return () => this.handlers.delete(handlers);
  }

  connect(): Promise<void> {
    if (this.closed) {
      return Promise.reject(new Error("Realtime client is closed"));
    }
    if (this.ready) return Promise.resolve();
    if (this.connectionPromise) return this.connectionPromise;

    this.connectionPromise = new Promise<void>((resolve, reject) => {
      this.resolveConnection = resolve;
      this.rejectConnection = reject;
    });
    try {
      const socket = this.socketFactory(this.url);
      this.socket = socket;
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        if (this.socket !== socket || this.closed) return;
        socket.send(
          JSON.stringify({
            type: "auth",
            access_token: this.accessToken,
          }),
        );
      };
      socket.onmessage = (event) => {
        if (this.socket !== socket || this.closed) return;
        this.handleSocketMessage(event.data);
      };
      socket.onerror = () => {
        if (this.socket !== socket || this.closed) return;
        const error = new Error("Realtime connection failed");
        this.notifyError(error);
        this.rejectPending(error);
        this.socket = undefined;
        this.ready = false;
        this.captureActive = false;
        this.activeStream = undefined;
        this.closed = true;
        this.detachSocket(socket);
        if (socket.readyState < 2) {
          socket.close();
        }
        this.handlers.clear();
      };
      socket.onclose = (event) => {
        if (this.socket !== socket) return;
        this.socket = undefined;
        this.ready = false;
        this.captureActive = false;
        this.activeStream = undefined;
        const wasClosedByClient = this.closed;
        this.closed = true;
        this.detachSocket(socket);
        if (!wasClosedByClient) {
          const error = new Error(
            "Realtime connection closed before ready",
          );
          this.rejectPending(error);
          for (const handler of this.handlers) {
            handler.onClose?.(event);
          }
        }
        this.handlers.clear();
      };
    } catch (error) {
      const connectionError =
        error instanceof Error
          ? error
          : new Error("Realtime connection failed");
      this.rejectPending(connectionError);
    }
    return this.connectionPromise;
  }

  startAudio(): void {
    if (!this.ready || this.captureActive) return;
    this.captureActive = true;
    this.sendJson({
      type: "audio.start",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
    });
  }

  sendPcm(frame: ArrayBuffer): void {
    if (
      !this.ready ||
      !this.captureActive ||
      frame.byteLength !== 640 ||
      !this.isSocketOpen()
    ) {
      return;
    }
    this.socket?.send(frame);
  }

  stopAudio(): void {
    if (!this.ready || !this.captureActive) return;
    this.captureActive = false;
    this.sendJson({ type: "audio.stop" });
  }

  interrupt(reason = "user_pressed_interrupt"): void {
    if (!this.ready) return;
    this.retireActiveStream();
    this.sendJson({ type: "turn.interrupt", reason });
  }

  endSession(): void {
    if (!this.ready) return;
    this.captureActive = false;
    this.retireActiveStream();
    this.sendJson({ type: "session.end" });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.ready = false;
    this.captureActive = false;
    this.activeStream = undefined;
    this.rejectPending(new Error("Realtime client closed"));
    const socket = this.socket;
    this.socket = undefined;
    if (!socket) return;
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    if (socket.readyState < 2) {
      socket.close();
    }
    this.handlers.clear();
  }

  private handleSocketMessage(data: unknown): void {
    if (data instanceof ArrayBuffer) {
      if (this.ready && this.activeStream) {
        for (const handler of this.handlers) {
          handler.onAudio?.(data, this.activeStream);
        }
      }
      return;
    }
    if (typeof data !== "string") {
      this.notifyError(new Error("Invalid realtime binary frame"));
      return;
    }
    let message: ServerMessage;
    try {
      message = parseServerMessage(data);
    } catch {
      this.notifyError(new Error("Invalid realtime message"));
      return;
    }

    if (message.type === "session.ready") {
      this.ready = true;
      this.resolveConnection?.();
      this.clearPendingConnection();
    } else if (message.type === "audio.start") {
      if (this.retiredTurnIds.has(message.turn_id)) return;
      this.activeStream = message;
    } else if (message.type === "audio.end") {
      if (
        this.activeStream?.stream_id === message.stream_id &&
        this.activeStream.turn_id === message.turn_id
      ) {
        this.activeStream = undefined;
      }
    } else if (message.type === "playback.interrupted") {
      if (
        message.turn_id === null ||
        this.activeStream?.turn_id === message.turn_id
      ) {
        this.retireActiveStream();
      }
      if (message.turn_id !== null) {
        this.rememberRetiredTurn(message.turn_id);
      }
    } else if (
      message.type === "turn.completed" &&
      this.activeStream?.turn_id === message.turn_id
    ) {
      this.activeStream = undefined;
    }

    for (const handler of this.handlers) {
      handler.onMessage?.(message);
    }
  }

  private sendJson(payload: Record<string, unknown>): void {
    if (!this.isSocketOpen()) return;
    this.socket?.send(JSON.stringify(payload));
  }

  private isSocketOpen(): boolean {
    return this.socket?.readyState === 1;
  }

  private retireActiveStream(): void {
    if (!this.activeStream) return;
    this.rememberRetiredTurn(this.activeStream.turn_id);
    this.activeStream = undefined;
  }

  private rememberRetiredTurn(turnId: string): void {
    this.retiredTurnIds.add(turnId);
    if (this.retiredTurnIds.size > 128) {
      const oldest = this.retiredTurnIds.values().next().value;
      if (typeof oldest === "string") {
        this.retiredTurnIds.delete(oldest);
      }
    }
  }

  private notifyError(error: Error): void {
    for (const handler of this.handlers) {
      handler.onError?.(error);
    }
  }

  private detachSocket(socket: WebSocket): void {
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
  }

  private rejectPending(error: Error): void {
    this.rejectConnection?.(error);
    this.clearPendingConnection();
  }

  private clearPendingConnection(): void {
    this.resolveConnection = undefined;
    this.rejectConnection = undefined;
  }
}
