import { describe, expect, it, vi } from "vitest";

import { LocalRealtimeClient } from "./LocalRealtimeClient";
import type { ServerMessage } from "./protocol";

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  binaryType: BinaryType = "blob";
  readyState = FakeWebSocket.CONNECTING;
  sent: (string | ArrayBufferLike | Blob | ArrayBufferView)[] = [];
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string) {}

  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close"));
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  receiveJson(payload: unknown): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(payload) }),
    );
  }

  receiveBinary(payload: ArrayBuffer): void {
    this.onmessage?.(new MessageEvent("message", { data: payload }));
  }

  serverClose(code = 1006): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.(new CloseEvent("close", { code }));
  }

  fail(): void {
    this.onerror?.(new Event("error"));
  }
}

function setupClient() {
  const socket = new FakeWebSocket("ws://localhost/realtime");
  const messages: ServerMessage[] = [];
  const audio: { frame: ArrayBuffer; streamId: string; turnId: string }[] = [];
  const client = new LocalRealtimeClient(
    socket.url,
    "a".repeat(32),
    () => socket as unknown as WebSocket,
  );
  client.subscribe({
    onMessage: (message) => messages.push(message),
    onAudio: (frame, stream) =>
      audio.push({
        frame,
        streamId: stream.stream_id,
        turnId: stream.turn_id,
      }),
  });
  return { audio, client, messages, socket };
}

describe("LocalRealtimeClient", () => {
  it("sends authentication first and resolves only after session readiness", async () => {
    const { client, socket } = setupClient();
    const connected = client.connect();
    const settled = vi.fn();
    void connected.then(settled);

    expect(socket.binaryType).toBe("arraybuffer");
    socket.open();
    expect(socket.sent).toEqual([
      JSON.stringify({ type: "auth", access_token: "a".repeat(32) }),
    ]);
    await Promise.resolve();
    expect(settled).not.toHaveBeenCalled();

    socket.receiveJson({ type: "session.ready", session_id: "session_1" });
    await expect(connected).resolves.toBeUndefined();
  });

  it("never sends PCM before session readiness or audio start", async () => {
    const { client, socket } = setupClient();
    const connected = client.connect();
    socket.open();
    const frame = new ArrayBuffer(640);

    client.sendPcm(frame);
    socket.receiveJson({ type: "session.ready", session_id: "session_1" });
    await connected;
    client.sendPcm(frame);
    client.startAudio();
    client.sendPcm(frame);

    expect(socket.sent).toEqual([
      JSON.stringify({ type: "auth", access_token: "a".repeat(32) }),
      JSON.stringify({
        type: "audio.start",
        sample_rate: 16000,
        channels: 1,
        encoding: "pcm_s16le",
      }),
      frame,
    ]);
  });

  it("routes binary only for the active stream and drops interrupted stale audio", async () => {
    const { audio, client, socket } = setupClient();
    const connected = client.connect();
    socket.open();
    socket.receiveJson({ type: "session.ready", session_id: "session_1" });
    await connected;
    const first = new ArrayBuffer(640);
    const stale = new ArrayBuffer(640);

    socket.receiveBinary(stale);
    socket.receiveJson({
      type: "audio.start",
      stream_id: "audio_turn_1",
      turn_id: "turn_1",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
    });
    socket.receiveBinary(first);
    socket.receiveJson({
      type: "playback.interrupted",
      turn_id: "turn_1",
      reason: "user_barge_in",
    });
    socket.receiveBinary(stale);
    socket.receiveJson({
      type: "audio.start",
      stream_id: "audio_turn_1",
      turn_id: "turn_1",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
    });
    socket.receiveBinary(stale);

    expect(audio).toEqual([
      {
        frame: first,
        streamId: "audio_turn_1",
        turnId: "turn_1",
      },
    ]);
  });

  it("ignores stale audio end while a newer stream is active", async () => {
    const { audio, client, socket } = setupClient();
    const connected = client.connect();
    socket.open();
    socket.receiveJson({ type: "session.ready", session_id: "session_1" });
    await connected;

    socket.receiveJson({
      type: "audio.start",
      stream_id: "audio_turn_2",
      turn_id: "turn_2",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
    });
    socket.receiveJson({
      type: "audio.end",
      stream_id: "audio_turn_1",
      turn_id: "turn_1",
    });
    const frame = new ArrayBuffer(640);
    socket.receiveBinary(frame);

    expect(audio).toHaveLength(1);
    expect(audio[0].frame).toBe(frame);
  });

  it("rejects a pending connection when the socket closes", async () => {
    const { client, socket } = setupClient();
    const connected = client.connect();
    socket.open();

    socket.serverClose();

    await expect(connected).rejects.toThrow(
      "Realtime connection closed before ready",
    );
    expect(socket.onopen).toBeNull();
    expect(socket.onmessage).toBeNull();
    expect(socket.onerror).toBeNull();
    expect(socket.onclose).toBeNull();
  });

  it("rejects and releases the socket when transport setup errors", async () => {
    const { client, socket } = setupClient();
    const connected = client.connect();

    socket.fail();

    await expect(connected).rejects.toThrow("Realtime connection failed");
    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
    expect(socket.onopen).toBeNull();
    expect(socket.onmessage).toBeNull();
    expect(socket.onerror).toBeNull();
    expect(socket.onclose).toBeNull();
  });

  it("cleans handlers and makes close idempotent", () => {
    const { client, socket } = setupClient();
    void client.connect().catch(() => undefined);
    socket.open();

    client.close();
    client.close();

    expect(socket.readyState).toBe(FakeWebSocket.CLOSED);
    expect(socket.onopen).toBeNull();
    expect(socket.onmessage).toBeNull();
    expect(socket.onerror).toBeNull();
    expect(socket.onclose).toBeNull();
  });
});
