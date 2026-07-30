import { createElement, StrictMode, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type {
  SessionCreated,
  SessionDescriptor,
  TurnResult,
} from "../api/client";
import type { PlaybackCallbacks } from "../realtime/audioPlayback";
import type {
  RealtimeClientHandlers,
} from "../realtime/LocalRealtimeClient";
import type {
  AudioStartMessage,
  ServerMessage,
} from "../realtime/protocol";
import { useSessionStore } from "../state/sessionStore";
import {
  type LocalRealtimeMediaDependencies,
  useLocalRealtimeMediaSession,
} from "./useLocalRealtimeMediaSession";

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

const SESSION: SessionCreated = {
  session_id: "session_local_1",
  status: "active",
  provider_mode: "local",
  camera_consent: false,
  fallback_capabilities: ["voice_text", "typed_text", "http_text"],
  access_token: "pst_private_token_with_more_than_32_characters",
};

const ENDED_SESSION: SessionDescriptor = {
  ...SESSION,
  status: "ended",
};

const STREAM: AudioStartMessage = {
  type: "audio.start",
  stream_id: "audio_turn_1",
  turn_id: "turn_1",
  sample_rate: 16000,
  channels: 1,
  encoding: "pcm_s16le",
};

class FakeRealtimeClient {
  isReady = false;
  private handlers?: RealtimeClientHandlers;
  private readonly connection = deferred<void>();

  constructor(private readonly order: string[]) {}

  subscribe = vi.fn((handlers: RealtimeClientHandlers) => {
    this.handlers = handlers;
    return () => {
      if (this.handlers === handlers) this.handlers = undefined;
    };
  });

  connect = vi.fn(() => {
    this.order.push("socket.connect");
    return this.connection.promise;
  });

  startAudio = vi.fn(() => this.order.push("socket.audio.start"));
  sendPcm = vi.fn((_frame: ArrayBuffer) => this.order.push("socket.pcm"));
  stopAudio = vi.fn(() => this.order.push("socket.audio.stop"));
  interrupt = vi.fn(() => this.order.push("socket.interrupt"));
  endSession = vi.fn(() => this.order.push("socket.session.end"));
  close = vi.fn(() => {
    this.order.push("socket.close");
    this.isReady = false;
  });

  ready(): void {
    this.isReady = true;
    this.handlers?.onMessage?.({
      type: "session.ready",
      session_id: SESSION.session_id,
    });
    this.connection.resolve();
  }

  emitMessage(message: ServerMessage): void {
    this.handlers?.onMessage?.(message);
  }

  emitAudio(frame: ArrayBuffer): void {
    this.handlers?.onAudio?.(frame, STREAM);
  }

  emitError(error = new Error("socket failed")): void {
    this.handlers?.onError?.(error);
  }
}

class FakeCapture {
  private frameHandler?: (frame: ArrayBuffer) => void;

  constructor(private readonly order: string[]) {}

  start = vi.fn(async (onFrame: (frame: ArrayBuffer) => void) => {
    this.order.push("microphone.start");
    this.frameHandler = onFrame;
  });

  stop = vi.fn(async () => {
    this.order.push("microphone.stop");
    this.frameHandler = undefined;
  });

  emit(frame: ArrayBuffer): void {
    this.frameHandler?.(frame);
  }
}

class FakePlayback {
  private active = false;

  constructor(
    private readonly callbacks: PlaybackCallbacks,
    private readonly order: string[],
  ) {}

  enqueue = vi.fn((_frame: ArrayBuffer) => {
    this.order.push("playback.enqueue");
    if (!this.active) {
      this.active = true;
      this.callbacks.onPlaybackStarted();
    }
  });

  cancel = vi.fn(() => {
    this.order.push("playback.cancel");
    const wasActive = this.active;
    this.active = false;
    if (wasActive) this.callbacks.onPlaybackIdle();
  });

  close = vi.fn(async () => {
    this.order.push("playback.close");
    this.active = false;
  });

  finish(): void {
    if (!this.active) return;
    this.active = false;
    this.callbacks.onPlaybackIdle();
  }
}

function makeHarness() {
  const order: string[] = [];
  const client = new FakeRealtimeClient(order);
  const capture = new FakeCapture(order);
  let playback: FakePlayback | undefined;
  const createSession = vi.fn(async () => {
    order.push("api.createSession");
    return SESSION;
  });
  const createTextTurn = vi.fn(async (): Promise<TurnResult> => ({
    status: "completed",
    risk_level: "GREEN",
    response: {
      spoken_text: "我们可以先从最困扰你的部分说起。",
      display_text: "我们可以先从最困扰你的部分说起。",
      support_mode: "listen",
    },
    delivery_mode: "text",
  }));
  const endSession = vi.fn(async () => {
    order.push("api.endSession");
    return ENDED_SESSION;
  });
  const api = {
    createSession,
    createRealtimeClient: vi.fn(() => client),
    createTextTurn,
    endSession,
  };
  const dependencies: LocalRealtimeMediaDependencies = {
    api,
    createCapture: () => capture,
    createPlayback: (callbacks) => {
      playback = new FakePlayback(callbacks, order);
      return playback;
    },
  };
  return {
    api,
    capture,
    client,
    createSession,
    createTextTurn,
    dependencies,
    endSession,
    get playback() {
      if (!playback) throw new Error("playback has not been created");
      return playback;
    },
    order,
  };
}

async function renderConnected(harness = makeHarness()) {
  const hook = renderHook(() =>
    useLocalRealtimeMediaSession(SESSION.session_id, harness.dependencies),
  );
  await waitFor(() => expect(harness.client.connect).toHaveBeenCalledOnce());
  act(() => harness.client.ready());
  await waitFor(() =>
    expect(hook.result.current.connectionState).toBe("connected"),
  );
  return { harness, ...hook };
}

beforeEach(() => {
  useSessionStore.setState({
    visionState: "off",
    agentState: "connecting",
  });
});

describe("useLocalRealtimeMediaSession", () => {
  it("creates the audited session before connecting and becomes ready only on session.ready", async () => {
    const harness = makeHarness();
    const { result } = renderHook(() =>
      useLocalRealtimeMediaSession(
        SESSION.session_id,
        harness.dependencies,
      ),
    );

    await waitFor(() => expect(harness.client.connect).toHaveBeenCalledOnce());
    expect(harness.order.slice(0, 2)).toEqual([
      "api.createSession",
      "socket.connect",
    ]);
    expect(result.current.connectionState).toBe("connecting");

    act(() => harness.client.ready());

    await waitFor(() =>
      expect(result.current.connectionState).toBe("connected"),
    );
    expect(result.current.agentState).toBe("listening");
    expect(result.current.providerLabel).toBe("本地 Qwen3.6");
  });

  it("maps captions and waits for ended playback before listening again", async () => {
    const { harness, result } = await renderConnected();

    act(() => {
      harness.client.emitMessage({
        type: "transcript.partial",
        kind: "PARTIAL",
        text: "最近工作",
        start_ms: 0,
        end_ms: 400,
        turn_id: "turn_1",
      });
      harness.client.emitMessage({
        type: "transcript.final",
        text: "最近工作压力很大",
        input_mode: "voice",
        turn_id: "turn_1",
      });
      harness.client.emitMessage({
        type: "assistant.response",
        display_text: "听起来你已经承受了一段时间。",
        turn_id: "turn_1",
      });
    });

    expect(result.current.userCaption).toBe("最近工作压力很大");
    expect(result.current.assistantCaption).toBe(
      "听起来你已经承受了一段时间。",
    );
    expect(result.current.agentState).toBe("thinking");

    act(() => harness.client.emitAudio(new ArrayBuffer(640)));
    expect(result.current.agentState).toBe("speaking");

    act(() => {
      harness.client.emitMessage({
        type: "audio.end",
        stream_id: STREAM.stream_id,
        turn_id: STREAM.turn_id,
      });
    });
    expect(result.current.agentState).toBe("speaking");

    act(() => harness.playback.finish());
    expect(result.current.agentState).toBe("listening");
  });

  it("shows a Chinese actionable fallback for realtime protocol errors", async () => {
    const { harness, result } = await renderConnected();

    act(() => {
      harness.client.emitMessage({
        type: "error",
        code: "PROVIDER_ERROR",
      });
    });

    expect(result.current.connectionState).toBe("error");
    expect(result.current.errorMessage).toMatch(/重试/);
    expect(result.current.errorMessage).toMatch(/文字输入/);
  });

  it("leaves no stale speaking state after a realtime transport failure", async () => {
    const { harness, result } = await renderConnected();
    act(() => harness.client.emitAudio(new ArrayBuffer(640)));
    expect(result.current.agentState).toBe("speaking");

    act(() => harness.client.emitError());

    expect(result.current.connectionState).toBe("error");
    expect(result.current.agentState).toBe("disconnected");
  });

  it("cancels playback before sending a barge-in interrupt", async () => {
    const { harness, result } = await renderConnected();
    act(() => harness.client.emitAudio(new ArrayBuffer(640)));
    harness.order.length = 0;

    await act(async () => result.current.interrupt());

    expect(harness.order.slice(0, 2)).toEqual([
      "playback.cancel",
      "socket.interrupt",
    ]);
    expect(result.current.agentState).toBe("listening");
  });

  it("starts and stops capture around the audio control messages", async () => {
    const { harness, result } = await renderConnected();

    await act(async () => result.current.toggleMicrophone());
    expect(result.current.microphoneEnabled).toBe(true);
    expect(harness.order).toContain("socket.audio.start");
    expect(harness.order).toContain("microphone.start");

    const frame = new ArrayBuffer(640);
    act(() => harness.capture.emit(frame));
    expect(harness.client.sendPcm).toHaveBeenCalledWith(frame);

    await act(async () => result.current.toggleMicrophone());
    expect(result.current.microphoneEnabled).toBe(false);
    expect(harness.order.indexOf("microphone.stop")).toBeLessThan(
      harness.order.indexOf("socket.audio.stop"),
    );
  });

  it("hangs up all local resources once and ends the HTTP session", async () => {
    const { harness, result } = await renderConnected();
    await act(async () => result.current.toggleMicrophone());
    harness.order.length = 0;

    await act(async () => result.current.hangUp());
    await act(async () => result.current.hangUp());

    expect(harness.capture.stop).toHaveBeenCalledOnce();
    expect(harness.playback.cancel).toHaveBeenCalledOnce();
    expect(harness.client.close).toHaveBeenCalledOnce();
    expect(harness.endSession).toHaveBeenCalledOnce();
    expect(harness.order.indexOf("microphone.stop")).toBeLessThan(
      harness.order.indexOf("socket.close"),
    );
    expect(harness.order.indexOf("socket.close")).toBeLessThan(
      harness.order.indexOf("api.endSession"),
    );
    expect(result.current.connectionState).toBe("disconnected");
  });

  it("sends nonblank text as an audited text-only fallback", async () => {
    const harness = makeHarness();
    const response = deferred<TurnResult>();
    harness.createTextTurn.mockReturnValueOnce(response.promise);
    const { result } = await renderConnected(harness);

    let sending!: Promise<void>;
    act(() => {
      sending = result.current.sendText("  我最近总是睡不好  ");
    });
    expect(result.current.agentState).toBe("thinking");
    expect(result.current.userCaption).toBe("我最近总是睡不好");

    response.resolve({
      status: "completed",
      risk_level: "GREEN",
      response: {
        spoken_text: "我们可以先看看睡前最难放松的时刻。",
        display_text: "我们可以先看看睡前最难放松的时刻。",
        support_mode: "listen",
      },
      delivery_mode: "text",
    });
    await act(async () => sending);

    expect(harness.createTextTurn).toHaveBeenCalledWith(
      SESSION.session_id,
      "我最近总是睡不好",
    );
    expect(result.current.assistantCaption).toBe(
      "我们可以先看看睡前最难放松的时刻。",
    );
    expect(result.current.agentState).toBe("listening");

    await act(async () => result.current.sendText("   "));
    expect(harness.createTextTurn).toHaveBeenCalledOnce();
  });

  it("shares session startup through StrictMode replay and cleans up once", async () => {
    const harness = makeHarness();
    const creation = deferred<SessionCreated>();
    harness.createSession.mockReturnValueOnce(creation.promise);
    const wrapper = ({ children }: { children: ReactNode }) => (
      createElement(StrictMode, null, children)
    );
    const { unmount } = renderHook(
      () =>
        useLocalRealtimeMediaSession(
          SESSION.session_id,
          harness.dependencies,
        ),
      { wrapper },
    );

    await waitFor(() => expect(harness.createSession).toHaveBeenCalledOnce());
    act(() => creation.resolve(SESSION));
    await waitFor(() => expect(harness.client.connect).toHaveBeenCalledOnce());
    act(() => harness.client.ready());

    unmount();

    await waitFor(() => expect(harness.endSession).toHaveBeenCalledOnce());
    expect(harness.client.close).toHaveBeenCalledOnce();
  });
});
