import { useCallback, useEffect, useRef, useState } from "react";

import {
  PsyAvatarApi,
  type SessionCreated,
  type SessionDescriptor,
  type TurnResult,
} from "../api/client";
import {
  PcmPlaybackQueue,
  type PlaybackCallbacks,
} from "../realtime/audioPlayback";
import { MicrophoneCapture } from "../realtime/audioCapture";
import type {
  RealtimeClientHandlers,
} from "../realtime/LocalRealtimeClient";
import type { ServerMessage } from "../realtime/protocol";
import { useSessionStore } from "../state/sessionStore";
import type { MediaSessionController } from "./useMediaSession";

interface RealtimeClientPort {
  readonly isReady: boolean;
  subscribe(handlers: RealtimeClientHandlers): () => void;
  connect(): Promise<void>;
  startAudio(): void;
  sendPcm(frame: ArrayBuffer): void;
  stopAudio(): void;
  interrupt(reason?: string): void;
  endSession(): void;
  close(): void;
}

interface MicrophoneCapturePort {
  start(onFrame: (frame: ArrayBuffer) => void): Promise<void>;
  stop(): Promise<void>;
}

interface PlaybackPort {
  enqueue(frame: ArrayBuffer): void;
  cancel(): void;
  close(): Promise<void>;
}

interface LocalRealtimeApi {
  createSession(sessionId: string): Promise<SessionCreated>;
  createRealtimeClient(sessionId: string): RealtimeClientPort;
  createTextTurn(sessionId: string, text: string): Promise<TurnResult>;
  endSession(sessionId: string): Promise<SessionDescriptor>;
}

export interface LocalRealtimeMediaDependencies {
  api: LocalRealtimeApi;
  createCapture: () => MicrophoneCapturePort;
  createPlayback: (callbacks: PlaybackCallbacks) => PlaybackPort;
}

function defaultDependencies(): LocalRealtimeMediaDependencies {
  const api = new PsyAvatarApi();
  return {
    api,
    createCapture: () => new MicrophoneCapture(),
    createPlayback: (callbacks) => new PcmPlaybackQueue(callbacks),
  };
}

const REALTIME_ERROR_MESSAGE =
  "语音连接出现问题，请检查本地服务后重试，或使用下方文字输入继续。";
const TEXT_ERROR_MESSAGE =
  "文字消息暂时未能发送，请稍后重试。若你正处于紧急危险中，请立即联系当地急救服务或可信赖的人。";

export function useLocalRealtimeMediaSession(
  sessionId: string,
  injectedDependencies?: LocalRealtimeMediaDependencies,
): MediaSessionController {
  const dependenciesRef = useRef<
    LocalRealtimeMediaDependencies | undefined
  >(undefined);
  if (!dependenciesRef.current) {
    dependenciesRef.current =
      injectedDependencies ?? defaultDependencies();
  }
  const dependencies = dependenciesRef.current;
  const visionState = useSessionStore((state) => state.visionState);
  const setVisionState = useSessionStore((state) => state.setVisionState);
  const agentState = useSessionStore((state) => state.agentState);
  const setAgentState = useSessionStore((state) => state.setAgentState);
  const [connectionState, setConnectionState] =
    useState<MediaSessionController["connectionState"]>("connecting");
  const [microphoneEnabled, setMicrophoneEnabled] = useState(false);
  const [userCaption, setUserCaption] = useState<string>();
  const [assistantCaption, setAssistantCaption] = useState<string>();
  const [providerLabel, setProviderLabel] = useState<string>();
  const [errorMessage, setErrorMessage] = useState<string>();

  const clientRef = useRef<RealtimeClientPort | undefined>(undefined);
  const captureRef = useRef<MicrophoneCapturePort | undefined>(undefined);
  const playbackRef = useRef<PlaybackPort | undefined>(undefined);
  const unsubscribeRef = useRef<(() => void) | undefined>(undefined);
  const sessionPromiseRef = useRef<
    Promise<SessionCreated> | undefined
  >(undefined);
  const sessionPromiseIdRef = useRef<string | undefined>(undefined);
  const lifecycleGenerationRef = useRef(0);
  const shutdownPromiseRef = useRef<Promise<void> | undefined>(undefined);
  const microphoneEnabledRef = useRef(false);
  const playbackActiveRef = useRef(false);
  const audioEndedRef = useRef(true);
  const textTurnPendingRef = useRef(false);

  const shutdown = useCallback(
    (updateState: boolean): Promise<void> => {
      if (shutdownPromiseRef.current) {
        return shutdownPromiseRef.current;
      }
      const capture = captureRef.current;
      const playback = playbackRef.current;
      const client = clientRef.current;
      const unsubscribe = unsubscribeRef.current;
      captureRef.current = undefined;
      playbackRef.current = undefined;
      clientRef.current = undefined;
      unsubscribeRef.current = undefined;
      microphoneEnabledRef.current = false;

      const shuttingDown = (async () => {
        try {
          await capture?.stop();
        } catch {
          // Continue releasing the remaining session resources.
        }
        playback?.cancel();
        try {
          await playback?.close();
        } catch {
          // Continue closing the transport and audited session.
        }
        client?.endSession();
        client?.close();
        unsubscribe?.();
        const sessionPromise = sessionPromiseRef.current;
        if (sessionPromise) {
          try {
            await sessionPromise;
            await dependencies.api.endSession(sessionId);
          } catch {
            // A failed creation has no server session left to end.
          }
        }
        if (updateState) {
          setMicrophoneEnabled(false);
          setConnectionState("disconnected");
          setAgentState("disconnected");
        }
      })();
      shutdownPromiseRef.current = shuttingDown;
      return shuttingDown;
    },
    [dependencies.api, sessionId, setAgentState],
  );

  useEffect(() => {
    const generation = ++lifecycleGenerationRef.current;
    let cancelled = false;
    shutdownPromiseRef.current = undefined;
    setConnectionState("connecting");
    setAgentState("connecting");
    setErrorMessage(undefined);

    if (
      !sessionPromiseRef.current ||
      sessionPromiseIdRef.current !== sessionId
    ) {
      sessionPromiseIdRef.current = sessionId;
      sessionPromiseRef.current =
        dependencies.api.createSession(sessionId);
    }
    const sessionPromise = sessionPromiseRef.current;

    void (async () => {
      try {
        const session = await sessionPromise;
        if (
          cancelled ||
          lifecycleGenerationRef.current !== generation
        ) {
          return;
        }
        setProviderLabel(
          session.provider_mode === "local"
            ? "本地 Qwen3.6"
            : session.provider_mode,
        );
        const playback = dependencies.createPlayback({
          onPlaybackStarted: () => {
            if (lifecycleGenerationRef.current !== generation) return;
            playbackActiveRef.current = true;
            setAgentState("speaking");
          },
          onPlaybackIdle: () => {
            if (lifecycleGenerationRef.current !== generation) return;
            playbackActiveRef.current = false;
            if (audioEndedRef.current) {
              setAgentState("listening");
            }
          },
        });
        const capture = dependencies.createCapture();
        const client =
          dependencies.api.createRealtimeClient(sessionId);
        playbackRef.current = playback;
        captureRef.current = capture;
        clientRef.current = client;

        const handleMessage = (message: ServerMessage) => {
          if (lifecycleGenerationRef.current !== generation) return;
          switch (message.type) {
            case "session.ready":
              setConnectionState("connected");
              setAgentState("listening");
              if (message.model) {
                setProviderLabel(`本地 ${message.model}`);
              }
              break;
            case "agent.state":
              if (
                message.state === "listening" ||
                message.state === "thinking" ||
                message.state === "speaking"
              ) {
                setAgentState(message.state);
              }
              break;
            case "transcript.partial":
              setUserCaption(message.text);
              break;
            case "transcript.final":
              setUserCaption(message.text);
              setAgentState("thinking");
              break;
            case "assistant.response":
              setAssistantCaption(message.display_text);
              break;
            case "audio.start":
              audioEndedRef.current = false;
              break;
            case "audio.end":
              audioEndedRef.current = true;
              if (!playbackActiveRef.current) {
                setAgentState("listening");
              }
              break;
            case "turn.completed":
              if (!playbackActiveRef.current) {
                setAgentState("listening");
              }
              break;
            case "playback.interrupted":
              audioEndedRef.current = true;
              playback.cancel();
              setAgentState("listening");
              break;
            case "response.displayed":
              setAssistantCaption(message.text);
              break;
            case "error":
              setConnectionState("error");
              setAgentState("listening");
              setErrorMessage(REALTIME_ERROR_MESSAGE);
              break;
            case "transcript.empty":
            case "pong":
            case "risk.updated":
            case "playback.started":
            case "vision.degraded":
            case "vision.cancelled":
            case "tts.degraded":
            case "avatar.degraded":
              break;
          }
        };
        const handlers: RealtimeClientHandlers = {
          onMessage: handleMessage,
          onAudio: (frame) => {
            if (lifecycleGenerationRef.current !== generation) return;
            playback.enqueue(frame);
          },
          onError: () => {
            if (lifecycleGenerationRef.current !== generation) return;
            setConnectionState("error");
            setAgentState("disconnected");
            setErrorMessage(REALTIME_ERROR_MESSAGE);
          },
          onClose: () => {
            if (lifecycleGenerationRef.current !== generation) return;
            setConnectionState("disconnected");
            setAgentState("disconnected");
            setErrorMessage(REALTIME_ERROR_MESSAGE);
          },
        };
        unsubscribeRef.current = client.subscribe(handlers);
        await client.connect();
        if (
          cancelled ||
          lifecycleGenerationRef.current !== generation
        ) {
          return;
        }
        setConnectionState("connected");
        setAgentState("listening");
      } catch {
        if (
          cancelled ||
          lifecycleGenerationRef.current !== generation
        ) {
          return;
        }
        setConnectionState("error");
        setAgentState("listening");
        setErrorMessage(REALTIME_ERROR_MESSAGE);
      }
    })();

    return () => {
      cancelled = true;
      queueMicrotask(() => {
        if (lifecycleGenerationRef.current === generation) {
          void shutdown(false);
        }
      });
    };
  }, [
    dependencies,
    sessionId,
    setAgentState,
    shutdown,
  ]);

  const publishCamera = useCallback(async () => {
    setVisionState("disabled");
    throw new Error(
      "本地语音模式暂不传输摄像头画面，你仍可继续语音或文字交流。",
    );
  }, [setVisionState]);

  const pauseVision = useCallback(async () => {
    setVisionState("paused");
  }, [setVisionState]);

  const toggleMicrophone = useCallback(async () => {
    const client = clientRef.current;
    const capture = captureRef.current;
    if (!client || !capture || !client.isReady) {
      setErrorMessage(REALTIME_ERROR_MESSAGE);
      return;
    }
    if (microphoneEnabledRef.current) {
      microphoneEnabledRef.current = false;
      setMicrophoneEnabled(false);
      await capture.stop();
      client.stopAudio();
      return;
    }
    setErrorMessage(undefined);
    client.startAudio();
    try {
      await capture.start((frame) => client.sendPcm(frame));
      microphoneEnabledRef.current = true;
      setMicrophoneEnabled(true);
    } catch {
      client.stopAudio();
      setErrorMessage(
        "未能开启麦克风，请检查浏览器权限，或使用下方文字输入继续。",
      );
    }
  }, []);

  const interrupt = useCallback(async () => {
    audioEndedRef.current = true;
    playbackRef.current?.cancel();
    clientRef.current?.interrupt("user_pressed_interrupt");
    setAgentState("listening");
  }, [setAgentState]);

  const sendText = useCallback(
    async (text: string) => {
      const normalized = text.trim();
      if (!normalized || textTurnPendingRef.current) return;
      textTurnPendingRef.current = true;
      setErrorMessage(undefined);
      setUserCaption(normalized);
      setAgentState("thinking");
      try {
        const result = await dependencies.api.createTextTurn(
          sessionId,
          normalized,
        );
        const displayText = result.response?.display_text.trim();
        if (result.status === "completed" && displayText) {
          setAssistantCaption(displayText);
          setAgentState("listening");
        } else if (result.status === "interrupted") {
          setAgentState("listening");
        } else {
          throw new Error("Text turn returned no display response");
        }
      } catch {
        setAgentState("listening");
        setErrorMessage(TEXT_ERROR_MESSAGE);
      } finally {
        textTurnPendingRef.current = false;
      }
    },
    [dependencies.api, sessionId, setAgentState],
  );

  const hangUp = useCallback(
    () => shutdown(true),
    [shutdown],
  );

  return {
    connectionState,
    microphoneEnabled,
    visionState,
    agentState,
    errorMessage,
    userCaption,
    assistantCaption,
    providerLabel,
    publishCamera,
    pauseVision,
    toggleMicrophone,
    interrupt,
    sendText,
    hangUp,
  };
}
