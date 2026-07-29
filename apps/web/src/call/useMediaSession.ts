import { useCallback, useEffect, useRef, useState } from "react";
import type { LocalVideoTrack, RemoteVideoTrack } from "livekit-client";

import {
  type AgentState,
  type VisionState,
  useSessionStore,
} from "../state/sessionStore";

export type MediaConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "error";

export interface MediaSessionController {
  connectionState: MediaConnectionState;
  microphoneEnabled: boolean;
  visionState: VisionState;
  agentState?: AgentState;
  avatarTrack?: RemoteVideoTrack;
  localCameraTrack?: LocalVideoTrack;
  localPreviewStream?: MediaStream;
  errorMessage?: string;
  publishCamera: () => Promise<void>;
  pauseVision: () => Promise<void>;
  toggleMicrophone: () => Promise<void>;
  interrupt: () => Promise<void>;
  hangUp: () => Promise<void>;
}

interface PauseVisionTransaction {
  stopCamera: () => Promise<void>;
  markPaused: () => void;
  revokeConsent: () => Promise<void>;
}

export async function pauseVisionTransaction({
  stopCamera,
  markPaused,
  revokeConsent,
}: PauseVisionTransaction): Promise<void> {
  await stopCamera();
  markPaused();
  await revokeConsent();
}


export function useMockMediaSession(): MediaSessionController {
  const visionState = useSessionStore((state) => state.visionState);
  const setVisionState = useSessionStore((state) => state.setVisionState);
  const agentState = useSessionStore((state) => state.agentState);
  const setAgentState = useSessionStore((state) => state.setAgentState);
  const [microphoneEnabled, setMicrophoneEnabled] = useState(true);
  const [localPreviewStream, setLocalPreviewStream] = useState<MediaStream>();
  const streamRef = useRef<MediaStream | undefined>(undefined);

  useEffect(() => {
    setAgentState("listening");
    return () => streamRef.current?.getTracks().forEach((track) => track.stop());
  }, [setAgentState]);

  const publishCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setVisionState("disabled");
      throw new Error("当前浏览器不支持摄像头");
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 360 } },
        audio: false,
      });
      streamRef.current = stream;
      setLocalPreviewStream(stream);
      setVisionState("active");
    } catch (error) {
      setVisionState("disabled");
      throw error;
    }
  }, [setVisionState]);

  const pauseVision = useCallback(async () => {
    await pauseVisionTransaction({
      stopCamera: async () => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = undefined;
      },
      markPaused: () => {
        setLocalPreviewStream(undefined);
        setVisionState("paused");
      },
      revokeConsent: async () => undefined,
    });
  }, [setVisionState]);

  const interrupt = useCallback(async () => {
    setAgentState("listening");
  }, [setAgentState]);

  const hangUp = useCallback(async () => {
    await pauseVision();
    setAgentState("disconnected");
  }, [pauseVision, setAgentState]);

  return {
    connectionState: agentState === "disconnected" ? "disconnected" : "connected",
    microphoneEnabled,
    visionState,
    agentState,
    localPreviewStream,
    publishCamera,
    pauseVision,
    toggleMicrophone: async () => setMicrophoneEnabled((enabled) => !enabled),
    interrupt,
    hangUp,
  };
}
