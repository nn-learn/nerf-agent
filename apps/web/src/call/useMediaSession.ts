import { useCallback, useEffect, useRef, useState } from "react";
import type { LocalVideoTrack, RemoteVideoTrack } from "livekit-client";

import type { AvatarPlan } from "../realtime/protocol";

import type {
  MemoryIngestionView,
  MemoryConflictView,
  MemoryChangeView,
  MemoryProfileView,
  MemoryRecallView,
  MemoryResearchConsentView,
  MemoryRetention,
  MemoryShadowReportView,
  MemoryView,
} from "../api/client";

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
  avatarPlan?: AvatarPlan;
  avatarTrack?: RemoteVideoTrack;
  localCameraTrack?: LocalVideoTrack;
  localPreviewStream?: MediaStream;
  errorMessage?: string;
  userCaption?: string;
  assistantCaption?: string;
  providerLabel?: string;
  publishCamera: () => Promise<void>;
  pauseVision: () => Promise<void>;
  toggleMicrophone: () => Promise<void>;
  interrupt: () => Promise<void>;
  sendText: (text: string) => Promise<void>;
  hangUp: () => Promise<void>;
  listMemories?: () => Promise<MemoryView[]>;
  getMemoryStatus?: () => Promise<MemoryIngestionView>;
  retryMemoryIngestion?: () => Promise<MemoryIngestionView>;
  decideMemory?: (
    memoryId: string,
    decision: "confirm" | "reject",
  ) => Promise<MemoryView>;
  deleteMemory?: (memoryId: string) => Promise<void>;
  updateMemory?: (
    memoryId: string,
    text: string,
    retention: MemoryRetention,
  ) => Promise<MemoryView>;
  getLatestMemoryRecall?: (
    memoryId: string,
  ) => Promise<MemoryRecallView | null>;
  getMemoryResearchConsent?: () => Promise<MemoryResearchConsentView>;
  setMemoryResearchConsent?: (
    granted: boolean,
    policyVersion: string,
  ) => Promise<MemoryResearchConsentView>;
  getMemoryShadowReport?: () => Promise<MemoryShadowReportView>;
  listMemoryProfiles?: () => Promise<MemoryProfileView[]>;
  decideMemoryProfile?: (
    profileId: string,
    decision: "confirm" | "reject",
  ) => Promise<MemoryProfileView | null>;
  listMemoryConflicts?: () => Promise<MemoryConflictView[]>;
  decideMemoryConflict?: (
    conflictId: string,
    decision: "select" | "dismiss",
    profileId?: string,
  ) => Promise<MemoryConflictView>;
  listMemoryChanges?: () => Promise<MemoryChangeView[]>;
  decideMemoryChange?: (
    changeId: string,
    decision: "apply" | "reject",
  ) => Promise<MemoryChangeView>;
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
    sendText: async () => undefined,
    hangUp,
    listMemories: async () => [],
    getMemoryStatus: async () => ({ state: "idle", retryable: false }),
    retryMemoryIngestion: async () => ({ state: "complete", retryable: false }),
    decideMemory: async () => {
      throw new Error("Mock memory has no candidates");
    },
    deleteMemory: async () => undefined,
    updateMemory: async () => {
      throw new Error("Mock memory has no editable items");
    },
    getLatestMemoryRecall: async () => null,
    listMemoryProfiles: async () => [],
    decideMemoryProfile: async () => null,
    listMemoryConflicts: async () => [],
    decideMemoryConflict: async () => {
      throw new Error("Mock memory has no conflicts");
    },
    listMemoryChanges: async () => [],
    decideMemoryChange: async () => {
      throw new Error("Mock memory has no changes");
    },
    getMemoryResearchConsent: async () => ({
      enabled: false,
      granted: false,
      policy_version: "memory-shadow-research-v1",
      strategy_version: "hybrid-bge-m3-v1.6@0.50",
      retained_fields: [],
      excluded_fields: ["query text", "memory text", "embedding"],
    }),
    setMemoryResearchConsent: async () => {
      throw new Error("Mock shadow evaluation is disabled");
    },
    getMemoryShadowReport: async () => ({
      enabled: false,
      consent_granted: false,
      aggregate: {
        run_count: 0,
        completed_count: 0,
        failed_count: 0,
        timeout_count: 0,
        circuit_open_count: 0,
        completion_rate: 0,
        reliable: false,
        minimum_reliable_runs: 100,
        warnings: ["INSUFFICIENT_COMPLETED_RUNS"],
        strategy_versions: {},
      },
    }),
  };
}
