import { create } from "zustand";

export type VisionState = "off" | "active" | "paused" | "disabled";
export type AgentState =
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "disconnected";

interface SessionState {
  sessionId: string;
  visionState: VisionState;
  agentState: AgentState;
  setVisionState: (state: VisionState) => void;
  setAgentState: (state: AgentState) => void;
}

function newSessionId(): string {
  const suffix = crypto.randomUUID?.() ?? Math.random().toString(36).slice(2);
  return `session_${suffix.replaceAll("-", "")}`;
}


export const useSessionStore = create<SessionState>((set) => ({
  sessionId: newSessionId(),
  visionState: "off",
  agentState: "connecting",
  setVisionState: (visionState) => set({ visionState }),
  setAgentState: (agentState) => set({ agentState }),
}));
