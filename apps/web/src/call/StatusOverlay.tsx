import { Eye, EyeOff, Radio, ShieldCheck } from "lucide-react";

import type {
  MediaConnectionState,
} from "./useMediaSession";

interface StatusOverlayProps {
  connectionState: MediaConnectionState;
  visionActive: boolean;
  agentState?: "connecting" | "listening" | "thinking" | "speaking" | "disconnected";
}

const agentLabels = {
  connecting: "正在进入陪伴空间",
  listening: "小澄正在倾听",
  thinking: "小澄正在组织回应",
  speaking: "小澄正在回应",
  disconnected: "通话已结束",
} as const;


export function StatusOverlay({
  connectionState,
  visionActive,
  agentState = "listening",
}: StatusOverlayProps) {
  const connected = connectionState === "connected";
  return (
    <div className="status-overlay">
      <div className={`status-pill ${connected ? "is-online" : ""}`}>
        <Radio aria-hidden="true" size={15} />
        {connected ? agentLabels[agentState] : "正在连接"}
      </div>
      <div
        className={`status-pill vision-pill ${
          visionActive ? "is-watching" : ""
        }`}
      >
        {visionActive ? (
          <Eye aria-hidden="true" size={15} />
        ) : (
          <EyeOff aria-hidden="true" size={15} />
        )}
        {visionActive ? "AI 正在看" : "视觉未开启"}
      </div>
      <div className="privacy-badge">
        <ShieldCheck aria-hidden="true" size={14} />
        原始音视频不落盘
      </div>
    </div>
  );
}
