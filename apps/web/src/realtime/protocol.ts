export type AgentRealtimeState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "interrupted"
  | "error";

export interface SessionReadyMessage {
  type: "session.ready";
  session_id?: string;
  provider_mode?: string;
  model?: string;
}

export interface AgentStateMessage {
  type: "agent.state";
  state: AgentRealtimeState;
  turn_id?: string;
}

export interface TranscriptPartialMessage {
  type: "transcript.partial";
  kind: string;
  text: string;
  start_ms: number;
  end_ms: number;
  turn_id: string;
}

export interface TranscriptFinalMessage {
  type: "transcript.final";
  text: string;
  input_mode: string;
  turn_id: string;
}

export interface TranscriptEmptyMessage {
  type: "transcript.empty";
  input_mode: string;
  turn_id: string;
}

export interface AssistantResponseMessage {
  type: "assistant.response";
  display_text: string;
  spoken_text?: string;
  support_mode?: string;
  risk_level?: string;
  evidence_ids?: string[];
  visual_observation_ids?: string[];
  action_proposals?: unknown[];
  memory_candidates?: unknown[];
  avatar_style?: string;
  turn_id?: string;
}

export interface AudioStartMessage {
  type: "audio.start";
  stream_id: string;
  turn_id: string;
  sample_rate: 16000;
  channels: 1;
  encoding: "pcm_s16le";
}

export interface AudioEndMessage {
  type: "audio.end";
  stream_id: string;
  turn_id: string;
}

export interface TurnCompletedMessage {
  type: "turn.completed";
  turn_id: string;
  delivery_mode: string;
}

export interface PlaybackInterruptedMessage {
  type: "playback.interrupted";
  turn_id: string | null;
  reason: string;
}

export interface ErrorMessage {
  type: "error";
  code: string;
  recoverable?: boolean;
}

export interface PongMessage {
  type: "pong";
  timestamp_ms?: number;
}

export interface RiskUpdatedMessage {
  type: "risk.updated";
  turn_id: string;
  level: string;
  reasons: unknown[];
  confidence: number;
  evidence_event_ids: unknown[];
}

export interface PlaybackStartedMessage {
  type: "playback.started";
  turn_id: string;
  delivery_mode: string;
}

export interface ResponseDisplayedMessage {
  type: "response.displayed";
  turn_id: string;
  text: string;
}

export interface DegradedMessage {
  type: "vision.degraded" | "tts.degraded" | "avatar.degraded";
  turn_id: string;
  fallback: string;
}

export interface VisionCancelledMessage {
  type: "vision.cancelled";
  turn_id: string;
  reason: string;
}

export type ServerMessage =
  | SessionReadyMessage
  | AgentStateMessage
  | TranscriptPartialMessage
  | TranscriptFinalMessage
  | TranscriptEmptyMessage
  | AssistantResponseMessage
  | AudioStartMessage
  | AudioEndMessage
  | TurnCompletedMessage
  | PlaybackInterruptedMessage
  | ErrorMessage
  | PongMessage
  | RiskUpdatedMessage
  | PlaybackStartedMessage
  | ResponseDisplayedMessage
  | DegradedMessage
  | VisionCancelledMessage;

const AGENT_STATES = new Set<AgentRealtimeState>([
  "idle",
  "listening",
  "thinking",
  "speaking",
  "interrupted",
  "error",
]);

function invalid(): never {
  throw new Error("Invalid realtime message");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasString(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return typeof value[key] === "string";
}

function hasNumber(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return typeof value[key] === "number" && Number.isFinite(value[key]);
}

function optionalString(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return value[key] === undefined || typeof value[key] === "string";
}

function optionalBoolean(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return value[key] === undefined || typeof value[key] === "boolean";
}

function optionalStringArray(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return (
    value[key] === undefined ||
    (Array.isArray(value[key]) &&
      value[key].every((item) => typeof item === "string"))
  );
}

function optionalArray(
  value: Record<string, unknown>,
  key: string,
): boolean {
  return value[key] === undefined || Array.isArray(value[key]);
}

function requireFields(
  value: Record<string, unknown>,
  strings: string[],
  numbers: string[] = [],
): void {
  if (
    !strings.every((key) => hasString(value, key)) ||
    !numbers.every((key) => hasNumber(value, key))
  ) {
    invalid();
  }
}

function asMessage(value: Record<string, unknown>): ServerMessage {
  return value as unknown as ServerMessage;
}

export function parseServerMessage(input: unknown): ServerMessage {
  let value = input;
  if (typeof input === "string") {
    try {
      value = JSON.parse(input) as unknown;
    } catch {
      invalid();
    }
  }
  if (!isRecord(value) || typeof value.type !== "string") {
    return invalid();
  }

  switch (value.type) {
    case "session.ready":
      if (
        !optionalString(value, "session_id") ||
        !optionalString(value, "provider_mode") ||
        !optionalString(value, "model") ||
        !["session_id", "provider_mode", "model"].some((key) =>
          hasString(value, key),
        )
      ) {
        return invalid();
      }
      return asMessage(value);
    case "agent.state":
      if (
        typeof value.state !== "string" ||
        !AGENT_STATES.has(value.state as AgentRealtimeState) ||
        !optionalString(value, "turn_id")
      ) {
        return invalid();
      }
      return asMessage(value);
    case "transcript.partial":
      requireFields(
        value,
        ["kind", "text", "turn_id"],
        ["start_ms", "end_ms"],
      );
      return asMessage(value);
    case "transcript.final":
      requireFields(value, ["text", "input_mode", "turn_id"]);
      return asMessage(value);
    case "transcript.empty":
      requireFields(value, ["input_mode", "turn_id"]);
      return asMessage(value);
    case "assistant.response":
    case "assistant.response.ready": {
      requireFields(value, ["display_text"]);
      if (
        !["spoken_text", "support_mode", "risk_level", "avatar_style", "turn_id"].every(
          (key) => optionalString(value, key),
        ) ||
        !["evidence_ids", "visual_observation_ids"].every((key) =>
          optionalStringArray(value, key),
        ) ||
        !["action_proposals", "memory_candidates"].every((key) =>
          optionalArray(value, key),
        )
      ) {
        return invalid();
      }
      const normalized = {
        ...value,
        type: "assistant.response" as const,
      };
      return asMessage(normalized);
    }
    case "audio.start":
      requireFields(value, ["stream_id", "turn_id"]);
      if (
        value.sample_rate !== 16000 ||
        value.channels !== 1 ||
        value.encoding !== "pcm_s16le"
      ) {
        return invalid();
      }
      return asMessage(value);
    case "audio.end":
      requireFields(value, ["stream_id", "turn_id"]);
      return asMessage(value);
    case "turn.completed":
      requireFields(value, ["turn_id", "delivery_mode"]);
      return asMessage(value);
    case "playback.interrupted":
      if (
        (value.turn_id !== null && typeof value.turn_id !== "string") ||
        !hasString(value, "reason")
      ) {
        return invalid();
      }
      return asMessage(value);
    case "error":
      if (!hasString(value, "code") || !optionalBoolean(value, "recoverable")) {
        return invalid();
      }
      return asMessage(value);
    case "pong":
      if (
        value.timestamp_ms !== undefined &&
        !hasNumber(value, "timestamp_ms")
      ) {
        return invalid();
      }
      return asMessage(value);
    case "risk.updated":
      requireFields(value, ["turn_id", "level"], ["confidence"]);
      if (
        !Array.isArray(value.reasons) ||
        !Array.isArray(value.evidence_event_ids)
      ) {
        return invalid();
      }
      return asMessage(value);
    case "playback.started":
      requireFields(value, ["turn_id", "delivery_mode"]);
      return asMessage(value);
    case "response.displayed":
      requireFields(value, ["turn_id", "text"]);
      return asMessage(value);
    case "vision.degraded":
    case "tts.degraded":
    case "avatar.degraded":
      requireFields(value, ["turn_id", "fallback"]);
      return asMessage(value);
    case "vision.cancelled":
      requireFields(value, ["turn_id", "reason"]);
      return asMessage(value);
    default:
      return invalid();
  }
}
