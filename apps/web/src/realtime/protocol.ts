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

function requireExactKeys(
  value: Record<string, unknown>,
  required: string[],
  optional: string[] = [],
): void {
  const allowed = new Set(["type", ...required, ...optional]);
  if (
    required.some(
      (key) => !Object.prototype.hasOwnProperty.call(value, key),
    ) ||
    Object.keys(value).some((key) => !allowed.has(key))
  ) {
    invalid();
  }
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
    case "session.ready": {
      requireExactKeys(value, [], [
        "session_id",
        "provider_mode",
        "model",
      ]);
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
      return {
        type: "session.ready",
        ...(typeof value.session_id === "string"
          ? { session_id: value.session_id }
          : {}),
        ...(typeof value.provider_mode === "string"
          ? { provider_mode: value.provider_mode }
          : {}),
        ...(typeof value.model === "string"
          ? { model: value.model }
          : {}),
      };
    }
    case "agent.state": {
      requireExactKeys(value, ["state"], ["turn_id"]);
      if (
        typeof value.state !== "string" ||
        !AGENT_STATES.has(value.state as AgentRealtimeState) ||
        !optionalString(value, "turn_id")
      ) {
        return invalid();
      }
      return {
        type: "agent.state",
        state: value.state as AgentRealtimeState,
        ...(typeof value.turn_id === "string"
          ? { turn_id: value.turn_id }
          : {}),
      };
    }
    case "transcript.partial": {
      requireExactKeys(
        value,
        ["kind", "text", "start_ms", "end_ms", "turn_id"],
      );
      requireFields(
        value,
        ["kind", "text", "turn_id"],
        ["start_ms", "end_ms"],
      );
      return {
        type: "transcript.partial",
        kind: value.kind as string,
        text: value.text as string,
        start_ms: value.start_ms as number,
        end_ms: value.end_ms as number,
        turn_id: value.turn_id as string,
      };
    }
    case "transcript.final": {
      requireExactKeys(value, ["text", "input_mode", "turn_id"]);
      requireFields(value, ["text", "input_mode", "turn_id"]);
      return {
        type: "transcript.final",
        text: value.text as string,
        input_mode: value.input_mode as string,
        turn_id: value.turn_id as string,
      };
    }
    case "transcript.empty": {
      requireExactKeys(value, ["input_mode", "turn_id"]);
      requireFields(value, ["input_mode", "turn_id"]);
      return {
        type: "transcript.empty",
        input_mode: value.input_mode as string,
        turn_id: value.turn_id as string,
      };
    }
    case "assistant.response":
    case "assistant.response.ready": {
      requireExactKeys(value, ["display_text"], [
        "spoken_text",
        "support_mode",
        "risk_level",
        "evidence_ids",
        "visual_observation_ids",
        "action_proposals",
        "memory_candidates",
        "avatar_style",
        "turn_id",
      ]);
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
      return {
        type: "assistant.response" as const,
        display_text: value.display_text as string,
        ...(typeof value.spoken_text === "string"
          ? { spoken_text: value.spoken_text }
          : {}),
        ...(typeof value.support_mode === "string"
          ? { support_mode: value.support_mode }
          : {}),
        ...(typeof value.risk_level === "string"
          ? { risk_level: value.risk_level }
          : {}),
        ...(Array.isArray(value.evidence_ids)
          ? { evidence_ids: [...value.evidence_ids] as string[] }
          : {}),
        ...(Array.isArray(value.visual_observation_ids)
          ? {
              visual_observation_ids: [
                ...value.visual_observation_ids,
              ] as string[],
            }
          : {}),
        ...(Array.isArray(value.action_proposals)
          ? { action_proposals: [...value.action_proposals] }
          : {}),
        ...(Array.isArray(value.memory_candidates)
          ? { memory_candidates: [...value.memory_candidates] }
          : {}),
        ...(typeof value.avatar_style === "string"
          ? { avatar_style: value.avatar_style }
          : {}),
        ...(typeof value.turn_id === "string"
          ? { turn_id: value.turn_id }
          : {}),
      };
    }
    case "audio.start": {
      requireExactKeys(value, [
        "stream_id",
        "turn_id",
        "sample_rate",
        "channels",
        "encoding",
      ]);
      requireFields(value, ["stream_id", "turn_id"]);
      if (
        value.sample_rate !== 16000 ||
        value.channels !== 1 ||
        value.encoding !== "pcm_s16le"
      ) {
        return invalid();
      }
      return {
        type: "audio.start",
        stream_id: value.stream_id as string,
        turn_id: value.turn_id as string,
        sample_rate: 16000,
        channels: 1,
        encoding: "pcm_s16le",
      };
    }
    case "audio.end": {
      requireExactKeys(value, ["stream_id", "turn_id"]);
      requireFields(value, ["stream_id", "turn_id"]);
      return {
        type: "audio.end",
        stream_id: value.stream_id as string,
        turn_id: value.turn_id as string,
      };
    }
    case "turn.completed": {
      requireExactKeys(value, ["turn_id", "delivery_mode"]);
      requireFields(value, ["turn_id", "delivery_mode"]);
      return {
        type: "turn.completed",
        turn_id: value.turn_id as string,
        delivery_mode: value.delivery_mode as string,
      };
    }
    case "playback.interrupted": {
      requireExactKeys(value, ["turn_id", "reason"]);
      if (
        (value.turn_id !== null && typeof value.turn_id !== "string") ||
        !hasString(value, "reason")
      ) {
        return invalid();
      }
      return {
        type: "playback.interrupted",
        turn_id: value.turn_id as string | null,
        reason: value.reason as string,
      };
    }
    case "error": {
      requireExactKeys(value, ["code"], ["recoverable"]);
      if (!hasString(value, "code") || !optionalBoolean(value, "recoverable")) {
        return invalid();
      }
      return {
        type: "error",
        code: value.code as string,
        ...(typeof value.recoverable === "boolean"
          ? { recoverable: value.recoverable }
          : {}),
      };
    }
    case "pong": {
      requireExactKeys(value, [], ["timestamp_ms"]);
      if (
        value.timestamp_ms !== undefined &&
        !hasNumber(value, "timestamp_ms")
      ) {
        return invalid();
      }
      return {
        type: "pong",
        ...(typeof value.timestamp_ms === "number"
          ? { timestamp_ms: value.timestamp_ms }
          : {}),
      };
    }
    case "risk.updated": {
      requireExactKeys(value, [
        "turn_id",
        "level",
        "reasons",
        "confidence",
        "evidence_event_ids",
      ]);
      requireFields(value, ["turn_id", "level"], ["confidence"]);
      if (
        !Array.isArray(value.reasons) ||
        !Array.isArray(value.evidence_event_ids)
      ) {
        return invalid();
      }
      return {
        type: "risk.updated",
        turn_id: value.turn_id as string,
        level: value.level as string,
        reasons: [...value.reasons],
        confidence: value.confidence as number,
        evidence_event_ids: [...value.evidence_event_ids],
      };
    }
    case "playback.started": {
      requireExactKeys(value, ["turn_id", "delivery_mode"]);
      requireFields(value, ["turn_id", "delivery_mode"]);
      return {
        type: "playback.started",
        turn_id: value.turn_id as string,
        delivery_mode: value.delivery_mode as string,
      };
    }
    case "response.displayed": {
      requireExactKeys(value, ["turn_id", "text"]);
      requireFields(value, ["turn_id", "text"]);
      return {
        type: "response.displayed",
        turn_id: value.turn_id as string,
        text: value.text as string,
      };
    }
    case "vision.degraded":
    case "tts.degraded":
    case "avatar.degraded": {
      requireExactKeys(value, ["turn_id", "fallback"]);
      requireFields(value, ["turn_id", "fallback"]);
      return {
        type: value.type,
        turn_id: value.turn_id as string,
        fallback: value.fallback as string,
      };
    }
    case "vision.cancelled": {
      requireExactKeys(value, ["turn_id", "reason"]);
      requireFields(value, ["turn_id", "reason"]);
      return {
        type: "vision.cancelled",
        turn_id: value.turn_id as string,
        reason: value.reason as string,
      };
    }
    default:
      return invalid();
  }
}
