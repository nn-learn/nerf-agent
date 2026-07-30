import { describe, expect, it } from "vitest";

import { parseServerMessage } from "./protocol";

describe("parseServerMessage", () => {
  it("parses a ready event", () => {
    const ready = parseServerMessage({
      type: "session.ready",
      provider_mode: "local",
      model: "qwen3.6:latest",
    });

    expect(ready.type).toBe("session.ready");
  });

  it("rejects invalid event fields", () => {
    expect(() =>
      parseServerMessage({
        type: "assistant.response",
        display_text: 42,
      }),
    ).toThrow("Invalid realtime message");
  });

  it("rejects response arrays whose elements do not match their typed contract", () => {
    expect(() =>
      parseServerMessage({
        type: "assistant.response",
        display_text: "先慢慢说。",
        evidence_ids: [42],
      }),
    ).toThrow("Invalid realtime message");
  });

  it.each([
    { type: "pong", unexpected: "raw-pcm-marker" },
    {
      type: "assistant.response.ready",
      display_text: "ok",
      raw_audio: "sensitive",
    },
    { type: "error", code: "X", message: 42 },
  ])("rejects undeclared fields on $type", (payload) => {
    expect(() => parseServerMessage(payload)).toThrow(
      "Invalid realtime message",
    );
  });

  it("normalizes the backend response-ready wire event", () => {
    const response = parseServerMessage({
      type: "assistant.response.ready",
      turn_id: "turn_1",
      spoken_text: "先慢慢说。",
      display_text: "先慢慢说。",
      support_mode: "supportive",
      risk_level: "LOW",
      evidence_ids: [],
      visual_observation_ids: [],
      action_proposals: [],
      memory_candidates: [],
      avatar_style: "gentle",
    });

    expect(response).toMatchObject({
      type: "assistant.response",
      turn_id: "turn_1",
      display_text: "先慢慢说。",
    });
  });

  it.each([
    { type: "agent.state", state: "thinking", turn_id: "turn_1" },
    {
      type: "transcript.partial",
      kind: "PARTIAL",
      text: "最近",
      start_ms: 0,
      end_ms: 300,
      turn_id: "turn_1",
    },
    {
      type: "transcript.final",
      text: "最近压力很大",
      input_mode: "voice",
      turn_id: "turn_1",
    },
    {
      type: "audio.start",
      stream_id: "audio_turn_1",
      turn_id: "turn_1",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
    },
    {
      type: "audio.end",
      stream_id: "audio_turn_1",
      turn_id: "turn_1",
    },
    {
      type: "turn.completed",
      turn_id: "turn_1",
      delivery_mode: "voice_text",
    },
    {
      type: "playback.interrupted",
      turn_id: "turn_1",
      reason: "user_barge_in",
    },
    { type: "error", code: "PROVIDER_ERROR", recoverable: false },
    { type: "pong" },
  ])("parses $type", (payload) => {
    expect(parseServerMessage(payload).type).toBe(payload.type);
  });

  it("rejects unknown message types", () => {
    expect(() => parseServerMessage({ type: "raw.audio" })).toThrow(
      "Invalid realtime message",
    );
  });
});
