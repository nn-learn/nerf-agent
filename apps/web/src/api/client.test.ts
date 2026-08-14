import { afterEach, expect, it, vi } from "vitest";

import { PsyAvatarApi } from "./client";


afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});


it("creates the audited backend session before realtime room setup", async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(
      JSON.stringify({
        session_id: "session_browser_1",
        status: "active",
        provider_mode: "mock",
        camera_consent: false,
        fallback_capabilities: ["http_text"],
        access_token: "pst_test_token_with_more_than_32_characters",
      }),
      {
        status: 201,
        headers: { "Content-Type": "application/json" },
      },
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");

  const session = await api.createSession("session_browser_1");

  expect(session.session_id).toBe("session_browser_1");
  expect(fetchMock).toHaveBeenCalledWith(
    "http://orchestrator.test/api/sessions",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        camera_consent: false,
        client_session_id: "session_browser_1",
      }),
    }),
  );
});


it("ends the backend session when the call hangs up", async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          session_id: "session_browser_1",
          status: "active",
          provider_mode: "mock",
          camera_consent: false,
          fallback_capabilities: ["http_text"],
          access_token: "pst_test_token_with_more_than_32_characters",
        }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )
    .mockResolvedValueOnce(new Response(
      JSON.stringify({
        session_id: "session_browser_1",
        status: "ended",
        provider_mode: "mock",
        camera_consent: false,
        fallback_capabilities: ["http_text"],
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      },
    ));
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");

  await api.createSession("session_browser_1");
  await api.endSession("session_browser_1");

  expect(fetchMock).toHaveBeenCalledWith(
    "http://orchestrator.test/api/sessions/session_browser_1",
    expect.objectContaining({
      method: "DELETE",
      headers: expect.objectContaining({
        Authorization: "Bearer pst_test_token_with_more_than_32_characters",
      }),
    }),
  );
});


it("reuses one in-flight creation during React effect replay", async () => {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(
      JSON.stringify({
        session_id: "session_browser_1",
        status: "active",
        provider_mode: "mock",
        camera_consent: false,
        fallback_capabilities: ["http_text"],
        access_token: "pst_test_token_with_more_than_32_characters",
      }),
      {
        status: 201,
        headers: { "Content-Type": "application/json" },
      },
    ),
  );
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");

  await Promise.all([
    api.createSession("session_browser_1"),
    api.createSession("session_browser_1"),
  ]);

  expect(fetchMock).toHaveBeenCalledTimes(1);
});


it("persists and reuses the opaque memory subject token", async () => {
  const subjectToken = `pms_${"s".repeat(43)}`;
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          session_id: "session_browser_1",
          status: "active",
          provider_mode: "mock",
          camera_consent: false,
          fallback_capabilities: ["http_text"],
          access_token: "pst_first_token_with_more_than_32_characters",
          memory_subject_token: subjectToken,
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    )
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          session_id: "session_browser_2",
          status: "active",
          provider_mode: "mock",
          camera_consent: false,
          fallback_capabilities: ["http_text"],
          access_token: "pst_second_token_with_more_than_32_characters",
          memory_subject_token: null,
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );
  vi.stubGlobal("fetch", fetchMock);

  await new PsyAvatarApi("http://orchestrator.test").createSession(
    "session_browser_1",
  );
  await new PsyAvatarApi("http://orchestrator.test").createSession(
    "session_browser_2",
  );

  expect(fetchMock).toHaveBeenNthCalledWith(
    2,
    "http://orchestrator.test/api/sessions",
    expect.objectContaining({
      body: JSON.stringify({
        camera_consent: false,
        client_session_id: "session_browser_2",
        memory_subject_token: subjectToken,
      }),
    }),
  );
});


it("manages memory candidates with the private session bearer", async () => {
  const memoryId = `memory_${"a".repeat(32)}`;
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({
      session_id: "session_browser_1",
      status: "active",
      provider_mode: "mock",
      camera_consent: false,
      fallback_capabilities: ["http_text"],
      access_token: "pst_test_token_with_more_than_32_characters",
    }), { status: 201, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify([]), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      memory_id: memoryId,
      state: "ACTIVE",
    }), { status: 200, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(null, { status: 204 }));
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");
  await api.createSession("session_browser_1");

  await api.listMemories("session_browser_1");
  await api.decideMemory("session_browser_1", memoryId, "confirm");
  await api.deleteMemory("session_browser_1", memoryId);

  expect(fetchMock).toHaveBeenNthCalledWith(
    3,
    `http://orchestrator.test/api/sessions/session_browser_1/memories/${memoryId}/decision`,
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ decision: "confirm" }),
      headers: expect.objectContaining({
        Authorization: "Bearer pst_test_token_with_more_than_32_characters",
      }),
    }),
  );
  expect(fetchMock).toHaveBeenLastCalledWith(
    `http://orchestrator.test/api/sessions/session_browser_1/memories/${memoryId}`,
    expect.objectContaining({ method: "DELETE" }),
  );
});


it("reads memory ingestion status and requests a retry", async () => {
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({
      session_id: "session_browser_1",
      status: "active",
      provider_mode: "mock",
      camera_consent: false,
      fallback_capabilities: ["http_text"],
      access_token: "pst_test_token_with_more_than_32_characters",
    }), { status: 201, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      state: "failed",
      retryable: true,
    }), { status: 200, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      state: "queued",
      retryable: false,
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");
  await api.createSession("session_browser_1");

  expect((await api.getMemoryStatus("session_browser_1")).state).toBe("failed");
  expect((await api.retryMemoryIngestion("session_browser_1")).state).toBe("queued");
  expect(fetchMock).toHaveBeenLastCalledWith(
    "http://orchestrator.test/api/sessions/session_browser_1/memories/retry",
    expect.objectContaining({
      method: "POST",
      headers: expect.objectContaining({
        Authorization: "Bearer pst_test_token_with_more_than_32_characters",
      }),
    }),
  );
});


it("updates memory retention and reads its latest recall explanation", async () => {
  const memoryId = `memory_${"c".repeat(32)}`;
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({
      session_id: "session_browser_1",
      status: "active",
      provider_mode: "mock",
      camera_consent: false,
      fallback_capabilities: ["http_text"],
      access_token: "pst_test_token_with_more_than_32_characters",
    }), { status: 201, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      memory_id: memoryId,
      text: "用户偏好：先给简短结论",
      state: "ACTIVE",
    }), { status: 200, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      used_at_ms: 2,
      score: 0.8,
      relevance_score: 0.7,
      reason_codes: ["USER_CONFIRMED", "TOPIC_MATCH"],
      turn_id: "turn_2",
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");
  await api.createSession("session_browser_1");

  await api.updateMemory(
    "session_browser_1",
    memoryId,
    "用户偏好：先给简短结论",
    "30_days",
  );
  const recall = await api.getLatestMemoryRecall("session_browser_1", memoryId);

  expect(fetchMock).toHaveBeenNthCalledWith(
    2,
    `http://orchestrator.test/api/sessions/session_browser_1/memories/${memoryId}`,
    expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({
        text: "用户偏好：先给简短结论",
        retention: "30_days",
      }),
    }),
  );
  expect(recall?.reason_codes).toContain("TOPIC_MATCH");
});

it("uses versioned memory-research consent and reads only aggregate shadow data", async () => {
  const policyVersion = "memory-shadow-research-v1";
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(new Response(JSON.stringify({
      session_id: "session_browser_1",
      status: "active",
      provider_mode: "local",
      camera_consent: false,
      fallback_capabilities: ["http_text"],
      access_token: "pst_test_token_with_more_than_32_characters",
    }), { status: 201, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      enabled: true,
      granted: true,
      policy_version: policyVersion,
      strategy_version: "hybrid-bge-m3-v1.6@0.50",
      retained_fields: ["IDs and scores"],
      excluded_fields: ["query text", "memory text", "embedding"],
    }), { status: 200, headers: { "Content-Type": "application/json" } }))
    .mockResolvedValueOnce(new Response(JSON.stringify({
      enabled: true,
      consent_granted: true,
      aggregate: {
        run_count: 1,
        completed_count: 1,
        reliable: false,
        minimum_reliable_runs: 100,
        warnings: ["INSUFFICIENT_COMPLETED_RUNS"],
      },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");
  await api.createSession("session_browser_1");

  const consent = await api.setMemoryResearchConsent(
    "session_browser_1",
    true,
    policyVersion,
  );
  const report = await api.getMemoryShadowReport("session_browser_1");

  expect(consent.granted).toBe(true);
  expect(report.aggregate.reliable).toBe(false);
  expect(fetchMock).toHaveBeenNthCalledWith(
    2,
    "http://orchestrator.test/api/sessions/session_browser_1/memories/research-consent",
    expect.objectContaining({
      method: "PUT",
      body: JSON.stringify({
        granted: true,
        acknowledged_policy_version: policyVersion,
      }),
    }),
  );
});


it("creates a typed text fallback turn with the private bearer token", async () => {
  const fetchMock = vi
    .fn()
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          session_id: "session_browser_1",
          status: "active",
          provider_mode: "local",
          camera_consent: false,
          fallback_capabilities: ["http_text"],
          access_token: "pst_test_token_with_more_than_32_characters",
        }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" },
        },
      ),
    )
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "completed",
          risk_level: "GREEN",
          response: {
            spoken_text: "我们先慢慢说。",
            display_text: "我们先慢慢说。",
            support_mode: "listen",
          },
          delivery_mode: "text",
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
  vi.stubGlobal("fetch", fetchMock);
  const api = new PsyAvatarApi("http://orchestrator.test");
  await api.createSession("session_browser_1");

  const result = await api.createTextTurn(
    "session_browser_1",
    "最近压力很大",
  );

  expect(result.delivery_mode).toBe("text");
  expect(result.response?.display_text).toBe("我们先慢慢说。");
  expect(fetchMock).toHaveBeenLastCalledWith(
    "http://orchestrator.test/api/sessions/session_browser_1/turns",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({
        text: "最近压力很大",
        visual_summary: "",
      }),
      headers: expect.objectContaining({
        Authorization: "Bearer pst_test_token_with_more_than_32_characters",
      }),
    }),
  );
});


it("builds a token-free realtime URL and passes auth only to the client factory", async () => {
  const token = "pst_private_token_with_more_than_32_characters";
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "session_browser_1",
          status: "active",
          provider_mode: "local",
          camera_consent: false,
          fallback_capabilities: ["http_text"],
          access_token: token,
        }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" },
        },
      ),
    ),
  );
  const api = new PsyAvatarApi("https://orchestrator.test/base?legacy=1");
  await api.createSession("session_browser_1");
  const factory = vi.fn((url: string, accessToken: string) => ({
    url,
    accessToken,
  }));

  const client = api.createRealtimeClient(
    "session_browser_1",
    factory,
  );

  expect(api.realtimeUrl("session_browser_1")).toBe(
    "wss://orchestrator.test/api/sessions/session_browser_1/realtime",
  );
  expect(factory).toHaveBeenCalledWith(
    "wss://orchestrator.test/api/sessions/session_browser_1/realtime",
    token,
  );
  expect(client.url).not.toContain(token);
});
