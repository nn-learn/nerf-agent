import { afterEach, expect, it, vi } from "vitest";

import { PsyAvatarApi } from "./client";


afterEach(() => {
  vi.unstubAllGlobals();
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
