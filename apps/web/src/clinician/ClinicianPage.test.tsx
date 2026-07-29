import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { ClinicianPage } from "./ClinicianPage";


it("shows a redacted handoff view and accepts the scoped case", async () => {
  const user = userEvent.setup();
  const acceptHandoff = vi.fn().mockResolvedValue({
    handoff_id: "handoff_1",
    state: "ACCEPTED",
  });
  const api = {
    getClinicianSummary: vi.fn().mockResolvedValue({
      session_id: "session_1",
      risk_level: "AMBER",
      reasons: ["distress_or_impairment_language"],
      key_turns: [
        {
          turn_id: "turn_1",
          transcript_excerpt: "最近持续低落，但我想先…",
          risk_level: "AMBER",
          support_mode: "listen",
        },
      ],
      visual_context_used: false,
      visual_summaries: [],
      handoff_state: "REQUESTED",
      handoff_id: "handoff_1",
      safety_policy: "AI support only",
    }),
    getClinicianTimeline: vi.fn().mockResolvedValue({
      session_id: "session_1",
      events: [
        {
          seq: 4,
          type: "risk.updated",
          timestamp_ms: 1_000,
          turn_id: "turn_1",
          trace_id: "trace_1",
          metadata: { level: "AMBER" },
        },
      ],
    }),
    acceptHandoff,
  };

  render(<ClinicianPage api={api} sessionId="session_1" />);

  expect((await screen.findAllByText("AMBER")).length).toBeGreaterThan(0);
  expect(screen.getByText("最近持续低落，但我想先…")).toBeInTheDocument();
  expect(screen.queryByText("完整对话")).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "接受演示接管" }));

  expect(acceptHandoff).toHaveBeenCalledWith("session_1", "handoff_1");
  expect(await screen.findByText("已由演示临床角色接管")).toBeInTheDocument();
});
