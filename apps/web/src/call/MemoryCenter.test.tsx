import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import type {
  MemoryChangeView,
  MemoryConflictView,
  MemoryProfileView,
  MemoryView,
} from "../api/client";
import { MemoryCenter } from "./MemoryCenter";

const pending: MemoryView = {
  memory_id: `memory_${"a".repeat(32)}`,
  text: "用户偏好：简短回答",
  aspect: "PREFERENCE",
  state: "AWAITING_CONSENT",
  contains_sensitive_content: false,
  confidence: 0.92,
  purpose_scope: "personalization",
  created_at_ms: 1,
  updated_at_ms: 1,
  source_turn_id: "turn_1",
};

const active: MemoryView = {
  ...pending,
  memory_id: `memory_${"b".repeat(32)}`,
  text: "对用户有效的支持方式：呼吸练习",
  aspect: "COPING_STRATEGY",
  state: "ACTIVE",
};

const profile: MemoryProfileView = {
  profile_id: `profile_${"c".repeat(32)}`,
  subject_key: "communication.response_style",
  aspect: "PREFERENCE",
  statement: "用户偏好：回答时先给简短结论",
  state: "AWAITING_CONFIRMATION",
  confidence: 0.83,
  evidence_count: 2,
  supporting_evidence_count: 2,
  conflicting_evidence_count: 0,
  distinct_session_count: 2,
  contains_sensitive_content: false,
  created_at_ms: 1,
  updated_at_ms: 2,
  user_edited: false,
  evidence: [
    {
      memory_id: pending.memory_id,
      text: "用户偏好：简短回答",
      relation: "SUPPORTS",
      valid_at_ms: 1,
      observed_at_ms: 1,
    },
  ],
};

const conflict: MemoryConflictView = {
  conflict_id: `conflict_${"d".repeat(32)}`,
  subject_key: "communication.response_style",
  state: "OPEN",
  created_at_ms: 1,
  updated_at_ms: 2,
  options: [
    { ...profile, profile_id: `profile_${"e".repeat(32)}`, state: "STALE" },
    {
      ...profile,
      profile_id: `profile_${"f".repeat(32)}`,
      statement: "用户偏好：回答时提供更详细的解释",
      state: "STALE",
    },
  ],
};

const change: MemoryChangeView = {
  change_id: `change_${"1".repeat(32)}`,
  subject_key: "communication.response_style",
  state: "OPEN",
  effective_at_ms: 40_000,
  observed_at_ms: 45_000,
  created_at_ms: 50_000,
  updated_at_ms: 50_000,
  previous_profile: { ...profile, state: "ACTIVE" },
  proposed_profile: {
    ...profile,
    profile_id: `profile_${"2".repeat(32)}`,
    statement: "用户偏好：回答时提供更详细的解释",
  },
};

function port(memories: MemoryView[] = [pending, active]) {
  return {
    listMemories: vi.fn().mockResolvedValue(memories),
    getMemoryStatus: vi.fn().mockResolvedValue({
      state: "complete",
      retryable: false,
    }),
    retryMemoryIngestion: vi.fn().mockResolvedValue({
      state: "queued",
      retryable: false,
    }),
    decideMemory: vi.fn().mockImplementation(
      async (_memoryId: string, decision: "confirm" | "reject") => ({
        ...pending,
        state: decision === "confirm" ? "ACTIVE" : "REJECTED",
      }),
    ),
    deleteMemory: vi.fn().mockResolvedValue(undefined),
    updateMemory: vi.fn().mockImplementation(
      async (_memoryId: string, text: string) => ({
        ...active,
        text,
        user_edited: true,
      }),
    ),
    getLatestMemoryRecall: vi.fn().mockResolvedValue(null),
    getMemoryResearchConsent: vi.fn().mockResolvedValue({
      enabled: true,
      granted: false,
      policy_version: "memory-shadow-research-v1",
      strategy_version: "hybrid-bge-m3-v1.6@0.50",
      retained_fields: ["IDs and scores"],
      excluded_fields: ["query text", "memory text", "embedding"],
    }),
    setMemoryResearchConsent: vi.fn().mockImplementation(
      async (granted: boolean) => ({
        enabled: true,
        granted,
        policy_version: "memory-shadow-research-v1",
        strategy_version: "hybrid-bge-m3-v1.6@0.50",
        retained_fields: ["IDs and scores"],
        excluded_fields: ["query text", "memory text", "embedding"],
      }),
    ),
    getMemoryShadowReport: vi.fn().mockResolvedValue({
      enabled: true,
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
    listMemoryProfiles: vi.fn().mockResolvedValue([]),
    decideMemoryProfile: vi.fn().mockResolvedValue({
      ...profile,
      state: "ACTIVE",
    }),
    listMemoryConflicts: vi.fn().mockResolvedValue([]),
    decideMemoryConflict: vi.fn().mockResolvedValue({
      ...conflict,
      state: "RESOLVED",
    }),
    listMemoryChanges: vi.fn().mockResolvedValue([]),
    decideMemoryChange: vi.fn().mockResolvedValue({
      ...change,
      state: "APPLIED",
    }),
  };
}

it("requires explicit confirmation for a cross-session long-term profile", async () => {
  const user = userEvent.setup();
  const memoryPort = port([]);
  memoryPort.listMemoryProfiles.mockResolvedValue([profile]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  expect(await screen.findByText(profile.statement)).toBeInTheDocument();
  expect(screen.getByText("2 次会话支持")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "确认这个理解" }));

  expect(memoryPort.decideMemoryProfile).toHaveBeenCalledWith(
    profile.profile_id,
    "confirm",
  );
});

it("lets the user resolve a contradictory profile instead of recalling it", async () => {
  const user = userEvent.setup();
  const memoryPort = port([]);
  memoryPort.listMemoryConflicts.mockResolvedValue([conflict]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  expect(await screen.findByText("你在不同时间表达过不同倾向")).toBeInTheDocument();
  await user.click(screen.getByRole("button", {
    name: /回答时提供更详细的解释/,
  }));

  expect(memoryPort.decideMemoryConflict).toHaveBeenCalledWith(
    conflict.conflict_id,
    "select",
    conflict.options[1].profile_id,
  );
});

it("shows a temporal preference change separately and requires confirmation", async () => {
  const user = userEvent.setup();
  const memoryPort = port([]);
  memoryPort.listMemoryChanges.mockResolvedValue([change]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  expect(await screen.findByText("系统识别到你的偏好可能发生了变化")).toBeInTheDocument();
  expect(screen.getByText("之前的理解")).toBeInTheDocument();
  expect(screen.getByText("新的理解")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "确认已经改变" }));

  expect(memoryPort.decideMemoryChange).toHaveBeenCalledWith(
    change.change_id,
    "apply",
  );
});

it("requires explicit opt-in and supports physical shadow-data withdrawal", async () => {
  const user = userEvent.setup();
  const memoryPort = port([]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  expect(await screen.findByText("帮助改进记忆检索")).toBeInTheDocument();
  expect(screen.getByText(/不另存：问题原文/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "阅读并同意参与" }));
  expect(memoryPort.setMemoryResearchConsent).toHaveBeenCalledWith(
    true,
    "memory-shadow-research-v1",
  );
  await user.click(await screen.findByRole("button", {
    name: "撤回并清除全部影子记录",
  }));
  expect(memoryPort.setMemoryResearchConsent).toHaveBeenLastCalledWith(
    false,
    "memory-shadow-research-v1",
  );
});

it("separates candidate memories from user-approved memories", async () => {
  render(
    <MemoryCenter
      ended
      onClose={vi.fn()}
      open
      port={port()}
    />,
  );

  expect(await screen.findByText("用户偏好：简短回答")).toBeInTheDocument();
  expect(screen.getByText("对用户有效的支持方式：呼吸练习")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "确认记住" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "不保存" })).toBeInTheDocument();
  expect(screen.getByText("只有你亲自确认的内容，才会用于以后的陪伴。")).toBeInTheDocument();
});

it("confirms a candidate and moves it into the approved section", async () => {
  const user = userEvent.setup();
  const memoryPort = port([pending]);
  render(
    <MemoryCenter ended onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", { name: "确认记住" }));

  expect(memoryPort.decideMemory).toHaveBeenCalledWith(
    pending.memory_id,
    "confirm",
  );
  expect(screen.getByText("0 条候选")).toBeInTheDocument();
  expect(screen.getByText("1 条")).toBeInTheDocument();
});

it("rejects a candidate and removes its text from the interface", async () => {
  const user = userEvent.setup();
  const memoryPort = port([pending]);
  render(
    <MemoryCenter ended onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", { name: "不保存" }));

  expect(memoryPort.decideMemory).toHaveBeenCalledWith(
    pending.memory_id,
    "reject",
  );
  expect(screen.queryByText(pending.text)).not.toBeInTheDocument();
});

it("physically deletes an approved memory after an explicit action", async () => {
  const user = userEvent.setup();
  const memoryPort = port([active]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", {
    name: `永久删除：${active.text}`,
  }));
  expect(memoryPort.deleteMemory).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", {
    name: `确认永久删除：${active.text}`,
  }));

  expect(memoryPort.deleteMemory).toHaveBeenCalledWith(active.memory_id);
  expect(screen.queryByText(active.text)).not.toBeInTheDocument();
});

it("offers retry after extraction failure", async () => {
  const user = userEvent.setup();
  const memoryPort = port([]);
  memoryPort.getMemoryStatus
    .mockResolvedValueOnce({ state: "failed", retryable: true })
    .mockResolvedValue({ state: "complete", retryable: false });
  render(
    <MemoryCenter ended onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", { name: "重试" }));

  expect(memoryPort.retryMemoryIngestion).toHaveBeenCalledTimes(1);
  expect(await screen.findByText("候选记忆已整理完成")).toBeInTheDocument();
});

it("lets the user correct memory text and choose a retention period", async () => {
  const user = userEvent.setup();
  const memoryPort = port([active]);
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", { name: "编辑" }));
  const textbox = screen.getByRole("textbox", { name: "修正记忆内容" });
  await user.clear(textbox);
  await user.type(textbox, "用户偏好：先给简短结论");
  await user.selectOptions(screen.getByLabelText("保留期限"), "30_days");
  await user.click(screen.getByRole("button", { name: "保存修改" }));

  expect(memoryPort.updateMemory).toHaveBeenCalledWith(
    active.memory_id,
    "用户偏好：先给简短结论",
    "30_days",
  );
  expect(await screen.findByText("用户偏好：先给简短结论")).toBeInTheDocument();
  expect(screen.getByText("由你修正")).toBeInTheDocument();
});

it("explains deterministic recall without claiming response causality", async () => {
  const user = userEvent.setup();
  const memoryPort = port([active]);
  memoryPort.getLatestMemoryRecall.mockResolvedValue({
    used_at_ms: 2,
    score: 0.87,
    relevance_score: 0.75,
    reason_codes: ["USER_CONFIRMED", "TOPIC_MATCH"],
    turn_id: "turn_2",
  });
  render(
    <MemoryCenter ended={false} onClose={vi.fn()} open port={memoryPort} />,
  );

  await user.click(await screen.findByRole("button", { name: "为何使用" }));

  expect(memoryPort.getLatestMemoryRecall).toHaveBeenCalledWith(active.memory_id);
  expect(screen.getByText("你曾明确确认 · 与本轮话题相关")).toBeInTheDocument();
  expect(screen.getByText(/不代表 Agent 的回答一定由它造成/)).toBeInTheDocument();
});
