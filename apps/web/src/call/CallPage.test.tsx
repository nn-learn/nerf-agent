import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { CallPage } from "./CallPage";


it("does not publish camera before explicit consent", async () => {
  const user = userEvent.setup();
  const publishCamera = vi.fn().mockResolvedValue(undefined);
  render(
    <CallPage
      media={{
        connectionState: "connected",
        microphoneEnabled: true,
        visionState: "off",
        publishCamera,
        pauseVision: vi.fn(),
        toggleMicrophone: vi.fn(),
        interrupt: vi.fn(),
        sendText: vi.fn(),
        hangUp: vi.fn(),
      }}
    />,
  );

  expect(screen.getByText("视觉未开启")).toBeInTheDocument();
  expect(publishCamera).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "开启摄像头" }));
  expect(publishCamera).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "同意并开启" }));

  expect(publishCamera).toHaveBeenCalledTimes(1);
  expect(await screen.findByText("AI 正在看")).toBeInTheDocument();
});


it("keeps voice available when camera permission is denied", async () => {
  const user = userEvent.setup();
  const publishCamera = vi.fn().mockRejectedValue(
    new DOMException("Permission denied", "NotAllowedError"),
  );
  render(
    <CallPage
      media={{
        connectionState: "connected",
        microphoneEnabled: true,
        visionState: "off",
        publishCamera,
        pauseVision: vi.fn(),
        toggleMicrophone: vi.fn(),
        interrupt: vi.fn(),
        sendText: vi.fn(),
        hangUp: vi.fn(),
      }}
    />,
  );

  await user.click(screen.getByRole("button", { name: "开启摄像头" }));
  await user.click(screen.getByRole("button", { name: "同意并开启" }));

  expect(await screen.findByText(/仍可继续语音或文字交流/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "关闭麦克风" })).toBeEnabled();
  expect(screen.getByText("视觉未开启")).toBeInTheDocument();
});


it("immediately removes the watching indicator when vision is paused", async () => {
  const user = userEvent.setup();
  const pauseVision = vi.fn().mockResolvedValue(undefined);
  render(
    <CallPage
      media={{
        connectionState: "connected",
        microphoneEnabled: true,
        visionState: "active",
        publishCamera: vi.fn(),
        pauseVision,
        toggleMicrophone: vi.fn(),
        interrupt: vi.fn(),
        sendText: vi.fn(),
        hangUp: vi.fn(),
      }}
    />,
  );

  expect(screen.getByText("AI 正在看")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "暂停视觉" }));

  expect(pauseVision).toHaveBeenCalledTimes(1);
  expect(await screen.findByText("视觉未开启")).toBeInTheDocument();
});


it("renders live user and assistant captions with the active speaker", () => {
  const media = {
    connectionState: "connected" as const,
    microphoneEnabled: true,
    visionState: "off" as const,
    agentState: "listening" as const,
    userCaption: "最近工作压力很大",
    assistantCaption: "听起来你已经承受了一段时间。",
    providerLabel: "本地 qwen3.6:latest",
    publishCamera: vi.fn(),
    pauseVision: vi.fn(),
    toggleMicrophone: vi.fn(),
    interrupt: vi.fn(),
    sendText: vi.fn(),
    hangUp: vi.fn(),
  };
  const { rerender } = render(<CallPage media={media} />);

  expect(screen.getByText("你")).toBeInTheDocument();
  expect(screen.getByText("最近工作压力很大")).toBeInTheDocument();
  expect(screen.getByText("本地 qwen3.6:latest")).toBeInTheDocument();

  rerender(
    <CallPage
      media={{
        ...media,
        agentState: "thinking",
      }}
    />,
  );

  expect(
    screen.getByText("小澄", { selector: ".caption-speaker" }),
  ).toBeInTheDocument();
  expect(screen.getByText("听起来你已经承受了一段时间。")).toBeInTheDocument();
});


it("shows realtime errors as alerts without restoring the canned caption", () => {
  render(
    <CallPage
      media={{
        connectionState: "error",
        microphoneEnabled: false,
        visionState: "off",
        agentState: "listening",
        errorMessage:
          "语音连接出现问题，请检查本地服务后重试，或使用下方文字输入继续。",
        publishCamera: vi.fn(),
        pauseVision: vi.fn(),
        toggleMicrophone: vi.fn(),
        interrupt: vi.fn(),
        sendText: vi.fn(),
        hangUp: vi.fn(),
      }}
    />,
  );

  expect(screen.getByRole("alert")).toHaveTextContent("使用下方文字输入");
  expect(
    screen.queryByText(
      "我在这里。你可以慢慢说，我们先从此刻最困扰你的事情开始。",
    ),
  ).not.toBeInTheDocument();
});


it("keeps text fallback usable during recoverable voice-channel loss", async () => {
  const user = userEvent.setup();
  const sendText = vi.fn().mockResolvedValue(undefined);
  const { container } = render(
    <CallPage
      media={{
        connectionState: "error",
        microphoneEnabled: false,
        visionState: "off",
        agentState: "listening",
        errorMessage: "Voice unavailable; continue with text.",
        publishCamera: vi.fn(),
        pauseVision: vi.fn(),
        toggleMicrophone: vi.fn(),
        interrupt: vi.fn(),
        sendText,
        hangUp: vi.fn(),
      }}
    />,
  );

  expect(container.querySelector(".ended-overlay")).not.toBeInTheDocument();
  const input = screen.getByRole("textbox");
  expect(input).toBeEnabled();
  await user.type(input, "Please continue in text");
  await user.click(screen.getByRole("button", { name: "发送文字" }));

  expect(sendText).toHaveBeenCalledWith("Please continue in text");
});


it("submits a nonblank text fallback and disables it while thinking", async () => {
  const user = userEvent.setup();
  const sendText = vi.fn().mockResolvedValue(undefined);
  const media = {
    connectionState: "connected" as const,
    microphoneEnabled: false,
    visionState: "off" as const,
    agentState: "listening" as const,
    publishCamera: vi.fn(),
    pauseVision: vi.fn(),
    toggleMicrophone: vi.fn(),
    interrupt: vi.fn(),
    sendText,
    hangUp: vi.fn(),
  };
  const { rerender } = render(<CallPage media={media} />);
  const input = screen.getByRole("textbox", {
    name: "语音不可用时输入文字",
  });
  const submit = screen.getByRole("button", { name: "发送文字" });

  expect(submit).toBeDisabled();
  await user.type(input, "最近睡不好");
  expect(submit).toBeEnabled();
  await user.click(submit);

  expect(sendText).toHaveBeenCalledWith("最近睡不好");

  rerender(
    <CallPage
      media={{
        ...media,
        agentState: "thinking",
      }}
    />,
  );
  expect(screen.getByRole("textbox", {
    name: "语音不可用时输入文字",
  })).toBeDisabled();
  expect(screen.getByRole("button", { name: "回应中…" })).toBeDisabled();
});
