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
        hangUp: vi.fn(),
      }}
    />,
  );

  expect(screen.getByText("AI 正在看")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "暂停视觉" }));

  expect(pauseVision).toHaveBeenCalledTimes(1);
  expect(await screen.findByText("视觉未开启")).toBeInTheDocument();
});
