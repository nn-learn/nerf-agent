import { vi } from "vitest";

import { pauseVisionTransaction } from "./useMediaSession";


it("stops camera before changing UI and revoking server consent", async () => {
  const order: string[] = [];
  const stopCamera = vi.fn(async () => {
    order.push("camera_stopped");
  });
  const markPaused = vi.fn(() => {
    order.push("ui_paused");
  });
  const revokeConsent = vi.fn(async () => {
    order.push("server_revoked");
  });

  await pauseVisionTransaction({
    stopCamera,
    markPaused,
    revokeConsent,
  });

  expect(order).toEqual(["camera_stopped", "ui_paused", "server_revoked"]);
});
