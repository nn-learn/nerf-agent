import { expect, test } from "@playwright/test";


test("camera remains off until the second consent action", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("视觉未开启")).toBeVisible();
  await page.getByRole("button", { name: "开启摄像头" }).click();
  await expect(page.getByText("开启视觉理解")).toBeVisible();
  await expect(page.locator("video.self-preview-video")).toHaveCount(0);
});
