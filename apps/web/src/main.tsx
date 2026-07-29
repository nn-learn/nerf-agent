import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import { CallPage } from "./call/CallPage";
import { useMockMediaSession } from "./call/useMediaSession";
import "./styles.css";

const LiveKitApp = lazy(() => import("./call/LiveKitApp"));

function MockApp() {
  const media = useMockMediaSession();
  return <CallPage demoMode media={media} />;
}

function App() {
  return import.meta.env.VITE_MEDIA_MODE === "livekit" ? (
    <Suspense fallback={<div className="boot-screen">正在连接实时房间…</div>}>
      <LiveKitApp />
    </Suspense>
  ) : (
    <MockApp />
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
