import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import { CallPage } from "./call/CallPage";
import { useMockMediaSession } from "./call/useMediaSession";
import "./styles.css";

const LiveKitApp = lazy(() => import("./call/LiveKitApp"));
const ClinicianPage = lazy(() => import("./clinician/ClinicianPage"));

function MockApp() {
  const media = useMockMediaSession();
  return <CallPage demoMode media={media} />;
}

function App() {
  const search = new URLSearchParams(window.location.search);
  if (search.get("view") === "clinician") {
    const sessionId = search.get("session");
    return (
      <Suspense fallback={<div className="boot-screen">正在加载接管台…</div>}>
        {sessionId ? (
          <ClinicianPage sessionId={sessionId} />
        ) : (
          <div className="boot-screen">缺少 session 参数</div>
        )}
      </Suspense>
    );
  }
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
