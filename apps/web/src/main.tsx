import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import { CallPage } from "./call/CallPage";
import { useLocalRealtimeMediaSession } from "./call/useLocalRealtimeMediaSession";
import { useMockMediaSession } from "./call/useMediaSession";
import { useSessionStore } from "./state/sessionStore";
import "./styles.css";

const LiveKitApp = lazy(() => import("./call/LiveKitApp"));
const ClinicianPage = lazy(() => import("./clinician/ClinicianPage"));

function MockApp() {
  const media = useMockMediaSession();
  return <CallPage demoMode media={media} />;
}

function LocalApp() {
  const sessionId = useSessionStore((state) => state.sessionId);
  const media = useLocalRealtimeMediaSession(sessionId);
  return <CallPage demoMode={false} media={media} />;
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
  if (import.meta.env.VITE_MEDIA_MODE === "local") {
    return <LocalApp />;
  }
  if (import.meta.env.VITE_MEDIA_MODE === "livekit") {
    return (
      <Suspense fallback={<div className="boot-screen">正在连接实时房间…</div>}>
        <LiveKitApp />
      </Suspense>
    );
  }
  return <MockApp />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
