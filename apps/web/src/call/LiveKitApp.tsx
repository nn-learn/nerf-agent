import { CallPage } from "./CallPage";
import { useLiveKitMediaSession } from "./useLiveKitMediaSession";
import { useSessionStore } from "../state/sessionStore";


export default function LiveKitApp() {
  const sessionId = useSessionStore((state) => state.sessionId);
  const media = useLiveKitMediaSession(sessionId);
  return <CallPage demoMode={false} media={media} />;
}
