import {
  type CSSProperties,
  type FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  BookOpenText,
  BrainCircuit,
  ChevronRight,
  Clock3,
  HeartHandshake,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { LocalVideoTrack, RemoteVideoTrack } from "livekit-client";

import type { AvatarPlan } from "../realtime/protocol";

import { CallControls } from "./CallControls";
import { CameraConsent } from "./CameraConsent";
import { MemoryCenter } from "./MemoryCenter";
import { StatusOverlay } from "./StatusOverlay";
import type { MediaSessionController } from "./useMediaSession";

interface CallPageProps {
  media: MediaSessionController;
  demoMode?: boolean;
}

function LiveKitVideo({
  track,
  className,
  muted = false,
}: {
  track: LocalVideoTrack | RemoteVideoTrack;
  className: string;
  muted?: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    track.attach(element);
    return () => {
      track.detach(element);
    };
  }, [track]);
  return <video autoPlay className={className} muted={muted} playsInline ref={ref} />;
}

function PreviewStream({ stream }: { stream: MediaStream }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream;
    return () => {
      if (ref.current) ref.current.srcObject = null;
    };
  }, [stream]);
  return <video autoPlay className="self-preview-video" muted playsInline ref={ref} />;
}

function MockAvatar({
  plan,
  speaking,
}: {
  plan?: AvatarPlan;
  speaking: boolean;
}) {
  const style = plan?.style ?? "neutral_listening";
  const speechCycle = plan ? 0.42 / plan.speech_rate : 0.42;
  return (
    <div
      aria-label="数字人小澄演示形象"
      className={`mock-avatar avatar-style-${style} ${speaking ? "is-speaking" : ""}`}
      data-avatar-style={style}
      data-gesture={plan?.gesture_intensity ?? "NONE"}
      role="img"
      style={{ "--speech-cycle": `${speechCycle}s` } as CSSProperties}
    >
      <div className="avatar-ambient avatar-ambient-one" />
      <div className="avatar-ambient avatar-ambient-two" />
      <div className="avatar-halo" />
      <svg
        aria-hidden="true"
        className="avatar-portrait"
        viewBox="0 0 440 560"
      >
        <defs>
          <linearGradient id="skin" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#f5c7aa" />
            <stop offset="1" stopColor="#d99783" />
          </linearGradient>
          <linearGradient id="shirt" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0" stopColor="#2f746d" />
            <stop offset="1" stopColor="#163c43" />
          </linearGradient>
          <filter id="soft">
            <feGaussianBlur stdDeviation="8" />
          </filter>
        </defs>
        <ellipse
          cx="220"
          cy="530"
          fill="#c7a270"
          filter="url(#soft)"
          opacity=".22"
          rx="175"
          ry="24"
        />
        <path
          d="M68 560c6-112 57-167 152-167s146 55 152 167Z"
          fill="url(#shirt)"
        />
        <path d="m168 391 52 57 53-57v-67H168Z" fill="url(#skin)" />
        <ellipse cx="220" cy="240" fill="url(#skin)" rx="112" ry="142" />
        <path
          d="M110 234c-11-99 29-164 112-168 88-4 131 62 113 176-20-35-31-75-38-116-51 34-107 50-169 49-4 24-10 44-18 59Z"
          fill="#25343a"
        />
        <path
          d="M119 159c25-77 122-113 187-54-47-13-87-6-120 20-23 17-46 28-67 34Z"
          fill="#354b4d"
          opacity=".9"
        />
        <ellipse cx="175" cy="247" fill="#2e3030" rx="10" ry="7" />
        <ellipse cx="267" cy="247" fill="#2e3030" rx="10" ry="7" />
        <path
          d="M157 222c13-9 27-11 42-4M244 218c15-7 29-5 41 4"
          fill="none"
          stroke="#4c3836"
          strokeLinecap="round"
          strokeWidth="6"
        />
        <path
          d="M218 251c-4 24-5 38 8 42"
          fill="none"
          stroke="#b87467"
          strokeLinecap="round"
          strokeWidth="5"
        />
        <path
          className="avatar-mouth"
          d="M184 319c24 18 49 18 73 0"
          fill="none"
          stroke="#9b4b50"
          strokeLinecap="round"
          strokeWidth="7"
        />
        <circle cx="302" cy="275" fill="#f2ad95" opacity=".25" r="27" />
        <circle cx="142" cy="275" fill="#f2ad95" opacity=".2" r="25" />
      </svg>
      <div className="voice-wave" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
        <i />
      </div>
    </div>
  );
}


export function CallPage({ media, demoMode = true }: CallPageProps) {
  const [consentOpen, setConsentOpen] = useState(false);
  const [consentBusy, setConsentBusy] = useState(false);
  const [consentError, setConsentError] = useState<string>();
  const [visionOverride, setVisionOverride] = useState<boolean | null>(null);
  const [textInput, setTextInput] = useState("");
  const [textBusy, setTextBusy] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const cameraActive =
    visionOverride ?? media.visionState === "active";
  const callEnded = media.connectionState === "disconnected";
  const textThinking = media.agentState === "thinking";
  const userIsSpeaking =
    media.agentState === "listening" && Boolean(media.userCaption);
  const captionText = userIsSpeaking
    ? media.userCaption
    : media.assistantCaption ??
      (media.errorMessage
        ? "语音暂时不可用，你可以使用文字输入继续。"
        : "我在这里。你可以慢慢说，我们先从此刻最困扰你的事情开始。");
  const memoryPort = media.listMemories &&
    media.getMemoryStatus &&
    media.retryMemoryIngestion &&
    media.decideMemory &&
    media.deleteMemory &&
    media.updateMemory &&
    media.getLatestMemoryRecall
    ? {
        listMemories: media.listMemories,
        getMemoryStatus: media.getMemoryStatus,
        retryMemoryIngestion: media.retryMemoryIngestion,
        decideMemory: media.decideMemory,
        deleteMemory: media.deleteMemory,
        updateMemory: media.updateMemory,
        getLatestMemoryRecall: media.getLatestMemoryRecall,
        getMemoryResearchConsent: media.getMemoryResearchConsent,
        setMemoryResearchConsent: media.setMemoryResearchConsent,
        getMemoryShadowReport: media.getMemoryShadowReport,
        listMemoryProfiles: media.listMemoryProfiles,
        decideMemoryProfile: media.decideMemoryProfile,
        listMemoryConflicts: media.listMemoryConflicts,
        decideMemoryConflict: media.decideMemoryConflict,
        listMemoryChanges: media.listMemoryChanges,
        decideMemoryChange: media.decideMemoryChange,
      }
    : undefined;

  const confirmCamera = async () => {
    setConsentBusy(true);
    setConsentError(undefined);
    try {
      await media.publishCamera();
      setVisionOverride(true);
      setConsentOpen(false);
    } catch {
      setConsentError("未能开启摄像头。你仍可继续语音或文字交流。");
    } finally {
      setConsentBusy(false);
    }
  };

  const toggleCamera = async () => {
    if (!cameraActive) {
      setConsentOpen(true);
      return;
    }
    await media.pauseVision();
    setVisionOverride(false);
  };

  const submitText = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = textInput.trim();
    if (!text || textBusy || textThinking) return;
    setTextBusy(true);
    try {
      await media.sendText(text);
      setTextInput("");
    } finally {
      setTextBusy(false);
    }
  };

  return (
    <main className="call-page">
      <header className="app-header">
        <a className="brand" href="/" aria-label="PsyAvatar Care 首页">
          <span className="brand-mark">
            <HeartHandshake aria-hidden="true" size={21} />
          </span>
          <span>
            <strong>LUMEN</strong>
            <small>数字人心理支持</small>
          </span>
        </a>
        <div className="header-meta">
          {(media.providerLabel || demoMode) && (
            <span className="demo-chip">
              {media.providerLabel ?? "LOCAL DEMO"}
            </span>
          )}
          <span className="session-clock">
            <Clock3 aria-hidden="true" size={15} />
            12:08
          </span>
          {memoryPort && (
            <button
              className="memory-header-button"
              onClick={() => setMemoryOpen(true)}
              type="button"
            >
              <BrainCircuit aria-hidden="true" size={15} />
              我的记忆
            </button>
          )}
        </div>
      </header>

      <div className="call-layout">
        <section className="video-stage" aria-label="数字人通话画面">
          <StatusOverlay
            agentState={media.agentState}
            connectionState={media.connectionState}
            visionActive={cameraActive}
          />

          {media.avatarTrack ? (
            <LiveKitVideo
              className="avatar-video"
              track={media.avatarTrack}
            />
          ) : (
            <MockAvatar
              plan={media.avatarPlan}
              speaking={media.agentState === "speaking"}
            />
          )}

          <div className="avatar-identity">
            <div className="identity-dot" />
            <div>
              <strong>小澄</strong>
              <span>AI 心理支持伙伴</span>
            </div>
          </div>

          {cameraActive && (media.localCameraTrack || media.localPreviewStream) && (
            <div className="self-preview">
              {media.localCameraTrack ? (
                <LiveKitVideo
                  className="self-preview-video"
                  muted
                  track={media.localCameraTrack}
                />
              ) : media.localPreviewStream ? (
                <PreviewStream stream={media.localPreviewStream} />
              ) : null}
              <span>你 · 仅本次会话</span>
            </div>
          )}

          <div className="live-caption" aria-live="polite">
            <span className="caption-speaker">
              {userIsSpeaking ? "你" : "小澄"}
            </span>
            <p>{captionText}</p>
          </div>

          {callEnded && (
            <div className="ended-overlay">
              <Sparkles aria-hidden="true" />
              <h2>本次陪伴已结束</h2>
              <p>感谢你照顾自己的感受。会话中的原始音视频没有被保存。</p>
              {memoryPort && (
                <button
                  className="button button-secondary"
                  onClick={() => setMemoryOpen(true)}
                  type="button"
                >
                  审核本次候选记忆
                </button>
              )}
              <button
                className="button button-primary"
                onClick={() => window.location.reload()}
                type="button"
              >
                开始新会话
              </button>
            </div>
          )}
        </section>

        <aside className="support-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">CARE PATH · 01</p>
              <h1>此刻，先陪你稳下来</h1>
            </div>
            <BrainCircuit aria-hidden="true" size={25} />
          </div>
          <p className="panel-intro">
            这是一段由 AI 提供的非诊断性心理支持。你决定说什么、展示什么，以及何时停止。
          </p>

          {media.errorMessage && (
            <div className="realtime-error" role="alert">
              {media.errorMessage}
            </div>
          )}

          <form
            className="text-fallback"
            onSubmit={(event) => void submitText(event)}
          >
            <label htmlFor="text-fallback-input">
              语音不可用时输入文字
            </label>
            <div>
              <input
                disabled={callEnded || textBusy || textThinking}
                id="text-fallback-input"
                maxLength={8000}
                onChange={(event) => setTextInput(event.target.value)}
                placeholder="写下此刻最想说的事"
                type="text"
                value={textInput}
              />
              <button
                className="button button-primary"
                disabled={
                  callEnded ||
                  textBusy ||
                  textThinking ||
                  !textInput.trim()
                }
                type="submit"
              >
                {textThinking || textBusy ? "回应中…" : "发送文字"}
              </button>
            </div>
          </form>

          <div className="conversation-rail">
            <article className="rail-card is-current">
              <span className="rail-index">01</span>
              <div>
                <strong>倾听与澄清</strong>
                <p>理解你此刻的处境，不急着给结论。</p>
              </div>
              <span className="rail-state">进行中</span>
            </article>
            <article className="rail-card">
              <span className="rail-index">02</span>
              <div>
                <strong>找到可行动的一小步</strong>
                <p>只使用经评审的心理教育材料。</p>
              </div>
            </article>
            <article className="rail-card">
              <span className="rail-index">03</span>
              <div>
                <strong>需要时转接真人</strong>
                <p>高风险场景进入明确的人工接管流程。</p>
              </div>
            </article>
          </div>

          <button className="resource-card" type="button">
            <span className="resource-icon">
              <BookOpenText aria-hidden="true" size={20} />
            </span>
            <span>
              <small>本轮可用工具</small>
              <strong>60 秒节律呼吸</strong>
            </span>
            <ChevronRight aria-hidden="true" size={20} />
          </button>

          <div className="safety-note">
            <ShieldCheck aria-hidden="true" size={18} />
            <p>
              <strong>安全边界</strong>
              不做诊断、不开药，也不会声称已联系急救服务。
            </p>
          </div>
        </aside>
      </div>

      <CallControls
        cameraActive={cameraActive}
        disabled={callEnded}
        microphoneEnabled={media.microphoneEnabled}
        onCamera={() => void toggleCamera()}
        onHangUp={() => void media.hangUp()}
        onInterrupt={() => void media.interrupt()}
        onMicrophone={() => void media.toggleMicrophone()}
      />

      <CameraConsent
        busy={consentBusy}
        error={consentError}
        onClose={() => setConsentOpen(false)}
        onConfirm={() => void confirmCamera()}
        open={consentOpen}
      />
      {memoryPort && (
        <MemoryCenter
          ended={callEnded}
          onClose={() => setMemoryOpen(false)}
          open={memoryOpen}
          port={memoryPort}
        />
      )}
    </main>
  );
}
