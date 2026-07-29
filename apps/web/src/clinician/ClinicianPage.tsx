import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowLeft,
  EyeOff,
  ShieldCheck,
  UserCheck,
} from "lucide-react";

import {
  type ClinicianSummary,
  type ClinicianTimeline,
  type HandoffRecord,
  PsyAvatarApi,
} from "../api/client";

interface ClinicianApi {
  getClinicianSummary: (sessionId: string) => Promise<ClinicianSummary>;
  getClinicianTimeline: (sessionId: string) => Promise<ClinicianTimeline>;
  acceptHandoff: (
    sessionId: string,
    handoffId: string,
  ) => Promise<HandoffRecord>;
}

interface ClinicianPageProps {
  sessionId: string;
  api?: ClinicianApi;
}


function formatTime(timestampMs: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestampMs));
}


export function ClinicianPage({
  sessionId,
  api: suppliedApi,
}: ClinicianPageProps) {
  const api = useMemo(
    () => suppliedApi ?? new PsyAvatarApi(),
    [suppliedApi],
  );
  const [summary, setSummary] = useState<ClinicianSummary>();
  const [timeline, setTimeline] = useState<ClinicianTimeline>();
  const [error, setError] = useState<string>();
  const [accepting, setAccepting] = useState(false);
  const [accepted, setAccepted] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      api.getClinicianSummary(sessionId),
      api.getClinicianTimeline(sessionId),
    ])
      .then(([nextSummary, nextTimeline]) => {
        if (cancelled) return;
        setSummary(nextSummary);
        setTimeline(nextTimeline);
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setError(
          reason instanceof Error ? reason.message : "无法加载演示接管信息",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [api, sessionId]);

  const accept = async () => {
    if (!summary?.handoff_id) return;
    setAccepting(true);
    try {
      await api.acceptHandoff(sessionId, summary.handoff_id);
      setAccepted(true);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "接管操作失败",
      );
    } finally {
      setAccepting(false);
    }
  };

  if (error) {
    return (
      <main className="clinician-shell clinician-center">
        <ShieldCheck aria-hidden="true" />
        <h1>临床演示台暂不可用</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!summary || !timeline) {
    return (
      <main className="clinician-shell clinician-center">
        <Activity aria-hidden="true" className="clinician-pulse" />
        <p>正在加载脱敏会话摘要…</p>
      </main>
    );
  }

  return (
    <main className="clinician-shell">
      <header className="clinician-header">
        <a href="/" className="clinician-back">
          <ArrowLeft aria-hidden="true" size={18} />
          返回数字人通话
        </a>
        <span className="clinician-role">
          <ShieldCheck aria-hidden="true" size={16} />
          CLINICIAN DEMO · 仅当前会话
        </span>
      </header>

      <section className="clinician-hero">
        <div>
          <p className="eyebrow">HUMAN HANDOFF</p>
          <h1>脱敏人工接管台</h1>
          <p className="clinician-session">{summary.session_id}</p>
        </div>
        <div className={`risk-card risk-${summary.risk_level.toLowerCase()}`}>
          <small>当前确定性风险分级</small>
          <strong>{summary.risk_level}</strong>
          <span>{summary.reasons.join(" · ") || "暂无升级原因"}</span>
        </div>
      </section>

      <div className="clinician-grid">
        <section className="clinician-panel">
          <div className="clinician-panel-title">
            <div>
              <p className="eyebrow">KEY TURNS</p>
              <h2>关键回合摘要</h2>
            </div>
            <EyeOff aria-label="内容已脱敏" size={20} />
          </div>
          <p className="redaction-note">
            仅显示短摘录；不提供原始音频、视频帧或完整逐字稿。
          </p>
          <div className="key-turn-list">
            {summary.key_turns.map((turn) => (
              <article key={turn.turn_id} className="key-turn">
                <div>
                  <span>{turn.risk_level ?? "UNKNOWN"}</span>
                  <small>{turn.support_mode ?? "support"}</small>
                </div>
                <p>{turn.transcript_excerpt}</p>
              </article>
            ))}
          </div>

          {accepted ? (
            <div className="handoff-accepted">
              <UserCheck aria-hidden="true" size={19} />
              已由演示临床角色接管
            </div>
          ) : summary.handoff_state === "REQUESTED" &&
            summary.handoff_id ? (
            <button
              className="button button-primary handoff-button"
              disabled={accepting}
              onClick={() => void accept()}
              type="button"
            >
              {accepting ? "正在记录接管…" : "接受演示接管"}
            </button>
          ) : (
            <div className="handoff-idle">当前没有待接管请求</div>
          )}
        </section>

        <section className="clinician-panel">
          <div className="clinician-panel-title">
            <div>
              <p className="eyebrow">AUDIT TIMELINE</p>
              <h2>不可变事件时间线</h2>
            </div>
            <Activity aria-hidden="true" size={20} />
          </div>
          <div className="audit-timeline">
            {timeline.events.map((event) => (
              <article key={event.seq} className="audit-row">
                <span className="audit-dot" />
                <div>
                  <strong>{event.type}</strong>
                  <small>
                    #{event.seq} · {formatTime(event.timestamp_ms)}
                  </small>
                </div>
                <code>{event.turn_id.replace("turn_", "").slice(0, 8)}</code>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}


export default ClinicianPage;
