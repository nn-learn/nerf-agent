import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  Check,
  Clock3,
  FilePenLine,
  Info,
  RefreshCw,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-react";

import type {
  MemoryIngestionView,
  MemoryConflictView,
  MemoryChangeView,
  MemoryProfileView,
  MemoryRecallView,
  MemoryResearchConsentView,
  MemoryRetention,
  MemoryShadowReportView,
  MemoryView,
} from "../api/client";

interface MemoryCenterPort {
  listMemories: () => Promise<MemoryView[]>;
  getMemoryStatus: () => Promise<MemoryIngestionView>;
  retryMemoryIngestion: () => Promise<MemoryIngestionView>;
  decideMemory: (
    memoryId: string,
    decision: "confirm" | "reject",
  ) => Promise<MemoryView>;
  deleteMemory: (memoryId: string) => Promise<void>;
  updateMemory: (
    memoryId: string,
    text: string,
    retention: MemoryRetention,
  ) => Promise<MemoryView>;
  getLatestMemoryRecall: (memoryId: string) => Promise<MemoryRecallView | null>;
  getMemoryResearchConsent?: () => Promise<MemoryResearchConsentView>;
  setMemoryResearchConsent?: (
    granted: boolean,
    policyVersion: string,
  ) => Promise<MemoryResearchConsentView>;
  getMemoryShadowReport?: () => Promise<MemoryShadowReportView>;
  listMemoryProfiles?: () => Promise<MemoryProfileView[]>;
  decideMemoryProfile?: (
    profileId: string,
    decision: "confirm" | "reject",
  ) => Promise<MemoryProfileView | null>;
  listMemoryConflicts?: () => Promise<MemoryConflictView[]>;
  decideMemoryConflict?: (
    conflictId: string,
    decision: "select" | "dismiss",
    profileId?: string,
  ) => Promise<MemoryConflictView>;
  listMemoryChanges?: () => Promise<MemoryChangeView[]>;
  decideMemoryChange?: (
    changeId: string,
    decision: "apply" | "reject",
  ) => Promise<MemoryChangeView>;
}

interface MemoryCenterProps {
  ended: boolean;
  onClose: () => void;
  open: boolean;
  port: MemoryCenterPort;
}

const ASPECT_LABEL: Record<MemoryView["aspect"], string> = {
  FACT: "个人事实",
  PREFERENCE: "互动偏好",
  GOAL: "长期目标",
  COPING_STRATEGY: "有效方法",
  BOUNDARY: "个人边界",
};

const RECALL_REASON_LABEL: Record<string, string> = {
  USER_CONFIRMED: "你曾明确确认",
  TOPIC_MATCH: "与本轮话题相关",
  STABLE_PREFERENCE: "属于稳定偏好或边界",
  RECENTLY_UPDATED: "近期更新过",
};

const RETENTION_LABEL: Record<MemoryRetention, string> = {
  "7_days": "保留 7 天",
  "30_days": "保留 30 天",
  "90_days": "保留 90 天",
  forever: "长期保留",
};

function activeStatusText(status: MemoryIngestionView["state"]): string {
  if (status === "queued") return "已进入本地记忆队列";
  if (status === "processing") return "正在本地提炼候选记忆";
  if (status === "failed") return "本次提炼未完成";
  if (status === "pending") return "正在等待后台任务";
  return "候选记忆已整理完成";
}

export function MemoryCenter({ ended, onClose, open, port }: MemoryCenterProps) {
  const portRef = useRef(port);
  portRef.current = port;
  const [memories, setMemories] = useState<MemoryView[]>([]);
  const [status, setStatus] = useState<MemoryIngestionView>({
    state: "idle",
    retryable: false,
  });
  const [loading, setLoading] = useState(false);
  const [actionId, setActionId] = useState<string>();
  const [deleteConfirmId, setDeleteConfirmId] = useState<string>();
  const [editId, setEditId] = useState<string>();
  const [editText, setEditText] = useState("");
  const [retention, setRetention] = useState<MemoryRetention>("forever");
  const [recalls, setRecalls] = useState<Record<string, MemoryRecallView | null>>({});
  const [pollCycle, setPollCycle] = useState(0);
  const [error, setError] = useState<string>();
  const [researchConsent, setResearchConsent] =
    useState<MemoryResearchConsentView>();
  const [shadowReport, setShadowReport] = useState<MemoryShadowReportView>();
  const [researchBusy, setResearchBusy] = useState(false);
  const [profiles, setProfiles] = useState<MemoryProfileView[]>([]);
  const [conflicts, setConflicts] = useState<MemoryConflictView[]>([]);
  const [changes, setChanges] = useState<MemoryChangeView[]>([]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [
        nextStatus,
        nextMemories,
        nextConsent,
        nextReport,
        nextProfiles,
        nextConflicts,
        nextChanges,
      ] = await Promise.all([
        portRef.current.getMemoryStatus(),
        portRef.current.listMemories(),
        portRef.current.getMemoryResearchConsent?.() ?? Promise.resolve(undefined),
        portRef.current.getMemoryShadowReport?.() ?? Promise.resolve(undefined),
        portRef.current.listMemoryProfiles?.() ?? Promise.resolve([]),
        portRef.current.listMemoryConflicts?.() ?? Promise.resolve([]),
        portRef.current.listMemoryChanges?.() ?? Promise.resolve([]),
      ]);
      setStatus(nextStatus);
      setMemories(nextMemories);
      setResearchConsent(nextConsent);
      setShadowReport(nextReport);
      setProfiles(nextProfiles);
      setConflicts(nextConflicts);
      setChanges(nextChanges);
      setError(undefined);
      return nextStatus;
    } catch {
      setError("暂时无法读取记忆，请稍后重试。");
      return undefined;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;

    const poll = async () => {
      const next = await refresh();
      if (
        cancelled ||
        !ended ||
        next?.state === "complete" ||
        next?.state === "failed"
      ) return;
      if (attempts >= 90) {
        setError("后台提炼用时较长，你可以稍后重新打开记忆中心。");
        return;
      }
      attempts += 1;
      timer = setTimeout(() => void poll(), 2_000);
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [ended, open, pollCycle, refresh]);

  const decide = async (
    memoryId: string,
    decision: "confirm" | "reject",
  ) => {
    setActionId(memoryId);
    try {
      const changed = await port.decideMemory(memoryId, decision);
      setMemories((current) =>
        decision === "reject"
          ? current.filter((item) => item.memory_id !== memoryId)
          : current.map((item) =>
              item.memory_id === memoryId ? changed : item,
            ),
      );
      setError(undefined);
    } catch {
      setError("没有完成这次记忆操作，请重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const deleteMemory = async (memoryId: string) => {
    setActionId(memoryId);
    try {
      await port.deleteMemory(memoryId);
      setMemories((current) =>
        current.filter((item) => item.memory_id !== memoryId),
      );
      setDeleteConfirmId(undefined);
      setError(undefined);
    } catch {
      setError("没有完成永久删除，请重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const retry = async () => {
    setLoading(true);
    try {
      const next = await port.retryMemoryIngestion();
      setStatus(next);
      setError(undefined);
      setPollCycle((cycle) => cycle + 1);
    } catch {
      setError("暂时无法重新提炼，请确认本地 Ollama 服务可用。");
    } finally {
      setLoading(false);
    }
  };

  const beginEdit = (item: MemoryView) => {
    setEditId(item.memory_id);
    setEditText(item.text);
    setRetention("forever");
  };

  const saveEdit = async (memoryId: string) => {
    const text = editText.trim();
    if (!text) return;
    setActionId(memoryId);
    try {
      const changed = await port.updateMemory(memoryId, text, retention);
      setMemories((current) =>
        current.map((item) => item.memory_id === memoryId ? changed : item),
      );
      setEditId(undefined);
      setError(undefined);
    } catch {
      setError("修改没有保存。请移除身份信息、危险内容或指令性文字后重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const explainRecall = async (memoryId: string) => {
    setActionId(memoryId);
    try {
      const recall = await port.getLatestMemoryRecall(memoryId);
      setRecalls((current) => ({ ...current, [memoryId]: recall }));
      setError(undefined);
    } catch {
      setError("暂时无法读取这条记忆的使用说明。");
    } finally {
      setActionId(undefined);
    }
  };

  const toggleResearchConsent = async () => {
    if (
      !researchConsent?.enabled ||
      !port.setMemoryResearchConsent ||
      !port.getMemoryShadowReport
    ) return;
    setResearchBusy(true);
    try {
      const nextConsent = await port.setMemoryResearchConsent(
        !researchConsent.granted,
        researchConsent.policy_version,
      );
      setResearchConsent(nextConsent);
      setShadowReport(await port.getMemoryShadowReport());
      setError(undefined);
    } catch {
      setError("没有完成研究授权操作。若策略版本已更新，请重新打开记忆中心后阅读。 ");
    } finally {
      setResearchBusy(false);
    }
  };

  const decideProfile = async (
    profileId: string,
    decision: "confirm" | "reject",
  ) => {
    if (!port.decideMemoryProfile) return;
    setActionId(profileId);
    try {
      await port.decideMemoryProfile(profileId, decision);
      await refresh();
      setError(undefined);
    } catch {
      setError("没有完成这次长期画像操作，请重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const decideConflict = async (
    conflictId: string,
    decision: "select" | "dismiss",
    profileId?: string,
  ) => {
    if (!port.decideMemoryConflict) return;
    setActionId(conflictId);
    try {
      await port.decideMemoryConflict(conflictId, decision, profileId);
      await refresh();
      setError(undefined);
    } catch {
      setError("没有完成这次冲突确认，请重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const decideChange = async (
    changeId: string,
    decision: "apply" | "reject",
  ) => {
    if (!port.decideMemoryChange) return;
    setActionId(changeId);
    try {
      await port.decideMemoryChange(changeId, decision);
      await refresh();
      setError(undefined);
    } catch {
      setError("没有完成这次偏好变化确认，请重试。");
    } finally {
      setActionId(undefined);
    }
  };

  const retentionText = (item: MemoryView): string => {
    if (item.expires_at_ms == null) return "长期保留";
    return `保留至 ${new Date(item.expires_at_ms).toLocaleDateString("zh-CN")}`;
  };

  if (!open) return null;
  const pending = memories.filter((item) => item.state === "AWAITING_CONSENT");
  const saved = memories.filter((item) => item.state !== "AWAITING_CONSENT");
  const processing = ended && status.state !== "complete";

  return (
    <div className="memory-backdrop" role="presentation">
      <section
        aria-labelledby="memory-center-title"
        aria-modal="true"
        className="memory-center"
        role="dialog"
      >
        <header className="memory-header">
          <div>
            <span className="memory-kicker">YOUR CONTROL</span>
            <h2 id="memory-center-title">我的记忆中心</h2>
            <p>只有你亲自确认的内容，才会用于以后的陪伴。</p>
          </div>
          <button aria-label="关闭记忆中心" className="icon-button" onClick={onClose}>
            <X aria-hidden="true" />
          </button>
        </header>

        {ended && (
          <div aria-live="polite" className={`memory-job memory-job-${status.state}`}>
            {processing ? (
              <RefreshCw aria-hidden="true" className={loading ? "is-spinning" : ""} />
            ) : (
              <Check aria-hidden="true" />
            )}
            <div>
              <strong>{activeStatusText(status.state)}</strong>
              <span>原始音视频不会进入长期记忆。</span>
            </div>
            {status.retryable && (
              <button disabled={loading} onClick={() => void retry()} type="button">
                重试
              </button>
            )}
          </div>
        )}

        {error && <div className="memory-error" role="alert">{error}</div>}

        {researchConsent && (
          <section aria-label="记忆检索研究授权" className="memory-research-card">
            <div className="memory-research-heading">
              <div>
                <span className="memory-kicker">V1.6 · SHADOW EVALUATION</span>
                <h3>帮助改进记忆检索</h3>
              </div>
              <span className={researchConsent.granted ? "is-granted" : ""}>
                {researchConsent.granted ? "已授权" : "未授权"}
              </span>
            </div>
            <p>
              开启后，BGE Hybrid 只在后台比较检索排名；实时回答仍使用当前稳定策略，
              不会因实验结果改变本轮对话。
            </p>
            <div className="memory-research-privacy">
              <span>影子评测仅保存：去标识化会话/回合/记忆 ID、排名、分数、延迟与版本</span>
              <span>影子评测不另存：问题原文、记忆原文、音视频或 embedding</span>
            </div>
            {shadowReport && shadowReport.aggregate.run_count > 0 && (
              <>
                <div className="memory-shadow-metrics" aria-label="影子评测摘要">
                  <span>
                    <strong>{shadowReport.aggregate.completed_count}</strong>次完成
                  </span>
                  <span>
                    <strong>
                      {shadowReport.aggregate.mean_overlap_at_5 == null
                        ? "—"
                        : `${Math.round(shadowReport.aggregate.mean_overlap_at_5 * 100)}%`}
                    </strong>
                    Top-5 重合
                  </span>
                  <span>
                    <strong>
                      {shadowReport.aggregate.shadow_latency_ms_p95 == null
                        ? "—"
                        : `${Math.round(shadowReport.aggregate.shadow_latency_ms_p95)} ms`}
                    </strong>
                    Hybrid P95
                  </span>
                </div>
                {!shadowReport.aggregate.reliable && (
                  <small className="memory-shadow-warning">
                    当前少于 {shadowReport.aggregate.minimum_reliable_runs} 次完成运行，
                    仅用于观察系统行为，不能证明 Hybrid 更好。
                  </small>
                )}
              </>
            )}
            <div className="memory-research-action">
              <small>
                策略 {researchConsent.strategy_version}
                {shadowReport?.runtime?.model_state === "warming" && " · BGE 后台预热中"}
                {shadowReport?.runtime?.model_state === "failed" && " · BGE 初始化失败"}
              </small>
              <button
                className={researchConsent.granted ? "button button-secondary" : "button button-primary"}
                disabled={!researchConsent.enabled || researchBusy}
                onClick={() => void toggleResearchConsent()}
                type="button"
              >
                {researchConsent.enabled
                  ? researchConsent.granted
                    ? "撤回并清除全部影子记录"
                    : "阅读并同意参与"
                  : "当前部署未开启研究模式"}
              </button>
            </div>
          </section>
        )}

        {(changes.length > 0 || conflicts.length > 0 || profiles.length > 0) && (
          <section aria-label="长期画像治理" className="memory-profile-layer">
            <div className="memory-section-heading">
              <div>
                <h3>Agent 对你的长期理解</h3>
                <span>Memory 2.2</span>
              </div>
              <p>稳定画像需跨会话验证；明确表达“现在改成”时，会单独请你确认时间变更。</p>
            </div>

            {changes.map((change) => (
              <article className="memory-change-card" key={change.change_id}>
                <div className="memory-change-heading">
                  <Clock3 aria-hidden="true" size={18} />
                  <div>
                    <strong>系统识别到你的偏好可能发生了变化</strong>
                    <span>
                      有效时间 {new Date(change.effective_at_ms).toLocaleString("zh-CN")}
                      {change.observed_at_ms !== change.effective_at_ms &&
                        ` · 系统于 ${new Date(change.observed_at_ms).toLocaleString("zh-CN")} 记录`}
                    </span>
                  </div>
                </div>
                <div className="memory-change-flow">
                  <div>
                    <small>之前的理解</small>
                    <p>{change.previous_profile.statement}</p>
                  </div>
                  <span aria-hidden="true">→</span>
                  <div>
                    <small>新的理解</small>
                    <p>{change.proposed_profile.statement}</p>
                  </div>
                </div>
                <p className="memory-change-note">
                  确认后旧理解只保留为历史记录，Agent 以后只使用新的理解。
                </p>
                <div className="memory-card-actions">
                  <button
                    className="button button-primary"
                    disabled={actionId === change.change_id}
                    onClick={() => void decideChange(change.change_id, "apply")}
                    type="button"
                  >
                    <Check aria-hidden="true" size={15} />确认已经改变
                  </button>
                  <button
                    className="button button-secondary"
                    disabled={actionId === change.change_id}
                    onClick={() => void decideChange(change.change_id, "reject")}
                    type="button"
                  >
                    <X aria-hidden="true" size={15} />不是这个意思
                  </button>
                </div>
              </article>
            ))}

            {conflicts.map((conflict) => (
              <article className="memory-conflict-card" key={conflict.conflict_id}>
                <div className="memory-conflict-title">
                  <AlertTriangle aria-hidden="true" size={18} />
                  <div>
                    <strong>你在不同时间表达过不同倾向</strong>
                    <span>在你选择前，相关长期画像已暂停使用。</span>
                  </div>
                </div>
                <div className="memory-conflict-options">
                  {conflict.options.map((option) => (
                    <button
                      disabled={actionId === conflict.conflict_id}
                      key={option.profile_id}
                      onClick={() => void decideConflict(
                        conflict.conflict_id,
                        "select",
                        option.profile_id,
                      )}
                      type="button"
                    >
                      <strong>{option.statement}</strong>
                      <span>
                        {option.supporting_evidence_count} 条支持证据 · {option.distinct_session_count} 次会话
                      </span>
                    </button>
                  ))}
                </div>
                <button
                  className="memory-conflict-dismiss"
                  disabled={actionId === conflict.conflict_id}
                  onClick={() => void decideConflict(conflict.conflict_id, "dismiss")}
                  type="button"
                >
                  暂不判断
                </button>
              </article>
            ))}

            <div className="memory-profile-grid">
              {profiles.map((profile) => (
                <article className="memory-profile-card" key={profile.profile_id}>
                  <div className="memory-card-meta">
                    <span>{ASPECT_LABEL[profile.aspect]}</span>
                    <span>{profile.distinct_session_count} 次会话支持</span>
                    {profile.state === "ACTIVE" && <em>已由你确认</em>}
                    {profile.contains_sensitive_content && <em>敏感内容</em>}
                  </div>
                  <p>{profile.statement}</p>
                  <details>
                    <summary>查看形成这条理解的证据</summary>
                    <ul>
                      {profile.evidence.map((evidence, index) => (
                        <li key={`${evidence.memory_id}-${index}`}>
                          <span>{evidence.relation === "SUPPORTS" ? "支持" : "冲突"}</span>
                          {evidence.text}
                        </li>
                      ))}
                    </ul>
                  </details>
                  {profile.state === "AWAITING_CONFIRMATION" && (
                    <div className="memory-card-actions">
                      <button
                        className="button button-primary"
                        disabled={actionId === profile.profile_id}
                        onClick={() => void decideProfile(profile.profile_id, "confirm")}
                        type="button"
                      >
                        <Check aria-hidden="true" size={15} />确认这个理解
                      </button>
                      <button
                        className="button button-secondary"
                        disabled={actionId === profile.profile_id}
                        onClick={() => void decideProfile(profile.profile_id, "reject")}
                        type="button"
                      >
                        <X aria-hidden="true" size={15} />这不准确
                      </button>
                    </div>
                  )}
                </article>
              ))}
            </div>
          </section>
        )}

        <div className="memory-section-heading">
          <div>
            <h3>等待你的决定</h3>
            <span>{pending.length} 条候选</span>
          </div>
          <p>模型的建议可能出错，请核对后再确认。</p>
        </div>
        <div className="memory-list">
          {pending.map((item) => (
            <article className="memory-card memory-card-pending" key={item.memory_id}>
              <div className="memory-card-meta">
                <span>{ASPECT_LABEL[item.aspect]}</span>
                <span>{Math.round(item.confidence * 100)}% 置信度</span>
                {item.contains_sensitive_content && <em>敏感内容</em>}
              </div>
              <p>{item.text}</p>
              <div className="memory-card-actions">
                <button
                  className="button button-primary"
                  disabled={actionId === item.memory_id}
                  onClick={() => void decide(item.memory_id, "confirm")}
                  type="button"
                >
                  <Check aria-hidden="true" size={15} />确认记住
                </button>
                <button
                  className="button button-secondary"
                  disabled={actionId === item.memory_id}
                  onClick={() => void decide(item.memory_id, "reject")}
                  type="button"
                >
                  <X aria-hidden="true" size={15} />不保存
                </button>
              </div>
            </article>
          ))}
          {!pending.length && !processing && (
            <div className="memory-empty">
              <BrainCircuit aria-hidden="true" />
              <strong>没有等待确认的内容</strong>
              <span>系统不会为了“显得懂你”而强行制造记忆。</span>
            </div>
          )}
        </div>

        <div className="memory-section-heading memory-saved-heading">
          <div>
            <h3>已经允许使用</h3>
            <span>{saved.length} 条</span>
          </div>
          <p>这些内容可能在相关话题中帮助 Agent 个性化回应。</p>
        </div>
        <div className="memory-list">
          {saved.map((item) => (
            <article className="memory-card" key={item.memory_id}>
              <div className="memory-card-meta">
                <ShieldCheck aria-hidden="true" size={14} />
                <span>{ASPECT_LABEL[item.aspect]}</span>
                {item.state === "EXPIRED" && <em>已过期</em>}
                {item.user_edited && <em>由你修正</em>}
              </div>
              {editId === item.memory_id ? (
                <div className="memory-edit-form">
                  <label htmlFor={`memory-edit-${item.memory_id}`}>修正记忆内容</label>
                  <textarea
                    id={`memory-edit-${item.memory_id}`}
                    maxLength={500}
                    onChange={(event) => setEditText(event.target.value)}
                    value={editText}
                  />
                  <label htmlFor={`memory-retention-${item.memory_id}`}>保留期限</label>
                  <select
                    id={`memory-retention-${item.memory_id}`}
                    onChange={(event) =>
                      setRetention(event.target.value as MemoryRetention)
                    }
                    value={retention}
                  >
                    {Object.entries(RETENTION_LABEL).map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                  <div className="memory-card-actions">
                    <button
                      className="button button-primary"
                      disabled={!editText.trim() || actionId === item.memory_id}
                      onClick={() => void saveEdit(item.memory_id)}
                      type="button"
                    >保存修改</button>
                    <button
                      className="button button-secondary"
                      onClick={() => setEditId(undefined)}
                      type="button"
                    >取消</button>
                  </div>
                </div>
              ) : (
                <p>{item.text}</p>
              )}
              {Object.prototype.hasOwnProperty.call(recalls, item.memory_id) && (
                <div className="memory-recall-explanation" role="status">
                  <Info aria-hidden="true" size={14} />
                  {recalls[item.memory_id] ? (
                    <div>
                      <strong>最近一次进入 Agent 上下文</strong>
                      <span>
                        {recalls[item.memory_id]!.reason_codes
                          .map((code) => RECALL_REASON_LABEL[code] ?? code)
                          .join(" · ")}
                      </span>
                      <small>
                        召回分数 {Math.round(recalls[item.memory_id]!.score * 100)}%；
                        这不代表 Agent 的回答一定由它造成。
                      </small>
                    </div>
                  ) : (
                    <span>还没有进入过后续会话的 Agent 上下文。</span>
                  )}
                </div>
              )}
              <div className="memory-card-footer">
                <span><Clock3 aria-hidden="true" size={13} />{retentionText(item)}</span>
                <span className="memory-secondary-actions">
                  <button
                    disabled={actionId === item.memory_id}
                    onClick={() => void explainRecall(item.memory_id)}
                    type="button"
                  ><Info aria-hidden="true" size={14} />为何使用</button>
                  <button onClick={() => beginEdit(item)} type="button">
                    <FilePenLine aria-hidden="true" size={14} />编辑
                  </button>
                </span>
                {deleteConfirmId === item.memory_id ? (
                  <span className="memory-delete-confirm">
                    <button
                      aria-label={`确认永久删除：${item.text}`}
                      disabled={actionId === item.memory_id}
                      onClick={() => void deleteMemory(item.memory_id)}
                      type="button"
                    >
                      <Trash2 aria-hidden="true" size={14} />确认永久删除
                    </button>
                    <button
                      onClick={() => setDeleteConfirmId(undefined)}
                      type="button"
                    >
                      取消
                    </button>
                  </span>
                ) : (
                  <button
                    aria-label={`永久删除：${item.text}`}
                    disabled={actionId === item.memory_id}
                    onClick={() => setDeleteConfirmId(item.memory_id)}
                    type="button"
                  >
                    <Trash2 aria-hidden="true" size={14} />永久删除
                  </button>
                )}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
