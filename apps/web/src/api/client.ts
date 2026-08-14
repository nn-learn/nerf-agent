import { LocalRealtimeClient } from "../realtime/LocalRealtimeClient";

export interface LiveKitCredentials {
  server_url: string;
  room_name: string;
  participant_token: string;
}

export interface SessionDescriptor {
  session_id: string;
  status: "active" | "ended";
  provider_mode: string;
  camera_consent: boolean;
  fallback_capabilities: string[];
}

export interface SessionCreated extends SessionDescriptor {
  access_token: string;
  memory_subject_token?: string | null;
}

export interface TurnResult {
  status: "completed" | "interrupted";
  risk_level?: string;
  response?: {
    spoken_text: string;
    display_text: string;
    support_mode: string;
  };
  delivery_mode: string;
}

export interface ClinicianKeyTurn {
  turn_id: string;
  transcript_excerpt: string;
  risk_level?: string;
  support_mode?: string;
}

export interface ClinicianSummary {
  session_id: string;
  risk_level: string;
  reasons: string[];
  key_turns: ClinicianKeyTurn[];
  visual_context_used: boolean;
  visual_summaries: string[];
  handoff_state: string;
  handoff_id?: string;
  safety_policy: string;
}

export interface ClinicianTimelineItem {
  seq: number;
  type: string;
  timestamp_ms: number;
  turn_id: string;
  trace_id: string;
  metadata: Record<string, unknown>;
}

export interface ClinicianTimeline {
  session_id: string;
  events: ClinicianTimelineItem[];
}

export interface HandoffRecord {
  handoff_id: string;
  state: string;
}

export interface MemoryView {
  memory_id: string;
  text: string;
  aspect: "FACT" | "PREFERENCE" | "GOAL" | "COPING_STRATEGY" | "BOUNDARY";
  state:
    | "CANDIDATE"
    | "AWAITING_CONSENT"
    | "ACTIVE"
    | "REVOKED"
    | "EXPIRED"
    | "REJECTED"
    | "SUPERSEDED"
    | "QUARANTINED";
  contains_sensitive_content: boolean;
  confidence: number;
  purpose_scope: string;
  created_at_ms: number;
  updated_at_ms: number;
  source_turn_id: string;
  expires_at_ms?: number | null;
  user_edited?: boolean;
}

export type MemoryRetention = "7_days" | "30_days" | "90_days" | "forever";

export interface MemoryRecallView {
  used_at_ms: number;
  score: number;
  relevance_score: number;
  reason_codes: string[];
  turn_id: string;
}

export interface MemoryIngestionView {
  state: "idle" | "queued" | "processing" | "complete" | "failed" | "pending";
  retryable: boolean;
}

export interface MemoryResearchConsentView {
  enabled: boolean;
  granted: boolean;
  policy_version: string;
  strategy_version: string;
  updated_at_ms?: number | null;
  retained_fields: string[];
  excluded_fields: string[];
}

export interface MemoryShadowAggregate {
  run_count: number;
  completed_count: number;
  failed_count: number;
  timeout_count: number;
  circuit_open_count: number;
  completion_rate: number;
  reliable: boolean;
  minimum_reliable_runs: number;
  warnings: string[];
  mean_overlap_at_5?: number | null;
  mean_rank_biased_overlap?: number | null;
  baseline_latency_ms_p50?: number | null;
  baseline_latency_ms_p95?: number | null;
  shadow_latency_ms_p50?: number | null;
  shadow_latency_ms_p95?: number | null;
  strategy_versions: Record<string, number>;
}

export interface MemoryShadowReportView {
  enabled: boolean;
  consent_granted: boolean;
  aggregate: MemoryShadowAggregate;
  runtime?: {
    policy_version: string;
    strategy_version: string;
    model_state: string;
    circuit_open: boolean;
    pending_count: number;
    max_concurrency: number;
    max_pending: number;
  } | null;
}

export interface MemoryProfileEvidenceView {
  memory_id: string;
  text: string;
  relation: "SUPPORTS" | "CONTRADICTS";
  valid_at_ms: number;
  observed_at_ms: number;
}

export interface MemoryProfileView {
  profile_id: string;
  subject_key: string;
  aspect: MemoryView["aspect"];
  statement: string;
  state: "AWAITING_CONFIRMATION" | "ACTIVE" | "STALE" | "HISTORICAL" | "REJECTED";
  confidence: number;
  evidence_count: number;
  supporting_evidence_count: number;
  conflicting_evidence_count: number;
  distinct_session_count: number;
  contains_sensitive_content: boolean;
  valid_from_ms?: number | null;
  valid_to_ms?: number | null;
  expires_at_ms?: number | null;
  created_at_ms: number;
  updated_at_ms: number;
  user_edited: boolean;
  evidence: MemoryProfileEvidenceView[];
}

export interface MemoryConflictView {
  conflict_id: string;
  subject_key: string;
  state: "OPEN" | "RESOLVED" | "DISMISSED";
  selected_profile_id?: string | null;
  created_at_ms: number;
  updated_at_ms: number;
  options: MemoryProfileView[];
}

export interface MemoryChangeView {
  change_id: string;
  subject_key: string;
  state: "OPEN" | "APPLIED" | "REJECTED";
  effective_at_ms: number;
  observed_at_ms: number;
  created_at_ms: number;
  updated_at_ms: number;
  previous_profile: MemoryProfileView;
  proposed_profile: MemoryProfileView;
}

const DEFAULT_API_BASE = "http://localhost:8000";
const MEMORY_SUBJECT_STORAGE_KEY = "psyavatar.memorySubjectToken.v1";


export class PsyAvatarApi {
  private accessToken?: string;
  private readonly sessionCreations = new Map<
    string,
    Promise<SessionCreated>
  >();

  constructor(
    private readonly baseUrl =
      import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE,
  ) {}

  createSession(sessionId: string): Promise<SessionCreated> {
    const existing = this.sessionCreations.get(sessionId);
    if (existing) return existing;
    const memorySubjectToken = this.readMemorySubjectToken();
    const creation = this.request<SessionCreated>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        camera_consent: false,
        client_session_id: sessionId,
        ...(memorySubjectToken
          ? { memory_subject_token: memorySubjectToken }
          : {}),
      }),
    })
      .then((session) => {
        this.accessToken = session.access_token;
        if (session.memory_subject_token) {
          this.writeMemorySubjectToken(session.memory_subject_token);
        }
        return session;
      })
      .catch((error: unknown) => {
        this.sessionCreations.delete(sessionId);
        throw error;
      });
    this.sessionCreations.set(sessionId, creation);
    return creation;
  }

  async endSession(sessionId: string): Promise<SessionDescriptor> {
    return this.request<SessionDescriptor>(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    });
  }

  async listMemories(sessionId: string): Promise<MemoryView[]> {
    return this.request<MemoryView[]>(`/api/sessions/${sessionId}/memories`, {
      method: "GET",
    });
  }

  async listMemoryProfiles(sessionId: string): Promise<MemoryProfileView[]> {
    return this.request<MemoryProfileView[]>(
      `/api/sessions/${sessionId}/memory-profiles`,
      { method: "GET" },
    );
  }

  async decideMemoryProfile(
    sessionId: string,
    profileId: string,
    decision: "confirm" | "reject",
  ): Promise<MemoryProfileView | null> {
    return this.request<MemoryProfileView | null>(
      `/api/sessions/${sessionId}/memory-profiles/${profileId}/decision`,
      { method: "POST", body: JSON.stringify({ decision }) },
    );
  }

  async listMemoryConflicts(sessionId: string): Promise<MemoryConflictView[]> {
    return this.request<MemoryConflictView[]>(
      `/api/sessions/${sessionId}/memory-conflicts`,
      { method: "GET" },
    );
  }

  async decideMemoryConflict(
    sessionId: string,
    conflictId: string,
    decision: "select" | "dismiss",
    profileId?: string,
  ): Promise<MemoryConflictView> {
    return this.request<MemoryConflictView>(
      `/api/sessions/${sessionId}/memory-conflicts/${conflictId}/decision`,
      {
        method: "POST",
        body: JSON.stringify({
          decision,
          ...(profileId ? { profile_id: profileId } : {}),
        }),
      },
    );
  }

  async listMemoryChanges(sessionId: string): Promise<MemoryChangeView[]> {
    return this.request<MemoryChangeView[]>(
      `/api/sessions/${sessionId}/memory-changes`,
      { method: "GET" },
    );
  }

  async decideMemoryChange(
    sessionId: string,
    changeId: string,
    decision: "apply" | "reject",
  ): Promise<MemoryChangeView> {
    return this.request<MemoryChangeView>(
      `/api/sessions/${sessionId}/memory-changes/${changeId}/decision`,
      {
        method: "POST",
        body: JSON.stringify({ decision }),
      },
    );
  }

  async getMemoryStatus(sessionId: string): Promise<MemoryIngestionView> {
    return this.request<MemoryIngestionView>(
      `/api/sessions/${sessionId}/memories/status`,
      { method: "GET" },
    );
  }

  async retryMemoryIngestion(sessionId: string): Promise<MemoryIngestionView> {
    return this.request<MemoryIngestionView>(
      `/api/sessions/${sessionId}/memories/retry`,
      { method: "POST", body: JSON.stringify({}) },
    );
  }

  async getMemoryResearchConsent(
    sessionId: string,
  ): Promise<MemoryResearchConsentView> {
    return this.request<MemoryResearchConsentView>(
      `/api/sessions/${sessionId}/memories/research-consent`,
      { method: "GET" },
    );
  }

  async setMemoryResearchConsent(
    sessionId: string,
    granted: boolean,
    policyVersion: string,
  ): Promise<MemoryResearchConsentView> {
    return this.request<MemoryResearchConsentView>(
      `/api/sessions/${sessionId}/memories/research-consent`,
      {
        method: "PUT",
        body: JSON.stringify({
          granted,
          acknowledged_policy_version: policyVersion,
        }),
      },
    );
  }

  async getMemoryShadowReport(sessionId: string): Promise<MemoryShadowReportView> {
    return this.request<MemoryShadowReportView>(
      `/api/sessions/${sessionId}/memories/shadow-report`,
      { method: "GET" },
    );
  }

  async decideMemory(
    sessionId: string,
    memoryId: string,
    decision: "confirm" | "reject",
  ): Promise<MemoryView> {
    return this.request<MemoryView>(
      `/api/sessions/${sessionId}/memories/${memoryId}/decision`,
      {
        method: "POST",
        body: JSON.stringify({ decision }),
      },
    );
  }

  async deleteMemory(sessionId: string, memoryId: string): Promise<void> {
    await this.request(`/api/sessions/${sessionId}/memories/${memoryId}`, {
      method: "DELETE",
    });
  }

  async updateMemory(
    sessionId: string,
    memoryId: string,
    text: string,
    retention: MemoryRetention,
  ): Promise<MemoryView> {
    return this.request<MemoryView>(
      `/api/sessions/${sessionId}/memories/${memoryId}`,
      {
        method: "PUT",
        body: JSON.stringify({ text, retention }),
      },
    );
  }

  async getLatestMemoryRecall(
    sessionId: string,
    memoryId: string,
  ): Promise<MemoryRecallView | null> {
    return this.request<MemoryRecallView | null>(
      `/api/sessions/${sessionId}/memories/${memoryId}/latest-recall`,
      { method: "GET" },
    );
  }

  createTextTurn(sessionId: string, text: string): Promise<TurnResult> {
    return this.request<TurnResult>(`/api/sessions/${sessionId}/turns`, {
      method: "POST",
      body: JSON.stringify({ text, visual_summary: "" }),
    });
  }

  realtimeUrl(sessionId: string): string {
    const url = new URL(this.baseUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.pathname = `/api/sessions/${sessionId}/realtime`;
    url.search = "";
    return url.toString();
  }

  createRealtimeClient(sessionId: string): LocalRealtimeClient;
  createRealtimeClient<Client>(
    sessionId: string,
    factory: (url: string, accessToken: string) => Client,
  ): Client;
  createRealtimeClient<Client>(
    sessionId: string,
    factory: (url: string, accessToken: string) => Client = (
      url,
      accessToken,
    ) => new LocalRealtimeClient(url, accessToken) as Client,
  ): Client {
    if (!this.accessToken) {
      throw new Error("Create the session before realtime connection");
    }
    return factory(this.realtimeUrl(sessionId), this.accessToken);
  }

  async getLiveKitToken(
    sessionId: string,
    participantName: string,
  ): Promise<LiveKitCredentials> {
    return this.request<LiveKitCredentials>("/api/livekit/token", {
      method: "POST",
      body: JSON.stringify({
        session_id: sessionId,
        participant_name: participantName,
      }),
    });
  }

  async setConsent(
    sessionId: string,
    kind: "camera" | "microphone",
    granted: boolean,
  ): Promise<void> {
    await this.request(`/api/sessions/${sessionId}/consents/${kind}`, {
      method: "POST",
      body: JSON.stringify({ granted }),
    });
  }

  async interrupt(sessionId: string): Promise<void> {
    await this.request(`/api/sessions/${sessionId}/interrupt`, {
      method: "POST",
      body: JSON.stringify({ reason: "user_pressed_interrupt" }),
    });
  }

  async getClinicianSummary(
    sessionId: string,
  ): Promise<ClinicianSummary> {
    return this.request<ClinicianSummary>(
      `/api/clinician/sessions/${sessionId}/summary`,
      {
        method: "GET",
        headers: this.clinicianHeaders(sessionId),
      },
    );
  }

  async getClinicianTimeline(
    sessionId: string,
  ): Promise<ClinicianTimeline> {
    return this.request<ClinicianTimeline>(
      `/api/clinician/sessions/${sessionId}/timeline`,
      {
        method: "GET",
        headers: this.clinicianHeaders(sessionId),
      },
    );
  }

  async acceptHandoff(
    sessionId: string,
    handoffId: string,
  ): Promise<HandoffRecord> {
    return this.request<HandoffRecord>(
      `/api/clinician/sessions/${sessionId}/accept`,
      {
        method: "POST",
        headers: this.clinicianHeaders(sessionId),
        body: JSON.stringify({ handoff_id: handoffId }),
      },
    );
  }

  private clinicianHeaders(sessionId: string): Record<string, string> {
    return {
      "X-Demo-Role": "CLINICIAN_DEMO",
      "X-Demo-Session": sessionId,
    };
  }

  private readMemorySubjectToken(): string | undefined {
    try {
      return globalThis.localStorage?.getItem(MEMORY_SUBJECT_STORAGE_KEY) ?? undefined;
    } catch {
      return undefined;
    }
  }

  private writeMemorySubjectToken(token: string): void {
    try {
      globalThis.localStorage?.setItem(MEMORY_SUBJECT_STORAGE_KEY, token);
    } catch {
      // Storage may be unavailable in privacy mode; the session remains usable.
    }
  }

  private async request<T = unknown>(
    path: string,
    init: RequestInit,
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(this.accessToken
          ? { Authorization: `Bearer ${this.accessToken}` }
          : {}),
        ...init.headers,
      },
    });
    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || `API request failed: ${response.status}`);
    }
    if (response.status === 204) {
      return undefined as T;
    }
    return response.json() as Promise<T>;
  }
}
