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

const DEFAULT_API_BASE = "http://localhost:8000";


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
    const creation = this.request<SessionCreated>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        camera_consent: false,
        client_session_id: sessionId,
      }),
    })
      .then((session) => {
        this.accessToken = session.access_token;
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
