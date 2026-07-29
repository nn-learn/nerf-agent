export interface LiveKitCredentials {
  server_url: string;
  room_name: string;
  participant_token: string;
}

const DEFAULT_API_BASE = "http://localhost:8000";


export class PsyAvatarApi {
  constructor(
    private readonly baseUrl =
      import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE,
  ) {}

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

  private async request<T = unknown>(
    path: string,
    init: RequestInit,
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
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
