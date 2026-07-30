/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MEDIA_MODE?: "mock" | "local" | "livekit";
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
