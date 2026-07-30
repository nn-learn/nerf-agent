# Local Ollama Realtime Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing PsyAvatar Care call UI to real microphone capture, faster-whisper, local Ollama `qwen3.6:latest`, Edge TTS, and the lightweight Web avatar without bypassing deterministic safety or persisting raw media.

**Architecture:** FastAPI owns one auditable turn from transcription through guarded response and TTS. A session WebSocket carries JSON control/events plus 16 kHz mono PCM, while the existing LangGraph is converted to asynchronous provider calls so Ollama cancellation remains effective. The React app uses an AudioWorklet, pure TypeScript PCM utilities, and a local realtime media hook; mock and LiveKit paths remain available.

**Tech Stack:** Python 3.12, FastAPI WebSocket, LangGraph, httpx, Pydantic, faster-whisper CPU INT8, WebRTC VAD, BGE-M3 CPU, Ollama REST API, Edge TTS/FFmpeg, React 19, TypeScript 7, Web Audio, Vitest, pytest.

## Global Constraints

- The product is an adult, non-diagnostic psychological-support demo; it must not claim to diagnose, treat, prescribe, or contact emergency/clinical services.
- `RiskLevel` remains owned by deterministic rules; Ollama, vision, memory, and RAG cannot change it.
- Raw microphone PCM, camera frames, and TTS bytes remain in memory only and must never enter SQLite events or logs.
- Local baseline is Windows with Python `>=3.12,<3.13`, no GPU, approximately 31.7 GB RAM.
- STT stays `faster-whisper small`, `device="cpu"`, `compute_type="int8"`.
- LLM defaults to local Ollama `qwen3.6:latest` at `http://127.0.0.1:11434`.
- Retrieval stays BGE-M3 plus BM25 over reviewed, unexpired knowledge only.
- TTS stays Edge TTS voice `zh-CN-XiaoxiaoNeural`; network/TTS failure degrades to text.
- Browser/server audio contract is exactly 16 kHz, mono, signed little-endian 16-bit PCM, 20 ms/640-byte frames.
- Ollama output is not spoken until the complete `AgentResponse` passes schema, evidence, risk-consistency, and output-guard validation.
- `mock`, `local`, and `real` provider modes remain explicit; configuration failure must never silently become mock output.
- Existing camera consent, clinician scope, high-risk recall, vision-boundary, privacy, and mock-mode tests must remain green.

---

## File Structure

### Backend files to create

- `services/orchestrator/app/providers/ollama_text.py` — native Ollama chat/readiness adapter.
- `services/orchestrator/app/providers/local_agent.py` — reviewed BGE context plus Ollama normal-response adapter and deterministic crisis response.
- `services/orchestrator/app/providers/runtime.py` — provider-mode assembly, lazy BGE index, Edge TTS bridge, lifecycle and readiness.
- `services/orchestrator/app/realtime/observer.py` — typed turn observer contract used by HTTP and WebSocket delivery.
- `services/orchestrator/app/realtime/utterance.py` — bounded VAD utterance state machine.
- `services/orchestrator/app/realtime/websocket.py` — authenticated realtime protocol and per-connection pipeline.
- `services/orchestrator/app/api/realtime.py` — FastAPI WebSocket route.
- `services/orchestrator/tests/providers/test_ollama_text.py`
- `services/orchestrator/tests/providers/test_local_agent.py`
- `services/orchestrator/tests/providers/test_runtime.py`
- `services/orchestrator/tests/realtime/test_observer_delivery.py`
- `services/orchestrator/tests/realtime/test_utterance.py`
- `services/orchestrator/tests/api/test_realtime.py`

### Backend files to modify

- `services/orchestrator/app/settings.py` — local Ollama, RAG, timeout and media settings.
- `services/orchestrator/app/providers/protocols.py` — asynchronous, cancel-aware `AgentProvider`.
- `services/orchestrator/app/providers/mock.py` — asynchronous mock implementation.
- `services/orchestrator/app/graph/state.py` — carry `turn_id` and `cancel_token`.
- `services/orchestrator/app/graph/build.py` — async context/reply/crisis nodes.
- `services/orchestrator/app/realtime/session.py` — injected coordinator/provider, audio-turn entry point and observer delivery.
- `services/orchestrator/app/api/sessions.py` — mark HTTP typed fallback as text-only.
- `services/orchestrator/app/main.py` — runtime assembly, lifespan cleanup, realtime route and accurate readiness.
- `services/orchestrator/pyproject.toml` — BGE optional dependency.
- Existing graph, health, integration and speech tests — async signatures and regression assertions.

### Frontend files to create

- `apps/web/src/realtime/protocol.ts` — discriminated JSON message types and parser.
- `apps/web/src/realtime/pcm.ts` — deterministic resampling, S16LE framing and decoding.
- `apps/web/src/realtime/audioCapture.ts` — microphone/AudioWorklet lifecycle.
- `apps/web/src/realtime/audioPlayback.ts` — cancellable PCM Web Audio queue.
- `apps/web/src/realtime/LocalRealtimeClient.ts` — authenticated mixed JSON/binary WebSocket client.
- `apps/web/src/realtime/protocol.test.ts`
- `apps/web/src/realtime/pcm.test.ts`
- `apps/web/src/realtime/LocalRealtimeClient.test.ts`
- `apps/web/src/call/useLocalRealtimeMediaSession.ts`
- `apps/web/src/call/useLocalRealtimeMediaSession.test.ts`
- `apps/web/public/pcm-capture-processor.js`

### Frontend and operations files to modify

- `apps/web/src/api/client.ts` and `client.test.ts` — expose session-scoped realtime connection and typed fallback.
- `apps/web/src/call/useMediaSession.ts` — captions, provider status and text fallback in the controller contract.
- `apps/web/src/call/CallPage.tsx`, `CallPage.test.tsx`, `styles.css` — live captions, errors and keyboard fallback.
- `apps/web/src/main.tsx` — select local media mode.
- `apps/web/src/vite-env.d.ts`, `.env.example` — typed local configuration.
- `scripts/bootstrap.ps1` — install local speech/RAG extras.
- `scripts/doctor.ps1` — local Ollama/model, FFmpeg and dependency diagnostics.
- `scripts/start-demo.ps1` — `local` mode without Docker.
- `scripts/smoke-local.ps1` — real typed-turn local smoke.
- `README.md`, `docs/architecture.md`, `services/orchestrator/.env.example` — operating and privacy documentation.

---

### Task 1: Native Ollama Provider and Local Settings

**Files:**
- Create: `services/orchestrator/app/providers/ollama_text.py`
- Create: `services/orchestrator/tests/providers/test_ollama_text.py`
- Modify: `services/orchestrator/app/settings.py`
- Modify: `services/orchestrator/tests/test_health.py`
- Modify: `services/orchestrator/.env.example`

**Interfaces:**
- Consumes: `CancellationRegistry`, `AgentResponse`, `RiskLevel`, `OutputGuard`.
- Produces: `OllamaTextProvider.respond(context, *, turn_id, cancel_token, risk_level) -> OllamaTextResult`, `OllamaTextProvider.readiness() -> OllamaReadiness`, and local settings used by Task 3.

- [ ] **Step 1: Write failing settings and provider contract tests**

Add tests that instantiate:

```python
settings = Settings(provider_mode="local")
assert settings.ollama_base_url == "http://127.0.0.1:11434"
assert settings.text_model == "qwen3.6:latest"
assert settings.ollama_keep_alive == "30m"
assert settings.ollama_timeout_seconds == 180
assert settings.ollama_num_ctx == 4096
assert settings.ollama_num_predict == 320
```

Use `httpx.MockTransport` to capture the Ollama request and return:

```python
{
    "model": "qwen3.6:latest",
    "message": {
        "role": "assistant",
        "content": json.dumps(valid_agent_response, ensure_ascii=False),
    },
    "done": True,
    "total_duration": 2_000_000_000,
    "load_duration": 500_000_000,
    "prompt_eval_count": 120,
    "eval_count": 48,
}
```

Assert the request has:

```python
assert captured["model"] == "qwen3.6:latest"
assert captured["stream"] is False
assert captured["think"] is False
assert captured["format"] == AgentResponse.model_json_schema()
assert captured["keep_alive"] == "30m"
assert captured["options"] == {
    "temperature": 0.2,
    "num_ctx": 4096,
    "num_predict": 320,
}
```

Add distinct tests for cancellation before request, cancellation after response,
model changing deterministic risk, an evidence ID not present in
`context["reviewed_evidence"]`, malformed JSON, timeout, and readiness when
`/api/tags` does not contain `qwen3.6:latest`.
Add a prewarm test asserting `prewarm()` posts `/api/generate` with:

```python
{
    "model": "qwen3.6:latest",
    "prompt": "",
    "stream": False,
    "keep_alive": "30m",
}
```

- [ ] **Step 2: Run tests and verify the new API is absent**

Run:

```powershell
cd services/orchestrator
.\.venv\Scripts\python.exe -m pytest tests/providers/test_ollama_text.py tests/test_health.py -q
```

Expected: collection fails because `app.providers.ollama_text` and local settings do not exist.

- [ ] **Step 3: Add exact local settings**

Change `Settings` to include:

```python
ProviderMode = Literal["mock", "local", "real"]

provider_mode: ProviderMode = "mock"
ollama_base_url: str = "http://127.0.0.1:11434"
text_model: str = "qwen3.6:latest"
ollama_keep_alive: str = "30m"
ollama_timeout_seconds: float = Field(default=180, gt=0, le=600)
ollama_num_ctx: int = Field(default=4096, ge=1024, le=32768)
ollama_num_predict: int = Field(default=320, ge=32, le=2048)
knowledge_manifest_path: Path = Path("../../knowledge/manifest.yaml")
rag_top_k: int = Field(default=3, ge=1, le=8)
rag_evidence_threshold: float = Field(default=0.62, ge=0, le=1)
websocket_auth_timeout_seconds: float = Field(default=5, gt=0, le=30)
utterance_end_silence_ms: int = Field(default=600, ge=200, le=2000)
utterance_max_ms: int = Field(default=20_000, ge=1000, le=60_000)
```

Keep `dashscope_api_key` and `vision_model` because the future `real`/vision path still uses them.

- [ ] **Step 4: Implement the native Ollama adapter**

Create:

```python
@dataclass(frozen=True, slots=True)
class OllamaReadiness:
    ready: bool
    version: str | None
    model: str
    reason: str | None = None


class OllamaMetrics(BaseModel):
    total_duration_ns: int = Field(ge=0)
    load_duration_ns: int = Field(ge=0)
    prompt_eval_count: int = Field(ge=0)
    eval_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class OllamaTextResult:
    response: AgentResponse
    metrics: OllamaMetrics


class OllamaTextProvider:
    def __init__(
        self,
        *,
        registry: CancellationRegistry,
        model: str = "qwen3.6:latest",
        base_url: str = "http://127.0.0.1:11434",
        keep_alive: str = "30m",
        timeout_seconds: float = 180,
        num_ctx: int = 4096,
        num_predict: int = 320,
        client: httpx.AsyncClient | None = None,
        output_guard: OutputGuard | None = None,
        generation_lock: asyncio.Lock | None = None,
    ) -> None:
        self._registry = registry
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._num_ctx = num_ctx
        self._num_predict = num_predict
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._guard = output_guard or OutputGuard()
        self._generation_lock = generation_lock or asyncio.Lock()
```

`respond()` must POST `/api/chat` with the schema shown in Step 1, validate with:

```python
agent_response = AgentResponse.model_validate_json(content)
if agent_response.risk_level is not risk_level:
    raise TextProviderResponseError(
        "text provider changed the deterministic risk level"
    )
allowed_ids = {
    str(item["chunk_id"])
    for item in context.get("reviewed_evidence", [])
    if isinstance(item, dict) and "chunk_id" in item
}
unknown_ids = set(agent_response.evidence_ids) - allowed_ids
if unknown_ids:
    raise TextProviderResponseError(
        "text provider cited evidence outside the reviewed turn bundle"
    )
if agent_response.evidence_ids and not context.get(
    "has_sufficient_evidence", False
):
    raise TextProviderResponseError(
        "text provider cited an insufficient evidence bundle"
    )
guarded = self._guard.validate(agent_response)
return OllamaTextResult(
    response=guarded,
    metrics=OllamaMetrics(
        total_duration_ns=payload.get("total_duration", 0),
        load_duration_ns=payload.get("load_duration", 0),
        prompt_eval_count=payload.get("prompt_eval_count", 0),
        eval_count=payload.get("eval_count", 0),
    ),
)
```

Run the request and response validation under `generation_lock`, defaulting to
one lock owned by the provider. This serializes local generations across
sessions so the 23 GB model is not driven into memory pressure by concurrent
turns. A concurrency test must hold the first mock request and assert the second
does not enter the transport until the first releases.

`readiness()` must call `/api/version` and `/api/tags`, compare exact names, and
return a reason code without exposing response bodies or secrets. Map transport,
HTTP, validation, and cancellation failures to the existing
`TextProviderCancelled`/`TextProviderResponseError` names imported from
`qwen_text.py`, so callers have one error vocabulary.

`prewarm()` uses `/api/generate` with the exact body tested above, checks the
HTTP status, ignores generated content, and never logs the response body.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/providers/test_ollama_text.py tests/test_health.py -q
.\.venv\Scripts\python.exe -m ruff check app/providers/ollama_text.py app/settings.py tests/providers/test_ollama_text.py
.\.venv\Scripts\python.exe -m mypy app
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add services/orchestrator/app/settings.py services/orchestrator/app/providers/ollama_text.py services/orchestrator/tests/providers/test_ollama_text.py services/orchestrator/tests/test_health.py services/orchestrator/.env.example
git commit -m "feat: add local Ollama text provider"
```

---

### Task 2: Async Safety Graph and Reviewed BGE Agent

**Files:**
- Create: `services/orchestrator/app/providers/local_agent.py`
- Create: `services/orchestrator/tests/providers/test_local_agent.py`
- Modify: `services/orchestrator/app/providers/protocols.py`
- Modify: `services/orchestrator/app/providers/mock.py`
- Modify: `services/orchestrator/app/graph/state.py`
- Modify: `services/orchestrator/app/graph/build.py`
- Modify: `services/orchestrator/tests/graph/test_routes.py`
- Modify: `services/orchestrator/pyproject.toml`

**Interfaces:**
- Consumes: `OllamaTextProvider.respond`, `HybridRetriever.retrieve`.
- Produces: async `AgentProvider` consumed by the graph and runtime assembly.

- [ ] **Step 1: Write failing async graph tests**

Convert graph tests to `pytest.mark.asyncio` and call
`await graph.ainvoke({"transcript": "最近压力很大", "visual_summary": "", "turn_id": "turn_1", "cancel_token": "ct_1", "visited": []})`.
Add a fake provider whose normal response records `turn_id` and `cancel_token`,
then invoke:

```python
result = await graph.ainvoke({
    "transcript": "最近压力很大",
    "visual_summary": "",
    "turn_id": "turn_1",
    "cancel_token": "ct_1",
    "visited": [],
})
assert fake.reply_call == ("turn_1", "ct_1", RiskLevel.GREEN)
```

For an emergency transcript, assert the fake Ollama responder was never called,
`context_fetch` was skipped, and deterministic `support_mode == "handoff"`.

- [ ] **Step 2: Run focused tests to demonstrate the synchronous mismatch**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/graph/test_routes.py tests/providers/test_local_agent.py -q
```

Expected: failure because `AgentProvider` and graph nodes are synchronous and `LocalAgentProvider` is absent.

- [ ] **Step 3: Make the provider protocol async and cancellation-aware**

Use these exact signatures:

```python
class AgentPlan(BaseModel):
    response: AgentResponse
    provider_metrics: dict[str, int | float | str] = Field(
        default_factory=dict
    )


class AgentProvider(Protocol):
    async def load_context(
        self,
        transcript: str,
        visual_summary: str,
        risk: RiskAssessment,
    ) -> dict[str, object]:
        raise NotImplementedError

    async def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AgentPlan:
        raise NotImplementedError

    async def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentPlan:
        raise NotImplementedError
```

Make `MockAgentProvider` methods async without changing its response copy.
Add `turn_id: str` and `cancel_token: str` to `AgentState`. Convert
`context_fetch`, `reply_planner`, and `crisis_policy` to `async def`, and pass
the exact state values to the provider. Each planning node writes
`plan.response` to `candidate_response` and `plan.provider_metrics` to a new
`provider_metrics: dict[str, int | float | str]` graph-state field.

- [ ] **Step 4: Implement LocalAgentProvider**

Use constructor injection so BGE tests never load a real model:

```python
class ReviewedRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        risk_level: RiskLevel,
        k: int,
    ) -> EvidenceBundle:
        raise NotImplementedError


class LocalAgentProvider:
    def __init__(
        self,
        *,
        text_provider: OllamaTextProvider,
        retriever: ReviewedRetriever,
        top_k: int = 3,
    ) -> None:
        self._text = text_provider
        self._retriever = retriever
        self._top_k = top_k
```

`load_context()` must await the retriever, serialize only reviewed
`EvidenceItem` objects, and return:

```python
{
    "transcript": transcript,
    "visual_summary": visual_summary,
    "reviewed_evidence": [
        item.model_dump(mode="json") for item in bundle.items
    ],
    "has_sufficient_evidence": bundle.has_sufficient_evidence,
    "working_memory": [],
    "long_term_memory": [],
}
```

`plan_reply()` delegates to `OllamaTextProvider.respond()` and returns:

```python
result = await self._text.respond(
    context,
    turn_id=turn_id,
    cancel_token=cancel_token,
    risk_level=risk.level,
)
return AgentPlan(
    response=result.response,
    provider_metrics=result.metrics.model_dump(),
)
```

`plan_crisis()` returns an `AgentPlan` containing fixed, reviewed handoff
wording and `{"provider": "deterministic_crisis"}`; it never calls Ollama.
Do not say that the system contacted anyone.

- [ ] **Step 5: Enforce evidence-subset behavior**

Add a test where the retriever returns `chunk_1`, Ollama returns
`evidence_ids=["chunk_2"]`, and assert `TextProviderResponseError`. Add a
no-evidence test where the model uses no evidence IDs and remains valid.

- [ ] **Step 6: Add the explicit BGE dependency group**

Add:

```toml
rag = [
  "sentence-transformers>=5,<6",
  "PyYAML>=6,<7",
]
```

Do not add a second embedding model and do not replace the existing
`BAAI/bge-m3`, CPU, 1024-dimensional contract.

- [ ] **Step 7: Run graph/provider tests and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/graph/test_routes.py tests/providers/test_local_agent.py tests/rag -q
.\.venv\Scripts\python.exe -m ruff check app/providers app/graph tests/providers tests/graph
.\.venv\Scripts\python.exe -m mypy app
```

Expected: all pass.

- [ ] **Step 8: Commit**

```powershell
git add services/orchestrator/app/providers services/orchestrator/app/graph services/orchestrator/tests/providers/test_local_agent.py services/orchestrator/tests/graph/test_routes.py services/orchestrator/pyproject.toml
git commit -m "feat: route reviewed context through local Qwen"
```

---

### Task 3: Runtime Assembly and Observable Streaming Turns

**Files:**
- Create: `services/orchestrator/app/providers/runtime.py`
- Create: `services/orchestrator/app/realtime/observer.py`
- Create: `services/orchestrator/tests/providers/test_runtime.py`
- Create: `services/orchestrator/tests/realtime/test_observer_delivery.py`
- Modify: `services/orchestrator/app/realtime/session.py`
- Modify: `services/orchestrator/app/api/sessions.py`
- Modify: `services/orchestrator/app/main.py`
- Modify: `services/orchestrator/tests/integration/test_full_turn.py`
- Modify: `services/orchestrator/tests/integration/test_barge_in.py`
- Modify: `services/orchestrator/tests/integration/test_degradation.py`
- Modify: `services/orchestrator/tests/test_health.py`

**Interfaces:**
- Consumes: async `AgentProvider`, `OllamaTextProvider`, `FasterWhisperProvider`, `EdgeTtsProvider`.
- Produces: `RuntimeProviders`, `TurnObserver`, `SessionManager.process_audio_turn`, and accurate app readiness consumed by Task 4.

- [ ] **Step 1: Write failing runtime assembly tests**

Assert:

```python
mock = build_runtime_providers(
    Settings(provider_mode="mock"),
    registry=registry,
    ollama_client=mock_http_client,
)
assert mock.transcriber is None
assert isinstance(mock.agent_provider, MockAgentProvider)

local = build_runtime_providers(
    Settings(provider_mode="local"),
    registry=registry,
    ollama_client=mock_http_client,
    embedding_provider=fake_bge,
)
assert isinstance(local.transcriber, FasterWhisperProvider)
assert isinstance(local.agent_provider, LocalAgentProvider)
assert local.provider_mode == "local"
```

Also assert `real` preserves the existing LiveKit-facing behavior and does not
claim local provider readiness.

- [ ] **Step 2: Write failing observer delivery test**

Define a recording observer:

```python
class RecordingObserver:
    events: list[tuple[str, dict[str, object]]]
    audio: list[PcmChunk]

    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append((event_type, payload))

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        self.audio.append(chunk)
```

Run a text turn and assert `assistant.response.ready` arrives before the first
audio chunk, every stored `tts.audio.chunk` contains metadata only, and no
`pcm_s16le` key or bytes exist in `EventStore`.
Assert `provider.agent.metrics` contains Ollama duration/token counts,
`provider.tts.metrics` contains first-chunk/total durations, and neither event
contains transcript, prompt, response text, access token, or binary values.

- [ ] **Step 3: Define observer and streaming audio protocols**

Create:

```python
class TurnObserver(Protocol):
    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        raise NotImplementedError

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        raise NotImplementedError


class NullTurnObserver:
    async def on_event(self, turn, event_type, payload) -> None:
        return None

    async def on_audio(self, turn, chunk) -> None:
        return None
```

Change `AudioBridge.synthesize()` to return `AsyncIterator[PcmChunk]`. Update
`MockAudioBridge` to yield one 640-byte chunk and implement an
`EdgeTtsAudioBridge` that delegates using:

```python
return self._provider.synthesize(
    text,
    turn_id=turn.turn_id,
    cancel_token=turn.cancel_token,
    start_pts_ms=0,
)
```

- [ ] **Step 4: Refactor SessionManager around one active-turn core**

Inject `TurnCoordinator` and `AgentProvider`:

```python
def __init__(
    self,
    *,
    store: EventStore,
    coordinator: TurnCoordinator | None = None,
    agent_provider: AgentProvider | None = None,
    avatar_client: AvatarClient | None = None,
    vision_bridge: VisionBridge | None = None,
    audio_bridge: AudioBridge | None = None,
    provider_mode: str = "mock",
) -> None:
```

Build the graph from the injected provider. Split existing processing into
these exact methods:

- `process_text_turn(self, session_id: str, *, text: str, visual_summary: str = "", synthesize_audio: bool = True, observer: TurnObserver | None = None) -> TurnResult`
- `process_audio_turn(self, session_id: str, *, chunks: AsyncIterator[PcmChunk], transcriber: FasterWhisperProvider, observer: TurnObserver | None = None) -> TurnResult`
- `_process_active_turn(self, runtime: SessionRuntime, active: ActiveTurn, *, transcript: str, input_mode: Literal["typed_text", "voice"], visual_summary: str, observer: TurnObserver) -> TurnResult`

`process_audio_turn()` starts the active turn before transcription, forwards
partial transcripts only to the observer, and calls `_process_active_turn()`
after the final transcript. If no final transcript exists, emit
`transcript.empty`, finish the turn, and return a text-only completed result
with `response=None`.

Measure transcription with `time.perf_counter_ns()` and emit
`provider.stt.metrics` containing only `duration_ns` and the fixed model/device
labels. Pass `turn_id` and `cancel_token` into `graph.ainvoke()`. After the
graph returns, emit `provider.agent.metrics` from its `provider_metrics` field;
the allowlist is exactly `provider`, `total_duration_ns`, `load_duration_ns`,
`prompt_eval_count`, and `eval_count`.

In the TTS loop, capture total duration and the first-chunk duration with
`time.perf_counter_ns()`:

```python
tts_started_ns = time.perf_counter_ns()
first_chunk_ns: int | None = None
async for chunk in self._audio.synthesize(response.spoken_text, turn=turn):
    if first_chunk_ns is None:
        first_chunk_ns = time.perf_counter_ns()
    emitted = await self._emit_current(
        runtime,
        active,
        "tts.audio.chunk",
        {
            "sequence": chunk.sequence,
            "pts_ms": chunk.pts_ms,
            "duration_ms": chunk.duration_ms,
        },
        observer=observer,
    )
    if not emitted:
        return self._interrupted_result(
            active,
            risk=risk,
            response=response,
        )
    await observer.on_audio(turn, chunk)
    audio_chunk_count += 1
await self._emit_current(
    runtime,
    active,
    "provider.tts.metrics",
    {
        "first_chunk_duration_ns": (
            0 if first_chunk_ns is None else first_chunk_ns - tts_started_ns
        ),
        "total_duration_ns": time.perf_counter_ns() - tts_started_ns,
        "chunk_count": audio_chunk_count,
    },
    observer=observer,
)
```

When `synthesize_audio=False`, emit `response.displayed`, skip TTS/avatar work,
and return `delivery_mode="text"`. The HTTP route in `app/api/sessions.py`
passes `synthesize_audio=False`; direct integration tests and realtime voice
keep the default `True`. This prevents the keyboard fallback from synthesizing
audio that HTTP cannot deliver.

Modify `_emit_current()` so it persists the metadata event first, then calls
`observer.on_event()` only while the token is still current. Observer failure
must cancel delivery and degrade the connection; it must not roll back the
already-audited event.

- [ ] **Step 5: Assemble providers with lazy BGE initialization**

Create a `LazyReviewedRetriever` guarded by `asyncio.Lock`. Its public
`retrieve()` method is async. On first retrieval, run model construction and
indexing inside `asyncio.to_thread` while holding the lock:

```python
documents = load_manifest(settings.knowledge_manifest_path)
retriever = HybridRetriever(
    embedding_provider=BgeM3EmbeddingProvider(),
    evidence_threshold=settings.rag_evidence_threshold,
)
retriever.index(documents, as_of=date.today())
```

After initialization, call the synchronous hybrid retrieval through
`asyncio.to_thread`. The lock prevents duplicate BGE loads.
`RuntimeProviders` contains:

```python
@dataclass(slots=True)
class RuntimeProviders:
    provider_mode: str
    agent_provider: AgentProvider
    transcriber: FasterWhisperProvider | None
    audio_bridge: AudioBridge
    ollama: OllamaTextProvider | None
    prewarm_task: asyncio.Task[None] | None = None

    def start_prewarm(self) -> None:
        if self.ollama is not None and self.prewarm_task is None:
            self.prewarm_task = asyncio.create_task(self.ollama.prewarm())

    async def readiness(self) -> dict[str, object]:
        if self.provider_mode != "local" or self.ollama is None:
            return {
                "ready": True,
                "provider_mode": self.provider_mode,
            }
        if self.prewarm_task is not None and not self.prewarm_task.done():
            return {
                "ready": False,
                "provider_mode": "local",
                "reason": "OLLAMA_WARMING",
            }
        if (
            self.prewarm_task is not None
            and not self.prewarm_task.cancelled()
            and self.prewarm_task.exception() is not None
        ):
            return {
                "ready": False,
                "provider_mode": "local",
                "reason": "OLLAMA_PREWARM_FAILED",
            }
        result = await self.ollama.readiness()
        return {
            "ready": result.ready,
            "provider_mode": "local",
            "ollama": {
                "ready": result.ready,
                "model": result.model,
                "reason": result.reason,
            },
            "stt": {"ready": self.transcriber is not None, "model": "small"},
            "tts": {"ready": True, "voice": "zh-CN-XiaoxiaoNeural"},
        }

    async def aclose(self) -> None:
        if self.prewarm_task is not None and not self.prewarm_task.done():
            self.prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.prewarm_task
        if self.ollama is not None:
            await self.ollama.aclose()
```

For `local`, instantiate one `TurnCoordinator` first and pass its
`tokens` registry to Ollama, whisper, and Edge TTS so all cancellation checks
refer to the same token registry.

- [ ] **Step 6: Wire application lifespan and health**

`create_app()` creates/injects one coordinator, provider bundle and manager.
The async lifespan calls `providers.start_prewarm()` without blocking liveness
and closes/cancels owned resources on shutdown. While the prewarm task is
running, readiness returns HTTP 503 with reason `OLLAMA_WARMING`; if it fails,
readiness returns `OLLAMA_PREWARM_FAILED` without the response body.
`/health/ready`
returns HTTP 200 for ready mock/local and HTTP 503 for an unavailable local
Ollama/model:

```json
{
  "status": "ready",
  "provider_mode": "local",
  "providers": {
    "ollama": {"ready": true, "model": "qwen3.6:latest"},
    "stt": {"ready": true, "model": "small", "device": "cpu"},
    "tts": {"ready": true, "voice": "zh-CN-XiaoxiaoNeural"}
  }
}
```

The response must contain no API keys, bearer tokens, prompt bodies, or local
media.

- [ ] **Step 7: Run focused and regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/providers/test_runtime.py tests/realtime/test_observer_delivery.py tests/integration tests/test_health.py -q
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m mypy app
```

Expected: all pass, including stale-output barge-in assertions.

- [ ] **Step 8: Commit**

```powershell
git add services/orchestrator/app/providers/runtime.py services/orchestrator/app/realtime/observer.py services/orchestrator/app/realtime/session.py services/orchestrator/app/api/sessions.py services/orchestrator/app/main.py services/orchestrator/tests
git commit -m "refactor: assemble observable local provider turns"
```

---

### Task 4: Bounded VAD and Authenticated Realtime WebSocket

**Files:**
- Create: `services/orchestrator/app/realtime/utterance.py`
- Create: `services/orchestrator/app/realtime/websocket.py`
- Create: `services/orchestrator/app/api/realtime.py`
- Create: `services/orchestrator/tests/realtime/test_utterance.py`
- Create: `services/orchestrator/tests/api/test_realtime.py`
- Modify: `services/orchestrator/app/main.py`
- Modify: `services/orchestrator/app/security/auth.py`

**Interfaces:**
- Consumes: `SessionManager.process_audio_turn`, `SessionManager.interrupt`, `TurnObserver`, `FasterWhisperProvider`.
- Produces: `WS /api/sessions/{session_id}/realtime` and the protocol consumed by Task 5.

- [ ] **Step 1: Write failing utterance-state tests**

Use a fake VAD with a predetermined speech sequence. Assert:

- 15 pre-roll frames are included;
- 30 silent frames (600 ms) finish the utterance;
- 1000 total frames (20 seconds) force a split;
- a frame not exactly 640 bytes is rejected;
- `speech_started=True` is emitted only on the speech transition;
- `flush()` emits an active utterance but returns `None` for pre-roll-only silence;
- `reset()` clears all buffered bytes.

Expected result type:

```python
@dataclass(frozen=True, slots=True)
class SegmentationResult:
    speech_started: bool = False
    utterance: tuple[PcmChunk, ...] | None = None
```

- [ ] **Step 2: Implement the bounded segmenter**

Create `VadUtteranceSegmenter` with:

```python
def __init__(
    self,
    *,
    vad: WebRtcVad,
    pre_roll_frames: int = 15,
    end_silence_frames: int = 30,
    max_utterance_frames: int = 1000,
) -> None:
```

Keep a bounded `deque(maxlen=pre_roll_frames)` while idle. Once speech starts,
copy pre-roll into the active list, append frames until end silence/max length,
emit an immutable tuple, then reset. `flush()` uses the same immutable result
path only when speech has started; it never treats idle pre-roll as speech.

- [ ] **Step 3: Write failing WebSocket security/protocol tests**

Using FastAPI `TestClient.websocket_connect`, cover:

1. wrong `Origin` closes with policy violation;
2. a non-loopback host over `ws://` closes and requires `wss://`;
3. binary audio before auth closes;
4. invalid access token closes;
5. auth timeout closes;
6. valid auth receives `session.ready`;
7. a non-640-byte binary frame returns `INVALID_AUDIO_FRAME`;
8. more than 1000 frames cannot grow one utterance;
9. disconnect stops capture, interrupts the active turn and ends observer sends;
10. event store payloads contain no byte values.

The first valid message is exactly:

```json
{"type":"auth","access_token":"pst_test_token_with_more_than_32_characters"}
```

- [ ] **Step 4: Implement discriminated client messages**

Define Pydantic models:

```python
class AuthMessage(BaseModel):
    type: Literal["auth"]
    access_token: str = Field(min_length=32, max_length=256)

class AudioStartMessage(BaseModel):
    type: Literal["audio.start"]
    sample_rate: Literal[16000]
    channels: Literal[1]
    encoding: Literal["pcm_s16le"]

class AudioStopMessage(BaseModel):
    type: Literal["audio.stop"]

class InterruptMessage(BaseModel):
    type: Literal["turn.interrupt"]
    reason: str = Field(default="user_barge_in", min_length=1, max_length=64)

class SessionEndMessage(BaseModel):
    type: Literal["session.end"]
```

Parse by `type`, reject unknown keys with `extra="forbid"`, and cap text message
length at 2048 bytes before JSON parsing.

- [ ] **Step 5: Implement WebSocketTurnObserver**

Serialize audited events into the frontend protocol. On the first TTS chunk:

```python
await send_json({
    "type": "audio.start",
    "stream_id": f"audio_{turn.turn_id}",
    "turn_id": turn.turn_id,
    "sample_rate": 16000,
    "channels": 1,
    "encoding": "pcm_s16le",
})
```

Then send each `chunk.pcm_s16le` under one `asyncio.Lock`. At turn completion,
send `audio.end` with the same IDs. Map `transcript.*`, `risk.updated`,
`assistant.response.ready`, `playback.interrupted`, degradation and completion
events to explicit JSON; never serialize raw event-store internals.

- [ ] **Step 6: Implement the connection loop**

Validate the `Origin` before media. Permit insecure `ws://` only when both the
request host and configured web origin resolve to loopback (`localhost`,
`127.0.0.1`, or `[::1]`); require `wss://` otherwise. Accept, wait at most
`settings.websocket_auth_timeout_seconds` for the auth frame, and call
`manager.has_access()`. After auth:

- send `session.ready`;
- require a valid `audio.start` control frame before accepting binary PCM and
  reject duplicate starts until `audio.stop`;
- translate each 640-byte binary message into a monotonic `PcmChunk`;
- feed the VAD segmenter;
- on `speech_started`, call
  `manager.interrupt(session_id, reason="user_barge_in")`
  if another turn is active;
- on a completed utterance, create one `process_audio_turn()` task;
- do not run two local LLM turns concurrently;
- on `audio.stop`, flush only a non-empty speech utterance;
- on disconnect, reset the segmenter, cancel the task and interrupt the manager.

Use WebSocket close code 1008 for policy/auth failures, 1009 for oversized
messages, and 1011 for unrecoverable provider errors.

- [ ] **Step 7: Register the router**

Register:

```python
app.include_router(
    create_realtime_router(
        settings=current,
        manager=session_manager,
        transcriber=providers.transcriber,
    )
)
```

When `transcriber is None`, the endpoint accepts no fake voice; it sends
`PROVIDER_MODE_UNAVAILABLE` and closes.

- [ ] **Step 8: Run backend WebSocket and privacy tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/realtime/test_utterance.py tests/api/test_realtime.py tests/evals/test_privacy_eval.py -q
.\.venv\Scripts\python.exe -m ruff check app/realtime app/api tests/realtime tests/api
.\.venv\Scripts\python.exe -m mypy app
```

Expected: all pass.

- [ ] **Step 9: Commit**

```powershell
git add services/orchestrator/app/realtime services/orchestrator/app/api/realtime.py services/orchestrator/app/main.py services/orchestrator/app/security/auth.py services/orchestrator/tests
git commit -m "feat: add authenticated realtime voice socket"
```

---

### Task 5: Browser PCM and WebSocket Primitives

**Files:**
- Create: `apps/web/src/realtime/protocol.ts`
- Create: `apps/web/src/realtime/pcm.ts`
- Create: `apps/web/src/realtime/audioCapture.ts`
- Create: `apps/web/src/realtime/audioPlayback.ts`
- Create: `apps/web/src/realtime/LocalRealtimeClient.ts`
- Create: `apps/web/src/realtime/protocol.test.ts`
- Create: `apps/web/src/realtime/pcm.test.ts`
- Create: `apps/web/src/realtime/LocalRealtimeClient.test.ts`
- Create: `apps/web/public/pcm-capture-processor.js`

**Interfaces:**
- Consumes: Task 4 JSON/binary protocol.
- Produces: `LocalRealtimeClient`, `MicrophoneCapture`, `PcmPlaybackQueue` consumed by Task 6.

- [ ] **Step 1: Write failing protocol parser tests**

Test parsing these events:

```typescript
const ready = parseServerMessage({
  type: "session.ready",
  provider_mode: "local",
  model: "qwen3.6:latest",
});
expect(ready.type).toBe("session.ready");

expect(() => parseServerMessage({
  type: "assistant.response",
  display_text: 42,
})).toThrow("Invalid realtime message");
```

Define unions for `session.ready`, `agent.state`, `transcript.partial`,
`transcript.final`, `assistant.response`, `audio.start`, `audio.end`,
`turn.completed`, `playback.interrupted`, `error`, and `pong`.

- [ ] **Step 2: Write failing PCM tests**

Pure tests must demonstrate:

```typescript
const resampler = new Pcm16Resampler(48_000, 16_000, 320);
const frames = resampler.push(new Float32Array(960).fill(0.5));
expect(frames).toHaveLength(1);
expect(frames[0].byteLength).toBe(640);
expect(new Int16Array(frames[0])[0]).toBe(16_384);
```

Also cover clipping `-2/+2`, uneven source batches retaining remainder, 44.1 kHz
input, monotonic frame order, and `decodePcm16()` returning normalized floats.

- [ ] **Step 3: Implement pure PCM utilities**

`Pcm16Resampler` stores fractional source position and pending samples. For
each 16 kHz destination sample, use linear interpolation between adjacent
source samples, clamp to `[-1, 1]`, convert with:

```typescript
const pcm = sample < 0
  ? Math.round(sample * 0x8000)
  : Math.round(sample * 0x7fff);
view.setInt16(offset, pcm, true);
```

Buffer exactly 320 output samples per `ArrayBuffer`. `decodePcm16()` rejects odd
byte lengths and reads little-endian signed integers.

- [ ] **Step 4: Implement the AudioWorklet capture wrapper**

The processor sends copied mono float blocks:

```javascript
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) this.port.postMessage(channel.slice());
    return true;
  }
}
registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
```

`MicrophoneCapture.start(onFrame)` must request:

```typescript
{
  audio: {
    channelCount: 1,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
  video: false,
}
```

Load `/pcm-capture-processor.js`, connect source -> worklet -> zero-gain node ->
destination to keep the graph alive without echo, resample each block, and call
`onFrame` for every 640-byte frame. `stop()` disconnects nodes, stops tracks,
closes the AudioContext and clears pending samples.

- [ ] **Step 5: Implement cancellable PCM playback**

`PcmPlaybackQueue.enqueue(bytes)` decodes 16 kHz mono PCM, creates an
`AudioBuffer`, and schedules it at `max(context.currentTime, nextStartTime)`.
`cancel()` stops all active sources, clears them, sets `nextStartTime` to
`context.currentTime`, and increments a generation number so old `onended`
callbacks cannot mark a new stream complete.

Expose callbacks:

```typescript
interface PlaybackCallbacks {
  onPlaybackStarted: () => void;
  onPlaybackIdle: () => void;
}
```

- [ ] **Step 6: Implement LocalRealtimeClient**

Constructor:

```typescript
constructor(
  private readonly url: string,
  private readonly accessToken: string,
  private readonly socketFactory: (url: string) => WebSocket =
    (url) => new WebSocket(url),
) {}
```

On open, send the auth JSON. Set `binaryType="arraybuffer"`. Route text through
`parseServerMessage` and binary frames only while a matching `audio.start`
stream is active. Methods:

```typescript
connect(): Promise<void>
startAudio(): void
sendPcm(frame: ArrayBuffer): void
stopAudio(): void
interrupt(reason?: string): void
endSession(): void
close(): void
```

Tests use a fake WebSocket and assert auth is first, PCM is never sent before
`session.ready`, stale binary after `playback.interrupted` is discarded, and
close rejects pending connection promises.

- [ ] **Step 7: Run web primitive tests and build**

Run:

```powershell
cd apps/web
npm run test -- --run src/realtime
npm run build
```

Expected: all pass.

- [ ] **Step 8: Commit**

```powershell
git add apps/web/src/realtime apps/web/public/pcm-capture-processor.js
git commit -m "feat: add browser realtime PCM transport"
```

---

### Task 6: Local Media Hook, Captions, and Text Fallback

**Files:**
- Create: `apps/web/src/call/useLocalRealtimeMediaSession.ts`
- Create: `apps/web/src/call/useLocalRealtimeMediaSession.test.ts`
- Modify: `apps/web/src/api/client.ts`
- Modify: `apps/web/src/api/client.test.ts`
- Modify: `apps/web/src/call/useMediaSession.ts`
- Modify: `apps/web/src/call/CallPage.tsx`
- Modify: `apps/web/src/call/CallPage.test.tsx`
- Modify: `apps/web/src/main.tsx`
- Modify: `apps/web/src/styles.css`
- Modify: `apps/web/src/vite-env.d.ts`
- Modify: `apps/web/.env.example`

**Interfaces:**
- Consumes: `LocalRealtimeClient`, `MicrophoneCapture`, `PcmPlaybackQueue`.
- Produces: a complete `MediaSessionController` for `VITE_MEDIA_MODE=local`.

- [ ] **Step 1: Write failing local hook state tests**

With injected fakes, assert:

- session creation precedes socket connection;
- `session.ready` sets connection to `connected` and agent to `listening`;
- partial/final transcript updates `userCaption`;
- `assistant.response` updates `assistantCaption` but does not set speaking;
- first PCM starts playback and sets speaking;
- `audio.end` plus playback idle returns to listening;
- `error` exposes a Chinese, user-actionable `errorMessage`;
- `interrupt()` cancels playback before sending interrupt;
- microphone toggle starts/stops capture and sends `audio.start`/`audio.stop`;
- hangup stops microphone, cancels playback, closes socket and ends the HTTP session.

- [ ] **Step 2: Extend the controller contract**

Add:

```typescript
userCaption?: string;
assistantCaption?: string;
providerLabel?: string;
sendText: (text: string) => Promise<void>;
```

Update mock and LiveKit controllers with safe defaults. Mock `sendText` may
return without generating fake new content; it must not pretend to call Qwen.

- [ ] **Step 3: Add typed fallback to PsyAvatarApi**

Add:

```typescript
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

createTextTurn(sessionId: string, text: string): Promise<TurnResult> {
  return this.request(`/api/sessions/${sessionId}/turns`, {
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
```

Keep the bearer token private in the API instance. Add
`createRealtimeClient(sessionId, factory?)` so it can pass the token to
`LocalRealtimeClient` without exposing it to React state or URLs.

- [ ] **Step 4: Implement useLocalRealtimeMediaSession**

On mount:

1. create the audited session;
2. construct playback and socket;
3. connect and wait for `session.ready`;
4. start the microphone only after the user has enabled it;
5. update Zustand states from typed server messages.

Use refs for client/capture/playback so React StrictMode effect replay does not
create duplicate sessions or sockets. Cleanup must be idempotent.

`sendText()` calls the HTTP fallback, shows thinking while pending, writes the
validated response caption, and leaves TTS to the server result only if the
realtime socket is active; otherwise it is explicitly text-only.

- [ ] **Step 5: Replace fixed captions and add keyboard fallback**

Render:

```tsx
<div className="live-caption" aria-live="polite">
  <span className="caption-speaker">
    {media.agentState === "listening" && media.userCaption ? "你" : "小澄"}
  </span>
  <p>
    {media.agentState === "listening" && media.userCaption
      ? media.userCaption
      : media.assistantCaption ??
        "我在这里。你可以慢慢说，我们先从此刻最困扰你的事情开始。"}
  </p>
</div>
```

Add a small form labeled “语音不可用时输入文字”. Disable blank submits and
while a turn is thinking. Show `errorMessage` in an `role="alert"` block; never
replace an error with the initial canned caption.

- [ ] **Step 6: Select local mode in main.tsx**

Use:

```tsx
if (import.meta.env.VITE_MEDIA_MODE === "local") {
  return <LocalApp />;
}
if (import.meta.env.VITE_MEDIA_MODE === "livekit") {
  return <LiveKitApp />;
}
return <MockApp />;
```

Type:

```typescript
interface ImportMetaEnv {
  readonly VITE_MEDIA_MODE?: "mock" | "local" | "livekit";
  readonly VITE_API_BASE_URL?: string;
}
```

- [ ] **Step 7: Run hook/UI tests and build**

Run:

```powershell
npm run test -- --run src/api src/realtime src/call
npm run build
```

Expected: all pass and TypeScript reports no unchecked protocol branches.

- [ ] **Step 8: Commit**

```powershell
git add apps/web/src apps/web/.env.example
git commit -m "feat: connect call UI to local voice agent"
```

---

### Task 7: Local Installation, Diagnostics, and Launch

**Files:**
- Modify: `scripts/bootstrap.ps1`
- Modify: `scripts/doctor.ps1`
- Modify: `scripts/start-demo.ps1`
- Create: `scripts/smoke-local.ps1`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `services/orchestrator/.env.example`
- Modify: `apps/web/.env.example`

**Interfaces:**
- Consumes: Task 1 settings, Task 3 health response and Task 6 media mode.
- Produces: reproducible zero-GPU local setup and launch commands.

- [ ] **Step 1: Write a failing diagnostics smoke**

Run:

```powershell
.\scripts\doctor.ps1 -ProviderMode local
```

Expected before changes: parameter validation rejects `local`.

- [ ] **Step 2: Install the local extras explicitly**

Change bootstrap installation to:

```powershell
& $venvPython -m pip install -e ".[dev,speech,rag]"
```

After installation, import-check exact modules:

```powershell
& $venvPython -c "import edge_tts, faster_whisper, sentence_transformers, webrtcvad"
```

Do not install CUDA packages and do not create the Python 3.10 avatar
environment for local mode.

- [ ] **Step 3: Add local doctor checks**

Extend both script `ValidateSet` declarations to `("mock", "local", "real")`.
For local mode:

- FFmpeg is required;
- Docker, LiveKit, avatar Python, RAD-NeRF assets and CUDA are skipped;
- `Invoke-RestMethod http://127.0.0.1:11434/api/version` must succeed;
- `Invoke-RestMethod http://127.0.0.1:11434/api/tags` must contain exact name
  from `PSYAVATAR_TEXT_MODEL`, defaulting to `qwen3.6:latest`;
- Python import checks for faster-whisper, Edge TTS, WebRTC VAD and
  sentence-transformers are required;
- no secret value is printed.

- [ ] **Step 4: Add local launch behavior**

Map:

```powershell
$env:VITE_MEDIA_MODE = switch ($ProviderMode) {
    "real" { "livekit" }
    "local" { "local" }
    default { "mock" }
}
```

Only `real` starts Docker. `local` starts FastAPI and Vite, waits for
`/health/ready`, and allows up to 240 seconds for the first Qwen3.6 prewarm;
intermediate 503 `OLLAMA_WARMING` responses continue waiting, while any other
503 reason fails visibly. Keep hidden process windows and the existing PID
file. Mock keeps its short current timeout.

- [ ] **Step 5: Create a real typed smoke script**

`smoke-local.ps1` must:

1. GET `/health/ready` and require local Ollama readiness;
2. POST `/api/sessions`;
3. POST a benign typed turn with its bearer;
4. require `status == "completed"`;
5. require non-empty `response.display_text`;
6. require the answer is not exactly the known mock sentence;
7. DELETE the session in `finally`;
8. print durations and status, never the bearer.

Use the benign prompt `最近工作压力有点大，我想先梳理一下。`.

- [ ] **Step 6: Document exact operation and limitations**

Document:

```powershell
.\scripts\bootstrap.ps1 -PythonExe C:\Path\To\Python312\python.exe
ollama list
.\scripts\start-demo.ps1 -ProviderMode local
.\scripts\smoke-local.ps1
```

State that `qwen3.6:latest` is about 23 GB on this machine, first load may be
slow, Edge TTS requires network access, BGE-M3 may download on first use, raw
media is not persisted, and `qwen:latest` is an explicit user-selected
real-time fallback rather than an automatic silent substitution.

- [ ] **Step 7: Run script syntax and mock regression checks**

Run:

```powershell
[void][scriptblock]::Create((Get-Content -Raw scripts\doctor.ps1))
[void][scriptblock]::Create((Get-Content -Raw scripts\start-demo.ps1))
[void][scriptblock]::Create((Get-Content -Raw scripts\smoke-local.ps1))
.\scripts\doctor.ps1 -ProviderMode mock
```

Expected: parse succeeds and mock diagnostics remain usable without Ollama.

- [ ] **Step 8: Commit**

```powershell
git add scripts README.md docs/architecture.md services/orchestrator/.env.example apps/web/.env.example
git commit -m "docs: add local voice agent operations"
```

---

### Task 8: Full Verification and Real Local Acceptance

**Files:**
- Modify only if a verification failure reveals an in-scope defect; use the
  owning task's test file and implementation file together.
- Record measured local timings in `README.md` only after observing them.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified baseline and reproducible evidence that responses are not mock.

- [ ] **Step 1: Run the complete automated suite**

Run:

```powershell
.\scripts\verify-demo.ps1
```

Expected:

- orchestrator pytest passes;
- Ruff passes;
- strict mypy passes;
- avatar CPU contract tests pass;
- web Vitest passes;
- production web build passes;
- fixed risk/vision/privacy acceptance report passes.

- [ ] **Step 2: Confirm the local model and clean launch**

Run:

```powershell
ollama list
.\scripts\doctor.ps1 -ProviderMode local
.\scripts\start-demo.ps1 -ProviderMode local
```

Expected: exact `qwen3.6:latest` is present, doctor passes, ready health reports
`provider_mode=local`, and the browser uses `VITE_MEDIA_MODE=local`.

- [ ] **Step 3: Run the typed real-model smoke**

Run:

```powershell
.\scripts\smoke-local.ps1
```

Expected: non-mock, context-relevant Chinese response, with the model/load/eval
timings observable but no token or raw media printed.

- [ ] **Step 4: Run the microphone acceptance path**

At `http://127.0.0.1:5173/`:

1. allow microphone access;
2. say `最近工作压力有点大，我想先梳理一下`;
3. confirm the real user transcript appears;
4. confirm thinking state appears;
5. confirm a context-relevant Qwen response replaces the canned caption;
6. confirm Edge TTS is audible;
7. confirm the Web avatar speaks only while audio plays;
8. press “打断 AI” and confirm audio stops immediately;
9. deny/pause camera and confirm voice remains available;
10. stop Ollama and confirm the UI shows a provider error rather than a fake response.

- [ ] **Step 5: Run high-risk and privacy acceptance**

Use the fixed evaluation suite, not an improvised production claim:

```powershell
cd services/orchestrator
.\.venv\Scripts\python.exe -m app.evals.report `
  --risk ..\..\evals\risk_cases.jsonl `
  --vision ..\..\evals\vision_cases.jsonl
```

Inspect the local session SQLite with the existing privacy evaluator and require:

- fixed risk cases `12/12`;
- high-risk recall `1.0`;
- vision cases `11/11`;
- zero accepted diagnosis;
- zero visual-only risk escalation;
- zero raw media payloads.

- [ ] **Step 6: Record measured latency honestly**

Record cold load, STT, Ollama total, TTS first chunk and end-to-end time from
the real run. Label the numbers as measurements from this 31.7 GB CPU-only
machine. Do not describe the result as Doubao-level realtime if measurements
do not support that claim.

- [ ] **Step 7: Run final diff and status checks**

Run:

```powershell
git diff --check
git status --short
git log --oneline -10
```

Expected: no whitespace errors, no generated runtime database/audio/model
files staged, and only intentional source/docs changes remain.

- [ ] **Step 8: Commit verification documentation if timings changed**

```powershell
git add README.md
git commit -m "docs: record local Qwen acceptance results"
```

Skip this commit when no documentation changed.
