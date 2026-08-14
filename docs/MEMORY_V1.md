# PsyAvatar Memory V1

## Outcome

Memory V1 is a runnable, CPU-only baseline for turning the append-only conversation event log into governed user memory:

```text
event log -> keyset reader -> adaptive windows -> bounded extraction batches
          -> fact/profile candidates -> policy -> consent state -> SQLite
          -> purpose-filtered retrieval -> compact untrusted model context
```

It intentionally does not put raw transcripts into a vector database and does not let model output write directly to active memory.

## Why this shape

Current long-memory research is moving beyond flat similarity search:

- [LongMemEval](https://arxiv.org/abs/2410.10813) evaluates extraction, multi-session reasoning, temporal reasoning, updates, and abstention.
- [ES-Mem](https://arxiv.org/abs/2601.07582) motivates event-boundary segmentation rather than rigid fixed chunks.
- [A-MEM](https://arxiv.org/abs/2502.12110) explores structured notes, links, and memory evolution.
- [Zep/Graphiti](https://arxiv.org/abs/2501.13956) motivates temporal invalidation instead of keeping contradictory facts active forever.
- [AgentPoison](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html) shows why retrieved memory must be treated as untrusted data.

V1 adopts the low-risk parts now: event-aware windows, structured provenance, temporal supersession, consent gates, purpose filters, and instruction screening. It defers autonomous graph construction until there is enough evaluated data to justify the complexity.

## Windowing and token control

Default window policy:

- soft target: 512 estimated tokens;
- hard limit: 768 estimated tokens or 32 messages;
- time boundary: 30 minutes;
- explicit topic-transition markers create an event boundary;
- carry-over context: at most 96 estimated tokens from the previous window;
- extraction batches: at most 8 windows and 4096 estimated prompt tokens.

Carry-over messages are read-only context. An extractor is allowed to create candidates only from `window.messages`, never from `context_messages`. This gives pronouns and short follow-ups enough context without duplicating full overlapping windows or duplicating memories.

The token estimator is deliberately local and deterministic. The future Ollama extractor should record the model-reported prompt token count and calibrate this estimate per model.

## Memory contract

Each candidate carries:

- semantic tier and user-facing aspect;
- normalized claim and stable `subject_key`;
- source turn, source message IDs, and source window;
- extraction confidence and integrity flags;
- purpose scope and time validity;
- whether the user explicitly confirmed it.

States are `CANDIDATE`, `AWAITING_CONSENT`, `ACTIVE`, `SUPERSEDED`, `REVOKED`, `EXPIRED`, `REJECTED`, and `QUARANTINED`.

Activating a new memory with the same `(user, subject_key)` supersedes the old active row. Revocation removes it from retrieval immediately; purge physically removes its content. V1 has no embedding cache, so there is no derived vector to delete yet.

Sensitive health candidates remain `AWAITING_CONSENT` even when ordinary memory consent is already enabled; they require item-level confirmation. Direct identifiers such as phone, identity-card, and address data are rejected from general personalization memory. Conflicting aspects with the same subject, such as “喜欢呼吸练习” followed by“不喜欢呼吸练习”, also supersede one another.

## Message ingestion

`EventMessageReader` reads only `transcript.final` and `assistant.response.ready` events. It uses `seq > cursor` keyset pagination, not `OFFSET`, so later pages do not slow down as history grows.

`MemoryIngestionService` advances a per-user, per-session cursor only after a successful pipeline run. Re-running an unchanged session therefore performs no extraction and creates no duplicate proposals.

The live avatar path is now wired as follows:

1. The first session receives a high-entropy `pms_...` memory-subject token. Only its SHA-256 digest is stored; the browser reuses the opaque token to bind later sessions to the same pseudonymous user.
2. Ending a session performs a non-blocking `put_nowait` into a bounded single-consumer queue. Extraction runs in a worker thread, never on the realtime audio or reply path.
3. Queue overflow does not block hang-up. The append-only event log and ingestion cursor make missed work recoverable at process startup.
4. Extracted facts remain `AWAITING_CONSENT`. Per-item confirm/reject and physical delete endpoints require the private session bearer and enforce memory-subject ownership.
5. Only `ACTIVE`, user-confirmed, purpose-matched memory is retrieved for a later normal turn. It is injected as `user_confirmed_data_not_instruction`; emergency turns bypass normal retrieval, and retrieval failure degrades to an empty memory list.

The browser token is acceptable for this local demo but is stored in `localStorage`, so production deployment should replace it with an authenticated account or an `HttpOnly`, `Secure`, `SameSite` cookie and add token rotation/revocation. Audit events contain only memory IDs, never deleted memory content.

### HTTP flow

```text
POST   /api/sessions
DELETE /api/sessions/{session_id}                 # enqueue extraction
GET    /api/sessions/{session_id}/memories        # pending + active + expired
POST   /api/sessions/{session_id}/memories/{id}/decision
DELETE /api/sessions/{session_id}/memories/{id}   # physical content deletion
```

Use `{"decision":"confirm"}` to activate a proposal or `{"decision":"reject"}` to revoke it. Set `PSYAVATAR_MEMORY_EXTRACTOR_MODE=ollama` to use the local Qwen extractor and tune `PSYAVATAR_MEMORY_INGESTION_QUEUE_SIZE` for the number of simultaneously ending sessions.

The web call screen now exposes a user-facing **My Memory** center. It separates model proposals from approved memories, polls the durable ingestion status after hang-up, supports retry after a local-model failure, requires an extra confirmation for permanent deletion, and removes rejected candidate content immediately. The UI deliberately avoids bulk approval so sensitive psychological facts are reviewed one at a time.

### V1.3: correction, retention, and recall transparency

Approved or pending memory can now be corrected by the user and assigned a 7-day, 30-day, 90-day, or indefinite retention policy. Every edit is re-screened for direct identifiers, safety-critical content, and instruction injection. User-edited text receives confidence `1.0` because it is an explicit user assertion, while confirmation state remains unchanged; editing a pending proposal does not silently activate it.

The retriever emits deterministic reason codes (`USER_CONFIRMED`, `TOPIC_MATCH`, `STABLE_PREFERENCE`, and `RECENTLY_UPDATED`) from its actual scoring features. When a memory enters a turn context, a separate ledger stores only memory/session/turn IDs, numeric scores, reason codes, and time—never the current user query. The memory center presents this as “entered Agent context”, not as proof that the generated response was caused by that memory.

Physical deletion also removes recall-ledger rows for that memory. Expired memories remain visible to the user for renewal or deletion but are excluded from all Agent retrieval.

### V1.4: multi-session retrieval and answer evaluation

V1.4 adds a de-identified, synthetic-but-realistic longitudinal fixture with five scenarios, ten sessions, and nine labelled queries. Each expected memory is anchored to an exact quote in a user turn. Retrieval labels are kept separate from extraction output so a missed extraction cannot be mistaken for a ranking failure.

The suite now measures:

- Recall@5 and graded nDCG@5;
- correct abstention when no memory should be used;
- forbidden retrieval split by superseded, expired, cross-user, consent, integrity, and third-party reasons;
- p50/p95 retrieval latency;
- answer-memory adherence using explicit per-case response contracts;
- false-memory adoption by watching for unique terms from stale, unrelated, or forbidden memories.

`GovernedEmbeddingMemoryRetriever` reuses the existing CPU BGE-M3 provider but keeps the same repository gates as the lexical baseline. It can only score `ACTIVE`, user-confirmed, purpose-matched, temporally valid, same-user memories, and it screens integrity flags again. Vectors are process-local and are not persisted. This branch is for offline A/B evaluation and is not enabled in the live Agent by V1.4.

The default BGE cosine threshold is `0.40`. On this tiny fixture, `0.25` recovered all relevant memories but reduced correct abstention to `0.50`; the two false positives had cosine scores `0.376` and `0.365`. Raising the threshold to `0.40` kept the semantic paraphrase hits while restoring correct abstention to `1.00`. This is preliminary calibration, not a production threshold: it must be revalidated on a larger independently annotated set.

An explicit quality gate defaults to Recall@5 >= 0.90, nDCG@5 >= 0.85, correct abstention >= 0.90, forbidden retrieval = 0, answer adherence >= 0.90, and false-memory adoption <= 0.01. Answer thresholds are enforced only when Ollama answer generation is requested.

## Current evaluation

Run:

```powershell
cd services/orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_evaluation
.\.venv\Scripts\python.exe -m app.memory.run_ollama_evaluation
.\.venv\Scripts\python.exe -m app.memory.run_multisession_evaluation
.\.venv\Scripts\python.exe -m app.memory.run_multisession_evaluation --with-bge
.\.venv\Scripts\python.exe -m app.memory.run_multisession_evaluation --with-bge --with-ollama-answers
.\.venv\Scripts\python.exe -m app.memory.benchmark
.\.venv\Scripts\python.exe -m pytest tests\memory -q
```

The initial 13-case Chinese gold set covers preference, boundary, goal, coping strategy, explicit fact, sensitive health content, direct identifiers, ordinary chat, transient emotion, safety content, and prompt injection.

Current deterministic baseline result:

```json
{
  "case_count": 13,
  "extraction_precision": 1.0,
  "extraction_recall": 1.0,
  "aspect_accuracy": 1.0,
  "policy_accuracy": 1.0,
  "forbidden_write_rate": 0.0
}
```

The synthetic longitudinal suite additionally reports:

```json
{
  "retrieval_recall_at_5": 1.0,
  "supersession_accuracy": 1.0,
  "correct_abstention": 1.0,
  "deletion_effectiveness": 1.0,
  "cross_user_leakage_rate": 0.0,
  "poisoning_attack_success_rate": 0.0,
  "user_correction_effectiveness": 1.0,
  "expiry_enforcement": 1.0,
  "recall_explanation_fidelity": 1.0
}
```

The V1.4 multi-session run on the current CPU-only development machine produced:

| Metric | Lexical baseline | BGE-M3 @ 0.40 |
| --- | ---: | ---: |
| Recall@5 | 0.600 | 1.000 |
| nDCG@5 | 0.600 | 0.967 |
| Correct abstention | 1.000 | 1.000 |
| Forbidden retrieval | 0.000 | 0.000 |
| Retrieval p50 | about 5 ms | 64-94 ms |
| Retrieval p95 | 11-13 ms | 235-279 ms |

With the local `qwen3.6:latest` answer probe over the BGE results, all nine answer contracts passed and none of the seven adoption-watch cases used a stale or forbidden memory term. A first run scored 8/9 because the evaluator required “可以/当然” rather than evidence of detailedness; manual review showed the answer did follow the memory (for example, “细节”“透彻”“拆解”). The contract was corrected to score the intended behavior, demonstrating why automated answer graders need human error analysis rather than silent label tuning.

These numbers are engineering regression evidence only. Five curated scenarios are far too small for a clinical or production claim, and deterministic term contracts can miss semantically valid or subtly unsafe responses.

### V1.5: hybrid retrieval and evaluation protocol

V1.5 implements the first scale-out increment without changing the live Agent's lexical default:

- `GovernedHybridMemoryRetriever` unions candidates from the governed lexical and BGE-M3 branches, deduplicates by memory ID, and applies deterministic reciprocal-rank fusion plus semantic/lexical relevance, cross-signal agreement, and a small boundary prior;
- the reranker is relevance-led. A boundary receives only a small tie-breaking prior, preventing a generic stable constraint from outranking a directly relevant preference;
- a process-local embedding cache allows a threshold sweep to reuse BGE vectors; no embedding or query is written to SQLite;
- `memory_v15_manifest.json` declares dataset status, development/test ownership, slices, label policy version, and minimum annotator count separately from the conversation fixture;
- independent annotations use a 0-3 relevance scale, two pseudonymous annotators, exact agreement, quadratic-weighted kappa, an explicit conflict list, and adjudicated labels with source annotation IDs;
- reports include paraphrase, temporal, consent, integrity, cross-user, expiry and abstention slices plus deterministic percentile-bootstrap intervals.

The runner scans thresholds only on `DEV`, freezes the best passing threshold, and then evaluates `TEST`. It never selects a threshold from held-out scores. Equal DEV objectives choose the higher threshold as the more conservative abstention policy.

Run with cached BGE-M3 weights:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
cd services/orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_v15_evaluation
```

To audit two-person annotations and apply a separately adjudicated label file:

```powershell
.\.venv\Scripts\python.exe -m app.memory.run_v15_evaluation `
  --annotations ..\..\evals\your_annotations.jsonl `
  --adjudicated-labels ..\..\evals\your_adjudicated_labels.jsonl
```

The included manifest deliberately says `ENGINEERING_FIXTURE`. Changing it to `INDEPENDENTLY_ANNOTATED` makes an adjudicated label file mandatory; this prevents synthetic labels from being silently presented as external human evaluation.

Observed CPU-only engineering-fixture result:

| Stage | Threshold | Recall@5 | nDCG@5 | Abstention | Forbidden retrieval |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEV | 0.35-0.50 | 1.000 | 1.000 | 1.000 | 0.000 |
| Frozen selection | 0.50 | - | - | - | - |
| Held-out TEST | 0.50 | 1.000 | 1.000 | 1.000 | 0.000 |

The held-out split contains only two cases and three queries. Its percentile-bootstrap intervals collapse to `[1.0, 1.0]`, which is statistically uninformative rather than strong evidence. The report therefore sets `reliable=false` and emits a warning for every metric with fewer than 30 observations.

This is a regression baseline, not evidence of clinical quality. The cases are small and rule-aligned. The next gold set must be independently labelled and include paraphrases, negation, temporally qualified facts, contradictions, third-party facts, implicit constraints, and adversarial content.

The implemented `OllamaMemoryExtractor` was also run against the local `qwen3.6:latest` model in two batches of at most eight windows. The final governed hybrid result was:

```json
{
  "extraction_precision": 1.0,
  "extraction_recall": 1.0,
  "aspect_accuracy": 1.0,
  "policy_accuracy": 1.0,
  "forbidden_write_rate": 0.0,
  "request_count": 2,
  "prompt_tokens": 1880,
  "completion_tokens_observed_range": [539, 636],
  "wall_elapsed_ms_observed_range": [18573.96, 92434.95],
  "rule_fallback_candidates_observed_range": [0, 1]
}
```

A model-only ablation reached 0.857 recall on an earlier run of this small set. The conservative rule floor supplied one obvious omitted memory in that run; a later cold-start regression needed no fallback and still achieved the same perfect governed score. The large wall-time spread is mainly model residency: the cold run reported about 36.21 seconds of load time. Forbidden safety, identifier, and prompt-injection content is not reintroduced by fallback.

On the current development machine, the synthetic 10,000-message benchmark produced:

```json
{
  "messages": 10000,
  "windows": 313,
  "extraction_batches": 52,
  "candidates": 500,
  "active_rows": 500,
  "context_token_ratio": 0.1526,
  "elapsed_ms_observed_range": [138.92, 166.05],
  "messages_per_second_observed_range": [60221.45, 71981.7]
}
```

This measures message normalization, windowing, rule extraction, policy evaluation, and one-transaction SQLite persistence. It excludes Ollama inference; model extraction will dominate latency.

## Implemented local-model increment

### V1.1: local Qwen extractor

`OllamaMemoryExtractor` now implements the existing `MemoryExtractor.extract_batch` contract with:

- one request per bounded window batch;
- source IDs constrained to user messages in the main window;
- an exact `evidence_quote` that must occur in a selected source message;
- temperature zero, JSON Schema validation, timeout, and fail-closed parsing;
- an in-memory prompt/result LRU keyed by content, model, and extractor version;
- one bounded local generation at a time by default to avoid memory exhaustion;
- deterministic re-screening for health sensitivity, identifiers, safety content, and prompt injection;
- actual prompt tokens, completion tokens, model/load latency, cache hits, invalid candidates, and fallback counts.

The safe default remains `PSYAVATAR_MEMORY_EXTRACTOR_MODE=rule`. Set it to `ollama` only for evaluated background extraction. The synchronous Ollama extractor must not run on the realtime audio/reply thread.

## Next increments

### V1.6: consented online shadow mode (implemented baseline)

V1.6 adds an operationally isolated BGE-M3 Hybrid shadow path. The live Agent
continues to use `GovernedMemoryRetriever`; the shadow ranking is never passed
to Qwen and therefore cannot change the current response. The BGE model is
constructed lazily only after both the deployment switch and the user's
versioned research consent are active.

The shadow ledger deliberately has no query, memory-content, audio/video, or
embedding column. It stores pseudonymous user/session/turn/memory IDs, arm,
rank, numeric scores, reason codes, latency, outcome, consent-policy version,
and strategy version. Revocation cancels the awaiting task and uses SQLite
secure deletion to physically remove all shadow runs and ranking rows for that
memory subject. A native embedding call already executing in a Python worker
thread cannot be forcibly interrupted; its result is rejected after revocation
and cannot be persisted.

Unlike the V1.5 offline threshold sweep, the online shadow runtime does not use
the process-level text/vector cache. Query text, memory text, and vectors exist
only as transient inference objects for that run. This trades CPU latency for a
clearer deletion boundary; the aggregate report measures that cost explicitly.

Operational safeguards include:

- default-off deployment configuration and exact policy-version acknowledgement;
- 30-day physical retention by default (configurable only from 1 to 90 days),
  purged on startup, new shadow activity, and report access;
- one CPU embedding run at a time by default, bounded pending work, and no work
  at all before consent;
- per-run timeout, consecutive-failure circuit breaker, cooldown, and explicit
  `INITIALIZATION_TIMEOUT`, `INITIALIZATION_ERROR`, `TIMEOUT`, `QUEUE_FULL`,
  `RETRIEVER_ERROR`, and `CIRCUIT_OPEN` outcomes;
- overlap@5 and rank-biased overlap for divergence detection, plus completion
  rate and shadow-latency p50/p95;
- aggregate-only user reporting in Memory Center, with immediate withdrawal and
  deletion.
- an explicit insufficient-data warning until at least 100 completed runs; rank
  overlap measures divergence from baseline, not relevance or clinical quality.

Enable this research path only on a prepared demo installation:

```powershell
$env:PSYAVATAR_MEMORY_SHADOW_ENABLED="true"
```

On the current CPU-only development machine, a cached offline BGE-M3 smoke took
about 61.86 s to construct the provider and 211 ms for one 1024-dimensional
embedding. V1.6 therefore starts a consent-triggered background prewarm with a
separate 120 s initialization limit and uses a 15 s retrieval limit only after
the model is resident. Runtime status distinguishes `cold`, `warming`, `ready`,
and `failed`. These are single-machine engineering measurements, not an SLO.

The user must then opt in under **My Memory -> Help improve memory retrieval**.
The relevant endpoints are:

```text
GET /api/sessions/{id}/memories/research-consent
PUT /api/sessions/{id}/memories/research-consent
GET /api/sessions/{id}/memories/shadow-report
```

An operator can export the same aggregate-only payload without printing the
pseudonymous subject ID or any per-turn row:

```powershell
cd services/orchestrator
.\.venv\Scripts\python.exe -m app.memory.run_shadow_report `
  --database runtime\events.sqlite3 --user-id <pseudonymous-user-id>
```

Overlap is not a relevance label and cannot show that Hybrid is better. This
implementation is an operational/behavioral comparison baseline only. It must
not promote Hybrid to the response path without the independent labelled study
below, privacy review, and an explicit answer-path latency and quality decision.

### V1.6 evaluation scale-out still requiring external work

Expand beyond the V1.4 engineering fixture with annotators who did not implement the retriever:

- at least 100 users, 500 queries, and multi-annotator relevance judgements;
- threshold calibration on a development split and one-time reporting on a held-out split;
- confidence intervals and slices for paraphrase, negation, time, conflict, consent, and attack cases;
- a hybrid candidate generator plus reranker, evaluated against BGE-only and lexical-only branches;
- a blinded human audit of answer adherence and over-personalization.
- promote hybrid retrieval to the response path only after held-out quality, privacy review, and latency SLOs pass.

### V2: hierarchical consolidation

The conservative V2.0 profile baseline is now implemented. It consolidates only
confirmed evidence seen in at least two independent sessions, requires a second
explicit confirmation for the derived profile, freezes recall on contradiction,
and cascades source revocation/deletion into derived state. See
[`MEMORY_V2.md`](MEMORY_V2.md) for architecture, APIs, governance metrics, and
the incremental 2.x roadmap. Bi-temporal change interpretation, independent
human labels, and evaluated graph/multi-hop retrieval remain future work.
