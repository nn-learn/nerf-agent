PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS consents (
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    granted INTEGER NOT NULL CHECK (granted IN (0, 1)),
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, kind),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    seq INTEGER NOT NULL CHECK (seq > 0),
    type TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    cancel_token TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_events_session_seq
    ON events(session_id, seq);

CREATE TABLE IF NOT EXISTS outbox (
    event_id TEXT PRIMARY KEY,
    published INTEGER NOT NULL DEFAULT 0 CHECK (published IN (0, 1)),
    created_at_ms INTEGER NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    level TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS visual_observations (
    observation_id TEXT PRIMARY KEY,
    source_event_id TEXT NOT NULL,
    trigger_turn_id TEXT NOT NULL,
    captured_at_ms INTEGER NOT NULL,
    valid_until_ms INTEGER NOT NULL,
    observation_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_items (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL,
    contains_sensitive_content INTEGER NOT NULL CHECK (contains_sensitive_content IN (0, 1)),
    source_turn_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    state TEXT NOT NULL,
    content TEXT NOT NULL,
    aspect TEXT NOT NULL DEFAULT 'FACT',
    subject_key TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    source_message_ids_json TEXT NOT NULL DEFAULT '[]',
    source_window_id TEXT,
    purpose_scope TEXT NOT NULL DEFAULT 'personalization',
    valid_from_ms INTEGER,
    expires_at_ms INTEGER,
    user_confirmed INTEGER NOT NULL DEFAULT 0 CHECK (user_confirmed IN (0, 1)),
    integrity_flags_json TEXT NOT NULL DEFAULT '[]',
    user_edited INTEGER NOT NULL DEFAULT 0 CHECK (user_edited IN (0, 1)),
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    supersedes_memory_id TEXT,
    source_type TEXT NOT NULL DEFAULT 'LEGACY',
    sensitivity TEXT NOT NULL DEFAULT 'GENERAL',
    allowed_uses_json TEXT NOT NULL DEFAULT '["PERSONALIZATION","RESPONSE_CONTEXT"]',
    observed_at_ms INTEGER,
    valid_to_ms INTEGER,
    derived_from_memory_ids_json TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS memory_retrievals (
    retrieval_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    score REAL NOT NULL,
    relevance_score REAL NOT NULL,
    reason_codes_json TEXT NOT NULL,
    used_at_ms INTEGER NOT NULL,
    UNIQUE(session_id, turn_id, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_retrieval_latest
    ON memory_retrievals(user_id, memory_id, used_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_observations (
    observation_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    valid_at_ms INTEGER NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    UNIQUE(user_id, memory_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_observations_user_time
    ON memory_observations(user_id, observed_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_profiles (
    profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    aspect TEXT NOT NULL,
    statement TEXT NOT NULL,
    state TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    supporting_evidence_count INTEGER NOT NULL,
    conflicting_evidence_count INTEGER NOT NULL,
    distinct_session_count INTEGER NOT NULL,
    contains_sensitive_content INTEGER NOT NULL CHECK (contains_sensitive_content IN (0, 1)),
    purpose_scope TEXT NOT NULL DEFAULT 'personalization',
    valid_from_ms INTEGER,
    valid_to_ms INTEGER,
    expires_at_ms INTEGER,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    profile_version TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    user_edited INTEGER NOT NULL DEFAULT 0 CHECK (user_edited IN (0, 1)),
    UNIQUE(user_id, subject_key, signature_hash)
);

CREATE INDEX IF NOT EXISTS idx_memory_profiles_active_scope
    ON memory_profiles(user_id, state, purpose_scope, expires_at_ms);

CREATE TABLE IF NOT EXISTS memory_profile_evidence (
    profile_id TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    PRIMARY KEY(profile_id, observation_id),
    FOREIGN KEY(profile_id) REFERENCES memory_profiles(profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_profile_evidence_memory
    ON memory_profile_evidence(memory_id, profile_id);

CREATE TABLE IF NOT EXISTS memory_conflict_groups (
    conflict_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    state TEXT NOT NULL,
    evidence_digest TEXT NOT NULL,
    selected_profile_id TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    UNIQUE(user_id, subject_key, evidence_digest)
);

CREATE INDEX IF NOT EXISTS idx_memory_conflicts_user_state
    ON memory_conflict_groups(user_id, state, updated_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_conflict_options (
    conflict_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    PRIMARY KEY(conflict_id, profile_id),
    FOREIGN KEY(conflict_id) REFERENCES memory_conflict_groups(conflict_id)
        ON DELETE CASCADE,
    FOREIGN KEY(profile_id) REFERENCES memory_profiles(profile_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_profile_rejections (
    user_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    rejected_at_ms INTEGER NOT NULL,
    PRIMARY KEY(user_id, subject_key, signature_hash)
);

CREATE TABLE IF NOT EXISTS memory_profile_changes (
    change_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    state TEXT NOT NULL,
    previous_profile_id TEXT NOT NULL,
    proposed_profile_id TEXT NOT NULL,
    effective_at_ms INTEGER NOT NULL,
    observed_at_ms INTEGER NOT NULL,
    evidence_digest TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    UNIQUE(user_id, subject_key, evidence_digest),
    FOREIGN KEY(previous_profile_id) REFERENCES memory_profiles(profile_id)
        ON DELETE CASCADE,
    FOREIGN KEY(proposed_profile_id) REFERENCES memory_profiles(profile_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_changes_user_state
    ON memory_profile_changes(user_id, state, updated_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_episode_summaries (
    summary_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    purpose_scope TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    ended_at_ms INTEGER NOT NULL,
    contains_sensitive_content INTEGER NOT NULL CHECK (
        contains_sensitive_content IN (0, 1)
    ),
    source_digest TEXT NOT NULL,
    summary_version TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    UNIQUE(user_id, session_id, purpose_scope, source_digest)
);

CREATE INDEX IF NOT EXISTS idx_memory_episode_summaries_user_time
    ON memory_episode_summaries(user_id, purpose_scope, ended_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_episode_members (
    summary_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY(summary_id, memory_id),
    FOREIGN KEY(summary_id) REFERENCES memory_episode_summaries(summary_id)
        ON DELETE CASCADE,
    FOREIGN KEY(memory_id) REFERENCES memory_items(memory_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_episode_members_memory
    ON memory_episode_members(memory_id, summary_id);

CREATE TABLE IF NOT EXISTS memory_research_consents (
    user_id TEXT PRIMARY KEY,
    granted INTEGER NOT NULL CHECK (granted IN (0, 1)),
    policy_version TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES memory_subjects(user_id)
);

CREATE TABLE IF NOT EXISTS memory_shadow_runs (
    shadow_run_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    consent_policy_version TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    status TEXT NOT NULL,
    baseline_latency_ms REAL NOT NULL,
    shadow_latency_ms REAL,
    overlap_at_5 REAL,
    rank_biased_overlap REAL,
    baseline_count INTEGER NOT NULL,
    shadow_count INTEGER,
    error_code TEXT,
    created_at_ms INTEGER NOT NULL,
    completed_at_ms INTEGER,
    expires_at_ms INTEGER NOT NULL,
    UNIQUE(session_id, turn_id, strategy_version)
);

CREATE INDEX IF NOT EXISTS idx_memory_shadow_runs_user_time
    ON memory_shadow_runs(user_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS memory_shadow_rankings (
    shadow_run_id TEXT NOT NULL,
    arm TEXT NOT NULL,
    rank INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    score REAL NOT NULL,
    relevance_score REAL NOT NULL,
    reason_codes_json TEXT NOT NULL,
    PRIMARY KEY (shadow_run_id, arm, rank),
    FOREIGN KEY (shadow_run_id) REFERENCES memory_shadow_runs(shadow_run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_ingestion_cursors (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0 CHECK (last_sequence >= 0),
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);

CREATE TABLE IF NOT EXISTS memory_subjects (
    user_id TEXT PRIMARY KEY,
    token_digest TEXT NOT NULL UNIQUE,
    created_at_ms INTEGER NOT NULL,
    last_seen_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_memory_subjects (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    bound_at_ms INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES memory_subjects(user_id)
);

CREATE INDEX IF NOT EXISTS idx_session_memory_subject_user
    ON session_memory_subjects(user_id, bound_at_ms);

CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    state TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS handoffs (
    handoff_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    risk_assessment_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL,
    accepted_by TEXT
);

CREATE TABLE IF NOT EXISTS avatar_metrics (
    metric_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp_ms INTEGER NOT NULL
);
