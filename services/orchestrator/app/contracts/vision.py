from pydantic import BaseModel, Field, model_validator

PROHIBITED_APPEARANCE_INFERENCES = (
    "抑郁症",
    "焦虑症",
    "自伤倾向",
    "自杀倾向",
    "人格障碍",
    "精神病",
    "depression",
    "anxiety disorder",
    "suicidal",
    "personality disorder",
)


class VisualObject(BaseModel):
    label: str = Field(min_length=1)
    attributes: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class VisualObservation(BaseModel):
    observation_id: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    trigger_turn_id: str = Field(min_length=1)
    captured_at_ms: int = Field(ge=0)
    valid_until_ms: int = Field(gt=0)
    frame_count: int = Field(ge=1, le=4)
    scene_summary: str
    objects: list[VisualObject] = Field(default_factory=list)
    visible_text_summary: str
    action_summary: str
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)
    contains_sensitive_content: bool
    prohibited_inferences_removed: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_visual_safety_boundary(self) -> "VisualObservation":
        if self.valid_until_ms != self.captured_at_ms + 10_000:
            raise ValueError("visual observation lifetime must be exactly 10 seconds")

        appearance_claims = f"{self.scene_summary} {self.action_summary}".casefold()
        if any(
            term.casefold() in appearance_claims
            for term in PROHIBITED_APPEARANCE_INFERENCES
        ):
            raise ValueError("psychiatric inference from appearance is forbidden")
        return self

    def is_fresh(self, now_ms: int) -> bool:
        return now_ms < self.valid_until_ms
