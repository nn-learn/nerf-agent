from pydantic import BaseModel, Field


class EventEnvelope[PayloadT](BaseModel):
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    seq: int = Field(gt=0)
    type: str = Field(min_length=1)
    timestamp_ms: int = Field(gt=0)
    cancel_token: str = Field(min_length=1)
    payload: PayloadT
