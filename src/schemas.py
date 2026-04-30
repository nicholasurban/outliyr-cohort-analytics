from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Direction = Literal["higher_better", "lower_better", "optimal_range", "track_only"]


class Endpoint(BaseModel):
    key: str
    label: str
    direction: Direction = "higher_better"
    role: Literal["primary", "secondary"] = "secondary"
    optimal_min: Optional[float] = None
    optimal_max: Optional[float] = None
    responder_threshold: Optional[float] = None


class ParticipantEndpoint(BaseModel):
    baseline_values: List[float] = Field(default_factory=list)
    intervention_values: List[float] = Field(default_factory=list)
    daily_values: List[Optional[float]] = Field(default_factory=list)


class Participant(BaseModel):
    participant_id: str
    status: str = "enrolled"
    verified_compliance_pct: Optional[float] = None
    dropout_reason: Optional[str] = None
    subgroups: Dict[str, str] = Field(default_factory=dict)
    endpoints: Dict[str, ParticipantEndpoint] = Field(default_factory=dict)


class TrialMeta(BaseModel):
    sponsored_trial_id: int
    title: str
    sponsor_name: str = ""
    protocol_hash: Optional[str] = None
    primary_endpoint: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


class AnalyzeRequest(BaseModel):
    trial: TrialMeta
    endpoints: List[Endpoint]
    participants: List[Participant]
    concurrent_interventions: List[Dict[str, Any]] = Field(default_factory=list)
    adverse_events: List[Dict[str, Any]] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    engine_version: str
    generated_at: str
    trial: Dict[str, Any]
    endpoint_results: List[Dict[str, Any]]
    dropout_sensitivity: Dict[str, Any]
    subgroup_analysis: Dict[str, Any]
    charts: Dict[str, Any]
    methodology: Dict[str, Any]
    report_hash: str
    pdf_base64: str
