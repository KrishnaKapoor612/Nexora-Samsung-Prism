"""Typed request and response models for the troubleshooting API."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class Condition(str, Enum):
    greater = "greater"
    equal = "equal"
    less = "less"


class ResultType(str, Enum):
    boolean = "boolean"
    integer = "integer"
    string = "str"
    float = "float"


class ActionCategory(str, Enum):
    auto = "auto"
    manual = "manual"
    critical = "critical"


class ValidationDeeplink(BaseModel):
    deeplink: str
    key: str
    resultType: ResultType | None = None
    condition: Condition | None = None
    value: str | None = None


class ActionableDeeplink(BaseModel):
    deeplink: str
    description: str
    message: str = ""
    classes: dict[str, str] | None = None
    originalType: str | None = None


class StepGroup(BaseModel):
    steps: list[str] = Field(min_length=1)
    validationDeeplink: ValidationDeeplink | None = None
    actionableDeeplink: ActionableDeeplink | None = None


class Action(BaseModel):
    actionName: str
    description: str
    stepGroups: list[StepGroup] = Field(min_length=1)
    category: ActionCategory = ActionCategory.manual


class Goal(BaseModel):
    goal: str
    title: str
    actions: list[Action] = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class TroubleshootingResponse(BaseModel):
    contexts: list[Goal] = Field(default_factory=list)


class TroubleshootRequest(BaseModel):
    query: str = Field(min_length=1)
    siis_response: str | None = None


class ResponseMeta(BaseModel):
    latency_ms: float = 0.0
    cache_hit: bool = False
    model: str = "deterministic"
    cost_usd: float = 0.0
    fallback: str | None = None
    reason: str | None = None


class TroubleshootEnvelope(BaseModel):
    query: str
    query_variations: list[str] = Field(default_factory=list)
    response: TroubleshootingResponse
    meta: ResponseMeta