"""Amazon Nova Lite structured troubleshooting generator."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import boto3
from botocore.config import Config
from pydantic import ValidationError

from app.evidence import Evidence
from app.models import TroubleshootingResponse


class NovaGenerationError(RuntimeError):
    """Raised when Nova returns unusable structured output."""


class NovaLitePlanGenerator:
    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_id = model_id or os.getenv(
            "NOVA_MODEL_ID",
            "amazon.nova-lite-v1:0",
        )
        self.region_name = region_name or os.getenv(
            "AWS_REGION",
            "us-east-1",
        )

        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=self.region_name,
            config=Config(
                read_timeout=3600,
                connect_timeout=10,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    def generate(
        self,
        query: str,
        evidence: Evidence,
    ) -> TroubleshootingResponse:
        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(query, evidence)

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[
                    {
                        "text": system_prompt,
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": user_prompt,
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "temperature": 0.0,
                    "topP": 0.1,
                    "maxTokens": 3000,
                },
            )
        except Exception as exc:
            raise NovaGenerationError(
                "Amazon Nova Lite request failed"
            ) from exc

        text = self._extract_text(response)
        payload = self._parse_json(text)

        try:
            resp = TroubleshootingResponse.model_validate(payload)
            from app.generator import _action_description, _action_category, _goal_sentence, _goal_title, _split_into_atomic_steps
            from app.models import ActionCategory
            category_rank = {
                ActionCategory.auto: 0,
                ActionCategory.manual: 1,
                ActionCategory.critical: 2,
            }
            for ctx in resp.contexts:
                ctx.title = _goal_title(ctx.title or evidence.title)
                ctx.goal = _goal_sentence(query, ctx.title)
                ctx.score = round(min(ctx.score or evidence.score, 1.0), 2)
                for action in ctx.actions:
                    action.category = _action_category(action.actionName)
                    action.description = _action_description(action.actionName)
                    for group in action.stepGroups:
                        final_steps = []
                        for step in group.steps:
                            if len(step) > 120:
                                split_steps = _split_into_atomic_steps(step)
                                if split_steps:
                                    final_steps.extend(split_steps)
                                else:
                                    final_steps.append(step)
                            else:
                                final_steps.append(step)
                        group.steps = final_steps
                ctx.actions.sort(key=lambda a: category_rank.get(a.category, 1))
            return resp
        except ValidationError as exc:
            raise NovaGenerationError(
                "Nova Lite returned schema-invalid troubleshooting JSON"
            ) from exc

    @staticmethod
    def _system_prompt() -> str:
        return """
You are a Samsung troubleshooting plan extractor.

Your task is to convert the supplied source evidence into a structured
troubleshooting response.

Strict rules:
1. Use only instructions supported by the supplied evidence.
2. Do not invent troubleshooting steps.
3. Do not use general knowledge to fill missing information.
4. Do not generate actionableDeeplink or validationDeeplink objects.
5. Do not generate URLs or Bixby deeplink strings.
6. Group steps belonging to the same physical screen into one action.
7. Order safe actions before disruptive or critical actions.
8. Use category values only: auto, manual, critical.
9. Return only valid JSON.
10. If the evidence does not contain enough actionable instructions, return:
   {"contexts": []}

The JSON must have this shape:
{
  "contexts": [
    {
      "goal": "Follow these steps to perform this <topic> Troubleshooting",
      "title": "Two or three words",
      "score": 0.0,
      "actions": [
        {
          "actionName": "Action name",
          "description": "It will help resolve this issue",
          "category": "auto",
          "stepGroups": [
            {
              "steps": [
                "Imperative step."
              ],
              "actionableDeeplink": null,
              "validationDeeplink": null
            }
          ]
        }
      ]
    }
  ]
}
""".strip()

    @staticmethod
    def _user_prompt(query: str, evidence: Evidence) -> str:
        return f"""
User query:
{query}

Evidence source:
{evidence.source}

Evidence title:
{evidence.title}

Evidence content:
{evidence.content}

Extract a grounded troubleshooting plan from this evidence.
""".strip()

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        try:
            content = response["output"]["message"]["content"]
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
        except (KeyError, TypeError) as exc:
            raise NovaGenerationError(
                "Unexpected Amazon Nova response shape"
            ) from exc

        text = "\n".join(text_parts).strip()
        if not text:
            raise NovaGenerationError("Amazon Nova returned empty output")
        return text

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise NovaGenerationError(
                "Amazon Nova output was not valid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise NovaGenerationError("Amazon Nova output must be a JSON object")

        return payload