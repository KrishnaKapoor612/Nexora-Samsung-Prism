"""FastAPI application for Smart Guided Troubleshooting."""

import os
import importlib

from fastapi import FastAPI

from app.config import load_assets
from app.evidence import EvidenceProvider, EvidenceProviderChain
from app.generator import DeterministicPlanGenerator
from app.models import TroubleshootEnvelope, TroubleshootRequest
from app.nova_generator import NovaLitePlanGenerator
from app.pipeline import TroubleshootingPipeline
from app.web_provider import WebEvidenceProvider


assets = load_assets()

# SIIS is the pipeline's primary provider. Web RAG is the only configured
# fallback and is attempted only after SIIS has no relevant evidence.
_fallback_providers: list[EvidenceProvider] = []
if os.getenv("ENABLE_WEB_RAG", "").lower() in {"1", "true", "yes"}:
    _fallback_providers.append(WebEvidenceProvider())
fallback_provider: EvidenceProvider | None = (
    EvidenceProviderChain(_fallback_providers) if _fallback_providers else None
)

_generation_provider = os.getenv("GENERATION_PROVIDER", "deterministic").lower()
if _generation_provider == "nova" or (
    _generation_provider == "deterministic"
    and os.getenv("ENABLE_NOVA", "").lower() in {"1", "true", "yes"}
):
    generator = NovaLitePlanGenerator()
elif _generation_provider == "samsung":
    # Samsung SDKs differ by deployment. An integration can provide a factory
    # returning an object with the shared generate(query, evidence) contract.
    factory_path = os.getenv("SAMSUNG_GENERATOR_FACTORY", "")
    if factory_path:
        module_name, attribute = factory_path.split(":", 1)
        generator = getattr(importlib.import_module(module_name), attribute)()
    else:
        # Keep startup independent of credentials and vendor SDK availability.
        generator = DeterministicPlanGenerator()
elif _generation_provider == "deterministic":
    generator = DeterministicPlanGenerator()
else:
    raise ValueError("GENERATION_PROVIDER must be samsung, nova, or deterministic")
pipeline = TroubleshootingPipeline(
    assets,
    fallback_provider=fallback_provider,
    generator=generator,
)
app = FastAPI(title="Smart_Guided_Troubleshooting_Engine_Team_Nexora")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/troubleshoot", response_model=TroubleshootEnvelope)
def troubleshoot(request: TroubleshootRequest) -> TroubleshootEnvelope:
    return pipeline.run(request)
