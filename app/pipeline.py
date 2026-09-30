"""End-to-end deterministic troubleshooting pipeline."""

from __future__ import annotations

import time

from app.cache import ResponseCache
from app.config import AppAssets
from app.deeplink_mapper import DeeplinkMapper
from app.evidence import (
    Evidence,
    EvidenceProvider,
    EvidenceProviderChain,
    LocalSIISProvider,
    select_evidence,
)
from app.generator import DeterministicPlanGenerator, StructuredPlanGenerator
from app.models import (
    ActionableDeeplink,
    ValidationDeeplink,
    ResponseMeta,
    TroubleshootEnvelope,
    TroubleshootRequest,
    TroubleshootingResponse,
)
from app.query_understanding import fingerprint_query, normalized_cache_key
from app.retrieval import normalize_text
from app.validators import ResponseValidationError, validate_response


def query_variations(query: str) -> list[str]:
    normalized = normalize_text(query)
    return list(
        dict.fromkeys(
            [
                normalized,
                f"troubleshoot {normalized}",
                f"Samsung device {normalized}",
                f"Galaxy issue {normalized}",
                f"phone problem {normalized}",
                f"how to fix {normalized}",
                f"{normalized} not working",
                f"help with {normalized}",
            ]
        )
    )[:10]


def _attach_deeplinks(
    response: TroubleshootingResponse, mapper: DeeplinkMapper
) -> TroubleshootingResponse:
    for goal in response.contexts:
        for action in goal.actions:
            for group in action.stepGroups:
                match = mapper.match(action.actionName, action.description, group.steps)
                if match is None:
                    continue
                group.actionableDeeplink = ActionableDeeplink.model_validate(
                    mapper.actionable_payload(match)
                )
                validation_payload = mapper.validation_payload(match)
                if validation_payload:
                    group.validationDeeplink = ValidationDeeplink.model_validate(
                        validation_payload
                    )
    return response


class TroubleshootingPipeline:
    def __init__(
        self,
        assets: AppAssets,
        evidence_provider: EvidenceProvider | None = None,
        fallback_provider: EvidenceProvider | None = None,
        generator: StructuredPlanGenerator | None = None,
    ) -> None:
        self.assets = assets
        self.mapper = DeeplinkMapper(assets)
        self.cache = ResponseCache()
        self.generator = generator or DeterministicPlanGenerator()
        self.deterministic_generator = DeterministicPlanGenerator()
        self.evidence_provider = evidence_provider or LocalSIISProvider(assets)
        self.fallback_provider = fallback_provider
        providers = [self.evidence_provider]
        if fallback_provider is not None:
            providers.append(fallback_provider)
        self.provider_chain = EvidenceProviderChain(providers)

    def _select_evidence(self, request: TroubleshootRequest) -> Evidence | None:
        candidates = self.provider_chain.search(request.query, limit=5)
        return select_evidence(request.query, candidates)

    def run(self, request: TroubleshootRequest) -> TroubleshootEnvelope:
        started = time.perf_counter()
        key = normalized_cache_key(request.query)
        fingerprint = fingerprint_query(request.query)
        cached = self.cache.get(key, fingerprint)
        if cached is not None:
            cached_envelope = TroubleshootEnvelope.model_validate(cached.value)
            cached_envelope.meta.cache_hit = True
            cached_envelope.meta.reason = (
                f"{cached.compatibility}_cache:{cached.similarity:.3f}"
            )
            return cached_envelope

        if request.siis_response:
            evidence = Evidence(
                source="request_payload",
                title="Provided SIIS Reference",
                content=request.siis_response,
                score=1.0,
                source_type="local"
            )
        else:
            evidence = self._select_evidence(request)
            
        if evidence is None:
            result = TroubleshootingResponse(contexts=[])
            envelope = self._envelope(request, result, started, False, "no_match")
            envelope.meta.reason = "evidence_not_found_or_incompatible"
            self.cache.put(
                key,
                envelope.model_dump(),
                fingerprint=fingerprint,
                cacheable=False,
            )
            return envelope

        try:
            result = self.generator.generate(request.query, evidence)
        except Exception:
            # The fallback consumes the same retrieved evidence and cannot add
            # instructions that were not present in that evidence.
            result = self.deterministic_generator.generate(request.query, evidence)
        if not result.contexts:
            envelope = self._envelope(request, result, started, False, "no_match")
            envelope.meta.reason = "generator_returned_no_steps"
            self.cache.put(
                key,
                envelope.model_dump(),
                fingerprint=fingerprint,
                cacheable=False,
            )
            return envelope
        # Deeplinks come from catalog entries only. The mapper can decline an
        # uncertain match, leaving the deeplink fields empty.
        result = _attach_deeplinks(result, self.mapper)
        try:
            result = validate_response(
                result,
                self.assets.deeplink_uris,
                f"{evidence.title}\n{evidence.content}",
                catalog_entries=self.assets.deeplinks,
                query_fingerprint=fingerprint,
            )
        except ResponseValidationError:
            result = TroubleshootingResponse(contexts=[])
            fallback = "no_match"
        else:
            fallback = None
        envelope = self._envelope(request, result, started, False, fallback)
        if fallback == "no_match":
            envelope.meta.reason = "generated_response_failed_validation"
            self.cache.put(key, envelope.model_dump(), fingerprint=fingerprint, cacheable=False)
        else:
            self.cache.put(key, envelope.model_dump(), fingerprint=fingerprint)
        return envelope

    def _envelope(
        self,
        request: TroubleshootRequest,
        response: TroubleshootingResponse,
        started: float,
        cache_hit: bool,
        fallback: str | None,
    ) -> TroubleshootEnvelope:
        return TroubleshootEnvelope(
            query=request.query,
            query_variations=query_variations(request.query),
            response=response,
            meta=ResponseMeta(
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                cache_hit=cache_hit,
                model=(
                    "nova-lite"
                    if self.generator.__class__.__name__ == "NovaLitePlanGenerator"
                    else "deterministic"
                ),
                fallback=fallback,
            ),
        )
