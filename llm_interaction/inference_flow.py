from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

from llm_interaction.system_output_parser import (
    ParsedSystemOutput,
    format_for_conversation_log,
    parse_system_output,
)


ProviderCallable = Callable[..., Union[str, Tuple[str, dict[str, Any]]]]


@dataclass(frozen=True)
class InferenceRequest:
    prompt: str
    provider: str
    model_name: str
    system_prompt: str | None = None
    gen: dict[str, Any] = field(default_factory=dict)
    hf_local: dict[str, Any] = field(default_factory=dict)
    easy_steer: dict[str, Any] = field(default_factory=dict)
    repeng: dict[str, Any] = field(default_factory=dict)
    anthropic: dict[str, Any] = field(default_factory=dict)
    stop_markers: list[str] | None = None


@dataclass(frozen=True)
class InferenceResponse:
    raw_text: str
    parsed: ParsedSystemOutput
    conversation_log_line: str
    metadata: dict[str, Any] = field(default_factory=dict)


def run_inference_turn(
    request: InferenceRequest,
    *,
    provider_func: Optional[ProviderCallable] = None,
) -> InferenceResponse:
    if provider_func is None:
        if request.provider.lower() == "stub":
            provider_func = lambda **kwargs: kwargs.get("gen", {}).get(
                "response", '{"type": "message", "content": "stub response"}'
            )
        else:
            from llm_interaction.providers.factory import generate_with_provider

            provider_func = generate_with_provider

    raw = provider_func(
        prompt=request.prompt,
        provider=request.provider,
        model_name=request.model_name,
        system_prompt=request.system_prompt,
        gen=request.gen,
        hf_local=request.hf_local,
        easy_steer=request.easy_steer,
        repeng=request.repeng,
        anthropic=request.anthropic,
        stop_markers=request.stop_markers,
    )

    metadata: dict[str, Any] = {}
    if isinstance(raw, tuple):
        raw_text, metadata = raw
    else:
        raw_text = raw

    parsed = parse_system_output(raw_text)
    return InferenceResponse(
        raw_text=raw_text,
        parsed=parsed,
        conversation_log_line=format_for_conversation_log(parsed),
        metadata=metadata,
    )
