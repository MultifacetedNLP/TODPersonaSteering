from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_interaction.contracts import ApiCall
from llm_interaction.system_output_parser import parse_system_output


@dataclass(frozen=True)
class ApiCallMetrics:
    method_accuracy: float
    key_accuracy: float
    value_accuracy: float
    full_accuracy: float


def _normalize_call(call: ApiCall | str | dict[str, Any] | None) -> ApiCall | None:
    if call is None or call == "":
        return None
    if isinstance(call, ApiCall):
        return call
    if isinstance(call, dict):
        method = call.get("method")
        parameters = call.get("parameters", {})
        if method:
            return ApiCall(method=str(method), parameters=dict(parameters or {}))
        return None
    if isinstance(call, str):
        parsed = parse_system_output(call)
        return parsed.api_call()
    return None


def score_api_call(reference: ApiCall | str | dict[str, Any], prediction: ApiCall | str | dict[str, Any] | None) -> ApiCallMetrics:
    ref = _normalize_call(reference)
    pred = _normalize_call(prediction)
    if ref is None:
        raise ValueError("reference API call could not be parsed")
    if pred is None:
        return ApiCallMetrics(0.0, 0.0, 0.0, 0.0)

    method_accuracy = 1.0 if pred.method == ref.method else 0.0
    ref_keys = set(ref.parameters)
    pred_keys = set(pred.parameters)

    if not ref_keys:
        key_accuracy = 1.0 if not pred_keys else 0.0
        value_accuracy = key_accuracy
    else:
        matched_keys = ref_keys & pred_keys
        key_accuracy = len(matched_keys) / len(ref_keys)
        matched_values = sum(1 for key in matched_keys if pred.parameters.get(key) == ref.parameters.get(key))
        value_accuracy = matched_values / len(ref_keys)

    full_accuracy = 1.0 if method_accuracy == key_accuracy == value_accuracy == 1.0 else 0.0
    return ApiCallMetrics(method_accuracy, key_accuracy, value_accuracy, full_accuracy)


def score_api_calls(reference_calls: list[Any], predicted_calls: list[Any] | None) -> list[ApiCallMetrics]:
    predicted_calls = predicted_calls or []
    padded_predictions = predicted_calls[: len(reference_calls)] + [None] * max(
        0, len(reference_calls) - len(predicted_calls)
    )
    return [score_api_call(ref, pred) for ref, pred in zip(reference_calls, padded_predictions)]
