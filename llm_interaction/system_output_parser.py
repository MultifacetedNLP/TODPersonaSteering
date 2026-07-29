"""
System Output Parser

Parses and sanitizes system model outputs to enforce a structured contract:
- Extracts JSON objects with type: "message" or "api_call"
- Falls back to extracting the first valid APICall(...) if no JSON found
- Strips any embedded User: turns, Search Results blocks, and other noise
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

from llm_interaction.contracts import ApiCall, SystemOutputType


@dataclass
class ParsedSystemOutput:
    """Structured result of parsing a system model output."""
    output_type: SystemOutputType
    content: Optional[str] = None  # For message type
    method: Optional[str] = None  # For api_call type
    parameters: Optional[Dict[str, Any]] = None  # For api_call type
    raw_api_call: Optional[str] = None  # Reconstructed APICall(...) string
    end_dialog: bool = False  # System signals the conversation is complete

    def api_call(self) -> Optional[ApiCall]:
        if self.output_type != "api_call" or not self.method:
            return None
        return ApiCall(method=self.method, parameters=self.parameters or {})


# Patterns for stripping noise
NOISE_PATTERNS = [
    # Strip anything after \nUser: (model continuing the transcript)
    (r'\n\s*User:.*', re.DOTALL),
    # Strip Search Results blocks (hallucinated tool output)
    (r'\n?\s*Search Results\s*\n.*?End Search Results', re.DOTALL),
    # Strip standalone Search Results without End marker (truncated)
    (r'\n?\s*Search Results\s*\n.*$', re.DOTALL),
    # Strip any System: prefix at the start (we add it ourselves)
    (r'^System:\s*', 0),
]


def sanitize_raw_output(raw_text: str) -> str:
    """
    Remove embedded User: turns, Search Results blocks, and other noise
    from raw model output before parsing.
    """
    text = raw_text
    for pattern, flags in NOISE_PATTERNS:
        if flags:
            text = re.sub(pattern, '', text, flags=flags)
        else:
            text = re.sub(pattern, '', text)
    # Strip common wrappers (code fences, stray quotes)
    text = text.strip()
    # Remove surrounding Markdown code fences if present
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    # Remove surrounding single/double quotes if the entire payload is quoted
    if (len(text) >= 2) and ((text[0] == text[-1]) and text[0] in ("'", '"')):
        text = text[1:-1].strip()
    return text.strip()


_UNQUOTED_KEY_RE = re.compile(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')


def _repair_almost_json(candidate: str) -> str:
    """
    Repair common "almost JSON" variants produced by LLMs:
    - Unquoted object keys: {type: "message"} -> {"type": "message"}
    - Escaped underscores: \\_ -> _
    This is intentionally conservative and only targets keys in object positions.
    """
    fixed = candidate.replace("\\_", "_").strip()
    # Quote unquoted keys that appear after '{' or ',' (object contexts)
    fixed = _UNQUOTED_KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}":', fixed)
    return fixed


def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid JSON object from text.
    Returns None if no valid JSON found.
    """
    # Try to find JSON object with braces
    # Look for patterns like { ... } that could be JSON
    brace_depth = 0
    start_idx = None
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                candidate = text[start_idx:i+1]
                # 1) Strict JSON
                obj = _try_parse_json(candidate)
                if obj is not None:
                    return obj
                # 2) Attempt to repair "almost JSON" (unquoted keys, etc.)
                repaired = _repair_almost_json(candidate)
                obj = _try_parse_json(repaired)
                if obj is not None:
                    return obj
                # Reset and continue searching
                start_idx = None
                continue
    return None


def extract_api_call_legacy(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first valid APICall(...) from text using legacy format.
    Returns dict with method and parameters if found.
    """
    # Match APICall(method='MethodName', parameters={...})
    # Be flexible with quotes and spacing
    api_pattern = r"APICall\s*\(\s*method\s*=\s*['\"]?([^'\",(]+)['\"]?\s*,\s*parameters\s*=\s*\{([^}]*)\}"
    match = re.search(api_pattern, text, re.IGNORECASE)
    
    if not match:
        return None
    
    method_name = match.group(1).strip().strip("'\"")
    params_str = match.group(2)
    
    # Parse parameters - handle both quoted and unquoted keys/values
    params = {}
    
    # Try pattern: key: value or 'key': 'value'
    pairs = re.findall(r"['\"]?(\w+)['\"]?\s*:\s*([^,}]+)", params_str)
    for key, value in pairs:
        key = key.strip().strip("'\"")
        value = value.strip().strip("'\"")
        params[key] = value
    
    if method_name:
        return {
            "method": method_name,
            "parameters": params
        }
    return None


def parse_system_output(raw_text: str) -> ParsedSystemOutput:
    """
    Parse and validate system model output.
    
    Returns a ParsedSystemOutput with:
    - output_type: "message" or "api_call"
    - For messages: content field populated
    - For api_calls: method, parameters, and raw_api_call populated
    
    Priority:
    1. Try to extract valid JSON with type field
    2. Fall back to legacy APICall(...) extraction
    3. If neither, treat remaining text as a message
    """
    # First sanitize to remove noise
    sanitized = sanitize_raw_output(raw_text)
    
    if not sanitized:
        return ParsedSystemOutput(
            output_type="message",
            content="I apologize, I couldn't generate a proper response."
        )
    
    # Try JSON extraction first
    json_obj = extract_json_object(sanitized)
    
    if json_obj:
        obj_type = json_obj.get("type", "").lower()
        
        if obj_type == "api_call":
            method = json_obj.get("method", "")
            params = json_obj.get("parameters", {})
            # Reconstruct canonical APICall string
            params_str = ", ".join(f"'{k}': '{v}'" for k, v in params.items())
            raw_call = f"APICall(method='{method}', parameters={{{params_str}}})"
            return ParsedSystemOutput(
                output_type="api_call",
                method=method,
                parameters=params,
                raw_api_call=raw_call
            )
        elif obj_type == "message":
            content = json_obj.get("content", "")
            end_dialog = bool(json_obj.get("end_dialog", False))
            return ParsedSystemOutput(
                output_type="message",
                content=content,
                end_dialog=end_dialog,
            )

    # If we didn't get a JSON object, try to salvage common almost-JSON patterns
    # Example: {"type": "message", content: "..."}, {"type":"api_call", method:"X", parameters:{...}}
    lowered = sanitized.lower()
    if "type" in lowered and "{" in sanitized and "}" in sanitized:
        # Extract message content if present
        if re.search(r'(["\']?type["\']?\s*:\s*["\']message["\'])|(type\s*:\s*["\']message["\'])', sanitized, re.IGNORECASE):
            m = re.search(r'(?:\"content\"|content)\s*:\s*\"(?P<content>(?:\\.|[^\"\\])*)\"', sanitized)
            if m:
                # Decode escapes safely by leveraging JSON string parsing
                try:
                    decoded = json.loads(f"\"{m.group('content')}\"")
                except Exception:
                    decoded = m.group("content")
                return ParsedSystemOutput(output_type="message", content=decoded.strip())
        # Extract api_call method + parameters if present
        if re.search(r'(["\']?type["\']?\s*:\s*["\']api_call["\'])|(type\s*:\s*["\']api_call["\'])', sanitized, re.IGNORECASE):
            mm = re.search(r'(?:\"method\"|method)\s*:\s*\"(?P<method>(?:\\.|[^\"\\])*)\"', sanitized)
            if mm:
                try:
                    method = json.loads(f"\"{mm.group('method')}\"")
                except Exception:
                    method = mm.group("method")
                # Try to extract parameters object substring and parse after repair
                params: Dict[str, Any] = {}
                p_idx = re.search(r'(?:\"parameters\"|parameters)\s*:\s*\{', sanitized)
                if p_idx:
                    start = p_idx.start()
                    # Find the first '{' after 'parameters:'
                    brace_start = sanitized.find("{", start)
                    if brace_start != -1:
                        depth = 0
                        end = None
                        for j in range(brace_start, len(sanitized)):
                            if sanitized[j] == "{":
                                depth += 1
                            elif sanitized[j] == "}":
                                depth -= 1
                                if depth == 0:
                                    end = j + 1
                                    break
                        if end:
                            params_candidate = sanitized[brace_start:end]
                            params_obj = _try_parse_json(params_candidate)
                            if params_obj is None:
                                params_obj = _try_parse_json(_repair_almost_json(params_candidate))
                            if isinstance(params_obj, dict):
                                params = params_obj
                params_str = ", ".join(f"'{k}': '{v}'" for k, v in params.items())
                raw_call = f"APICall(method='{method}', parameters={{{params_str}}})"
                return ParsedSystemOutput(
                    output_type="api_call",
                    method=method,
                    parameters=params,
                    raw_api_call=raw_call
                )

    # Handle truncated JSON objects where the closing quote/brace is missing.
    # Example (missing closing quote and '}'):
    #   {"type": "message", content: "Hello there,
    # or ending with a dangling comma:
    #   {"type": "message", content: "Hello there",
    if "type" in lowered and "{" in sanitized:
        # Truncated message case
        if re.search(r'(["\']?type["\']?\s*:\s*["\']message["\'])|(type\s*:\s*["\']message["\'])', sanitized, re.IGNORECASE):
            # Prefer a properly closed quote if present; otherwise capture to end
            m_closed = re.search(r'(?:\"content\"|content)\s*:\s*\"(?P<content>(?:\\.|[^\"\\])*)\"', sanitized)
            if m_closed:
                try:
                    decoded = json.loads(f"\"{m_closed.group('content')}\"")
                except Exception:
                    decoded = m_closed.group("content")
                return ParsedSystemOutput(output_type="message", content=decoded.strip())

            m_open = re.search(r'(?:\"content\"|content)\s*:\s*\"(?P<content>.*)$', sanitized, flags=re.DOTALL)
            if m_open:
                raw = m_open.group("content")
                # Remove trailing punctuation that typically follows truncated JSON
                raw = re.sub(r'[\s,}]*$', '', raw)
                # Basic unescape (we cannot rely on json.loads without a closing quote)
                raw = raw.replace(r'\\', '\\').replace(r'\"', '"').replace(r'\n', '\n').replace(r'\t', '\t')
                return ParsedSystemOutput(output_type="message", content=raw.strip())

        # Truncated api_call case (best-effort salvage)
        if re.search(r'(["\']?type["\']?\s*:\s*["\']api_call["\'])|(type\s*:\s*["\']api_call["\'])', sanitized, re.IGNORECASE):
            mm_closed = re.search(r'(?:\"method\"|method)\s*:\s*\"(?P<method>(?:\\.|[^\"\\])*)\"', sanitized)
            mm_open = re.search(r'(?:\"method\"|method)\s*:\s*\"(?P<method>.*)$', sanitized)
            method_val = None
            if mm_closed:
                try:
                    method_val = json.loads(f"\"{mm_closed.group('method')}\"")
                except Exception:
                    method_val = mm_closed.group("method")
            elif mm_open:
                method_val = re.sub(r'[\s,}]*$', '', mm_open.group("method")).strip()
                method_val = method_val.replace(r'\\', '\\').replace(r'\"', '"')
            if method_val:
                return ParsedSystemOutput(
                    output_type="api_call",
                    method=method_val,
                    parameters={},
                    raw_api_call=f"APICall(method='{method_val}', parameters={{}})"
                )
    
    # Fall back to legacy APICall extraction
    api_call = extract_api_call_legacy(sanitized)
    
    if api_call:
        method = api_call["method"]
        params = api_call["parameters"]
        params_str = ", ".join(f"'{k}': '{v}'" for k, v in params.items())
        raw_call = f"APICall(method='{method}', parameters={{{params_str}}})"
        
        # Check if there's text before the APICall (preamble message)
        api_match = re.search(r"APICall\s*\(", sanitized, re.IGNORECASE)
        preamble = ""
        if api_match and api_match.start() > 0:
            preamble = sanitized[:api_match.start()].strip()
        
        return ParsedSystemOutput(
            output_type="api_call",
            content=preamble if preamble else None,
            method=method,
            parameters=params,
            raw_api_call=raw_call
        )
    
    # No API call found - treat as plain message
    # Clean up any remaining APICall-like fragments that didn't parse
    clean_content = re.sub(r'APICall\s*\([^)]*$', '', sanitized).strip()
    
    return ParsedSystemOutput(
        output_type="message",
        content=clean_content if clean_content else sanitized
    )


def format_for_conversation_log(parsed: ParsedSystemOutput) -> str:
    """
    Format parsed output for appending to conversation log.
    Always prefixes with 'System: '.
    """
    if parsed.output_type == "api_call":
        if parsed.content:
            # Has preamble message before API call
            return f"System: {parsed.content} {parsed.raw_api_call}"
        return f"System: {parsed.raw_api_call}"
    else:
        return f"System: {parsed.content}"
