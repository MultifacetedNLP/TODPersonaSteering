"""Shared utilities and abstract base for all chat-simulator variants.

Concrete implementation lives in:
- ``sgd_chat_simulator.py`` — SGD dataset (single- and multi-domain)
"""

from __future__ import annotations

import os
import re
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Any

from llm_interaction._simulator_common import load_sample_conversation as _load_sample_conversation

logger = logging.getLogger(__name__)


def load_schema(schema_file: str, domain: str) -> str:
    """Extract the schema section for *domain* from *schema_file*."""
    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found at {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    domain_sections: list[str] = []
    collecting = False
    current_section: list[str] = []

    for line in lines:
        stripped = line.strip()
        service_match = re.match(r"^service_name:\s*(\S+)", stripped)
        if service_match:
            current_service = service_match.group(1)
            if collecting:
                if current_service != domain:
                    domain_sections.append("".join(current_section))
                current_section = []
                collecting = False
            if current_service == domain:
                collecting = True
                current_section.append(line)
            continue
        if collecting:
            current_section.append(line)

    if collecting and current_section:
        domain_sections.append("".join(current_section))

    if not domain_sections:
        raise ValueError(f"Domain '{domain}' not found in the schema file.")

    return "\n".join(domain_sections).strip()


def method_exists_in_schema(schema: str, method_name: str) -> bool:
    """Return True if *method_name* appears as an intent in *schema*."""
    inside_intents = False
    for line in schema.splitlines():
        stripped = line.strip()
        if stripped.startswith("Intents:"):
            inside_intents = True
            continue
        if inside_intents and stripped.startswith("Slots:"):
            inside_intents = False
            continue
        if inside_intents:
            match = re.match(r"^name:\s*(.+)", stripped)
            if match and match.group(1).strip() == method_name:
                return True
    return False


def create_chat_gpt_input(
    instruction_prompt: str,
    schema: str,
    domains_text: str,
    conversation: list[dict],
    add_example: bool = True,
    system_custom_instruction: str | None = None,
) -> str:
    """Build the system-side prompt string for the LLM."""
    lines = [
        instruction_prompt,
        "",
        "Schema:",
        "",
        schema,
        "end of schema",
        "",
    ]

    if add_example:
        lines.extend([
            "Here is a sample Dialog conversation between a System and a User:",
            "",
        ])
        for row in conversation:
            speaker = row["Speaker"].lower()
            utterance = row["Utterance"].strip('"')
            if speaker == "tool":
                lines.append(utterance)
            elif speaker in ("system", "user"):
                lines.append(f"{row['Speaker'].capitalize()}: {utterance}")
        lines.extend([
            "End of Conversation",
            "",
            "Understand the above structure of conversation between a User and a System. "
            "Learn how to interact with the User and generate the most human-like conversational "
            "response to the User's intent. You may need to make API calls and use the API call results.",
            f"Based on the above instructions and examples from the {domains_text} domain(s), learn how to "
            "interact with a User to generate the most human-like conversational response to the User's "
            "current intent.",
            "End of Instructions.",
            "",
        ])
    else:
        lines.extend([
            f"Based on the above instructions from the {domains_text} domain(s), learn how to interact "
            "with a User to generate the most human-like conversational response to the User's current intent.",
            "End of Instructions.",
            "",
        ])

    lines.extend([
        "Here are a few general guidelines to follow:",
        "- Please avoid asking too many slots in one turn; ideally, ask one slot at a time.",
        "- Don't overwhelm the User with too many questions or choices in one turn.",
        "- Confirm the slot values with the User before finalizing the call.",
        "- Follow the structure of API call from the above example whenever you are making an API Call.",
        "- If you're unsure about something, it's always better to ask or confirm with the User.",
        "- Do not provide all the information in the search results to the User. Provide details only if the User requests them.",
        "- If you feel the User is confused, guide the User with relevant suggestions and ensure it's relevant to their current intent.",
        "- You generate only one system response at a time and do not produce search results yourself; search results will be provided to you.",
    ])

    if system_custom_instruction:
        lines.extend(["", system_custom_instruction])

    lines.extend([
        "",
        "=== CRITICAL OUTPUT FORMAT REQUIREMENT ===",
        "You MUST respond with ONLY a valid JSON object. Do NOT write 'System:', 'User:', or any other role prefix.",
        'All JSON keys MUST be double-quoted (for example: "type", "content", "method", "parameters").',
        "Do NOT output JavaScript-style objects (no unquoted keys like content: \"...\" and no trailing commas).",
        "Do NOT generate 'Search Results' blocks. Those are provided externally after your API calls.",
        "",
        "Your response must be ONE of these two JSON formats:",
        "",
        "1. For a message to the user:",
        '   {"type": "message", "content": "Your response text here"}',
        "",
        "2. For an API call:",
        '   {"type": "api_call", "method": "MethodName", "parameters": {"slot1": "value1", "slot2": "value2"}}',
        "",
        "Examples:",
        '- {"type": "message", "content": "What date would you like to depart?"}',
        '- {"type": "api_call", "method": "SearchRoundtripFlights", "parameters": {"origin_airport": "Las Vegas", "destination_airport": "Seattle", "departure_date": "2019-03-04", "return_date": "2019-03-14"}}',
        "",
        "=== ENDING THE DIALOG ===",
        'YOU (the system) decide when the conversation is over. To end the dialog, add `"end_dialog": true` to a message response. The message content should be a natural farewell — do not leave it empty.',
        "",
        "Rules for setting end_dialog=true:",
        "- ONLY set it on a `message` response, never on an `api_call`.",
        "- Set it ONLY AFTER every transactional action the user requested has actually executed (e.g., the booking/purchase/reservation API call has been made AND its result has come back in a Search Results block). Never end the dialog while a confirmed transaction is still pending — execute the API call first.",
        "- If the user has just confirmed a booking/purchase/reservation but you have not yet issued the corresponding api_call, DO NOT end. Issue the api_call first; you may end on a later turn after delivering the outcome.",
        "- Once the task is complete and the user has nothing further (e.g., they say 'thanks, that's all', 'no, nothing else', 'goodbye'), respond with a brief farewell and set end_dialog=true.",
        "- If the user explicitly says goodbye or declines further help and there are no pending actions, end the dialog on this turn.",
        "",
        "Examples of ending messages:",
        '- {"type": "message", "content": "You\'re all set! Your ticket has been booked. Have a great time!", "end_dialog": true}',
        '- {"type": "message", "content": "Glad I could help. Have a great day!", "end_dialog": true}',
        "",
        "=== END FORMAT REQUIREMENT ===",
        "",
        "Here is a Conversation history:",
    ])

    return "\n".join(lines)


def load_instruction(instruction_file: str, domains: list[str], api_call_example: str = "", add_example: bool = True) -> str:
    """Load an instruction prompt YAML and substitute domain/API-call placeholders."""
    with open(instruction_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    prompt: str = data.get("instruction_prompt", "No instruction prompt found.")
    prompt = prompt.replace("{domain_name}", ", ".join(domains))
    if add_example and api_call_example:
        prompt = prompt.replace(
            "APICall (method='intent_name', parameters= {required_slot_1: value, required_slot_2: value})",
            api_call_example,
        )
    return prompt


def load_sample_conversation(csv_file: str, api_call_marker: str = "APICall"):
    """Thin wrapper around _simulator_common.load_sample_conversation."""
    return _load_sample_conversation(csv_file, api_call_marker=api_call_marker)


class BaseChatSimulator(ABC):
    """Abstract base for SGD chat simulators."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        self.dialog_id: str
        self.conversation_log: list[str] = []

    def __call__(self) -> str:
        return self.dialog_id

    @abstractmethod
    def interactive_chat(self, conversation_log: list[str]) -> str | None:
        """Run one system turn and append response(s) to *conversation_log*."""
