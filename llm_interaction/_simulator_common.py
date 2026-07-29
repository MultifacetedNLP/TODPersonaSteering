"""Helpers shared by the SGD chat simulator modules."""

from __future__ import annotations

import csv
from typing import Tuple


def load_sample_conversation(
    csv_file: str,
    *,
    api_call_marker: str = "ApiCall",
) -> Tuple[list[dict], str, str]:
    """Load a sample few-shot conversation from a CSV file.

    Each row of the CSV must have ``Speaker`` and ``Utterance`` columns. The
    function additionally extracts:

    - ``dialog_id`` from the row whose Speaker is ``dialogue_id:``.
    - ``api_call_example`` from the first row whose Utterance contains
      ``api_call_marker`` (``"APICall"`` for SGD).
    """
    full_conversation: list[dict] = []
    api_call_example: str | None = None
    dialog_id: str | None = None

    with open(csv_file, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            full_conversation.append(row)
            if row["Speaker"].strip().lower() == "dialogue_id:":
                dialog_id = row["Utterance"].strip()
            if api_call_marker in row["Utterance"]:
                api_call_example = row["Utterance"].replace("SYSTEM,", "").strip()

    if not full_conversation:
        raise ValueError(f"No conversations found in the CSV file: {csv_file}")
    if not dialog_id:
        raise ValueError(f"No dialogue_id found in the CSV file: {csv_file}")
    if not api_call_example:
        api_call_example = "No API call found."

    return full_conversation, api_call_example, dialog_id
