"""SGD chat simulator — single and multi-domain unified.

Replaces the four separate simulator modules:
- ``llm_chat_simulator.py`` (SGD single-domain)
- ``llm_chat_simulator_multi_domain.py`` (SGD multi-domain)

Both are now handled by a single ``SGDChatSimulator`` class parameterised by
a ``services`` list.  Single-domain is ``services=[domain]``.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from llm_interaction.base_chat_simulator import (
    BaseChatSimulator,
    create_chat_gpt_input,
    load_instruction,
    load_sample_conversation,
    load_schema,
    method_exists_in_schema,
)
from llm_interaction.providers.factory import generate_with_provider
from llm_interaction.system_output_parser import (
    format_for_conversation_log,
    parse_system_output,
)
from llm_interaction.utils import (
    api_params_check,
    check_is_transactional,
    domain_name_schema,
    fetch_service_results_from_csv,
    fetch_service_results_from_csv_no_params,
    format_service_results,
    get_result_slots,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.getenv("PERSONA_TOD_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CHATGPT_DIR = PROJECT_ROOT / "llm_interaction"

_NULLABLE_INT_COLS = {
    "party_size": "Int64",
    "number_of_seats": "Int64",
    "number_of_tickets": "Int64",
    "stay_length": "Int64",
    "amount": "Int64",
    "transfer_amount": "Int64",
}


def _fill_missing_columns(
    df_existing: "pd.DataFrame",
    new_entry: dict,
    query_params: dict,
) -> dict:
    """Best-effort fill of *new_entry* columns missing from *query_params* using fuzzy matching."""
    all_columns = df_existing.columns.tolist()
    missing_columns = [col for col in all_columns if col not in new_entry]
    if not missing_columns:
        logger.info("No missing columns to fill from previous entries.")
        return new_entry

    logger.info("Missing columns in the new entry: %s", missing_columns)
    matching_records = df_existing.copy()

    def _match_count(row: "pd.Series") -> int:
        count = 0
        for key, val in query_params.items():
            if key not in row or not pd.notna(row[key]):
                continue
            row_val = int(row[key]) if key in _NULLABLE_INT_COLS else row[key]
            if isinstance(row_val, str):
                if fuzz.partial_ratio(str(val).lower(), row_val.lower()) >= 90:
                    count += 1
            elif str(val).lower() == str(row_val).lower():
                count += 1
        return count

    matching_records["match_count"] = matching_records.apply(_match_count, axis=1)
    matching_records = matching_records[matching_records["match_count"] >= 2]

    if not matching_records.empty:
        best_match = matching_records.sort_values("match_count", ascending=False).iloc[0]
        for col in missing_columns:
            if col in best_match and pd.notna(best_match[col]):
                new_entry[col] = best_match[col]
    else:
        logger.info("No matching records found with at least 2 matching columns.")

    return new_entry


def process_api_call_from_parsed(
    method_name: str,
    query_params: dict,
    api_source: str,
    model_name: str,
    chat_gpt_input: str,
    conversation_log: list[str],
    service_results_paths: dict[str, str],
    schema_file: str,
    domains: list[str],
    schemas: list[str],
    activate_retry: bool = True,
    system_response_handler: Any = None,
    hf_local: Any = None,
    easy_steer: Any = None,
    repeng: Any = None,
    anthropic: Any = None,
    system_prompt: str | None = None,
) -> None:
    """Process an API call whose method/params have already been extracted.

    Performs domain resolution across *domains*/*schemas*, fetches service
    results, and handles transactional inserts.  Retries up to 3 times if
    required parameters are missing.
    """
    # Resolve which domain owns this method
    found_domain = "NoDomainFound"
    if method_name:
        for domain, schema in zip(domains, schemas):
            if method_exists_in_schema(schema, method_name):
                found_domain = domain
                logger.info("Method '%s' found in domain '%s'.", method_name, found_domain)
                break

    api_check = api_params_check(method_name, query_params, schema_file, found_domain)

    if api_check["missing_required_params"]:
        if not activate_retry:
            logger.info("Failed Dialog")
            conversation_log.append("<Failed Dialog>")
            return

        llm_conversation_log = conversation_log.copy()
        num_tries = 1
        while num_tries <= 3:
            if not api_check["missing_required_params"]:
                conversation_log[-1] = format_for_conversation_log(parse_system_output(
                    conversation_log[-1]
                ))
                break

            missing = api_check["missing_required_params"]
            if missing == ["method name wrong pattern"]:
                msg = "You are generating the method name in the wrong pattern"
            elif missing == ["method name not found"]:
                msg = "Method name is wrong because it is not found in the schema"
            else:
                msg = "You are missing the required parameters in the API call"
                if not api_check["provided_query_params"]:
                    msg += ", also you might be generating the API call with the wrong pattern"

            logger.info(msg)
            llm_conversation_log.append(f"\n Inform: {msg}, please generate the API call again as valid JSON.")
            dialog_history_prompt = "\n ".join(llm_conversation_log)

            llm_response = generate_with_provider(
                chat_gpt_input + "\n " + dialog_history_prompt,
                provider=api_source,
                model_name=model_name,
                system_prompt=system_prompt,
                hf_local=hf_local,
                easy_steer=easy_steer,
                repeng=repeng,
                anthropic=anthropic,
            )
            if isinstance(llm_response, tuple):
                llm_response = llm_response[0]

            parsed = parse_system_output(llm_response)
            formatted = format_for_conversation_log(parsed)
            llm_conversation_log.append(formatted)

            if parsed.output_type == "api_call" and parsed.method:
                method_name = parsed.method
                query_params = parsed.parameters or {}
                # Re-resolve domain
                found_domain = "NoDomainFound"
                for domain, schema in zip(domains, schemas):
                    if method_exists_in_schema(schema, method_name):
                        found_domain = domain
                        break
                api_check = api_params_check(method_name, query_params, schema_file, found_domain)
                if not api_check["missing_required_params"]:
                    conversation_log[-1] = formatted
                    break
            else:
                conversation_log[-1] = formatted
                return
            num_tries += 1

        if num_tries == 4:
            conversation_log.append("<Failed Dialog>")
            return

    if api_check["provided_query_params"]:
        logger.info("Provided query parameters: %s", query_params)

        service_results_path = service_results_paths.get(found_domain)
        if not service_results_path:
            logger.info("No service results path for domain '%s'.", found_domain)
            conversation_log.append("Search Results\n\nEnd Search Results")
            return

        logger.info("Fetching service results for the API Call from %s...", service_results_path)
        service_results = fetch_service_results_from_csv(service_results_path, query_params, limit=3)

        if service_results is not None and not service_results.empty:
            formatted = format_service_results(service_results)
            for result in formatted:
                if system_response_handler:
                    system_response_handler(result)
                else:
                    logger.info("Tool: %s", result)
            conversation_log.append(f"Search Results\n{formatted}\nEnd Search Results")
        else:
            no_results = "No service results found."
            if system_response_handler:
                system_response_handler(no_results)
            else:
                logger.info("Tool: %s", no_results)

            is_transactional = check_is_transactional(schema_file, found_domain, method_name)
            logger.info("Method '%s' is transactional: %s", method_name, is_transactional)

            if is_transactional:
                logger.info("Preparing to insert a new record into the database.")
                new_entry = query_params.copy()

                if os.path.exists(service_results_path):
                    df_existing = pd.read_csv(service_results_path, index_col=0, dtype=_NULLABLE_INT_COLS)
                    all_columns = df_existing.columns.tolist()
                    new_entry = _fill_missing_columns(df_existing, new_entry, query_params)
                else:
                    all_columns = get_result_slots(schema_file, found_domain, method_name)

                if "Unnamed: 0" in all_columns:
                    all_columns.remove("Unnamed: 0")

                ordered_entry = {col: new_entry.get(col, "") for col in all_columns}
                logger.info("Service Entry to be inserted: %s", ordered_entry)

                insertion_message = "The new record has been inserted into the database."
                if system_response_handler:
                    system_response_handler(insertion_message)
                else:
                    logger.info("SYSTEM: %s", insertion_message)
                conversation_log.append(f"Search Results\n[{ordered_entry}]\nEnd Search Results")
            else:
                conversation_log.append("Search Results\n\nEnd Search Results")
    else:
        logger.info("No provided query parameters to process.")
        service_results_path = service_results_paths.get(found_domain)
        if service_results_path:
            service_results = fetch_service_results_from_csv_no_params(service_results_path, limit=3)
            if service_results is not None and not service_results.empty:
                formatted = format_service_results(service_results)
                for result in formatted:
                    logger.info("Tool: %s", result)
                conversation_log.append(f"Search Results\n{formatted}\nEnd Search Results")
            else:
                conversation_log.append("Search Results\n\nEnd Search Results")
        else:
            conversation_log.append("Search Results\n\nEnd Search Results")


class SGDChatSimulator(BaseChatSimulator):
    """SGD chat simulator supporting both single- and multi-domain dialogues.

    Pass a list of one domain for single-domain behaviour:
    ``SGDChatSimulator(services=['Restaurants_2'], ...)``

    Pass multiple domains for multi-domain:
    ``SGDChatSimulator(services=['Music_3', 'Movies_1'], ...)``
    """

    def __init__(
        self,
        services: list[str],
        api_source: str = "openai",
        model_name: str = "gpt-4o",
        schema_file: str = str(CHATGPT_DIR / "schemas" / "schema.txt"),
        conversation_dir: str | None = None,
        activate_retry: bool = True,
        add_example: bool = True,
        system_response_handler: Any = None,
        hf_local: Any = None,
        easy_steer: Any = None,
        repeng: Any = None,
        anthropic: Any = None,
        system_custom_instruction: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.activate_retry = activate_retry
        self.api_source = api_source
        self.model_name = model_name
        self.schema_file = schema_file
        self.system_response_handler = system_response_handler
        self.hf_local = hf_local
        self.easy_steer = easy_steer
        self.repeng = repeng
        self.anthropic = anthropic
        self.system_prompt = system_prompt
        self.conversation_log: list[str] = []
        self.sa_coeffs: list = []
        self.baseline_counterfactuals: list = []
        self.dialog_ended: bool = False

        self.domains = services

        # Load schemas for all domains
        self.schemas: list[str] = []
        self.schema = ""
        for domain in self.domains:
            domain_schema = load_schema(schema_file, domain)
            self.schemas.append(domain_schema)
            self.schema += domain_schema + "\n"

        # Locate the sample-conversation CSV
        is_single = len(self.domains) == 1
        if is_single:
            _conv_dir = conversation_dir or str(CHATGPT_DIR / "sample_convs")
            domain_name = domain_name_schema(schema_file, self.domains[0])
            conversation_csv = os.path.join(_conv_dir, f"{domain_name}_conversation.csv")
        else:
            _conv_dir = conversation_dir or str(CHATGPT_DIR / "sample_convs_multi_domain")
            services_combination = "NoFile"
            for file_name in os.listdir(_conv_dir):
                if all(d in file_name for d in self.domains):
                    services_combination = file_name
                    break
            conversation_csv = os.path.join(_conv_dir, services_combination)

        if not os.path.exists(conversation_csv):
            raise FileNotFoundError(f"Sample conversation CSV not found: {conversation_csv}")

        self.conversation, self.api_call_example, self.dialog_id = load_sample_conversation(
            conversation_csv, api_call_marker="APICall"
        )

        # Load instruction prompt
        if is_single:
            prompt_name = "instruction_prompt.yml" if add_example else "instruction_prompt_no_example.yml"
        else:
            prompt_name = "instruction_prompt_multi_domain.yml" if add_example else "instruction_prompt_multi_domain_no_example.yml"
        instruction_file = str(CHATGPT_DIR / "prompts" / prompt_name)

        self.instruction_prompt = load_instruction(
            instruction_file, self.domains, self.api_call_example, add_example
        )
        self.chat_GPT_input = create_chat_gpt_input(
            self.instruction_prompt,
            self.schema,
            ", ".join(self.domains),
            self.conversation,
            add_example,
            system_custom_instruction=system_custom_instruction,
        )

        # Build service-results path dict (same structure for single and multi)
        self.service_results_paths: dict[str, str] = {}
        for domain in self.domains:
            if is_single:
                dn = domain_name_schema(schema_file, domain)
                path = str(CHATGPT_DIR / "service_results" / f"{dn}.csv")
            else:
                path = str(CHATGPT_DIR / "service_results" / f"{domain}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Service results CSV for domain '{domain}' not found at {path}"
                )
            self.service_results_paths[domain] = path
            logger.info("Using service results from: %s", path)

    def interactive_chat(self, conversation_log: list[str]) -> str | None:
        """Run one system turn; returns dialog_id or None if the LLM returned nothing."""
        self.conversation_log = conversation_log
        self.dialog_ended = False
        dialog_history_prompt = "\n ".join(self.conversation_log)
        dialog_id = self.dialog_id
        logger.info(dialog_id)

        # Pass only the last few conversation turns to the sa judge so it
        # receives focused context rather than the full 8KB+ system instruction prompt.
        recent_turns = [t for t in self.conversation_log if isinstance(t, str) and
                        (t.startswith("User:") or t.startswith("System:"))]
        judge_context = "\n".join(recent_turns[-6:]) if recent_turns else None

        llm_response = generate_with_provider(
            self.chat_GPT_input + "\n " + dialog_history_prompt,
            provider=self.api_source,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            hf_local=self.hf_local,
            easy_steer=self.easy_steer,
            repeng=self.repeng,
            anthropic=self.anthropic,
            judge_context=judge_context,
        )

        if isinstance(llm_response, tuple):
            llm_response, coeffs = llm_response
        else:
            coeffs = {}

        self.sa_coeffs.append(coeffs)

        if self.api_source in ("sa_steer", "sa"):
            baseline_resp = generate_with_provider(
                self.chat_GPT_input + "\n " + dialog_history_prompt,
                provider="local",
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                hf_local=self.hf_local,
            )
            if isinstance(baseline_resp, tuple):
                baseline_resp = baseline_resp[0]
            self.baseline_counterfactuals.append(baseline_resp)
        else:
            self.baseline_counterfactuals.append("")

        if not llm_response:
            return None

        parsed = parse_system_output(llm_response)
        formatted_response = format_for_conversation_log(parsed)

        if self.system_response_handler:
            self.system_response_handler(formatted_response)
        else:
            logger.info("SYSTEM (parsed): %s", formatted_response)

        self.conversation_log.append(formatted_response)

        # Only honor end_dialog on message responses — never end on the same turn
        # as an api_call (the call's result must come back before we can close).
        if parsed.output_type == "message" and parsed.end_dialog:
            self.dialog_ended = True

        if parsed.output_type == "api_call" and parsed.method:
            process_api_call_from_parsed(
                parsed.method,
                parsed.parameters or {},
                self.api_source,
                self.model_name,
                self.chat_GPT_input,
                self.conversation_log,
                self.service_results_paths,
                self.schema_file,
                self.domains,
                self.schemas,
                self.activate_retry,
                self.system_response_handler,
                self.hf_local,
                self.easy_steer,
                self.repeng,
                self.anthropic,
                system_prompt=self.system_prompt,
            )

        return dialog_id
