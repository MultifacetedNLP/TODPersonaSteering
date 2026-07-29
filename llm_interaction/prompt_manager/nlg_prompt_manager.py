from llm_interaction.prompt_manager.prompt_constants import NlgPromptType
from my_enums import ContextType
from data.sgd_dstc8.dstc_dataclasses import (
    DstcSchema,
)


class NlgPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}.",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call or a response to the user.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                # "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request can only use the required slots and optional slots from that Intent.",
                # f"You will be provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                f"You will be provided with a Schema for domain: {domain}.",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Dialog History:",
                dialog_history,
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                ". Using the Dialog History, Search Results, and by following the Instructions please generate the response for the chat assistant.",
            ]
        )
        return prompt_text
    
    def convert_list_to_string(self, input_list):
        result = []
        for sublist in input_list:
            if len(sublist) > 1:
                result.append(f"({', '.join(sublist)})")
            else:
                result.append(sublist[0])
        return ', '.join(result)
    
    def get_prompt_simulator(
        self,
        APICall: str,
        dialog_history: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are a User Simulator.",
                "Instructions: As an User Simulator, your task is to generate the most appropriate and realistic response that a user might provide.",
                f". You are provided with the next API Call that will be made by the chat assistant:",
                f"Next API Call:",
                APICall,
                f". You are provided with an incomplete dialog between a user and a chat assistant:",
                "Dialog History:",
                dialog_history,
                ". Using the aforementioned information, and by following the Instructions, please generate a relastic user response.",
            ]
        )
        return prompt_text
    
    def get_prompt_simulator_v2(
        self,
        domain: str,
        # schema: str,
        next_api_call: str,
        next_user_req_slots: list[str],
        last_api_call: str,
        last_user_req_slots: list[str],
        dialog_history: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are a User Simulator for the domain: {domain}.",
                "Generate a single realistic USER utterance (only the utterance text).",
                "Do NOT include role tags like 'User:' or 'System:'.",
                "Do NOT emit any end-of-dialog marker — the SYSTEM is responsible for closing the conversation. Just write a natural user utterance.",
                "",
                "GROUNDING AND CONSISTENCY RULES (CRITICAL):",
                "- The Dialog History is authoritative. Your utterance must be consistent with it.",
                "- Do NOT introduce new goals, requests, or tasks that are not implied by the Dialog History.",
                "- You are given the Next API Call the SYSTEM is expected to execute next. Your job is to steer the dialog so the SYSTEM uses that call (and only that call).",
                "- Do NOT request actions that would require an API call other than the provided Next API Call.",
                "",
                "HOW TO USE THE NEXT API CALL INFORMATION:",
                "- If 'Next API Call' is provided (not N/A), your utterance should naturally provide missing info the SYSTEM needs for that call, or ask for/confirm information aligned with the 'Next required slots'.",
                "- If 'Next API Call' is N/A, follow these rules:",
                "  1. You MUST NOT request any new action from the SYSTEM (no booking, reserving, canceling, buying, searching again, or follow-up tasks).",
                "  2. If the last SYSTEM message asks a question, answer it directly without adding new requests.",
                "  3. If the last SYSTEM message is an offer to help further (for example: 'Can I assist you with anything else?'), respond politely that you have nothing else (e.g., 'No, that's all, thanks.'). The SYSTEM will then close the dialog on its next turn.",
                "",
                f"Next API Call: {next_api_call or 'N/A'}",
                f"Next required slots: {', '.join(next_user_req_slots) if next_user_req_slots else 'N/A'}",
                f"Last API Call: {last_api_call or 'N/A'}",
                f"Last required slots: {', '.join(last_user_req_slots) if last_user_req_slots else 'N/A'}",
                "Dialog History:",
                dialog_history
            ]
        )
        return prompt_text

    def get_prompt_simulator_v3_prompting(
        self,
        domain: str,
        trait: str,
        personality_definition: str,
        # schema: str,
        next_api_call: str,
        next_user_req_slots: list[str],
        last_api_call: str,
        last_user_req_slots: list[str],
        dialog_history: str,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are a User Simulator for the domain: {domain}.",
                "You should adapt your conversation style and responses based on your personality trait: {trait}.",
                "Context: {personality_definition}.", 
                "",
                "Expressing likes, dislikes, or changes in preference based on your personality trait.",
                "Generate a single realistic USER utterance (only the utterance text).",
                "Do NOT include role tags like 'User:' or 'System:'.",
                "Do NOT emit any end-of-dialog marker — the SYSTEM is responsible for closing the conversation. Just write a natural user utterance.",
                "",
                "GROUNDING AND CONSISTENCY RULES (CRITICAL):",
                "- The Dialog History is authoritative. Your utterance must be consistent with it.",
                "- Do NOT introduce new goals, requests, or tasks that are not implied by the Dialog History.",
                "- You are given the Next API Call the SYSTEM is expected to execute next. Your job is to steer the dialog so the SYSTEM uses that call (and only that call).",
                "- Do NOT request actions that would require an API call other than the provided Next API Call.",
                "",
                "HOW TO USE THE NEXT API CALL INFORMATION:",
                "- If 'Next API Call' is provided (not N/A), your utterance should naturally provide missing info the SYSTEM needs for that call, or ask for/confirm information aligned with the 'Next required slots'.",
                "- If 'Next API Call' is N/A, follow these rules:",
                "  1. You MUST NOT request any new action from the SYSTEM (no booking, reserving, canceling, buying, searching again, or follow-up tasks).",
                "  2. If the last SYSTEM message asks a question, answer it directly without adding new requests.",
                "  3. If the last SYSTEM message is an offer to help further (for example: 'Can I assist you with anything else?'), respond politely that you have nothing else (e.g., 'No, that's all, thanks.'). The SYSTEM will then close the dialog on its next turn.",
                "",
                f"Next API Call: {next_api_call or 'N/A'}",
                f"Next required slots: {', '.join(next_user_req_slots) if next_user_req_slots else 'N/A'}",
                f"Last API Call: {last_api_call or 'N/A'}",
                f"Last required slots: {', '.join(last_user_req_slots) if last_user_req_slots else 'N/A'}",
                "Dialog History:",
                dialog_history
            ]
        )
        return prompt_text
class NlgMultidomainPrompt:
    def get_prompt(
        self,
        current_domain: str,
        current_domain_schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain and other domains
        """

        prompt_text = "\n".join(
            [
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be a service call or a response to the user.",
                f"Previously, you were an expert chat assistant for the domain: {other_domain}",
                other_domain_schema,
                f"You were provided with a Schema for the domain: {other_domain}.",
                f"Now you have switched domains and are an expert chat assistant for the domain: {current_domain}",
                current_domain_schema,
                f"You have been provided with a Schema for domain: {current_domain}, which is in the same format as the {other_domain} and contains the relevant Intents for the {current_domain}.",
                f"Use the {current_domain} schema in a similar way you used it in the {other_domain}.",
                "Dialog History:",
                dialog_history,
                f"You have been provided with a Schema for domain: {current_domain}, an incomplete dialog between a user and a chat assistant, and an optional search results.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                # "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request can only use the required slots and optional slots from that Intent.\n",
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                "Using the Dialog History, Search Results and by following the Instructions please generate the response for the chat assistant.",
            ]
        )

        return prompt_text


class KetodPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = "\n".join(
            [
                f"You are an expert chat assistant for the domain: {domain}",
                "Instructions: As an expert, you must generate the most appropriate response for the chat assistant.",
                "The response can be an api call, entity query or a response to the user.",
                # "If there are search results, you should use information from the search results to generate the response.",
                # "If you think you need more information to answer the user request, you can request information from the user.",
                "Based on the Last User Utterance, you must find the relevant Intent from the Schema and your request should use the required slots and optional slots from that Intent.",
                # f"You will be provided with a Schema for domain: {domain}, which contains the relevant Intents for the domain. Each Intent has a list of required and optional slots.",
                f"You will be provided with a Schema for domain: {domain}",
                schema,
                f"You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.",
                "Dialog History:",
                dialog_history,
                # "Using the Dialog History, Search Results, relevant Intent from the Schema and by following the Instructions please generate the response for the chat assistant.",
                "Using the Dialog History, Search Results and by following the Instructions please generate the response for the chat assistant.",
            ]
        )
        return prompt_text


class ChatGptPrompt:
    def get_prompt(
        self,
        domain: str,
        schema: str,
        dialog_history: str,
        other_domain: str = None,
        other_domain_schema: str = None,
        all_schema: dict[str, DstcSchema] = None,
        domains_original: str = None,
    ) -> str:
        """
        Returns the NLG prompt for the given domain
        """
        prompt_text = (
            f"You are an expert chat assistant for the DOMAIN: {domain}. \n"
            f"Instructions:\nGenerate a natural and helpful SYSTEM response for the given task-oriented dialog context.\n"
            f"Here are important slots related to each intent:\n"
        )
        domains = domains_original.split(",")
        for dom in domains:
            dstc_schema = all_schema[dom]
            # for intent, req_slots, opt_slots in intents_slots:
            for dstc_intent in dstc_schema.intents:
                req_slots_str = ", ".join(dstc_intent.required_slots)
                opt_slots_str = ", ".join(dstc_intent.optional_slots.keys())
                prompt_text += (
                    f"\nIntent: {dstc_intent.name}\n"
                    f"required slots: {req_slots_str}\n"
                    f"optional slots: {opt_slots_str}\n"
                )

        prompt_text += (
            f"\nYou can request the values of any number of slots from the USER in order to fulfill the user's current intent."
            f"Generally, required slots are more important than the optional ones. Moreover, you should restrict your conversation to the slots that are relevant to the user's current intent. "
        )
        prompt_text += (
            f"\nYou can assume that you have access to the following API calls: "
        )

        # for intent, _, _ in intents_slots:
        for dom in domains:
            dstc_schema = all_schema[dom]
            # for intent, req_slots, opt_slots in intents_slots:
            for dstc_intent in dstc_schema.intents:
                req_slots_str = ", ".join(dstc_intent.required_slots)
                opt_slots_str = ", ".join(dstc_intent.optional_slots.keys())
                prompt_text += f"method={dstc_intent.name}, parameters={req_slots_str}, {opt_slots_str}. "

        prompt_text += (
            f"Use column names as parameters for API calls. While making the call, Make sure you ask all the required slots from the user before making an API call, you can skip unwanted parameters. "
            f"Here is an example: ApiCall(method='intent_i', parameters={{'slot_1': 'value_1', 'slot_2': 'value_2', ...}}). "
            f"Try to match the parameters in the API calls with the column names from the dataframe. Match the required and optional slots with the column names and use the columns names to search in the API calls.Use date format YYYY-MM-DD if needed to search in API call. "
            f"When you think you have enough information about slot values, you must output the API call. Right after your API call, you will be provided with search result of the API call. "
            f"You will be provided with an incomplete dialog context between USER and SYSTEM, and optionally, search results from dataframe. "
            f"Here a few general guidelines: Don't overload the USER with too many choices. Confirm the values of slots and Ask the USER to confirm their choice before finalizing the transaction. "
            f"Whenever confused, it is always better to ask or confirm with the user. If you feel like that user is confused, you may guide the user with relevant suggestions. "
            f"Task:\n Based on the last USER utterance and dialog context you have to generate a conversational response or an API call in order to fulfill the user's current intent. "
            f"You may need to make API calls and use the results of the API call. \n Dialog Context: {dialog_history}."
        )

        return prompt_text


class NlgPromptFactory:
    @classmethod
    def get_handler(
        self, prompt_type: str, context_type: ContextType = ContextType.NLG_API_CALL
    ) -> NlgPrompt:
        if context_type in [ContextType.KETOD_API_CALL.value,ContextType.KETOD_GPT_API_CALL.value]:
            return KetodPrompt()
        if prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            return NlgMultidomainPrompt()
        if prompt_type == NlgPromptType.DEFAULT.value:
            return NlgPrompt()
        if prompt_type == NlgPromptType.CHATGPT.value:
            return ChatGptPrompt()
        all_prompts = ",".join(NlgPromptType.list())
        raise NotImplementedError(
            f"Prompt type {prompt_type} is not implemented. It must be one of the following values {all_prompts}."
        )
        
        
"""
        [
                f"You are a User Simulator for the domain: {domain}.",
                "Instructions: As an User Simulator, your task is to generate the most appropriate and realistic response that a user might provide.",
                # f"You are provided with a Schema for domain: {domain}.",
                # schema,
                f"You are provided with the very last API Call that the chat assistant executed:" if previous_api_call else "No last API call is available.",
                previous_api_call if previous_api_call else "",
                f"You are provided with the next API Call that the chat assistant will execute:" if next_api_call else "No next API call is available.",
                next_api_call if next_api_call else "",
                f"You are provided with the next required slots that you will ask:" if next_user_req_slots else "No required slot is available.",
                ', '.join(next_user_req_slots) if next_user_req_slots else "",
                f"You are provided with an incomplete dialog between a user and a chat assistant:",
                "Dialog History:",
                dialog_history,
                "Using the aforementioned information, and by following the Instructions, please generate a relastic user response.",
            ]
"""
