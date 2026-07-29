from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

from dataclasses_json import dataclass_json
from data.sgd_dstc8.dstc_constants import (
    DstcSpecialTokens,
    DstcConstants,
    DstcSteps,
    DstcSpeaker,
)

from data.sgd_dstc8 import dstc_utils


class DstcObjectAttributeException(Exception):
    pass


@dataclass
class DstcServiceCall:
    method: str
    parameters: Dict[str, str]


@dataclass
class DstcState:
    active_intent: str
    slot_values: Dict[str, List[str]]
    requested_slots: Optional[List[str]] = None


@dataclass
class DstcAction:
    act: str
    canonical_values: List[str]
    slot: str
    values: List[str]


@dataclass
class DstcFrame:
    actions: List[DstcAction]
    slots: List[any]
    service: str
    state: Optional[DstcState] = None
    service_call: Optional[DstcServiceCall] = None
    service_results: Optional[List[Dict[str, str]]] = None

    def __init__(
        self,
        actions: List[DstcAction],
        slots: List[any],
        service: str,
        state: Optional[DstcState] = None,
        service_call: Optional[DstcServiceCall] = None,
        service_results: Optional[List[Dict[str, str]]] = None,
    ):
        self.actions = actions
        self.slots = slots
        self.state = state
        self.short_service = dstc_utils.get_dstc_service_name(service)
        self.service = service
        self.service_call = service_call
        self.service_results = service_results


@dataclass
class DstcRequestedSlot:
    domain: str
    slot_name: str

    @classmethod
    def from_string(self, text: str) -> "DstcRequestedSlot":
        try:
            domain, slot_name = text.split(DstcConstants.DOMAIN_SLOT_SEPARATOR)
        except ValueError:
            return self("", text)
        return self(domain, slot_name)

    def __str__(self) -> str:
        return "".join(
            [
                self.domain,
                DstcConstants.DOMAIN_SLOT_SEPARATOR,
                self.slot_name,
            ]
        )

    def __eq__(self, other: any) -> bool:
        if not hasattr(other, "domain"):
            raise DstcObjectAttributeException(
                "Object being compared to does not have propery `domain`"
            )
        if not hasattr(other, "slot_name"):
            raise DstcObjectAttributeException(
                "Object being compared to does not have propery `slot_name`"
            )
        return self.domain == other.domain and self.slot_name == other.slot_name


@dataclass
class DstcTurn:
    frames: List[DstcFrame]
    speaker: str
    utterance: str

    def get_active_intent(self) -> Optional[str]:
        if self.speaker == DstcSpeaker.SYSTEM:
            return None
        for frame in self.frames:
            if frame.state is not None:
                return frame.state.active_intent
        return None

    def get_requested_slots(self) -> Optional[List[str]]:
        if self.speaker == DstcSpeaker.SYSTEM:
            return None
        for frame in self.frames:
            if frame.state is None or frame.state.requested_slots is None:
                continue
            if len(frame.state.requested_slots):
                return [
                    DstcRequestedSlot(frame.short_service, s)
                    for s in frame.state.requested_slots
                ]
        return None


@dataclass_json
@dataclass
class DstcDialog:
    dialogue_id: str
    turns: List[DstcTurn]
    services: List[str]

    def __init__(
        self,
        dialogue_id: str,
        turns: List[DstcTurn],
        services: List[str],
    ):
        self.dialogue_id = dialogue_id
        self.turns = turns
        self.short_services = [dstc_utils.get_dstc_service_name(s) for s in services]
        self.services = services


@dataclass
class DstcSchemaIntent:
    name: str
    description: str
    is_transactional: bool
    required_slots: List[str]
    optional_slots: any
    result_slots: List[str]

    def __init__(
        self,
        name: str,
        description: str,
        is_transactional: bool,
        required_slots: List[str],
        optional_slots: any,
        result_slots: List[str],
    ) -> None:
        self.name = name
        self.description = description
        self.is_transactional = is_transactional
        self.required_slots = required_slots
        self.optional_slots = optional_slots
        self.result_slots = result_slots

    def nlg_repr(self):
        name = dstc_utils.remove_underscore(self.name)
        required_slots = ",".join(map(dstc_utils.remove_underscore, self.required_slots))
        optional_slots = ",".join(map(dstc_utils.remove_underscore, self.optional_slots))
        result_slots = ",".join(map(dstc_utils.remove_underscore, self.result_slots))

        # return f"{name} \n required slots: {required_slots} \n optional slots: {optional_slots} \n result slots: {result_slots}"
        return "\n".join([
           f"Intent: {name}",
           f"required slots: {required_slots}" if required_slots else "",
           f"optional slots: {optional_slots}" if optional_slots else "", 
        ])
        
    def nlg_repr_with_results_slots(self):
        name = dstc_utils.remove_underscore(self.name)
        required_slots = ",".join(map(dstc_utils.remove_underscore, self.required_slots))
        optional_slots = ",".join(map(dstc_utils.remove_underscore, self.optional_slots))
        result_slots = ",".join(map(dstc_utils.remove_underscore, self.result_slots))

        return "\n".join([
           f"Intent: {name}",
           f"required slots: {required_slots}" if required_slots else "",
           f"optional slots: {optional_slots}" if optional_slots else "", 
           f"result slots: {result_slots}" if result_slots else ""
        ])

    def __str__(self):
        return "".join(
            [
                DstcSpecialTokens.begin_schema_intent,
                self.name,
                DstcSpecialTokens.intent_required_slots,
                " ".join(self.required_slots),
                DstcSpecialTokens.intent_optional_slots if self.optional_slots else "",
                " ".join(self.optional_slots),
            ]
        )


@dataclass
class DstcSchemaSlot:
    name: str
    description: str
    is_categorical: bool
    possible_values: List[str]

    def __eq__(self, slot_name: str) -> bool:
        return self.name == slot_name

    def nlg_repr(self, should_add_possible_values:bool =False) -> str:
        name = dstc_utils.remove_underscore(self.name)
        if not should_add_possible_values:
            return name
        possible_values = ",".join(map(dstc_utils.remove_underscore, self.possible_values))
        return f"{name} \n possible values: {possible_values}"

    def __str__(self):
        return "".join(
            [
                self.name,
                DstcSpecialTokens.possible_values if self.possible_values else "",
                " ".join(self.possible_values),
                DstcSpecialTokens.end_possible_values if self.possible_values else "",
            ]
        )


@dataclass_json
@dataclass
class DstcSchema:
    service_name: str
    description: str
    slots: List[DstcSchemaSlot]
    intents: List[DstcSchemaIntent]
    step: Optional[str] = None

    def get_intents(self) -> List[str]:
        return [i.name for i in self.intents]

    def get_slot_names(self) -> List[str]:
        return [s.name for s in self.slots]

    def get_possible_values(self, slot_name: str=None) -> List[str]:
        out = []
        for s in self.slots:
            out.append(s.possible_values)
        return

    def get_nlg_repr(self) -> str:
        """
        Returns a string representation of the schema, used when context type is nlg.

        Returns:
            str: A string representation of the schema.
        """
        return "\n".join([
            f"Schema for {dstc_utils.extract_characters(self.service_name)}",
            # self.description,
            "\n".join([i.nlg_repr() for i in self.intents]),
            # "\n".join([s.nlg_repr() for s in self.slots]),
        ])
        
    def get_nlg_repr_with_results_slots(self) -> str:
        """
        Returns a string representation of the schema, used when context type is nlg.

        Returns:
            str: A string representation of the schema.
        """
        return "\n".join([
            f"Schema for {dstc_utils.extract_characters(self.service_name)}",
            # self.description,
            "\n".join([i.nlg_repr_with_results_slots() for i in self.intents]),
            # "\n".join([s.nlg_repr() for s in self.slots]),
        ])    

    def get_slot_repr(self) -> str:
        return "".join(
            [
                DstcSpecialTokens.begin_schema,
                dstc_utils.get_dstc_service_name(self.service_name),
                DstcSpecialTokens.begin_schema_slot,
                " ".join(map(str, self.slots)),
                DstcSpecialTokens.end_schema,
            ]
        )

    def get_full_repr(self) -> str:
        return "".join(
            [
                DstcSpecialTokens.begin_schema,
                dstc_utils.get_dstc_service_name(self.service_name),
                "".join(map(str, self.intents)),
                DstcSpecialTokens.begin_schema_slot,
                " ".join(map(str, self.slots)),
                DstcSpecialTokens.end_schema,
            ]
        )

    def __eq__(self, other: any) -> bool:
        return self.service_name == other.service_name

    def __hash__(self) -> str:
        return hash(self.service_name)

    def __str__(self):
        return self.get_full_repr()


def get_schemas(data_root: Path, step: str) -> Dict[str, DstcSchema]:
    schemas = {}
    path = data_root / step / "schema.json"
    schema_json = dstc_utils.read_json(path)
    for s in schema_json:
        schema: DstcSchema = DstcSchema.from_dict(s)
        schema.step = step
        schemas[schema.service_name] = schema
    return schemas


def get_slot_categories(data_root: Path) -> Dict[str, bool]:
    schemas = get_schemas(data_root, DstcSteps.TEST.value)
    out = defaultdict(bool)
    for s in schemas.values():
        for slot in s.slots:
            out[slot.name] = slot.is_categorical
    return out
