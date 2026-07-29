from enum import Enum


class DstcSteps(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

    @classmethod
    def get_index(self, step_text):
        return DstcSteps.list().index(step_text.lower())

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class DstcSpeaker(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class DstcConstants(str, Enum):
    DOMAIN_SLOT_SEPARATOR = "$"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class DstcSpecialTokens(str, Enum):
    begin_schema = "<|beginschema|>"
    end_schema = "<|endschema|>"
    schema_name = "<|schemaname|>"
    schema_description = "<|schemadescription|>"

    begin_schema_intent = "<|beginschemaintent|>"
    end_schema_intent = "<|endschemaintent|>"
    intent_required_slots = "<|intentrequiredslots|>"
    intent_result_slots = "<|intentresultslots|>"
    intent_optional_slots = "<|intentoptionalslots|>"
    possible_values = "<|possiblevalues|>"
    end_possible_values = "<|endpossiblevalues|>"

    begin_schema_slot = "<|beginschemaslot|>"
    end_schema_slot = "<|endschemaslot|>"
    schema_slot_values = "<|schemaslotvalues|>"

    begin_service_results = "<|beginserviceresults|>"
    end_service_results = "<|endserviceresults|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
