from dataclasses import dataclass
from typing import Optional
from my_enums import TurnRowType
from data.tod.nlg.ke_tod_context import KeTodContext

from data.tod.nlg.nlg_tod_target import NlgTodTarget
from data.sgd_dstc8.dstc_dataclasses import DstcSchema


@dataclass
class KeTodTurn:
    context: KeTodContext
    target: NlgTodTarget
    schemas: list[DstcSchema]
    schema_str: str
    domains: list[str]
    domains_original: list[str]
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    turn_row_type: Optional[TurnRowType] = TurnRowType.RESPONSE.value
    is_retrieval: Optional[int] = 0
    is_slot_fill: Optional[int] = 0
    is_multi_domain_api_call: Optional[int] = 0
    user_req_slots: Optional[list[str]] = None
