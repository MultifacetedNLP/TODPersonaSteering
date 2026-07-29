from config.python.dataprep_config import DataPrepConfig
from my_enums import ContextType
from data.tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from data.tod.turns.multi_task_csv_row import MultiTaskCsvRow
from data.tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow
from data.tod.turns.turn_csv_row_base import TurnCsvRowBase


class TurnCsvRowFactory:
    @classmethod
    def get_handler(self, cfg: DataPrepConfig) -> TurnCsvRowBase:
        csv_row_cls = GeneralTurnCsvRow()
        if cfg.is_multi_task:
            csv_row_cls = MultiTaskCsvRow()
        if cfg.context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
        ]:
            csv_row_cls = ApiCallTurnCsvRow()
        return csv_row_cls
