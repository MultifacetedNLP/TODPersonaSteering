from data.data_prep.nlg_data_prep import NlgDataPrep
from data.data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from data.data_prep.zstod_data_prep import ZsTodDataPrep
from my_enums import ContextType


class DataPrepStrategyResolver:
    """
    Class for resolving data preparation strategy.
    """

    @classmethod
    def resolve(self, cfg):
        """
        Resolves data preparation strategy.
        """
        if cfg.context_type == ContextType.NLG.value:
            return NlgDataPrep(cfg)
        if cfg.context_type == ContextType.SHORT_REPR.value:
            return ZsTodDataPrep(cfg)
        if cfg.context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
        ]:
            return NlgApiCallStrategy(cfg)
        raise ValueError(f"Unknown data prep step: {cfg.context_type}")
