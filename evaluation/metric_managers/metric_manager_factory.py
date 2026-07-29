from evaluation.metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from my_enums import ContextType


class MetricManagerFactory:

    @classmethod
    def get_metric_manager(self, context_type: str, tokenizer, logger):
        if context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
        ]:
            return NlgApiCallMetricManager(logger, tokenizer)
