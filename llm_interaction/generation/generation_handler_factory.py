from llm_interaction.generation.simple_generation import SimpleGeneration
from llm_interaction.generation.t5_generation import T5Generation
import utils


class GenerationHandlerFactory:
    @classmethod
    def get_handler(self, cfg, model=None, tokenizer=None):
        model = model or cfg.model
        tokenizer = tokenizer or cfg.tokenizer
        if utils.is_t5_model(cfg.model_type.model_name):
            return T5Generation(model, tokenizer)
        return SimpleGeneration(model, tokenizer)
