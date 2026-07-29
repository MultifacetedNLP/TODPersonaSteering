# TOD Inference Config Reference

This file documents every property in `tod_inference.yaml`.

## Composition

- `defaults`: Hydra composition list. This config loads `data_size/dstc_rest_movie_inference.yaml` and `dataset/dstc.yaml`.
- `hydra.run.dir`: Directory where Hydra writes the composed runtime config. SLURM overrides this to the run output folder.

## Model Providers

- `tokenizer_name`: Hugging Face tokenizer used for data preparation and prompt length accounting. Defaults from `PERSONA_TOD_TOKENIZER_NAME`, then `Qwen/Qwen2.5-7B-Instruct`.
- `user.provider`: Provider used to generate user simulator utterances. Must be provider-backed, such as `openai` or `local_steer_anthropic`.
- `user.model_name`: User simulator model name passed to the provider.
- `system.provider`: Provider used to generate system utterances.
- `system.model_name`: System model name passed to the provider.
- `system.anthropic.vector_path`: Steering vector path for system providers that use anthropic/persona steering. Defaults from `PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH`.
- `system.anthropic.coef`: System steering coefficient. `0.0` is the baseline setting.
- `system.anthropic.layer`: Model layer where the system steering vector is applied.

## Run Behavior

- `activate_retry`: Enables retry behavior in the chat simulator/provider path.
- `add_example`: Includes examples in prompts where the prompt builder supports it.
- `num_dialogues_per_domain`: Maximum number of dialogs sampled for each top-level entry in `test_domain_settings`. Use `-1` to disable this fixed-size sampling and fall back to the multi-domain filtering branch.
- `selection_seed`: Seed used when sampling dialogs. Use a non-negative value for deterministic selection; use `-1` for random OS entropy.
- `num_workers`: Worker count passed to data preparation/loading paths.
- `test_prompt_max_len`: Prompt-token budget reserved for the test prompt.
- `max_token_len`: Total token budget used by tokenizer and generation paths.
- `should_test`: Keeps the config in inference/test mode.
- `num_turns`: Maximum number of user/system turns to simulate for a dialog.
- `max_turns`: Maximum dialog length retained during data preparation. Multi-domain dialogs need a larger value than single-domain dialogs.

## Prompt Contents

- `should_add_schema`: Adds service schema text to prompts.
- `should_add_user_actions`: Adds user action annotations when preparing prompts.
- `should_add_service_results`: Adds service/API search results to prompts and saved dialog metadata.
- `should_add_service_calls`: Adds previous service/API calls to prompts.
- `should_add_results_slots`: Adds result slot names from schemas.
- `should_add_user_req_slots`: Adds user requested-slot information.
- `prompt_type`: Prompt-template selector consumed by the prompt builder. `default` is the standard TOD prompt.

## Paths And Output

- `project_root`: Project root used to resolve relative paths. In SLURM this is the repository root.
- `resume_checkpoint`: Historical checkpoint field. Provider-backed inference does not use local user checkpoints.
- `out_dir`: Historical output field. Runtime output is resolved from Hydra and SLURM run paths.

## Data Splits And Domains

- `epochs`: Historical training field retained for config compatibility. Inference does not train.
- `overwrite`: Three flags for train/dev/test data preparation cache overwrite behavior. The third value applies to test data.
- `is_scale_grad`: Historical training/data-prep option. Keep `false` for inference.
- `train_domain_settings`: Domain selection for train data preparation paths.
- `dev_domain_settings`: Domain selection for dev data preparation paths.
- `test_domain_settings`: List of test-domain buckets. Each top-level entry is one independent output folder and must be a list of service names. The default runs one restaurant bucket and one movies bucket.
- `multi_domain`: Enables multi-domain dialog handling in data preparation and interactive session selection.
- `should_train`: Must remain `false` for provider-backed TOD inference.

## W&B Metadata

- `wandb.project`: Historical W&B project name.
- `wandb.entity`: Historical W&B entity name.
- `wandb.notes`: Historical run notes.
- `wandb.task`: Historical W&B task label.

## Environment Overrides

- `PERSONA_TOD_TEST_DOMAINS`: Overrides `test_domain_settings`. It must be a Python nested list string, for example `[["Restaurants_2"],["Movies_1","Movies_3"]]`.
- `PERSONA_TOD_SYSTEM_PROVIDER`, `PERSONA_TOD_SYSTEM_MODEL_NAME`, `PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH`, `PERSONA_TOD_SYSTEM_ANTHROPIC_COEF`, and `PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER`: Override system provider settings.
- `PERSONA_TOD_USER_PROVIDER`, `PERSONA_TOD_USER_MODEL_NAME`, `PERSONA_TOD_USER_ANTHROPIC_VECTOR_PATH`, `PERSONA_TOD_USER_ANTHROPIC_COEF`, and `PERSONA_TOD_USER_ANTHROPIC_LAYER`: Override user provider settings.
