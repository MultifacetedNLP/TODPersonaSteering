# Activation-Steered Personas in TOD

Code and generated dialogues for **"Activation-Steered Personas in Task-Oriented LLM
Agent Simulations"** (Shoaeinaeini et al., 2026), presented at the
[COLM 2026 Social Simulation with LLMs Workshop](https://sites.google.com/view/social-sims-with-llms)
(non-archival).

We study **activation steering** as an inference-time mechanism for inducing and
adapting Big Five personality in **task-oriented dialogue (TOD)** simulation. A single
LLM is instantiated twice — once as a simulated **user agent** and once as a **system
agent** — and steered via trait-pole activation vectors injected at layer 20.

## Framework

We evaluate four steering conditions. In each, a **system LLM** and a **user-simulator
LLM** converse across multi-turn dialogues on the SGD dataset (Restaurants + Movies
domains); Big Five activation vectors are optionally injected into one or both LLMs to
steer personality traits. Every condition chains inference → task-metric evaluation →
LLM-judge trait/quality scoring → perplexity scoring automatically via Slurm
dependencies.

| Condition | System side | User side | Slurm script |
|---|---|---|---|
| **Baseline** | unsteered | unsteered | `submit_baseline.sh` |
| **SA** (System-side Adaptive steering) | a predict-then-steer mechanism composes a turn-level steering vector from the dialogue context | unsteered | `submit_sa.sh` |
| **US** (User Steering) | unsteered | a fixed Big Five trait-pole vector, scalar α ∈ {−1, 0, 1, 2}, injected at every user turn (10 traits × 4 scalars = 40 runs) | `submit_user_steer.sh` |
| **US+SA** | SA | same user sweep | `submit_user_steer_sa.sh` |

User-steer traits: `calm, careless, compassionate, consistent, dependable, inventive, nervous, outgoing, self-interested, solitary`  
Scalars: `-1, 0, 1, 2`

## Models

- **Qwen2.5-7B-Instruct** — primary model.
- **Llama-3.1-8B-Instruct** — cross-model robustness setting.

Steering is applied at **layer 20**. We use the publicly released persona vectors from
the [`xiachongfeng/persona`](https://huggingface.co/datasets/xiachongfeng/persona)
repository for both models.

## Dataset

**SGD (Schema-Guided Dialogue)** — Restaurants and Movies domains. Each dialogue is
initialized with an instruction prompt, the domain schema, and a sample conversation,
and is grounded in schema-defined slots, API calls, and returned search results.

## Evaluation

**Standard task metrics:** Method Accuracy, Dialogue Success, Full API Accuracy, BLEU,
Inform Accuracy, Dialogue Completion Rate.

**LLM-judged pairwise metrics:** Constraint Satisfaction, Truthfulness, and
Personality-Conditioned User Satisfaction.

## Requirements

- Python 3.10+
- Linux shell with Bash 4+
- SLURM cluster with A100 GPUs (`gpuA100x4` partition)
- HF-accessible model weights for `meta-llama/Llama-3.1-8B-Instruct`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your credentials:

```
HF_TOKEN=...                  # Hugging Face token (model downloads)
OPENAI_API_KEY=...            # LLM-judge evaluation
OPENROUTER_API_KEY=...        # if using OpenRouter for the user simulator
PERSONA_TOD_REPO_USERNAME=...      # your cluster username
```

SLURM scripts source `slurm/common_env.sh`, which loads `.env` automatically at submission time.

## Running the Experiments

### Run all four at once (recommended)

```bash
bash slurm/run_all_experiments.sh
```

This submits all four submitter jobs. Each submitter internally chains its own eval jobs. All four run independently in parallel.

To group all outputs under a single parent directory:

```bash
PERSONA_TOD_OUTPUT_ROOT=/work/hdd/shared/exp-2026-05 \
  bash slurm/run_all_experiments.sh
```

Preview what would be submitted without actually running:

```bash
bash slurm/run_all_experiments.sh --dry-run
```

### Run a single condition

Each condition is submitted as a Slurm job that internally dispatches child jobs:

```bash
# Baseline
bash slurm/submit_baseline.sh

# Persona-flow system steering
bash slurm/submit_sa.sh

# User-steering sweep (40 runs)
bash slurm/submit_user_steer.sh

# User-steer + sa combined (40 runs)
bash slurm/submit_user_steer_sa.sh
```

Optional overrides (pass via `--export` or export beforehand):

| Variable | Default | Description |
|---|---|---|
| `PERSONA_TOD_OUTPUT_ROOT` | `<project_root>/output` | Root directory for all output |
| `N_DIALOG_SHARDS` | `5` | Number of dialog shards per run |
| `MAX_CONCURRENT` | `5–8` | Max parallel Slurm array tasks |
| `TRAIT_EVAL_MODE` | `both` | `traits`, `quality`, or `both` |
| `DOMAIN` | `both` | `both`, `movies`, or `rest` (user-steer-sa only) |

### Run eval only (on existing results)

To re-run LLM-judge trait/quality eval on a finished experiment:

```bash
bash slurm/submit_trait_eval.sh \
  /path/to/output/my-experiment [max_concurrent] [eval_mode]

# Example — quality only, 8 concurrent:
EVAL_MODE=quality bash slurm/submit_trait_eval.sh \
  output/llama-user-steer-2026-05-13 8
```

To re-run perplexity eval:

```bash
bash slurm/submit_ppl_eval.sh \
  /path/to/output/my-experiment [max_concurrent]
```

## Output Layout

Each experiment writes under `PERSONA_TOD_OUTPUT_ROOT/<experiment-name>/`:

```
output/<experiment-name>/
  manifest.txt          # experiment metadata, run manifest, job IDs
  metadata.json         # machine-readable metadata
  log.out               # submitter log
  runs/
    <run-label>/        # one dir per run (e.g. baseline, calm__s1)
      dialogs/
        <domain>/
          dialog_*.json
      metrics.json      # task-success + trait + PPL scores
      log.out
  logs/
    slurm/              # raw Slurm stdout per array task
    ppl_eval/           # PPL eval manifests + Slurm logs
    trait_eval/         # trait eval manifests + Slurm logs
```

## Monitoring

```bash
# All your active jobs
squeue -u $USER

# Tail the submitter log
tail -f output/<experiment-name>/log.out

# Per-run log (e.g. user-steer, calm scalar 1)
tail -f output/<experiment-name>/runs/calm__s1/log.out
```

## Analysis

After experiments complete, build a summary Excel workbook:

```bash
python analysis/build_excel.py --input-dir output/<experiment-name>
```

Generate insights across conditions:

```bash
python analysis/generate_insights.py
```

## Local / Interactive Runs

Without Slurm, run the inference loop directly:

```bash
# Interactive inference (SGD, sa capable):
python -m entrypoints.inference.t5_tod_interactive_inference_modified

# Evaluate generated dialogs:
python -m entrypoints.evaluation.evaluation --json-dir output/<run>

# Trait scoring only:
python -m entrypoints.evaluation.evaluation \
  --evaluate-traits --only-traits --json-dir output/<run>

# Figure reproduction, evaluation how-to:
# see docs/HOWTO.md
```

## Configuration

- Hydra TOD config: `config/hydra/tod_runs/tod_inference.yaml`  
  Field reference: `config/hydra/tod_runs/CONFIG_REFERENCE.md`
- Environment variables, architecture, and full run/eval/figure instructions: [`docs/HOWTO.md`](docs/HOWTO.md)
- SLURM path/env helpers: `slurm/common_env.sh`, `slurm/output_helpers.sh`

## Repository Structure

See [`docs/HOWTO.md`](docs/HOWTO.md) for the conceptual architecture (simulators, persona-vector steering, evaluation layers) and where results/analysis live per model.

```
entrypoints/                  # CLI entrypoints
  inference/                  #   dialog generation (t5_tod_interactive_inference_modified.py)
  evaluation/                 #   evaluation.py, pairwise_metrics.py, trait_expression_eval.py,
                               #   perplexity_eval.py, run_pairwise_comparison.sh
llm_interaction/               # simulator + provider backends
  providers/                  #   provider dispatch (openai/local/vllm/local_steer_anthropic/stub)
  generation/                 #   HF/T5 generation handlers
  prompt_manager/, prompts/   #   NLG prompt construction + instruction-prompt YAML
  system_output_parser.py, contracts.py, inference_flow.py, llm_models_api.py
  sgd_chat_simulator.py       #   owns one dialogue session (turn loop)
evaluation/                   # task-success metric managers + Big-Five LLM-judge
  metric_managers/, metrics/  #   API accuracy, dialog completion, requested-slot answering
  big5/                       #   Big-Five trait scoring
data/                         # dataset loaders and TOD data models
  data_prep/                  #   dataset preprocessing scripts
  sgd_dstc8/                  #   canonical SGD/DSTC8 dataclasses
config/                       # Hydra configs
  hydra/tod_runs/             #   tod_inference.yaml (canonical run config)
  python/                     #   DataModuleConfig, DataPrepConfig dataclasses
utilities/                    # shared helpers — runtime_config.py is the canonical env/path resolver
logger/, tools/               # inference logger dataclasses; standalone repair utilities
slurm/                        # SLURM submitters, workers, shared env/path libraries
  run_all_experiments.sh      #   entry point: launch all four conditions
  submit_baseline.sh          #   entry point: baseline condition
  submit_sa.sh                #   entry point: SA condition
  submit_user_steer.sh        #   entry point: user-steer sweep
  submit_user_steer_sa.sh     #   entry point: user-steer + SA
  submit_std_eval.sh          #   standalone: batch standard-metric eval (SLURM array)
  submit_ppl_eval.sh          #   standalone: re-run PPL eval
  submit_trait_eval.sh        #   standalone: re-run trait/quality eval
  workers/                    #   dispatched automatically — do not call directly
  common_env.sh, output_helpers.sh, persona_vector_paths.sh, load_dotenv.sh   # sourced libraries
analysis/                     # post-hoc analysis scripts
  build_excel.py, generate_insights.py, domain_comparison.py
  plot_paper_figures.py       #   reproduces the paper's Figures 1-3 from checked-in results
llama/, qwen/qwen-v2/         # per-model results: raw generated dialogues, evaluation/,
                               #   pairwise_comparison/, paired_dialogues/, analysis/ (write-ups)
paper_figures/<model>/        # rendered Figures 1-3, output of analysis/plot_paper_figures.py
docs/HOWTO.md                 # consolidated run/eval/figure-reproduction guide
INSTRUCTIONS.md               # fresh-clone environment setup (Python env, .env, dataset, vectors)
utils.py, my_enums.py, base_datamodule.py, simple_tod_dataclasses.py   # legacy import-only modules
```

