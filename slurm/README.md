# slurm/

Scripts in this directory fall into three tiers. Only the **entry points** are meant to be called by hand; everything else runs automatically.

```
slurm/
├── run_all_experiments.sh       ← start here
├── submit_baseline.sh
├── submit_sa.sh
├── submit_user_steer.sh
├── submit_user_steer_sa.sh
├── submit_ppl_eval.sh
├── submit_trait_eval.sh
│
├── workers/                     ← dispatched by submitters, never call directly
│   ├── inference_baseline.slurm
│   ├── inference_sa.slurm
│   ├── inference_user_steer.slurm
│   ├── inference_user_steer_sa.slurm
│   ├── eval_metrics_and_traits.slurm
│   ├── eval_metrics.slurm
│   ├── eval_traits_array.slurm
│   └── eval_ppl_array.slurm
│
├── common_env.sh                ← sourced library (paths, PROJECT_ROOT)
├── output_helpers.sh            ← sourced library (experiment/run identity)
├── persona_vector_paths.sh      ← sourced library (vector path resolution)
└── load_dotenv.sh               ← sourced library (.env loader)
```

---

## Entry points

### `run_all_experiments.sh`

Submits all four conditions at once. Recommended starting point.

```bash
bash slurm/run_all_experiments.sh

# Group outputs under a shared parent:
PERSONA_TOD_OUTPUT_ROOT=/path/to/results bash slurm/run_all_experiments.sh

# Preview without submitting:
bash slurm/run_all_experiments.sh --dry-run
```

### `submit_*.sh` — per-condition submitters

Run directly from a login node with `bash`. Each script creates the experiment
directory, submits the inference worker array, and chains CPU metric eval +
LLM-judge trait eval + PPL eval as Slurm dependencies. Nothing further needs to
be done manually after running.

| Script | What it runs |
|---|---|
| `submit_baseline.sh` | Unsteered system + unsteered user (1 run) |
| `submit_sa.sh` | Persona-flow system + unsteered user (1 run) |
| `submit_user_steer.sh` | Unsteered system + steered user (40 runs: 10 traits × 4 scalars) |
| `submit_user_steer_sa.sh` | Persona-flow system + steered user (40 runs) |

```bash
bash slurm/submit_baseline.sh
bash slurm/submit_sa.sh
bash slurm/submit_user_steer.sh
bash slurm/submit_user_steer_sa.sh
```

Key env overrides (export before running):

| Variable | Default | Effect |
|---|---|---|
| `PERSONA_TOD_OUTPUT_ROOT` | `<project_root>/output` | Where experiment dirs are created |
| `N_DIALOG_SHARDS` | `5` | Shards per run (parallelism within a run) |
| `MAX_CONCURRENT` | `5–8` | Max parallel Slurm array tasks |
| `TRAIT_EVAL_MODE` | `both` | `traits`, `quality`, or `both` |
| `DOMAIN` | `both` | `both`, `movies`, or `rest` (`submit_user_steer_sa.sh` only) |

### `submit_ppl_eval.sh` and `submit_trait_eval.sh`

Standalone re-runners for when you want to re-evaluate an already-finished experiment
without re-running inference. Both are called automatically by the submitters above.

```bash
# Re-run LLM-judge trait + quality eval
bash slurm/submit_trait_eval.sh /path/to/output/my-experiment [max_concurrent] [eval_mode]

# Re-run perplexity scoring
bash slurm/submit_ppl_eval.sh /path/to/output/my-experiment [max_concurrent]
```

---

## Workers (`workers/`)

Dispatched automatically by the submitters. Never submit these directly.

| Script | Resource | Called by |
|---|---|---|
| `inference_baseline.slurm` | GPU | `submit_baseline.sh` |
| `inference_sa.slurm` | GPU | `submit_sa.sh` |
| `inference_user_steer.slurm` | GPU | `submit_user_steer.sh` |
| `inference_user_steer_sa.slurm` | GPU | `submit_user_steer_sa.sh` |
| `eval_metrics_and_traits.slurm` | CPU | `submit_baseline.sh`, `submit_sa.sh` |
| `eval_metrics.slurm` | CPU | `submit_user_steer.sh`, `submit_user_steer_sa.sh` |
| `eval_traits_array.slurm` | CPU array | `submit_trait_eval.sh` |
| `eval_ppl_array.slurm` | GPU array | `submit_ppl_eval.sh` |

`eval_metrics_and_traits.slurm` runs task-success metrics and automatically includes
LLM-judge trait scoring if `OPENAI_API_KEY` is present. `eval_metrics.slurm` runs
task-success metrics only — trait scoring is handled in bulk by `submit_trait_eval.sh`
for the large 40-run user-steer sweeps.

---

## Library scripts

Sourced by submitters and workers — not executable on their own.

| Script | Purpose |
|---|---|
| `common_env.sh` | Sets `PROJECT_ROOT`, `REPO_ROOT`, `HF_CACHE_DIR` |
| `output_helpers.sh` | `persona_tod_init_experiment_identity`, `persona_tod_init_run_identity`, `persona_tod_redirect_run_log`, `persona_tod_write_shell_metadata` |
| `persona_vector_paths.sh` | Resolves `.pt` vector file paths from the sibling `persona` repo |
| `load_dotenv.sh` | Sources `.env` from `PROJECT_ROOT` (used when `common_env.sh` is already loaded) |
