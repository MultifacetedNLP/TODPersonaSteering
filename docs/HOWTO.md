# HOWTO — Run Dialogues, Run Evaluation, Reproduce the Paper's Figures

Single consolidated guide, replacing the previous set of separate `docs/*.md`
files. For first-time environment setup (Python env, `.env`, downloading the
SGD dataset and PERSONA vectors), see [`../INSTRUCTIONS.md`](../INSTRUCTIONS.md)
first — this file assumes that's done.

---

## 1. How the pieces fit together

Two LLMs (a **system** agent and a **user simulator**) converse turn-by-turn to
complete a Schema-Guided Dialogue (SGD) goal. Either agent can have a PERSONA
activation-steering vector injected at inference time.

```
entrypoints/inference/*.py        ── CLI: run a dialogue-generation experiment
        │
        ▼
llm_interaction/llm_chat_simulator*.py   ── owns one dialogue session:
        │                                   user turn → system turn → (API call?
        │                                   fetch DB rows → system turn again) → repeat
        ▼
llm_interaction/inference_flow.py, providers/factory.py, llm_models_api.py
        ── one model call: provider dispatch (openai/local/vllm/local_steer_anthropic),
           PERSONA vector loading + injection at the configured layer
        │
        ▼
llm_interaction/system_output_parser.py, contracts.py   ── parses system output
        │                                                  (message / api_call / end_dialog)
        ▼
entrypoints/evaluation/*.py        ── CLI: score generated dialogues
        ├── evaluation.py              standard task metrics
        ├── pairwise_metrics.py        LLM-judge head-to-head comparison
        └── trait_expression_eval.py   blind trait-recognition judge
```

Config: Hydra YAML in `config/hydra/tod_runs/` (canonical: `tod_inference.yaml`).
Path/credential/provider selection is entirely environment-variable driven —
`utilities/runtime_config.py` is the resolver; `slurm/common_env.sh` sources
`.env` and derives sensible defaults at job-submission time.

### System output contract

The system model must emit one of:

```json
{"type": "message", "content": "..."}
{"type": "api_call", "method": "MethodName", "parameters": {"slot": "value"}}
{"type": "message", "content": "...", "end_dialog": true}
```

`end_dialog: true` may only appear on a `message` (never with an `api_call`),
and only once every requested API call has been executed and its result
delivered. Setting it writes `"dialog_completed": true` into the dialog JSON —
this is what `dialog_completion_rate` measures. A legacy
`APICall(method='...', parameters={...})` text format is also recognized for
backward compatibility.

### Dialogue termination

A dialogue ends one of five ways: (1) clean `end_dialog` close, (2)
`<Failed Dialog>` — API call missing required params after 3 retries, (3)
`<Looping Dialog>` — 3 consecutive turns ≥90% similar, (4) forced wrap-up after
2 user turns once `next_api_call` is `N/A`, or (5) hitting `max_turns`. Only
(1) counts toward `dialog_completion_rate`.

---

## 2. How to run dialogue generation

**Via SLURM (recommended — matches how the paper's runs were produced):**

```bash
# All four conditions (baseline, sa, user-steer, user-steer+sa) in parallel:
bash slurm/run_all_experiments.sh

# Or one at a time:
bash slurm/submit_baseline.sh
bash slurm/submit_sa.sh
bash slurm/submit_user_steer.sh        # 10 traits x 4 scalars = 40 runs
bash slurm/submit_user_steer_sa.sh     # same 40-run sweep, + sa on the system side

# Preview without submitting:
bash slurm/run_all_experiments.sh --dry-run
```

Useful overrides: `PERSONA_TOD_OUTPUT_ROOT` (output root dir), `N_DIALOG_SHARDS`,
`MAX_CONCURRENT`, `TRAIT_EVAL_MODE` (`traits`/`quality`/`both`), `DOMAIN`
(`both`/`movies`/`rest`).

**Locally, without SLURM (no GPU-cluster required if using an API provider):**

```bash
python -m entrypoints.inference.t5_tod_interactive_inference_modified
```

Provider/model selection is via env vars: `PERSONA_TOD_SYSTEM_PROVIDER`,
`PERSONA_TOD_SYSTEM_MODEL_NAME`, `PERSONA_TOD_USER_PROVIDER`, `PERSONA_TOD_USER_MODEL_NAME`
(providers: `openai`, `openrouter`, `groq`, `local` (HF), `vllm`,
`local_steer_anthropic` (HF + activation steering), `stub` (tests)). Persona
steering: `PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH` / `_COEF` / `_LAYER`, and the
`PERSONA_TOD_USER_*` equivalents. Full variable list: `utilities/runtime_config.py`.

**Output layout** (one folder per experiment):

```
output/{experiment-name}/
  manifest.txt / metadata.json / log.out
  runs/{run-label}/            # e.g. "baseline" or "calm__s1"
    dialogs/{domain}/dialog_*.json
    metrics.json
    log.out
  logs/slurm/, logs/ppl_eval/, logs/trait_eval/
```

---

## 3. How to run evaluation experiments

**Standard task metrics** (method/API accuracy, dialogue success, BLEU, inform
accuracy, completion rate) on an already-generated run:

```bash
python -m entrypoints.evaluation.evaluation --json-dir output/<run>
```

To batch this across every run×domain directory under a full experiment sweep
(this is what produced every `<model>/evaluation/{us,us_sa}/<trait>__s<scalar>/<domain>/metrics.json`
file referenced throughout this repo's analysis docs) via a SLURM array instead
of one-by-one:

```bash
slurm/submit_std_eval.sh <experiment_dir> <condition_label> [max_concurrent]

# Example:
OUTPUT_BASE_DIR=$PWD/llama/evaluation \
  slurm/submit_std_eval.sh llama/llama-user-steer-2026-05-27-07-43 us 8
```

**Trait-pole / quality scoring only** (append to `metrics.json` rather than
recomputing task metrics):

```bash
python -m entrypoints.evaluation.evaluation \
  --evaluate-traits --only-traits --json-dir output/<run>
```

**Blind trait-recognition judge** (RQ2 in the paper — "does the user's steered
trait become observable?"): `entrypoints/evaluation/trait_expression_eval.py`,
config in `entrypoints/evaluation/trait_expression.yml`. This is what backs
`figure1_trait_observability.png` (§4 below).

**Pairwise LLM-judge comparison** (constraint / truthfulness / user
satisfaction win-rate between two conditions — this is what backs Figures 2–3
and Tables 2–8):

```bash
# All comparisons for one model (baseline-vs-sa + the 40-run US-vs-US+SA sweep):
bash entrypoints/evaluation/run_pairwise_comparison.sh llama
bash entrypoints/evaluation/run_pairwise_comparison.sh qwen

# Or a single manual comparison:
python -m entrypoints.evaluation.pairwise_metrics \
  --dir-a  llama/llama-user-steer-2026-05-27-07-43/runs/calm__s1/dialogs \
  --dir-b  llama/llama-user-steer-sa-both-2026-05-27-07-43/runs/calm__s1/dialogs \
  --condition-a "user-steer" --condition-b "user-steer+sa" \
  --output pairwise_calm_s1.json
```

Judge model defaults to `gpt-4.1-mini`; ~30 concurrent requests via asyncio
semaphore. Rough cost: ~$0.05 for a 50-pair baseline-vs-sa run, ~$2.00 for a
full 40-run × 50-pair US-vs-US+SA sweep. Output JSON has a `summary` block
(win/tie rates overall + `per_domain`) and a `per_pair` list (one verdict +
reason string per dialogue pair) — this is the format every file under
`<model>/pairwise_comparison/` follows.

**Re-run eval on an existing experiment / PPL only:**

```bash
bash slurm/submit_trait_eval.sh /path/to/output/my-experiment [max_concurrent] [eval_mode]
bash slurm/submit_ppl_eval.sh /path/to/output/my-experiment [max_concurrent]
```

**Summary workbook across a whole experiment:**

```bash
python analysis/build_excel.py --input-dir output/<experiment-name>
python analysis/generate_insights.py
```

---

## 4. How to reproduce the paper's figures

```bash
python analysis/plot_paper_figures.py --model qwen --out paper_figures/qwen
python analysis/plot_paper_figures.py --model llama --out paper_figures/llama
```

Reads directly from the checked-in `<model>/pairwise_comparison/*/summary.json`
(and `<model>/analysis/trait_expression_us.md` for Figure 1) — no GPU or API
calls needed, just the results already in this repo. Produces:

- `figure1_trait_observability.png` — target-trait presence rate by scalar
  (Figure 1; Qwen only, needs the blind trait-recognition judge data — skipped
  automatically for Llama, which doesn't have this data)
- `figure2_trait_advantage.png` — US+SA minus US win-rate advantage per trait,
  split into constraint/truthfulness/usersat bars (Figure 2 / Table 3 for
  Qwen; Table 7 for Llama). **Note the two models use different scalar
  conventions, matched automatically by the script**: Qwen averages over all 4
  scalars including α=0 (per the paper's Figure 2 caption); Llama excludes
  α=0 (per Table 7's caption, `s ∈ {-1,1,2}`) and is recomputed from the raw
  per-dialogue judge files rather than `summary.json`'s `per_trait` block,
  since that block doesn't separate out α=0 for this model.
- `figure3_scalar_winrates.png` — win rate vs. scalar strength, one line each
  for US and US+SA, one panel per metric (Figure 3 / Table 4 for Qwen; Table 8
  for Llama)

Additional exploratory plots (heatmaps, per-run breakdowns, box plots — not
figures in the paper itself but useful for digging further) already exist as
static PNGs under `<model>/pairwise_comparison/analysis_output/` and
`<model>/pairwise_comparison/baseline_vs_sa/analysis_output_baseline_vs_sa/`,
generated by `<model>/pairwise_comparison/analyze_results.py` (re-runnable
against fresh `summary.json` data if you regenerate the pairwise comparisons).

---

## 5. Where the paper's results and analysis write-ups live

Each model has a parallel structure — `llama/` and `qwen/qwen-v2/`:

| Path | Contents |
|---|---|
| `<model>-{baseline,sa,user-steer,user-steer-sa-both}-*/` | Raw generated dialogues, one folder per condition |
| `evaluation/` | Per-run `metrics.json` (standard task metrics), and `evaluation/trait_expression/` (blind trait-judge raw output) |
| `pairwise_comparison/` | Per-dialogue-pair LLM-judge verdicts + `summary.json` per metric; `baseline_vs_sa/` subfolder for that comparison specifically |
| `paired_dialogues/` | The matched baseline/sa dialogue pairs used for judging |
| `analysis/` | Markdown write-ups backing the paper's tables/discussion — e.g. `pairwise_summary_US-US+SA.md`, `pairwise_summary_Baseline-SA.md`, `significance_tests.md`, `evaluation_summary*.md`, `trait_expression_us.md`, plus the results workbook (`*_v2_results.xlsx`) |

`paper_figures/<model>/` (generated by §4 above) holds the actual rendered
Figure 1–3 images.

---

## 6. Statistical significance testing

Pairwise judge win/loss/tie counts are paired per-dialogue data — use
**McNemar's exact test** on the discordant pairs (ties are concordant, and
excluded). Standard task metrics (rates, not paired binary outcomes at the
dialogue level in the same way) use a **Wilcoxon signed-rank test** over the
(trait × scalar × domain) matched cells. See `llama/analysis/significance_tests.md`
for a full worked example (both tests, both comparisons, with the exact code
used to compute them) — the same approach applies unchanged to Qwen.

---

## 7. Contributing

- Branch from `master`; keep PRs focused with a short test plan.
- Never commit secrets/tokens/`.env` files — use `.env.example` placeholders only.
- Before a PR: `python -m py_compile $(git ls-files '*.py')`.
- Keep generated artifacts (`storage/`, `hf_cache/`, `logs/`, `output/`) out of git.
- New runnable scripts go under `entrypoints/`, not the repo root.
