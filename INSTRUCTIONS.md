## Instructions for a fresh clone of `Activation-Steered-Personas`

The GitHub repo is published as [`Activation-Steered-Personas`](https://github.com/MaryamShoaei1/Activation-Steered-Personas); the checkout folder name can be anything, but these instructions assume `Activation-Steered-Personas`.

Requirements: Python 3.10+, Linux with Bash 4+, and (for full experiments) a SLURM cluster with A100 GPUs.

---

### 1) Pick a workspace layout (recommended)
Keep both repos as siblings in the same parent folder:

```bash
mkdir -p ~/work/activation-steered-personas && cd ~/work/activation-steered-personas

git clone https://github.com/MaryamShoaei1/Activation-Steered-Personas
git clone https://github.com/xcfcode/persona
```

This matters because the code defaults to finding persona vectors at `../persona/vectors/<model-name>/` (relative to `Activation-Steered-Personas`). The sibling `persona` repo is also used as a fallback import path for activation-steering utilities.

---

### 2) Create Python env and install deps
From inside `Activation-Steered-Personas`:

```bash
cd ~/work/activation-steered-personas/Activation-Steered-Personas

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3) Create `.env` and set credentials (only what you need)
```bash
cp .env.example .env
```

Edit `.env` and set at least:

- **`HF_TOKEN`**: needed for gated model downloads and Hugging Face CLI downloads
- **`OPENAI_API_KEY`**: needed for LLM-judge evaluation (traits/quality)
- **`PERSONA_TOD_REPO_USERNAME`**: your cluster username (required by SLURM submitters when `$USER` is unset)

Optional, depending on provider choices:

- **`OPENROUTER_API_KEY`**: if using OpenRouter-backed simulators
- **`WANDB_PROJECT`**: defaults to `persona-vectors`

Authenticate the Hugging Face CLI once (uses `HF_TOKEN` if set):

```bash
hf auth login
# or: export HF_TOKEN=... before running hf commands
```

Install the `hf` CLI if it is not already available:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

---

### 4) Download DSTC8-SGD dataset into `storage/`
The repo expects the raw SGD/DSTC8 dataset at `storage/data/dstc8-schema-guided-dialogue`. Clone it there:

```bash
cd ~/work/activation-steered-personas/Activation-Steered-Personas
mkdir -p storage/data
git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue \
  storage/data/dstc8-schema-guided-dialogue
```

Processed CSVs under `storage/processed_data/simple_tod/` are generated automatically on the first inference run; you do not need to create them manually.

`llm_interaction/service_results/` (domain CSV lookup tables) is already included in the repo.

---

### 5) Download persona vectors into `persona/vectors/<model-name>/`
The default vector location is:

- `../persona/vectors/<model-name>/`

Where `<model-name>` is the final path segment of the HF model name, for example:
- `meta-llama/Llama-3.1-8B-Instruct` → `Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` → `Qwen2.5-7B-Instruct`

Vectors are hosted on Hugging Face at [`xiachongfeng/persona`](https://huggingface.co/datasets/xiachongfeng/persona/tree/main/vectors). Available model folders:

- `Meta-Llama-3.1-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen2.5-14B-Instruct`
- `Qwen3-4B-Instruct-2507`

Download with the Hugging Face CLI (`hf download`). Example for Qwen2.5-7B-Instruct:

```bash
cd ~/work/activation-steered-personas/Activation-Steered-Personas
source .venv/bin/activate

MODEL_DIR=Qwen2.5-7B-Instruct
PERSONA_ROOT=../persona

hf download xiachongfeng/persona \
  --repo-type dataset \
  --include "vectors/${MODEL_DIR}/*" \
  --local-dir "$PERSONA_ROOT"
```

This writes files to `../persona/vectors/<model-name>/*.pt`.

For the default Llama experiments, download the HF folder and rename it because the Hub folder name differs from the model-id suffix the code expects:

```bash
MODEL_DIR=Meta-Llama-3.1-8B-Instruct
PERSONA_ROOT=../persona

hf download xiachongfeng/persona \
  --repo-type dataset \
  --include "vectors/${MODEL_DIR}/*" \
  --local-dir "$PERSONA_ROOT"

mv "$PERSONA_ROOT/vectors/Meta-Llama-3.1-8B-Instruct" \
   "$PERSONA_ROOT/vectors/Llama-3.1-8B-Instruct"
```

Repeat for any other model(s) you plan to run. Steering expects `*_response_avg_diff.pt` files (for example `compassionate_response_avg_diff.pt`).

---

### 6) Quick sanity check (no GPU, no API keys required)
```bash
cd ~/work/activation-steered-personas/Activation-Steered-Personas
source .venv/bin/activate
python -c "import utils, utilities.runtime_config, llm_interaction.system_output_parser; print('imports OK')"
```

---

### 7) (If using SLURM) validate paths before submitting jobs

Check that persona vector files resolve correctly:

```bash
cd ~/work/activation-steered-personas/Activation-Steered-Personas
source slurm/common_env.sh
source slurm/persona_vector_paths.sh

VECTOR_DIR="$(persona_default_vector_dir "meta-llama/Llama-3.1-8B-Instruct")"
echo "Vector dir: $VECTOR_DIR"
ls "$VECTOR_DIR"/*_response_avg_diff.pt
```

Preview what Slurm jobs would be submitted without actually running them:

```bash
bash slurm/run_all_experiments.sh --dry-run
```

To run experiments for real, see [`README.md`](README.md) and [`slurm/README.md`](slurm/README.md).
