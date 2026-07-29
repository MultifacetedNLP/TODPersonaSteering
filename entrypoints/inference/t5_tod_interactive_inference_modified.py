"""Interactive E2E-TOD inference entrypoint (SGD, sa capable).

This is the canonical interactive inference entrypoint. It drives an SGD
dialogue session via :class:`InteractiveChatSession` (single-domain) or
:class:`MultiDomainInteractiveChatSession` and writes per-dialog outputs to
the run directory.

The workflow is provider-backed: both user and system utterances are generated
through the provider layer instead of local user-simulator checkpoints.

Run as a module:

    python -m entrypoints.inference.t5_tod_interactive_inference_modified

or directly from the file path:

    python entrypoints/inference/t5_tod_interactive_inference_modified.py

Hydra config defaults live under ``config/hydra/tod_runs/``; selected fields
can be overridden via env vars (see :func:`_apply_env_overrides`).
"""

import os
import sys
import re

from omegaconf import DictConfig, OmegaConf


# sys.path.insert(0, os.path.abspath("./src"))
from simple_tod_dataclasses import NlgTestDataBatch
from llm_interaction.prompt_manager.prompt_constants import NlgPromptType
from my_enums import Steps
import hydra
import ast
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from data.tod.turns.zs_tod_turn import TodTurnCsvRow
from pathlib import Path
from dotmap import DotMap
from torch.utils.data import Dataset
from accelerate import Accelerator
import torch
from datasets import Dataset
from utilities import text_utilities
from transformers import AutoTokenizer
from datetime import datetime
import utils
from tqdm import tqdm
import numpy as np
import logging
import socket
import subprocess

logger = logging.getLogger(__name__)

# import wandb


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# dataset libraries
from llm_interaction.prompt_manager.nlg_prompt_manager import NlgPrompt
from typing import List, Union
from dataclasses import dataclass, asdict
import copy
from data.tod.turns.zs_tod_turn import (
    TodTurnCsvRow,
    TodTurnApiCallCsvRow,
    TodTurnApiCallSimulatorCsvRow
)
from config.python.dm_config import DataModuleConfig
from config.python.dataprep_config import DataPrepConfig
from data.data_prep.data_prep_strategy_resolver import DataPrepStrategyResolver
from data.data_prep.dstc_base_data_prep import DstcBaseDataPrep
from data.data_prep.simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
from llm_interaction.sgd_chat_simulator import SGDChatSimulator
import json
from llm_interaction.providers.factory import generate_with_provider

from fuzzywuzzy import fuzz

IGNORE_INDEX = -100


def _apply_env_overrides(cfg: DictConfig) -> list[str]:
    """
    Allow overriding selected Hydra config values via environment variables.

    Intended for Slurm usage so multiple jobs can share the same YAML config
    without editing it in place.

    System env vars:
      - PERSONA_TOD_SYSTEM_PROVIDER
      - PERSONA_TOD_SYSTEM_MODEL_NAME
      - PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH
      - PERSONA_TOD_SYSTEM_ANTHROPIC_COEF
      - PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER

    User env vars:
      - PERSONA_TOD_USER_PROVIDER
      - PERSONA_TOD_USER_MODEL_NAME
      - PERSONA_TOD_USER_ANTHROPIC_VECTOR_PATH
      - PERSONA_TOD_USER_ANTHROPIC_COEF
      - PERSONA_TOD_USER_ANTHROPIC_LAYER

    Run env vars:
      - PERSONA_TOD_TEST_DOMAINS: a Python nested list such as
        "[[\"Restaurants_2\"],[\"Movies_1\",\"Movies_3\"]]".
    """

    def _set(path: str, raw_value: str, caster):
        value = caster(raw_value)
        OmegaConf.update(cfg, path, value, merge=True, force_add=True)
        return f"{path}={value!r}"

    def _needs_anthropic(provider: str) -> bool:
        return any(k in provider for k in ("anthropic", "steer", "sa"))

    def _parse_test_domains(raw_value: str) -> list[list[str]]:
        parsed = ast.literal_eval(raw_value)
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(isinstance(group, list) and group for group in parsed)
            or not all(isinstance(domain, str) for group in parsed for domain in group)
        ):
            raise ValueError(
                "PERSONA_TOD_TEST_DOMAINS must be a Python nested list like "
                "'[[\"Restaurants_2\"], [\"Movies_1\", \"Movies_3\"]]'"
            )
        return parsed

    applied: list[str] = []

    # System model overrides
    v = os.environ.get("PERSONA_TOD_SYSTEM_PROVIDER")
    if v is not None and v != "":
        applied.append(_set("system.provider", v, str))

    v = os.environ.get("PERSONA_TOD_SYSTEM_MODEL_NAME")
    if v is not None and v != "":
        applied.append(_set("system.model_name", v, str))

    sys_provider = OmegaConf.select(cfg, "system.provider", default="")
    if _needs_anthropic(str(sys_provider)):
        v = os.environ.get("PERSONA_TOD_SYSTEM_ANTHROPIC_VECTOR_PATH")
        if v is not None and v != "":
            applied.append(_set("system.anthropic.vector_path", v, str))

        v = os.environ.get("PERSONA_TOD_SYSTEM_ANTHROPIC_COEF")
        if v is not None and v != "":
            applied.append(_set("system.anthropic.coef", v, float))

        v = os.environ.get("PERSONA_TOD_SYSTEM_ANTHROPIC_LAYER")
        if v is not None and v != "":
            applied.append(_set("system.anthropic.layer", v, int))

    # User model overrides
    v = os.environ.get("PERSONA_TOD_USER_PROVIDER")
    if v is not None and v != "":
        applied.append(_set("user.provider", v, str))

    v = os.environ.get("PERSONA_TOD_USER_MODEL_NAME")
    if v is not None and v != "":
        applied.append(_set("user.model_name", v, str))

    user_provider = OmegaConf.select(cfg, "user.provider", default="")
    if _needs_anthropic(str(user_provider)):
        v = os.environ.get("PERSONA_TOD_USER_ANTHROPIC_VECTOR_PATH")
        if v is not None and v != "":
            applied.append(_set("user.anthropic.vector_path", v, str))

        v = os.environ.get("PERSONA_TOD_USER_ANTHROPIC_COEF")
        if v is not None and v != "":
            applied.append(_set("user.anthropic.coef", v, float))

        v = os.environ.get("PERSONA_TOD_USER_ANTHROPIC_LAYER")
        if v is not None and v != "":
            applied.append(_set("user.anthropic.layer", v, int))

    v = os.environ.get("PERSONA_TOD_TEST_DOMAINS")
    if v is not None and v != "":
        applied.append(_set("test_domain_settings", v, _parse_test_domains))

    v = os.environ.get("PERSONA_TOD_NUM_DIALOG_SHARDS")
    if v is not None and v != "":
        applied.append(_set("num_dialog_shards", v, int))

    v = os.environ.get("PERSONA_TOD_DIALOG_SHARD_ID")
    if v is not None and v != "":
        applied.append(_set("dialog_shard_id", v, int))

    return applied


def _resolve_run_paths(project_root: Path) -> tuple[str, Path, Path]:
    try:
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        config_dir = hydra_output_dir if hydra_output_dir.name == "config" else hydra_output_dir / "config"
        run_root = config_dir.parent
        return run_root.name, run_root, config_dir
    except Exception:
        run_id = os.environ.get("PERSONA_TOD_RUN_ID") or os.environ.get("SLURM_JOB_ID")
        if not run_id:
            run_id = f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_root = project_root / "storage" / str(run_id)
        config_dir = run_root / "config"
        return str(run_id), run_root, config_dir


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_run_metadata(
    run_root: Path,
    run_name: str,
    project_root: Path,
    cfg_container: dict,
    status: str,
) -> None:
    metadata_path = run_root / "metadata.json"
    previous: dict = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                previous = json.load(f)
        except Exception:
            previous = {}

    now = datetime.now().astimezone().isoformat()
    metadata = {
        **previous,
        "experiment_name": os.environ.get("PERSONA_TOD_EXPERIMENT_NAME"),
        "experiment_description": os.environ.get("PERSONA_TOD_EXPERIMENT_DESCRIPTION", ""),
        "run_name": os.environ.get("PERSONA_TOD_RUN_NAME") or run_name,
        "run_description": os.environ.get("PERSONA_TOD_RUN_DESCRIPTION", ""),
        "status": status,
        "updated_at": now,
        "command": os.environ.get("PERSONA_TOD_COMMAND"),
        "project_root": str(project_root),
        "experiment_dir": os.environ.get("PERSONA_TOD_EXPERIMENT_DIR"),
        "output_dir": str(run_root),
        "dialogs_dir": str(run_root / "dialogs"),
        "config_name": os.environ.get("CONFIG_NAME") or os.environ.get("PERSONA_TOD_CONFIG_NAME"),
        "git_commit": previous.get("git_commit") or _git_commit(project_root),
        "hostname": socket.gethostname(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "environment": {
            "experiment_id": os.environ.get("EXPERIMENT_ID"),
            "run_label": os.environ.get("RUN_LABEL"),
            "domain": os.environ.get("DOMAIN"),
            "python": sys.executable,
        },
        "config_summary": {
            "dataset_type": cfg_container.get("dataset", {}).get("dataset_type"),
            "test_domain_settings": cfg_container.get("test_domain_settings"),
            "num_dialogues_per_domain": cfg_container.get("num_dialogues_per_domain"),
            "system_provider": cfg_container.get("system", {}).get("provider"),
            "system_model_name": cfg_container.get("system", {}).get("model_name"),
            "user_provider": cfg_container.get("user", {}).get("provider"),
            "user_model_name": cfg_container.get("user", {}).get("model_name"),
        },
    }
    metadata.setdefault("created_at", now)
    _write_json(metadata_path, metadata)

@dataclass(frozen=True)
class StepData:
    name: Steps
    num_dialog: int
    overwrite: bool
    split_percent: float
    domain_settings: Union[list[str], str]
    
class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[TodTurnCsvRow],
    ):
        self.data: list[TodTurnCsvRow] = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TodTurnCsvRow:
        return self.data[idx]
    
@dataclass(frozen=False)
class TodTrainRowCollator:
    input_tokens: torch.IntTensor
    label: torch.IntTensor
    attention_mask: torch.IntTensor

# end of libraries

class T5Tod:
    def __init__(self, cfg, original_cfg_snapshot=None):
        self.original_cfg = cfg
        # Capture a snapshot of the initial composed config to avoid later mutations/edits affecting saved config.json
        self.original_cfg_snapshot = (
            original_cfg_snapshot
            if original_cfg_snapshot is not None
            else OmegaConf.to_container(cfg, resolve=True)
        )
        self.cfg = DotMap(cfg)
        # Hydra may change cwd to the job output directory, so resolve relative
        # project_root against Hydra's original cwd (repo root in our jobs).
        project_root_cfg = Path(str(cfg.project_root))
        if project_root_cfg.is_absolute():
            self.cfg.project_root = project_root_cfg
        else:
            self.cfg.project_root = (Path(get_original_cwd()) / project_root_cfg).resolve()
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.dataset.raw_data_root
        self.cfg.context_type = self.cfg.dataset.context_type
        self.cfg.data_prep_out_root = self.cfg.dataset.data_prep_out_root
        self.cfg.num_dialogs = self.cfg.data_size.num_dialogs
        self.cfg.data_split_percent = self.cfg.data_size.data_split_percent
        self.run_id, self.run_root, self.config_dir = _resolve_run_paths(self.cfg.project_root)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.config_dir / "t5_tod.log"
        # logging.basicConfig(
        #     filename=str(log_file), level=logging.INFO, encoding="utf-8", format='%(message)s'
        # )
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger
        # self.logger = logging
        self.cfg.out_dir = self.run_root
        
    def replace_last_system(self, text):
        # Find the last occurrence of the term "System"
        last_index = text.rfind("System")
        
        # If "System" is found, replace it
        if last_index != -1:
            text = text[:last_index] + "Last System Utterance" + text[last_index + len("System"):]
        
        return text
    
    def extract_dialog_data(self, last_turn, api_calls):
        user_schema = re.sub(r'^(required|optional) slots:.*\n?', '', last_turn.schema, flags=re.MULTILINE)
        system_schema = re.sub(r'^result slots:.*\n?', '', last_turn.schema, flags=re.MULTILINE)
        search_results = re.findall(r'Search Results\n(.*?)\nEnd Search Results', last_turn.context, re.DOTALL)
        dialog = last_turn.context.split("\nEnd Dialog History")[0] + "\nLast System Utterance: " + (last_turn.target or "")
        
        
        return {
            "api_calls": api_calls,
            "search_results": search_results,
            "user_schema": user_schema,
            "system_schema": system_schema,
            "domains": last_turn.domains,
            "dialog": dialog
        }
        
    ###### Start of loading the training ######
    def setup_single_run(
        self, step: str, step_data: StepData, domain_setting: Union[str, list[str]]
    ) -> "SimpleTodDataSet":
        self.data_simulator_cfg = copy.deepcopy(self.data_cfg)
        self.data_simulator_cfg.step_name = step_data.name
        self.data_simulator_cfg.num_dialogs = step_data.num_dialog
        self.data_simulator_cfg.overwrite = step_data.overwrite
        self.data_simulator_cfg.domain_setting = domain_setting

        data_prep = self.get_data_prep_class(self.data_simulator_cfg)
        self.prepare_data(data_prep)
        csv_path = utils.get_csv_data_path(
            step,
            step_data.num_dialog,
            cfg=data_prep.cfg,
        )
        try:
            dialog_data = utils.read_csv_dataclass(csv_path, TodTurnApiCallSimulatorCsvRow)
        except FileNotFoundError:
            dialog_data = []
            
        
        # create a dictionary which contains the information of each dialog
        dialog_dict = {}
        api_calls = []
        all_user_req_slots = []
        user_req_slots = []
        last_turn = None
        for turn in tqdm(dialog_data):
            if last_turn is not None and last_turn.dialog_id != turn.dialog_id:
                
                all_user_req_slots.append(user_req_slots)
                all_user_req_slots = all_user_req_slots[1:]  # drop the initial empty entry added before the first API call
                search_results = re.findall(r'Search Results\n(.*?)\nEnd Search Results', last_turn.context, re.DOTALL)
                dialog = last_turn.context.split("\nEnd Dialog History")[0] + "\nLast System Utterance: " + (last_turn.target or "")
                dialog_dict[last_turn.dialog_id] = {
                    "api_calls": api_calls,
                    "search_results": search_results,
                    "user_req_slots": all_user_req_slots,
                    "system_schema": last_turn.schema,
                    "domains": last_turn.domains,
                    "domains_original": last_turn.domains_original,
                    "dialog": dialog
                }
                
                api_calls = [] # empty the api_calls list for the next dialog
                all_user_req_slots = [] # empty the user_req_slots list for the next dialog
            
            last_turn = turn
            context = asdict(turn)["context"]
            if "Last User Utterance:" not in context:
                continue
            
            if turn.target and turn.target.startswith("ApiCall("):
                api_calls.append(turn.target)
                
                all_user_req_slots.append(user_req_slots)
                user_req_slots = []
                
            if turn.user_req_slots:
                user_req_slots.extend(ast.literal_eval(turn.user_req_slots))
        
        # for the last dialog
        if last_turn is not None:
            all_user_req_slots.append(user_req_slots)
            all_user_req_slots = all_user_req_slots[1:]  # drop the initial empty entry added before the first API call
            search_results = re.findall(r'Search Results\n(.*?)\nEnd Search Results', last_turn.context, re.DOTALL)
            dialog = last_turn.context.split("\nEnd Dialog History")[0] + "\nLast System Utterance: " + (last_turn.target or "")
            dialog_dict[last_turn.dialog_id] = {
                "api_calls": api_calls,
                "search_results": search_results,
                "user_req_slots": all_user_req_slots,
                "system_schema": last_turn.schema,
                "domains": last_turn.domains,
                "domains_original": last_turn.domains_original,
                "dialog": dialog
            }
        
        if self.cfg.num_dialogues_per_domain != -1:
            # set the random seed for reproducibility when requested
            try:
                selection_seed = int(getattr(self.cfg, "selection_seed", -1))
            except Exception:
                selection_seed = -1
            if selection_seed >= 0:
                np.random.seed(selection_seed)
            else:
                # None → unpredictably seed from OS entropy
                np.random.seed(None)
            
            multi_dialog_ids = []
            single_dialog_ids = []
            for dialog_id, dialog_info in dialog_dict.items():
                if len(dialog_info["domains"].split(",")) > 1: # multi-domain dialog
                    multi_dialog_ids.append(dialog_id)
                else:
                    single_dialog_ids.append(dialog_id)
            
            # Keep each test-domain bucket to a fixed size while preserving a mix
            # of single-service and multi-service dialogs when possible.
            num_multi = int(self.cfg.num_dialogues_per_domain / 2)
            num_single = self.cfg.num_dialogues_per_domain - num_multi
            
            final_dialog_ids = []
            if len(multi_dialog_ids) >= num_multi:
                final_dialog_ids.extend(np.random.choice(multi_dialog_ids, num_multi, replace=False).tolist())
            else:
                final_dialog_ids.extend(multi_dialog_ids)
                num_single += (num_multi - len(multi_dialog_ids))

            if len(single_dialog_ids) >= num_single:
                final_dialog_ids.extend(np.random.choice(single_dialog_ids, num_single, replace=False).tolist())
            else:
                final_dialog_ids.extend(single_dialog_ids)
            
            # Randomly select num_dialogues_per_domain from the keep_dialog_ids.
            # if we still have fewer than requested due to small population, take all
            sample_size = min(len(final_dialog_ids), self.cfg.num_dialogues_per_domain)
            keep_dialog_ids = np.random.choice(final_dialog_ids, sample_size, replace=False)
            
            dialog_dict = {k: v for k, v in dialog_dict.items() if k in keep_dialog_ids}
        else:
        
            if self.data_cfg.multi_domain:
                keep_dialog_ids = []
                for dialog_id, dialog_info in dialog_dict.items():
                    if len(dialog_info["domains"].split(",")) > 1: # multi-domain dialog
                        keep_dialog_ids.append(dialog_id)
                        
                dialog_dict = {k: v for k, v in dialog_dict.items() if k in keep_dialog_ids}
            else:
                keep_dialog_ids = []
                for dialog_id, dialog_info in dialog_dict.items():
                    if len(dialog_info["domains"].split(",")) == 1: # single-domain dialog
                        keep_dialog_ids.append(dialog_id)
                        
                dialog_dict = {k: v for k, v in dialog_dict.items() if k in keep_dialog_ids}
            
            """
            folder_path = str(Path(self.cfg.project_root) / "storage" / "interactive_generated_single_domain_gpt-4o (copy)")
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        dialog_id = filename[7:-5]
                        user_req_slots = dialog_dict[dialog_id]["user_req_slots"]
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        data["user_req_slots"] = user_req_slots
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4)  # Pretty-print with indentation

                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
            """
        
        return dialog_dict

    def get_data_prep_class(self, cfg: DataModuleConfig):
        if isinstance(cfg.raw_data_root, str):
            cfg.raw_data_root = Path(cfg.raw_data_root)
        dp_cfg = DataPrepConfig.from_dm_config(cfg)
        strategy = DataPrepStrategyResolver.resolve(dp_cfg)
        return DstcBaseDataPrep(dp_cfg, strategy)

    def prepare_data(self, stdp: SimpleTODDSTCDataPrep):
        # with self.cfg.accelerator.main_process_first():
        if self.data_simulator_cfg.accelerator.is_main_process:
            stdp.run()
        self.data_simulator_cfg.accelerator.wait_for_everyone()
        
    def get_step_data(self, step: Steps) -> StepData:
        index = Steps.get_index(step)
        
        self.domain_step_map = {
            Steps.TRAIN: f"{Steps.TRAIN.value}_domain_settings",
            Steps.DEV: f"{Steps.DEV.value}_domain_settings",
            Steps.TEST: f"{Steps.TEST.value}_domain_settings",
            }
        
        return StepData(
            step,
            self.cfg.num_dialogs[index],
            self.cfg.overwrite[index],
            self.cfg.data_split_percent[index],
            getattr(self.cfg, self.domain_step_map[step]),
        )
        
    ###### End of loading the training ######
        
    
    ###### start of trainer collate ######
    def my_tokenize(self, text: str, max_len: int = None):
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=max_len)
        return tokens.to(dtype=torch.int32)[0]
    
    def get_t5_item(
        self,
        context_tokens,
        target_tokens,
        pad_tokens,
        target_max_len,
    ) -> TodTrainRowCollator:
        input_tokens = torch.cat(
            [
                pad_tokens,
                context_tokens,
            ]
        )
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int32)

        target_unused_len = target_max_len - len(target_tokens)
        if target_unused_len < 0:
            raise Exception("Target is too long")
        label = torch.cat(
            [
                target_tokens,
                torch.full([1], self.tokenizer.eos_token_id), # TODO: the T5 model already has end of sentence token, do we need to add eos_token_id again
                torch.full([target_unused_len], IGNORE_INDEX), # TODO: it might be better to use the ignore label id (IGNORE_INDEX = -100). Previous Code: torch.full([target_unused_len], self.tokenizer.pad_token_id)
            ]
        )
        return TodTrainRowCollator(input_tokens, label, attention_mask)
        
    def collate_single_item(
        self, item: TodTurnCsvRow,
    ):
        other_domain, other_domain_schema = None, None
        if self.cfg.prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            other_domain, other_domain_schema = self.get_other_domain(item)
            
        if isinstance(item, TodTurnApiCallSimulatorCsvRow):
            context_text = NlgPrompt().get_prompt_simulator_v2(
                item.domains, 
                # item.schema, 
                item.next_api_call,
                item.next_user_req_slots,
                item.last_api_call,
                item.last_user_req_slots,
                item.context
            )
        else:
            context_text = NlgPrompt().get_prompt(
                item.domains,
                item.schema,
                item.context
            )
        context_tokens = self.my_tokenize(context_text)
        context_unused_len = self.cfg.test_prompt_max_len - len(context_tokens)
        if context_unused_len < 0:
            context_tokens = self.trim_dialog_history(
                item, -context_unused_len, other_domain, other_domain_schema
            )
            context_unused_len = self.cfg.test_prompt_max_len - len(context_tokens)
        pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
        
        input_tokens = torch.cat(
            [
                pad,
                context_tokens,
            ]
        )
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int32)
        
        return input_tokens, attention_mask
        
        
    def trim_dialog_history(
        self,
        item: TodTurnCsvRow,
        trim_len: int,
        other_domain: str,
        other_domain_schema: str,
    ):
        dialog_history_tokens = self.my_tokenize(item.context)
        trimmed_history_tokens = dialog_history_tokens[trim_len + 15 :]
        trimmed_history_text = self.tokenizer.decode(trimmed_history_tokens)
        if isinstance(item, TodTurnApiCallSimulatorCsvRow):
            context_text = NlgPrompt().get_prompt_simulator_v2(
                item.domains, 
                # item.schema, 
                item.next_api_call,
                item.next_user_req_slots,
                item.last_api_call,
                item.last_user_req_slots,
                trimmed_history_text
            )
        else:
            context_text = NlgPrompt().get_prompt(
                item.domains,
                item.schema,
                trimmed_history_text
            )
        context_tokens = self.my_tokenize(context_text)
        if len(context_tokens) > self.cfg.test_prompt_max_len:
            overflow_tokens = len(context_tokens) - self.cfg.test_prompt_max_len
            return self.trim_dialog_history(
                item, -overflow_tokens, other_domain, other_domain_schema
            )
        return context_tokens
    
    ###### End of trainer collate ######
    
    
    
    ###### Start of testing collate ######
    
    def my_tokenizer_pad(self, text: str, max_len: int = None):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
    
    def tod_test_collate(self, batch: list[TodTurnApiCallCsvRow]):
        data = DotMap(
            {
                key: []
                for key in [
                    "input_tokens",
                    # "contexts",
                    "attention_masks",
                    "dialog_ids",
                    "turn_ids",
                    "labels",
                    "turn_row_types",
                    "is_retrievals",
                    "is_slot_fills",
                    "is_multi_domain_api_calls",
                    "domain_ids",
                ]
            }
        )
        max_lengths = DotMap(
            domains=100,
        )
        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len, is_test=True)
            data.input_tokens.append(row.input_tokens)
            data.attention_masks.append(row.attention_mask)
            data.labels.append(row.label)
            data.turn_row_types.append(torch.tensor(item.turn_row_type))
            data.is_retrievals.append(torch.tensor(item.is_retrieval))
            data.is_slot_fills.append(torch.tensor(item.is_slot_fill))
            data.turn_ids.append(torch.tensor(int(item.turn_id)))
            data.dialog_ids.append(torch.tensor(int(item.dialog_id)))
            multi_dom = getattr(item, "is_multi_domain_api_call", 0)
            if not multi_dom:
                multi_dom = 0
            data.is_multi_domain_api_calls.append(torch.tensor(int(multi_dom)))
            domain_tokens = self.my_tokenizer_pad(
                item.domains_original, max_lengths.domains
            )
            data.domain_ids.append(domain_tokens["input_ids"][0])
        return NlgTestDataBatch(
            dialog_ids=torch.stack(data.dialog_ids),
            turn_ids=torch.stack(data.turn_ids),
            input_ids=torch.stack(data.input_tokens),
            attention_masks=torch.stack(data.attention_masks),
            labels=torch.stack(data.labels),
            turn_row_types=torch.stack(data.turn_row_types),
            is_retrievals=torch.stack(data.is_retrievals),
            is_slot_fills=torch.stack(data.is_slot_fills),
            is_multi_domain_api_calls=torch.stack(data.is_multi_domain_api_calls),
            domain_ids=torch.stack(data.domain_ids),
        )

    ###### End of testing collate ######





    def __old_init__(self, cfg, **kwargs):
        self.cfg = DotMap(dict(cfg))
        self.cfg.project_root = Path(self.cfg.project_root)
        base_out_dir = self.cfg.project_root / "playground" / "t5_tod_out"
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        self.cfg.out_dir = base_out_dir / date_str / time_str
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.cfg.out_dir)
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.out_dir = Path(os.getcwd())
        log_file = self.cfg.out_dir / "t5_tod.log"
        logging.basicConfig(filename=log_file, level=logging.INFO, encoding="utf-8")
        self.logger = logging
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.raw_data_root
        self.logger.info(self.cfg)
        print(self.cfg)

    def remove_padding(self, gen):
        return [row[row != self.tokenizer.pad_token_id] for row in gen]
    
    def remove_pad_decode(self, text, skip_special_tokens=False):
        if skip_special_tokens:
            return self.tokenizer.batch_decode(
                text, skip_special_tokens=skip_special_tokens
            )
        txt_no_pad = self.remove_padding(text)
        return self.tokenizer.batch_decode(txt_no_pad, skip_special_tokens=False)
    
    
    def extract_api_call(self, text):
        # Regular expression to extract the API call
        match = re.search(r'APICall\(.*?\)', text)

        if match:
            api_call = match.group(0)
        else:
            api_call = None
        
        return api_call
            
    def save_dialog_data(self, dialog, dialog_id, output_dir):
        
        # Create full path for the file
        filename = f"dialog_{dialog_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        if "predicted_api_calls" not in dialog:
            dialog["predicted_api_calls"] = []
        
        if "predicted_search_results" not in dialog:
            dialog["predicted_search_results"] = []
            
        # Create a new dictionary with all data except search_results
        output_data = {
            'api_calls': dialog['api_calls'],
            'user_req_slots': dialog['user_req_slots'],
            'system_schema': dialog['system_schema'],
            'domains': dialog['domains'],
            'domains_original': dialog['domains_original'],
            'dialog': dialog['dialog'],
            'predicted_api_calls': dialog['predicted_api_calls'],
            'generated_dialog': dialog['generated_dialog'],
            "search_results": dialog['search_results'],
            "predicted_search_results": dialog['predicted_search_results'],
            "sa_coeffs": dialog.get('sa_coeffs', []),
            "baseline_counterfactuals": dialog.get('baseline_counterfactuals', []),
            "dialog_completed": dialog.get('dialog_completed', False),
        }
        
        # Create the output directory if it doesn't exist
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data to a JSON file with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
    def extract_first_user_utterance_from_dialog(self, dialog_text: str):
        """
        Return the first ground-truth user utterance from a serialized dialog string.
        Lines are prefixed with 'User:' and 'System:'.
        """
        if not dialog_text:
            return None
        for line in dialog_text.splitlines():
            s = line.strip()
            if s.startswith("User:"):
                return s.split("User:", 1)[1].strip()
        return None
    
    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)

        tokenizer_name = (
            OmegaConf.select(self.original_cfg, "tokenizer_name")
            or OmegaConf.select(self.original_cfg, "user.model_name")
            or OmegaConf.select(self.original_cfg, "system.model_name")
        )
        if not tokenizer_name:
            raise ValueError("Set tokenizer_name, user.model_name, or system.model_name in the TOD config.")

        user_provider = OmegaConf.select(self.original_cfg, "user.provider")
        if not user_provider:
            raise ValueError(
                "user.provider must be set. Valid values: local, openai, local_steer_anthropic."
            )

        # The tokenizer is only used for data preparation and prompt length accounting.
        self.tokenizer  = AutoTokenizer.from_pretrained(
            tokenizer_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        self.tokenizer .add_special_tokens(
            {"additional_special_tokens": ["<SYSTEM>", "<USER>", "{", "}"]}
            # {"additional_special_tokens": ["<SYSTEM>", "<USER>"]}
        )
        self.tokenizer.model_max_length = 1024
        
        output_dir_path = self.run_root
        config_dir_path = self.config_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)
        config_dir_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(output_dir_path)
        # Persist the used config for later identification
        try:
            cfg_container = self.original_cfg_snapshot
            _write_json(config_dir_path / "config.json", cfg_container)
            _write_json(output_dir_path / "config.json", cfg_container)
        except Exception as e:
            # Best-effort fallback: write a minimal error note if config can't be serialized
            cfg_container = {"error": f"Failed to serialize config: {str(e)}"}
            _write_json(config_dir_path / "config.json", cfg_container)
            _write_json(output_dir_path / "config.json", cfg_container)
        _write_run_metadata(output_dir_path, self.run_id, self.cfg.project_root, cfg_container, "inference_started")
        
        # prepare the dialog data
        self.data_cfg = DataModuleConfig(tokenizer=self.tokenizer , **self.cfg)  
        test_datasets: list[SimpleTodDataSet] = []
        step = Steps.TEST.value
        step_data = self.get_step_data(step)
        for domain_setting in step_data.domain_settings:
            test_datasets.append(
                self.setup_single_run(step, step_data, domain_setting)
            )
        
        index = 0
        for test_dataset, domain_names_list in zip(
            test_datasets, self.cfg.test_domain_settings
        ):
            domain_names = ",".join(domain_names_list)
            if not len(test_dataset):
                print(f"No data for {domain_names}")
                continue
            print(f"testing {domain_names}")
            utils.log(self.logger, f"testing {domain_names}")
            
            all_dialog_ids = sorted(test_dataset.keys())
            num_shards = int(getattr(self.cfg, 'num_dialog_shards', 1) or 1)
            shard_id = int(getattr(self.cfg, 'dialog_shard_id', 0) or 0)
            if num_shards > 1:
                all_dialog_ids = [d for i, d in enumerate(all_dialog_ids) if i % num_shards == shard_id]

            max_iterations = len(all_dialog_ids)
            progress_bar = tqdm(total=max_iterations)

            domain_slug = domain_names.replace(",", "_").replace(" ", "_")
            output_dir = str(Path(self.output_dir) / "dialogs" / domain_slug)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            if os.path.exists(output_dir):
                generated_dialog_ids = {
                    dialog_file.stem[len("dialog_"):]
                    for dialog_file in Path(output_dir).glob("dialog_*.json")
                }
                ready_dialogs = [d for d in all_dialog_ids if d not in generated_dialog_ids]
                progress_bar.update(max_iterations - len(ready_dialogs))
            else:
                ready_dialogs = all_dialog_ids
            
            # If there are no dialogs left to process, still run one steered test so output is visible
            # if len(ready_dialogs) == 0:
            #     sys_cfg = getattr(self.cfg, "system", None)
            #     sys_provider = getattr(self.cfg, "api_source", None)
            #     sys_model_name = getattr(self.cfg, "llm_model_name", None)
            #     sys_hf_local = getattr(self.cfg, "hf_local", None)
            #     sys_easy_steer = getattr(self.cfg, "easy_steer", None)
            #     sys_repeng = None
            #     if sys_cfg:
            #         sys_provider = sys_cfg.get("provider", sys_provider)
            #         sys_model_name = sys_cfg.get("model_name", sys_model_name)
            #         sys_hf_local = sys_cfg.get("hf_local", sys_hf_local)
            #         sys_easy_steer = sys_cfg.get("easy_steer", sys_easy_steer)
            #         sys_repeng = sys_cfg.get("repeng", None)

            #     test_prompt = "<|im_start|>user Give me a one-sentence pitch for a TV show. <|im_end|>\n<|im_start|>assistant"
            #     test_repeng = dict(sys_repeng or {})
            #     gen_txt = generate_with_provider(
            #         prompt=test_prompt,
            #         provider=sys_provider,
            #         model_name=sys_model_name,
            #         gen={"max_new_tokens": 64},
            #         hf_local=sys_hf_local,
            #         easy_steer=sys_easy_steer,
            #         repeng=test_repeng,
            #     )
            #     print("STEER TEST (system):", gen_txt, flush=True)

            # target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
            
            for dialog_id in ready_dialogs:
                
                dialog_info = test_dataset[dialog_id].copy()
                
                # Resolve System provider/model with backward compatibility
                sys_cfg = getattr(self.cfg, "system", None)
                sys_provider = getattr(self.cfg, "api_source", None)
                sys_model_name = getattr(self.cfg, "llm_model_name", None)
                sys_hf_local = getattr(self.cfg, "hf_local", None)
                sys_easy_steer = getattr(self.cfg, "easy_steer", None)
                sys_repeng = None
                sys_custom_instruction = None
                sys_system_prompt = None
                if sys_cfg:
                    sys_provider = sys_cfg.get("provider", sys_provider)
                    sys_model_name = sys_cfg.get("model_name", sys_model_name)
                    sys_hf_local = sys_cfg.get("hf_local", sys_hf_local)
                    sys_easy_steer = sys_cfg.get("easy_steer", sys_easy_steer)
                    sys_repeng = sys_cfg.get("repeng", None)
                    sys_anthropic = sys_cfg.get("anthropic", None)
                    sys_custom_instruction = sys_cfg.get("customInstruction", None)
                    sys_system_prompt = sys_cfg.get("systemPrompt", None)

                services = dialog_info['domains_original'].split(",")
                sim_kwargs = dict(
                    services=services,
                    api_source=sys_provider,
                    model_name=sys_model_name,
                    activate_retry=self.cfg.activate_retry,
                    add_example=self.cfg.add_example,
                    hf_local=sys_hf_local,
                    easy_steer=sys_easy_steer,
                    repeng=sys_repeng,
                    anthropic=sys_anthropic,
                    system_prompt=sys_system_prompt,
                )
                if sys_custom_instruction:
                    sim_kwargs["system_custom_instruction"] = sys_custom_instruction
                chat_session = SGDChatSimulator(**sim_kwargs)
                    
                excluding_dialog_id = chat_session()
                    
                if excluding_dialog_id == dialog_id:
                    progress_bar.update(1)
                    test_dataset[dialog_id]["generated_dialog"] = "Ignoring the dialog"
                    continue
            
                # Seed the first user message from ground-truth dialog before starting the loop
                if 'log_user_context' not in dialog_info:
                    dialog_info['log_user_context'] = []
                if 'log_system_context' not in dialog_info:
                    dialog_info['log_system_context'] = []
                if len(dialog_info['log_user_context']) == 0 and len(dialog_info['log_system_context']) == 0:
                    first_user_msg = self.extract_first_user_utterance_from_dialog(dialog_info.get('dialog', ''))
                    if first_user_msg:
                        dialog_info['log_user_context'].append(f"User: {first_user_msg}")
                        dialog_info['log_system_context'].append(f"User: {first_user_msg}")
                        dialog_info['turn_tracker'] = "system"
                
                dialog_info['turn_id'] = -1
                
                while(True):
                
                    # add turn_id to the dialog_info
                    if 'turn_id' in dialog_info:
                        dialog_info['turn_id'] += 1 

                    ############################################################ context
                    # add the context to the dialog_info
                    # I need to context: one context is for user which does not have search results and api calls and the other context is for system which has search results and api calls
                                
                    if 'log_user_context' not in dialog_info:
                        # initializing some of the dialog_info
                        dialog_info['log_user_context'] = []
                                
                    if 'log_system_context' not in dialog_info:
                        # initializing some of the dialog_info
                        dialog_info['log_system_context'] = []
                                
                    ############################################################ context
                    
                    ############################################################ turn_tracker
                    if "turn_tracker" not in dialog_info:
                        dialog_info["turn_tracker"] = "user"
                    ############################################################ turn_tracker
                            
                    ############################################################ api_tracker
                    if "api_tracker" not in dialog_info:
                        dialog_info["api_tracker"] = 0
                            
                    if dialog_info["api_tracker"] < len(dialog_info['api_calls']):
                        next_api_call = dialog_info['api_calls'][dialog_info["api_tracker"]]
                        next_user_req_slots = dialog_info['user_req_slots'][dialog_info["api_tracker"]]
                    else:
                        next_api_call = None
                        next_user_req_slots = None
                            
                    last_api_tracker = dialog_info["api_tracker"] - 1
                    if last_api_tracker >= 0:
                        last_api_call = dialog_info['api_calls'][last_api_tracker]
                        last_user_req_slots = dialog_info['user_req_slots'][last_api_tracker]
                    else:
                        last_api_call = None
                        last_user_req_slots = None
                    ############################################################ api_tracker
                    
                    if dialog_info["turn_tracker"] == "user":
                        
                        # change the turn_tracker to system
                        dialog_info["turn_tracker"] = "system"
                        
                        input_context = copy.deepcopy(dialog_info['log_user_context'])
                        if len(dialog_info['log_user_context']) > 0:
                            input_context[-1] = "Last System Utterance" + input_context[-1][6:]
                            
                        
                        item = TodTurnApiCallSimulatorCsvRow(dialog_id=dialog_id,
                                                            turn_id=dialog_info['turn_id'],
                                                            context="\n".join(input_context), #dialog_info['user_context'],
                                                            domains=dialog_info['domains'],
                                                            next_api_call=next_api_call,
                                                            next_user_req_slots=next_user_req_slots,
                                                            last_api_call=last_api_call,
                                                            last_user_req_slots=last_user_req_slots)
                        # User utterances now always come from a provider-backed model.
                        user_cfg = getattr(self.cfg, "user", None)
                        user_provider = None
                        user_model_name = None
                        user_hf_local = None
                        user_easy_steer = None
                        user_repeng = None
                        user_gen = None
                        user_custom_instruction = None
                        user_system_prompt = None
                        if user_cfg:
                            user_provider = user_cfg.get("provider", None)
                            user_model_name = user_cfg.get("model_name", None)
                            user_hf_local = user_cfg.get("hf_local", None)
                            user_easy_steer = user_cfg.get("easy_steer", None)
                            user_repeng = user_cfg.get("repeng", None)
                            user_anthropic = user_cfg.get("anthropic", None)
                            user_gen = user_cfg.get("gen", None)
                            user_custom_instruction = user_cfg.get("customInstruction", None)
                            user_system_prompt = user_cfg.get("systemPrompt", None)

                        if not user_provider:
                            raise ValueError("user.provider must be set in the TOD config.")

                        context_text = NlgPrompt().get_prompt_simulator_v2(
                            item.domains,
                            item.next_api_call,
                            item.next_user_req_slots,
                            item.last_api_call,
                            item.last_user_req_slots,
                            item.context
                        )
                        if user_custom_instruction:
                            context_text = user_custom_instruction + "\n\n" + context_text
                        gen_txt = generate_with_provider(
                            context_text,
                            provider=user_provider,
                            model_name=user_model_name or sys_model_name,
                            system_prompt=user_system_prompt,
                            gen=user_gen,
                            hf_local=user_hf_local,
                            easy_steer=user_easy_steer,
                            repeng=user_repeng,
                            anthropic=user_anthropic,
                            stop_markers=["<|endofdialog|>", "endofdialog"],
                        )

                        # Defensive: strip any stray end-of-dialog marker the user model may still emit.
                        # Ending is now the system's responsibility (via the end_dialog flag on its message JSON).
                        cleaned_gen_txt = re.sub(r"<?\|?\s*end\s*of\s*dialog\s*\|?>?", "", gen_txt, flags=re.IGNORECASE).strip()
                        if not cleaned_gen_txt:
                            cleaned_gen_txt = "Thanks, that's all."

                        dialog_info['log_user_context'].append(f"User: {cleaned_gen_txt}")
                        dialog_info['log_system_context'].append(f"User: {cleaned_gen_txt}")

                        # Fix 5: hard cap on wrap-up turns after API calls are exhausted.
                        # Once next_api_call is N/A, allow at most a couple more user turns
                        # before forcing termination (system tends to loop here).
                        if next_api_call is None:
                            dialog_info['wrap_up_turns'] = dialog_info.get('wrap_up_turns', 0) + 1
                            if dialog_info['wrap_up_turns'] >= 2:
                                progress_bar.update(1)
                                test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                                break
                    else:
                    
                        chat_session.interactive_chat(dialog_info['log_system_context'])
                        test_dataset[dialog_id]["sa_coeffs"] = getattr(chat_session, 'sa_coeffs', [])
                        test_dataset[dialog_id]["baseline_counterfactuals"] = getattr(chat_session, 'baseline_counterfactuals', [])
                        # interactive_chat_with_llm_user(dialog_info['log_system_context'], dialog_info['domains_original'])
                        
                        # always add the generated text to the system context
                        last_system_turn = dialog_info['log_system_context'][-1]
                        
                        if last_system_turn == "<Failed Dialog>":
                            progress_bar.update(1)
                            
                            # save the predicted api calls to the test_dataset
                            if 'predicted_api_calls' not in test_dataset[dialog_id]:
                                test_dataset[dialog_id]['predicted_api_calls'] = []
                                
                            if 'predicted_search_results' not in test_dataset[dialog_id]:
                                test_dataset[dialog_id]['predicted_search_results'] = []
                            
                            test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                            # Finish the current dialog
                            break
                        
                        # Fix 4: repetition / loop detection across the recent System turns.
                        # If the model keeps re-emitting near-duplicate messages, abort the dialog.
                        recent_system_turns = [
                            t for t in dialog_info['log_system_context']
                            if isinstance(t, str) and t.startswith("System:")
                        ]
                        if len(recent_system_turns) >= 3:
                            a, b, c = recent_system_turns[-3:]
                            if fuzz.ratio(a, b) >= 90 and fuzz.ratio(b, c) >= 90:
                                progress_bar.update(1)
                                if 'predicted_api_calls' not in test_dataset[dialog_id]:
                                    test_dataset[dialog_id]['predicted_api_calls'] = []
                                if 'predicted_search_results' not in test_dataset[dialog_id]:
                                    test_dataset[dialog_id]['predicted_search_results'] = []
                                dialog_info['log_system_context'].append("<Looping Dialog>")
                                test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                                break

                        if last_system_turn.startswith("Search Results\n"):
                                        
                            # save the predicted api calls to the test_dataset
                            if 'predicted_api_calls' not in test_dataset[dialog_id]:
                                test_dataset[dialog_id]['predicted_api_calls'] = []
                            test_dataset[dialog_id]["predicted_api_calls"].append(self.extract_api_call(dialog_info['log_system_context'][-2]))
                            
                            if 'predicted_search_results' not in test_dataset[dialog_id]:
                                test_dataset[dialog_id]['predicted_search_results'] = []
                            test_dataset[dialog_id]["predicted_search_results"].append(last_system_turn)
                                        
                            # end the dialog when the number of generated api calls gets larger than the actual numbers of the generated api calls
                            if dialog_info['api_tracker'] >= len(dialog_info['api_calls']):
                                progress_bar.update(1)
                                test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                                # Finish the current dialog
                                break
                                        
                            # add search results to the system context
                            #if not search_results:
                            #    raise ValueError("search results are needed")
                            
                            dialog_info['api_tracker'] += 1
                        else:
                            # change the turn_tracker to user
                            dialog_info["turn_tracker"] = "user"

                            # add the generated text to the user context if the generated text is not an api call
                            # it already has "System:" at the beginning
                            dialog_info['log_user_context'].append(last_system_turn)

                            # System signaled end_dialog on its final message — close the dialog.
                            if getattr(chat_session, 'dialog_ended', False):
                                progress_bar.update(1)
                                if 'predicted_api_calls' not in test_dataset[dialog_id]:
                                    test_dataset[dialog_id]['predicted_api_calls'] = []
                                if 'predicted_search_results' not in test_dataset[dialog_id]:
                                    test_dataset[dialog_id]['predicted_search_results'] = []
                                test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                                test_dataset[dialog_id]["dialog_completed"] = True
                                break

                    # specifying the maximum number of turns
                    if dialog_info['turn_id'] >= self.data_cfg.max_turns:
                        progress_bar.update(1)
                                
                        # save the predicted api calls to the test_dataset
                        if 'predicted_api_calls' not in test_dataset[dialog_id]:
                            test_dataset[dialog_id]['predicted_api_calls'] = []
                            
                        if 'predicted_search_results' not in test_dataset[dialog_id]:
                            test_dataset[dialog_id]['predicted_search_results'] = []
                        
                        test_dataset[dialog_id]["generated_dialog"] = dialog_info['log_system_context']
                        
                        break
                
                
            progress_bar.close()

            # save dialogs
            for dialog_id, dialog_info in tqdm(test_dataset.items()):
                if "generated_dialog" in dialog_info:
                    if dialog_info["generated_dialog"] == "Ignoring the dialog":
                        continue
                    self.save_dialog_data(dialog_info, dialog_id, output_dir)

            logger.info("Saved the generated dialogs.")

        _write_run_metadata(output_dir_path, self.run_id, self.cfg.project_root, cfg_container, "inference_finished")


@hydra.main(config_path="../../config/hydra/tod_runs", config_name="tod_inference", version_base="1.1")
def hydra_start(cfg: DictConfig) -> None:
    # Enable small test subset by capping dialogues per test-domain bucket.
    applied = _apply_env_overrides(cfg)
    if applied:
        logger.info("Applied env overrides:")
        for item in applied:
            logger.info("  - %s", item)

    logger.info("Fetched config yaml")
    if hasattr(cfg, "test_mode") and cfg.test_mode:
        if (
            hasattr(cfg, "test_num_dialogues_per_domain")
            and cfg.test_num_dialogues_per_domain is not None
        ):
            cfg.num_dialogues_per_domain = cfg.test_num_dialogues_per_domain
        else:
            cfg.num_dialogues_per_domain = 10

    # Snapshot the composed configuration after runtime overrides (env + test mode)
    original_cfg_snapshot = OmegaConf.to_container(cfg, resolve=True)
    logger.info("Config: %s", original_cfg_snapshot)
    t5tod = T5Tod(cfg, original_cfg_snapshot=original_cfg_snapshot)
    t5tod.run()
    # sys.stdout.close()


if __name__ == "__main__":
    hydra_start()
