import os
import sys
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import time
import requests
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import Dict, Tuple, Any, List
from utilities.runtime_config import persona_repo_root, require_env

load_dotenv()

logger = logging.getLogger(__name__)

_anth_load_hf_model = None
_anth_load_vllm_model = None
_AnthActivationSteerer = None

PERSONA_BIG5_POLES = {
   "agreeableness": {"+": "compassionate", "-": "self-interested"},
   "conscientiousness": {"+": "dependable", "-": "careless"},
   "extraversion": {"+": "outgoing", "-": "solitary"},
   "neuroticism": {"+": "nervous", "-": "calm"},
   "openness": {"+": "inventive", "-": "consistent"},
}


def persona_big5_pole_file(trait: str, coefficient: float) -> str:
   trait_key = trait.strip().lower()
   if trait_key not in PERSONA_BIG5_POLES:
      raise ValueError(f"Unknown Big Five trait: {trait}")
   sign = "+" if coefficient >= 0 else "-"
   pole = PERSONA_BIG5_POLES[trait_key][sign]
   return f"{pole}_response_avg_diff.pt"


def _add_persona_utility_paths():
   this_file = Path(__file__).resolve()
   project_root = this_file.parents[1]
   repo_parent = project_root.parent
   persona_repo = persona_repo_root()
   anth_repo = repo_parent / "anthropic_persona_vectors"

   for p in (
      persona_repo,
      persona_repo / "eval",
      repo_parent,
      anth_repo,
      anth_repo / "eval",
   ):
      if p.is_dir() and str(p) not in sys.path:
         sys.path.insert(0, str(p))


def _try_import_anthropic_utils():
   global _anth_load_hf_model, _anth_load_vllm_model, _AnthActivationSteerer
   # 1) Prefer PERSONA utilities from the sibling repo.
   try:
      from eval.model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
      _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
   except Exception:
      try:
         from model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
         _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
      except Exception:
         # 2) Prefer vendored copies within this repo (explicit relative import)
         try:
            from .anth_model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
            _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
         except Exception:
            logger.info('COULD NOT IMPORT FROM COPY')
            # Fallback to absolute within this repo (if imported as top-level)
            try:
               from llm_interaction.anth_model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
               _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
            except Exception:
               # 3) Legacy package-style imports (external)
               try:
                  from persona_vectors.anthropic_persona_vectors.eval.model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
                  _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
               except Exception:
                  try:
                     from anthropic_persona_vectors.eval.model_utils import load_model as lm_hf, load_vllm_model as lm_vllm  # type: ignore
                     _anth_load_hf_model, _anth_load_vllm_model = lm_hf, lm_vllm
                  except Exception:
                     _anth_load_hf_model, _anth_load_vllm_model = None, None
   # Activation steerer - prefer PERSONA, then vendored copies.
   try:
      from activation_steer import ActivationSteerer as AS  # type: ignore
      _AnthActivationSteerer = AS
   except Exception:
      try:
         from .activation_steer import ActivationSteerer as AS  # type: ignore
         _AnthActivationSteerer = AS
      except Exception:
         try:
            from llm_interaction.activation_steer import ActivationSteerer as AS  # type: ignore
            _AnthActivationSteerer = AS
         except Exception:
            try:
               from persona_vectors.anthropic_persona_vectors.eval.activation_steer import ActivationSteerer as AS  # type: ignore
               _AnthActivationSteerer = AS
            except Exception:
               try:
                  from persona_vectors.anthropic_persona_vectors.activation_steer import ActivationSteerer as AS  # type: ignore
                  _AnthActivationSteerer = AS
               except Exception:
                  _AnthActivationSteerer = None

_add_persona_utility_paths()
_try_import_anthropic_utils()

# If not found, refresh external utility paths and retry.
if _anth_load_hf_model is None or _AnthActivationSteerer is None:
   try:
      _add_persona_utility_paths()
      _try_import_anthropic_utils()
   except Exception:
      # Ignore; we'll error later if needed
      pass



_HF_CACHE = {}
_VLLM_CACHE: Dict[Tuple[str, int], Any] = {}
_STEER_REQ_CACHE: Dict[Tuple[str, float, Tuple[int, ...]], Any] = {}
_ANTH_HF_CACHE: Dict[Tuple[str], Tuple[Any, Any]] = {}
_ANTH_VLLM_CACHE: Dict[Tuple[str], Tuple[Any, Any, Any]] = {}

# Default stop markers to prevent system model from generating user turns or fake tool output
DEFAULT_STOP_MARKERS = ["\nUser:", "\nSystem:", "\nSearch Results", "End Search Results"]


class RoleStoppingCriteria(StoppingCriteria):
    """
    Stops generation when any of the specified text markers appear in the decoded output.
    This prevents the model from continuing the transcript with User: turns or fake tool output.
    """
    def __init__(self, tokenizer, stop_markers: List[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_markers = stop_markers
        self.prompt_length = prompt_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only decode the generated portion (after prompt)
        if input_ids.shape[1] <= self.prompt_length:
            return False
        
        generated_ids = input_ids[0, self.prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        for marker in self.stop_markers:
            if marker in generated_text:
                return True
        return False

def hf_local_completion(
   prompt: str,
   model_name: str,
   system_prompt: str | None = None,
   max_new_tokens: int = 256,
   temperature: float = 0.0,
   top_p: float = 0.9,
   device_map: str = "auto",
   dtype: str = "bfloat16",
   load_in_4bit: bool = True,
   trust_remote_code: bool = True,
   stop_markers: List[str] = None,
):
   """
   Run local generation using a Hugging Face causal LM (e.g., Qwen/Qwen2.5-7B-Instruct).
   Caches tokenizer/model across calls. Supports 4-bit loading when bitsandbytes is available.
   
   Args:
       stop_markers: List of text patterns to stop generation at. If None, uses
                     DEFAULT_STOP_MARKERS to prevent role/tool hallucination.
   """
   if stop_markers is None:
      stop_markers = DEFAULT_STOP_MARKERS
   cache_key = (model_name, device_map, dtype, load_in_4bit, trust_remote_code)
   if cache_key not in _HF_CACHE:
      # Load tokenizer
      tokenizer = AutoTokenizer.from_pretrained(
         model_name,
         trust_remote_code=trust_remote_code,
         use_fast=False,
      )
      # Prepare model kwargs
      model_kwargs = {
         "device_map": device_map,
         "trust_remote_code": trust_remote_code,
      }
      if load_in_4bit:
         model_kwargs["load_in_4bit"] = True
      else:
         dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
         }
         model_kwargs["torch_dtype"] = dtype_map.get(dtype, torch.bfloat16)
      # Load model with graceful fallback if bitsandbytes isn't available
      try:
         model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
      except Exception as e:
         if load_in_4bit and "bitsandbytes" in str(e).lower():
            # Fallback to non-4bit load
            dtype_map = {
               "bfloat16": torch.bfloat16,
               "float16": torch.float16,
               "float32": torch.float32,
            }
            model = AutoModelForCausalLM.from_pretrained(
               model_name,
               device_map=device_map,
               trust_remote_code=trust_remote_code,
               torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            )
         else:
            raise
      # Ensure a pad token exists
      if tokenizer.pad_token_id is None:
         try:
            tokenizer.pad_token = tokenizer.eos_token
         except Exception:
            pass
      _HF_CACHE[cache_key] = (tokenizer, model)

   tokenizer, model = _HF_CACHE[cache_key]
   # Build input ids using chat template when available (Qwen supports this)
   if hasattr(tokenizer, "apply_chat_template"):
      messages = []
      if system_prompt:
         messages.append({"role": "system", "content": system_prompt})
      messages.append({"role": "user", "content": prompt})
      input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
   else:
      if system_prompt:
         prompt = f"{system_prompt}\n\n{prompt}"
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids

   device = getattr(model, "device", None)
   if device is None or (isinstance(device, torch.device) and device.type == "cpu" and torch.cuda.is_available()):
      # Move inputs to the first available device of the model if necessary
      try:
         device = next(model.parameters()).device
      except StopIteration:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   input_ids = input_ids.to(device)

   gen_kwargs = {
      "max_new_tokens": max_new_tokens,
      "do_sample": temperature > 0.0,
      "temperature": temperature,
      "top_p": top_p,
      "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
   }
   
   # Add stopping criteria to prevent role/tool hallucination
   if stop_markers:
      stopping_criteria = StoppingCriteriaList([
         RoleStoppingCriteria(tokenizer, stop_markers, input_ids.shape[-1])
      ])
      gen_kwargs["stopping_criteria"] = stopping_criteria
   
   with torch.no_grad():
      output = model.generate(input_ids, **gen_kwargs)
   # Only decode the generated continuation
   gen_tokens = output[0][input_ids.shape[-1]:]
   text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
   
   # Post-process: strip any text after stop markers (in case stopping was delayed)
   for marker in (stop_markers or []):
      if marker in text:
         text = text.split(marker)[0]
   
   return text.strip()

def vllm_steer_completion(
   prompt: str,
   model_name: str,
   steer_vector_path: str,
   system_prompt: str | None = None,
   max_new_tokens: int = 256,
   temperature: float = 0.0,
   tensor_parallel_size: int = 1,
   target_layers: Any = None,
   scale: float = 2.0,
):
   """
   Run local generation using vLLM + EasySteer for steer vector control.
   Caches the vLLM LLM instance per (model_name, tensor_parallel_size)
   and the SteerVectorRequest per (steer_vector_path, scale, target_layers).
   """
   if not steer_vector_path or not os.path.exists(steer_vector_path):
      raise ValueError("easy_steer.steer_vector_path must point to an existing file")
   try:
      from vllm import LLM, SamplingParams
      from vllm.steer_vectors.request import SteerVectorRequest
   except Exception as e:
      raise RuntimeError(f"vLLM/EasySteer not available: {e}")

   # Default target layers if not provided
   if target_layers is None:
      target_layers = list(range(10, 26))
   target_layers_tuple = tuple(target_layers)

   cache_key_llm = (model_name, tensor_parallel_size)
   if cache_key_llm not in _VLLM_CACHE:
      llm = LLM(
         model=model_name,
         enable_steer_vector=True,
         enforce_eager=True,
         tensor_parallel_size=tensor_parallel_size,
         enable_chunked_prefill=False,
      )
      _VLLM_CACHE[cache_key_llm] = llm
   else:
      llm = _VLLM_CACHE[cache_key_llm]

   # If tokenizer supports chat templates, inject system_prompt as a system message
   if system_prompt:
      try:
         tok = llm.get_tokenizer()
         if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            try:
               prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except TypeError:
               # Older tokenizers may not support tokenize=False
               ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
               prompt = tok.decode(ids[0], skip_special_tokens=False)
      except Exception:
         prompt = f"{system_prompt}\n\n{prompt}"

   cache_key_req = (steer_vector_path, float(scale), target_layers_tuple)
   if cache_key_req not in _STEER_REQ_CACHE:
      steer_req = SteerVectorRequest(
         "steer",
         1,
         steer_vector_local_path=steer_vector_path,
         scale=scale,
         target_layers=list(target_layers_tuple),
         prefill_trigger_tokens=[-1],
         generate_trigger_tokens=[-1],
      )
      _STEER_REQ_CACHE[cache_key_req] = steer_req
   else:
      steer_req = _STEER_REQ_CACHE[cache_key_req]

   sampling_params = SamplingParams(
      temperature=temperature,
      max_tokens=max_new_tokens,
   )
   outputs = llm.generate(prompt, steer_vector_request=steer_req, sampling_params=sampling_params)
   if not outputs or not outputs[0].outputs:
      return ""
   return (outputs[0].outputs[0].text or "").strip()

def vllm_steer_completion_anthropic(
   prompt: str,
   model_name: str,
   *,
   vector_path: str,
   layer: int,
   coef: float,
   system_prompt: str | None = None,
   max_new_tokens: int = 256,
   temperature: float = 0.0,
):
   """
   Generate with an Anthropic-style steering vector.
   - coef == 0: run the same HF model unsteered.
   - coef != 0: apply steering via HF ActivationSteerer (mirrors eval/eval_persona.py).
   """
   if coef == 0:
      if _anth_load_hf_model is None:
         raise RuntimeError("Anthropic HF model utilities not available in import path")
      cache_key = (model_name,)
      if cache_key not in _ANTH_HF_CACHE:
         hf_model, tokenizer = _anth_load_hf_model(model_name)
         _ANTH_HF_CACHE[cache_key] = (hf_model, tokenizer)
      else:
         hf_model, tokenizer = _ANTH_HF_CACHE[cache_key]
      if tokenizer.pad_token is None:
         try:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
         except Exception:
            pass
      if hasattr(tokenizer, "apply_chat_template"):
         messages = []
         if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
         messages.append({"role": "user", "content": prompt})
         input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
         )
      else:
         if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
      device = getattr(hf_model, "device", None)
      if device is None or (isinstance(device, torch.device) and device.type == "cpu" and torch.cuda.is_available()):
         try:
            device = next(hf_model.parameters()).device
         except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      input_ids = input_ids.to(device)
      gen_kwargs = {
         "max_new_tokens": max_new_tokens,
         "do_sample": temperature > 0.0,
         "temperature": temperature,
      }
      with torch.no_grad():
         output = hf_model.generate(input_ids, **gen_kwargs)
      gen_tokens = output[0][input_ids.shape[-1]:]
      text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
      return text.strip()

   if _anth_load_hf_model is None or _AnthActivationSteerer is None:
      raise RuntimeError("Anthropic HF steering utilities not available in import path")
   # Cache anthropic HF model/tokenizer across calls to avoid re-loading each turn
   cache_key = (model_name,)
   if cache_key not in _ANTH_HF_CACHE:
      hf_model, tokenizer = _anth_load_hf_model(model_name)
      _ANTH_HF_CACHE[cache_key] = (hf_model, tokenizer)
   else:
      hf_model, tokenizer = _ANTH_HF_CACHE[cache_key]
   if tokenizer.pad_token is None:
      try:
         tokenizer.pad_token = tokenizer.eos_token
         tokenizer.pad_token_id = tokenizer.eos_token_id
      except Exception:
         pass
   if hasattr(tokenizer, "apply_chat_template"):
      messages = []
      if system_prompt:
         messages.append({"role": "system", "content": system_prompt})
      messages.append({"role": "user", "content": prompt})
      input_ids = tokenizer.apply_chat_template(
         messages,
         add_generation_prompt=True,
         return_tensors="pt",
      )
   else:
      if system_prompt:
         prompt = f"{system_prompt}\n\n{prompt}"
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids
   device = getattr(hf_model, "device", None)
   if device is None or (isinstance(device, torch.device) and device.type == "cpu" and torch.cuda.is_available()):
      try:
         device = next(hf_model.parameters()).device
      except StopIteration:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   input_ids = input_ids.to(device)
   vec_obj = torch.load(vector_path, weights_only=False)
   try:
      steer_vec = vec_obj[layer]
   except Exception:
      raise ValueError(f"Vector file does not contain layer {layer}")
   gen_kwargs = {
      "max_new_tokens": max_new_tokens,
      "do_sample": temperature > 0.0,
      "temperature": temperature,
   }
   # Convention matches upstream persona_vectors:
   # - Saved vector tensor has shape [n_layers+1, hidden_dim] with an almost-zero row 0.
   # - `layer` is treated as 1-based transformer block index.
   #   We therefore take `vec_obj[layer]` and hook `model.layers[layer-1]`.
   with _AnthActivationSteerer(hf_model, steer_vec, coeff=coef, layer_idx=layer - 1, positions="response"):
      with torch.no_grad():
         output = hf_model.generate(input_ids, **gen_kwargs)
   gen_tokens = output[0][input_ids.shape[-1]:]
   text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
   return text.strip()

def repeng_steer_completion(
   prompt: str,
   model_name: str,
   *,
   system_prompt: str | None = None,
   dataset_jsonl_path: str | None = None,
   vector_gguf_path: str | None = None,
   strength: float = 2.0,
   target_layers: Any = None,
   max_new_tokens: int = 256,
   temperature: float = 0.0,
   device_map: str = "auto",
   dtype: str = "bfloat16",
   load_in_4bit: bool = True,
   trust_remote_code: bool = True,
):
   """
   Run local generation using Hugging Face + RepEng control vectors.
   Trains a control vector from a JSONL dataset of {positive, negative} pairs,
   then applies it at inference with the given strength.
   """
   # Fast-path: if a GGUF vector is provided, delegate to vLLM/EasySteer which natively supports GGUF
   if vector_gguf_path:
      if not os.path.exists(vector_gguf_path):
         raise ValueError(f"repeng.vector_gguf_path not found: {vector_gguf_path}")
      return vllm_steer_completion(
         prompt=prompt,
         model_name=model_name,
         system_prompt=system_prompt,
         steer_vector_path=vector_gguf_path,
         max_new_tokens=max_new_tokens,
         temperature=temperature,
         tensor_parallel_size=1,
         target_layers=target_layers if target_layers is not None else list(range(10, 26)),
         scale=strength,
      )

   try:
      from repeng import ControlVector, ControlModel, DatasetEntry
   except Exception as e:
      raise RuntimeError(f"RepEng library not available: {e}")

   # Reuse HF tokenizer/model cache, mirroring hf_local_completion
   cache_key = (model_name, device_map, dtype, load_in_4bit, trust_remote_code)
   if cache_key not in _HF_CACHE:
      # Load tokenizer
      tokenizer = AutoTokenizer.from_pretrained(
         model_name,
         trust_remote_code=trust_remote_code,
         use_fast=False,
      )
      # Prepare model kwargs
      model_kwargs = {
         "device_map": device_map,
         "trust_remote_code": trust_remote_code,
      }
      if load_in_4bit:
         model_kwargs["load_in_4bit"] = True
      else:
         dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
         }
         model_kwargs["torch_dtype"] = dtype_map.get(dtype, torch.bfloat16)
      # Load model with graceful fallback if bitsandbytes isn't available
      try:
         model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
      except Exception as e:
         if load_in_4bit and "bitsandbytes" in str(e).lower():
            dtype_map = {
               "bfloat16": torch.bfloat16,
               "float16": torch.float16,
               "float32": torch.float32,
            }
            model = AutoModelForCausalLM.from_pretrained(
               model_name,
               device_map=device_map,
               trust_remote_code=trust_remote_code,
               torch_dtype=dtype_map.get(dtype, torch.bfloat16),
            )
         else:
            raise
      if tokenizer.pad_token_id is None:
         try:
            tokenizer.pad_token = tokenizer.eos_token
         except Exception:
            pass
      _HF_CACHE[cache_key] = (tokenizer, model)

   tokenizer, base_model = _HF_CACHE[cache_key]

   # Select target layers if none provided (repeng examples use negative indices for last blocks)
   if target_layers is None:
      target_layers = list(range(-5, -18, -1))

   # Wrap the model with RepEng control
   repeng_model = ControlModel(base_model, target_layers)

   # Build dataset entries from JSONL if provided
   dataset_entries = []
   if dataset_jsonl_path:
      if not os.path.exists(dataset_jsonl_path):
         raise ValueError(f"repeng.dataset_jsonl_path not found: {dataset_jsonl_path}")
      with open(dataset_jsonl_path, "r", encoding="utf-8") as f:
         for line in f:
            line = line.strip()
            if not line:
               continue
            try:
               obj = json.loads(line)
               pos = obj.get("positive", "")
               neg = obj.get("negative", "")
               if pos and neg:
                  dataset_entries.append(DatasetEntry(positive=pos, negative=neg))
            except Exception:
               continue
   else:
      raise ValueError("repeng.dataset_jsonl_path is required for local_steer_repeng when vector_gguf_path is not provided")

   if not dataset_entries:
      raise ValueError("No valid entries found in dataset_jsonl_path for RepEng training")

   # Train control vector
   vector = ControlVector.train(repeng_model, tokenizer, dataset_entries)

   # Apply control
   repeng_model.set_control(vector, strength)

   # Prepare inputs (use chat template when available)
   if hasattr(tokenizer, "apply_chat_template"):
      messages = []
      if system_prompt:
         messages.append({"role": "system", "content": system_prompt})
      messages.append({"role": "user", "content": prompt})
      input_ids = tokenizer.apply_chat_template(
         messages,
         add_generation_prompt=True,
         return_tensors="pt",
      )
   else:
      if system_prompt:
         prompt = f"{system_prompt}\n\n{prompt}"
      input_ids = tokenizer(prompt, return_tensors="pt").input_ids

   # Ensure device match
   device = getattr(base_model, "device", None)
   if device is None or (isinstance(device, torch.device) and device.type == "cpu" and torch.cuda.is_available()):
      try:
         device = next(base_model.parameters()).device
      except StopIteration:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   input_ids = input_ids.to(device)

   gen_kwargs = {
      "max_new_tokens": max_new_tokens,
      "do_sample": temperature > 0.0,
      "temperature": temperature,
      "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
   }
   with torch.no_grad():
      output = repeng_model.generate(input_ids, **gen_kwargs)
   gen_tokens = output[0][input_ids.shape[-1]:]
   text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
   return text.strip()

def openai_chatgpt_models(prompt, model_name, system_prompt: str | None = None):
   api_key = require_env("OPENAI_API_KEY", "OpenAI ChatGPT provider calls")
   client = OpenAI(
         api_key=api_key
         )
   try:
         messages = []
         if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
         messages.append({"role": "user", "content": prompt})
         response = client.chat.completions.create(
               model= model_name,  # Corrected model name
               messages=messages,
               max_tokens=250,
               n=1,
               stop=None,
               temperature=0.0,  # Adjust as needed
         )
         return response.choices[0].message.content
   except Exception as e:
         raise Exception(f"Error while calling OpenAI API: {e}")

def open_route_ai_models(prompt, model_name, max_retries=10, max_wait=128, system_prompt: str | None = None):
   openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
   if not openrouter_api_key:
      raise ValueError("OPENROUTER_API_KEY is not set. Add it to your environment or .env file.")
   client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key=openrouter_api_key,
   )
    
   wait_time = 2  # initial wait time in seconds
   
   for attempt in range(max_retries):
      try:
         messages = []
         if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
         messages.append({"role": "user", "content": prompt})
         response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
               "Authorization": f"Bearer {openrouter_api_key}",
            },
            data=json.dumps({
               "model": model_name, # Optional
               "messages": messages,
               "provider": {
                  "order": [
                  "Lepton",
                  "DeepInfra",
                  ]
               },
               "max_tokens":250,
               "stop":None,
               "temperature":0.0,  # Adjust as needed
            })
         )
         
         # Raise for HTTP errors before parsing JSON
         response.raise_for_status()
         
         # Parse the JSON response
         response_data = response.json()
         # Extract the relevant part of the response
         completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
         
         # Check if response is valid and not empty
         if not completion:
            raise ValueError("Received empty response or no choices.")
            
         # If everything is fine, return the content
         return completion
        
      except Exception as e:
         # If we haven't reached the last retry, wait and then retry
         if attempt < max_retries - 1:
            logger.info(f"Attempt {attempt + 1} failed: {e}\nRetrying in {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, max_wait)
         else:
            # Raise exception if all retries have failed
            raise Exception(f"Error after {max_retries} attempts: {e}")

_PERSONA_FLOW_POLE_CACHE = {}

def sa_steer_completion(
    prompt: str,
    model_name: str,
    *,
    vector_dir: str,
    layer: int,
    system_prompt: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    judge_context: str | None = None,
):
    logger.info(f"[DEBUG] sa_steer_completion: prompt_len={len(prompt)} layer={layer}")
    import re

    JUDGE_PROMPT_TEMPLATE = """The AI has these default traits:
• Agreeableness: High (cooperative, trusting, helpful)
• Conscientiousness: High (organized, reliable, disciplined)
• Extraversion: Moderate (balanced social energy)
• Openness: Moderate (balanced creativity and practicality)
• Neuroticism: Low (generally calm and stable)

Context to Analyze:
• Persona Context: You are a helpful AI assistant.
• Current Input: {current_input}

Your Task:
Determine which traits need adjustment (-2.0 to +2.0) based on what would be most effective for this specific interaction. Consider both directions equally and choose based on situational demands.

Trait Adjustment Guidelines:
Extraversion:
• Increase (+) for: Group activities, public speaking, networking, team leadership
• Decrease (-) for: Individual work, quiet reflection, solo creative tasks
Agreeableness:
• Increase (+) for: Conflict resolution, team building, emotional support
• Decrease (-) for: Critical feedback, boundary setting, competitive situations
Conscientiousness:
• Increase (+) for: Detailed planning, precision work, deadline management
• Decrease (-) for: Spontaneous responses, creative brainstorming, crisis situations
Neuroticism:
• Increase (+) for: Appropriate caution, emotional sensitivity, risk awareness
• Decrease (-) for: Calm leadership, confident decisions, crisis management
Openness:
• Increase (+) for: Creative problem-solving, exploring new ideas, innovation
• Decrease (-) for: Following procedures, traditional approaches, proven solutions

Decision Principles:
• Situational Fit: Choose traits that best serve the interaction goals
• Context Sensitivity: Consider what the human needs from this specific interaction
• Balanced Assessment: Evaluate both positive and negative adjustments equally
• Natural Baseline: Use 0.0 when baseline personality already fits the situation well

Output Format:
Provide only the numerical adjustment scores:
Extraversion: [score]
Agreeableness: [score]
Conscientiousness: [score]
Neuroticism: [score]
Openness: [score]"""

    # 1. Predict Coefficients using OpenAI
    # Use judge_context (recent turns) when available; fall back to tail of full prompt.
    # Sending the full prompt (~8KB system instruction + history) causes the judge to return
    # 0.0 for every trait most of the time, defeating the purpose of sa steering.
    if judge_context:
        judge_input = judge_context
    else:
        # Extract last few User:/System: turns from the full prompt as a best-effort fallback
        turn_lines = [l.strip() for l in prompt.split('\n') if l.strip().startswith(('User:', 'System:'))]
        judge_input = '\n'.join(turn_lines[-6:]) if turn_lines else prompt[-800:]
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    if openrouter_key:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
        judge_model = "openai/gpt-4o-mini"
    else:
        client = OpenAI(api_key=require_env("OPENAI_API_KEY", "sa coefficient judging"))
        judge_model = "gpt-4o-mini"
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": JUDGE_PROMPT_TEMPLATE.format(current_input=judge_input)}],
            temperature=0.0
        )
        text_out = response.choices[0].message.content
        logger.info(f"[DEBUG] Judge raw output:\n{text_out.strip()}")
    except Exception as e:
        logger.info(f"Failed to fetch coefficients from OpenAI: {e}")
        text_out = ""

    coeffs = {}
    for line in text_out.strip().split('\n'):
        line = line.strip()
        if not line: continue
        for trait in ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]:
            if trait in line and ':' in line:
                try:
                    val = float(line.split(':')[-1].strip().strip('[]'))
                    coeffs[trait] = val
                except ValueError:
                    pass
    
    for trait in ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]:
        if trait not in coeffs:
            coeffs[trait] = 0.0

    # 2. Filter, Gate, and Composition
    for t, v in coeffs.items():
        v_clamped = max(-2.0, min(2.0, v))
        if abs(v_clamped) < 0.5:
            logger.info(f"[DEBUG] Trait '{t}' gated to 0.0 (raw={v_clamped})")
            v_final = 0.0
        else:
            logger.info(f"[DEBUG] Trait '{t}' ACTIVE: {v_clamped}")
            v_final = v_clamped
        coeffs[t] = v_final
        
    global _PERSONA_FLOW_POLE_CACHE
    composite_vector = None
    
    for trait, coeff in coeffs.items():
        if coeff == 0.0:
            continue
        file_name = persona_big5_pole_file(trait, coeff)
        file_path = os.path.join(vector_dir, file_name)
        
        if file_path not in _PERSONA_FLOW_POLE_CACHE:
            try:
                vec_obj = torch.load(file_path, weights_only=False)
                _PERSONA_FLOW_POLE_CACHE[file_path] = vec_obj[layer]
            except Exception as e:
                logger.info(f"Warning: Failed to load {file_path}. Error: {e}")
                continue
                
        # Normalize and compute product
        vec = _PERSONA_FLOW_POLE_CACHE[file_path]
        vec_norm = vec / vec.norm(dim=-1, keepdim=True)
        scaled_vec = vec_norm * abs(coeff)
        
        if composite_vector is None:
            composite_vector = scaled_vec.clone()
        else:
            composite_vector += scaled_vec

    # 3. Model Generation
    if _anth_load_hf_model is None or _AnthActivationSteerer is None:
        raise RuntimeError("Anthropic HF steering utilities not available in import path")
        
    cache_key = (model_name,)
    if cache_key not in _ANTH_HF_CACHE:
        hf_model, tokenizer = _anth_load_hf_model(model_name)
        _ANTH_HF_CACHE[cache_key] = (hf_model, tokenizer)
    else:
        hf_model, tokenizer = _ANTH_HF_CACHE[cache_key]
        
    if tokenizer.pad_token is None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except AttributeError:
            pass
            
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
        
    device = getattr(hf_model, "device", None)
    if device is None or (isinstance(device, torch.device) and device.type == "cpu" and torch.cuda.is_available()):
        try:
            device = next(hf_model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    input_ids = input_ids.to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature,
    }
    
    if composite_vector is not None:
        composite_vector = composite_vector.to(device)
        logger.info(f"[DEBUG] ACTIVATING ActivationSteerer at layer {layer} with vector norm {composite_vector.norm().item()}")
        with _AnthActivationSteerer(hf_model, composite_vector, coeff=1.0, layer_idx=layer - 1, positions="response"):
            with torch.no_grad():
                output = hf_model.generate(input_ids, **gen_kwargs)
    else:
        logger.info("[DEBUG] No composite vector (all traits gated to 0 or missing). Running unsteered.")
        with torch.no_grad():
            output = hf_model.generate(input_ids, **gen_kwargs)
            
    gen_tokens = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text, coeffs
