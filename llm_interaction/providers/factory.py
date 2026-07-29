from typing import Optional, Dict, Any, List, Union

from llm_interaction.llm_models_api import (
    openai_chatgpt_models,
    open_route_ai_models,
    hf_local_completion,
    vllm_steer_completion,
    vllm_steer_completion_anthropic,
    repeng_steer_completion,
    sa_steer_completion,
)


def generate_with_provider(
    prompt: str,
    provider: str,
    model_name: str,
    system_prompt: Optional[str] = None,
    *,
    gen: Optional[Dict[str, Any]] = None,
    hf_local: Optional[Dict[str, Any]] = None,
    easy_steer: Optional[Dict[str, Any]] = None,
    repeng: Optional[Dict[str, Any]] = None,
    anthropic: Optional[Dict[str, Any]] = None,
    stop_markers: Optional[List[str]] = None,
    judge_context: Optional[str] = None,
) -> Union[str, tuple[str, dict]]:
    provider = (provider or "").lower()
    gen = gen or {}
    hf_local = hf_local or {}
    easy_steer = easy_steer or {}
    repeng = repeng or {}
    anthropic = anthropic or {}

    if provider == "stub":
        return gen.get("response", '{"type": "message", "content": "stub response"}')
    if provider == "openai":
        return openai_chatgpt_models(prompt, model_name, system_prompt=system_prompt)
    if provider in ("openroute", "open_router", "open_route"):
        return open_route_ai_models(prompt, model_name, system_prompt=system_prompt)
    if provider in ("local", "local_finetuned"):
        return hf_local_completion(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            max_new_tokens=gen.get("max_new_tokens", hf_local.get("max_new_tokens", 256)),
            temperature=gen.get("temperature", hf_local.get("temperature", 0.0)),
            top_p=gen.get("top_p", hf_local.get("top_p", 0.9)),
            device_map=hf_local.get("device_map", "auto"),
            dtype=hf_local.get("dtype", "bfloat16"),
            load_in_4bit=hf_local.get("load_in_4bit", True),
            trust_remote_code=hf_local.get("trust_remote_code", True),
            stop_markers=stop_markers,
        )
    if provider == "local_steer":
        steer_path = easy_steer.get("steer_vector_path")
        if not steer_path:
            raise ValueError("easy_steer.steer_vector_path must be provided for provider=local_steer")
        return vllm_steer_completion(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            steer_vector_path=steer_path,
            max_new_tokens=gen.get("max_new_tokens", easy_steer.get("max_new_tokens", 256)),
            temperature=gen.get("temperature", easy_steer.get("temperature", 0.0)),
            tensor_parallel_size=easy_steer.get("tensor_parallel_size", 1),
            target_layers=easy_steer.get("target_layers", list(range(10, 26))),
            scale=easy_steer.get("scale", 2.0),
        )
    if provider in ("local_steer_anthropic", "local_anthropic", "anthropic_steer"):
        vector_path = anthropic.get("vector_path")
        layer = anthropic.get("layer")
        coef = anthropic.get("coef")
        if not vector_path or layer is None or coef is None:
            raise ValueError("anthropic.vector_path, anthropic.layer and anthropic.coef are required for provider=local_steer_anthropic")
        return vllm_steer_completion_anthropic(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            vector_path=vector_path,
            layer=int(layer),
            coef=float(coef),
            max_new_tokens=gen.get("max_new_tokens", anthropic.get("max_new_tokens", 256)),
            temperature=gen.get("temperature", anthropic.get("temperature", 0.0)),
        )
    if provider in ("local_steer_repeng", "local_repeng"):
        dataset_path = repeng.get("dataset_jsonl_path")
        vector_gguf_path = repeng.get("vector_gguf_path")
        if not dataset_path and not vector_gguf_path:
            raise ValueError("Provide repeng.dataset_jsonl_path or repeng.vector_gguf_path for provider=local_steer_repeng")
        return repeng_steer_completion(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            dataset_jsonl_path=dataset_path,
            vector_gguf_path=vector_gguf_path,
            strength=repeng.get("strength", 2.0),
            target_layers=repeng.get("target_layers", list(range(-5, -18, -1))),
            max_new_tokens=gen.get("max_new_tokens", repeng.get("max_new_tokens", 256)),
            temperature=gen.get("temperature", repeng.get("temperature", 0.0)),
            device_map=repeng.get("device_map", "auto"),
            dtype=repeng.get("dtype", "bfloat16"),
            load_in_4bit=repeng.get("load_in_4bit", True),
            trust_remote_code=repeng.get("trust_remote_code", True),
        )
    if provider in ("sa_steer", "sa"):
        vector_dir = anthropic.get("vector_dir") or anthropic.get("vector_path")
        layer = anthropic.get("layer", 15)  # Default target layer
        if not vector_dir:
            raise ValueError("anthropic.vector_dir must be provided for provider=sa")
        return sa_steer_completion(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            vector_dir=vector_dir,
            layer=int(layer),
            max_new_tokens=gen.get("max_new_tokens", anthropic.get("max_new_tokens", 256)),
            temperature=gen.get("temperature", anthropic.get("temperature", 0.0)),
            judge_context=judge_context,
        )
    raise ValueError(f"Unknown provider: {provider}")


