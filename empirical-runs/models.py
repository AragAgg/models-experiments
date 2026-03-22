"""
Model abstraction layer for multi-vendor LLM support.
Supports: OpenAI, Anthropic, Google Gemini

Usage:
    from models import call_model
    
    response = call_model(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        system_prompt="You are helpful.",
        api_keys={"openai": "sk-...", "anthropic": "sk-ant-...", "google": "..."},
        max_retries=3,
        enable_backoff=True
    )
"""

import time
import random
from typing import Optional


# Model to provider mapping (latest models as of Oct 2025)
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "o1": "openai",
    "o1-mini": "openai",
    
    # Anthropic models
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-opus-20240229": "anthropic",
    "claude-3-sonnet-20240229": "anthropic",
    "claude-3-haiku-20240307": "anthropic",
    
    # Google Gemini models
    "gemini-3-pro-preview": "google",
    "gemini-2.5-pro": "google",
    "gemini-2.5-flash": "google",
    "gemini-2.0-flash": "google",
    "gemini-1.5-pro": "google",
}


def get_provider(model: str) -> str:
    """Get provider for a model. Supports prefix matching for flexibility."""
    if model in MODEL_PROVIDERS:
        return MODEL_PROVIDERS[model]
    
    # Prefix matching for unlisted models
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"
    
    raise ValueError(f"Unknown model: {model}. Add it to MODEL_PROVIDERS or use a recognized prefix.")


def _call_openai(model: str, messages: list, system_prompt: str, api_key: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)
    
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
    )
    return response.choices[0].message.content


def _call_anthropic(model: str, messages: list, system_prompt: str, api_key: str) -> str:
    """Call Anthropic API."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt if system_prompt else "",
        messages=messages,
    )
    return response.content[0].text


def _call_google(model: str, messages: list, system_prompt: str, api_key: str) -> str:
    """Call Google Gemini API (using new google.genai package)."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    
    # Build contents list
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    
    # Config with system instruction
    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=1024),  # Low thinking for speed
    )
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    
    return response.text


def call_model(
    model: str,
    messages: list,
    system_prompt: str,
    api_keys: dict,
    max_retries: int = 3,
    enable_backoff: bool = True,
    base_delay: float = 1.0,
) -> tuple[str, Optional[str]]:
    """
    Call an LLM model with retry logic and exponential backoff.
    
    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-latest")
        messages: List of message dicts [{"role": "user/assistant", "content": "..."}]
        system_prompt: System prompt string
        api_keys: Dict with keys "openai", "anthropic", "google"
        max_retries: Maximum retry attempts
        enable_backoff: Enable exponential backoff
        base_delay: Base delay in seconds for backoff
        
    Returns:
        Tuple of (response_text, error_message)
        - On success: (response_text, None)
        - On failure: (None, error_message)
    """
    provider = get_provider(model)
    api_key = api_keys.get(provider)
    
    if not api_key:
        return None, f"No API key provided for provider: {provider}"
    
    call_fn = {
        "openai": _call_openai,
        "anthropic": _call_anthropic,
        "google": _call_google,
    }[provider]
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = call_fn(model, messages, system_prompt, api_key)
            return response, None
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                if enable_backoff:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                else:
                    time.sleep(base_delay)
    
    return None, f"Failed after {max_retries} attempts. Last error: {last_error}"
