"""Answer generation layer (provider-independent).

Status: evidence assembly, prompts and the grounding checker are implemented.
The LLM call itself is NOT implemented yet; a free OpenRouter model will be
plugged in through ``llm_client.LLMClient``.
"""
