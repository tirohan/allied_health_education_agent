import json
from typing import Any


async def call_openai_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 1200,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call OpenAI chat completions expecting a strict JSON object back.

    Shared by every direct-LLM call site (extraction, critic, chat) so the
    client construction and retry policy live in exactly one place. Exceptions
    are never caught or re-wrapped here -- callers (e.g. extraction_node's
    `exc.__class__.__name__` trace label) depend on the original exception
    type surfacing unchanged.
    """
    from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
    from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

    client = AsyncOpenAI(api_key=api_key, timeout=timeout)
    retryer = AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        reraise=True,
    )
    async for attempt in retryer:
        with attempt:
            response = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)
