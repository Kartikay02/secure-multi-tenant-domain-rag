"""Production-ready OpenAI-compatible hosted LLM provider adapter."""

import asyncio
import json
import random
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.exceptions import (
    GenerationAuthenticationError,
    GenerationError,
    GenerationRateLimitError,
    GenerationTimeoutError,
)
from app.core.logging import get_logger
from app.rag.generation.domain import (
    GeneratedResponse,
    GenerationParameters,
    StructuredGenerationPayload,
)
from app.rag.generation.interfaces import LLMProviderProtocol

logger = get_logger("app.rag.generation.openai")

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _mask_secret(secret: str) -> str:
    """Mask sensitive API key for safe logging."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:3]}...{secret[-4:]}"


class OpenAILLMProvider(LLMProviderProtocol):
    """Production HTTP adapter for OpenAI-compatible Chat Completion endpoints.

    Features:
    - Structured JSON outputs via response_format / JSON Schema
    - Robust Pydantic response validation
    - SSE streaming support
    - Safe bounded exponential backoff on transient errors (429, 5xx)
    - Request timeout management
    - Connection pooling and credential masking
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        max_backoff_seconds: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}")

        self._api_key = api_key.strip()
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds

        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=30),
        )

        logger.info(
            "Initialized OpenAILLMProvider",
            extra={
                "model": self.model,
                "api_base": self.api_base,
                "api_key_masked": _mask_secret(self._api_key),
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            },
        )

    def __repr__(self) -> str:
        return f"OpenAILLMProvider(model={self.model!r}, api_base={self.api_base!r})"

    async def close(self) -> None:
        """Close the HTTP client if not externally injected."""
        if not self._external_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAILLMProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
        structured_schema: type[BaseModel] | None = None,
    ) -> GeneratedResponse:
        """Execute chat completion with safe retry and Pydantic validation."""
        if not self._api_key:
            raise GenerationAuthenticationError(
                provider="openai",
                reason="API key is missing or empty. Set LLM_API_KEY or OPENAI_API_KEY.",
            )

        params = parameters or GenerationParameters()
        endpoint = f"{self.api_base}/chat/completions"

        target_schema = structured_schema or StructuredGenerationPayload

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
        }
        if params.stop:
            payload["stop"] = params.stop

        # Configure structured JSON output
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": target_schema.__name__,
                "schema": target_schema.model_json_schema(),
                "strict": True,
            },
        }

        start_time = time.perf_counter()
        attempt = 0

        while True:
            try:
                logger.debug(
                    f"Sending LLM chat completion attempt {attempt + 1}/{self.max_retries + 1}"
                )
                response = await self._client.post(
                    endpoint,
                    json=payload,
                    headers=self._build_headers(),
                )

                # 1. Non-retryable authentication failure
                if response.status_code in (401, 403):
                    logger.error(f"LLM authentication failed with status {response.status_code}")
                    raise GenerationAuthenticationError(
                        provider="openai",
                        reason=f"HTTP {response.status_code}: {response.text}",
                    )

                # 2. Transient status codes for retry
                if response.status_code in TRANSIENT_STATUS_CODES:
                    is_rate_limit = response.status_code == 429
                    if attempt >= self.max_retries:
                        if is_rate_limit:
                            raise GenerationRateLimitError(provider="openai")
                        raise GenerationError(
                            f"LLM provider error after {self.max_retries} retries: HTTP {response.status_code}",
                            details={"status_code": response.status_code, "model": self.model},
                        )

                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = min(self.max_backoff_seconds, float(retry_after))
                    else:
                        base = self.backoff_factor * (2**attempt)
                        jitter = random.uniform(0.0, 0.3)
                        delay = min(self.max_backoff_seconds, base + jitter)

                    logger.warning(
                        f"Transient HTTP {response.status_code} from LLM provider. Retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # 3. Non-transient client error
                if response.is_error:
                    logger.error(
                        f"LLM completion request failed with HTTP {response.status_code}: {response.text}"
                    )
                    raise GenerationError(
                        f"LLM API client error {response.status_code}: {response.text}",
                        details={"status_code": response.status_code, "model": self.model},
                    )

                # 4. Parse response JSON
                data = response.json()
                latency = time.perf_counter() - start_time
                choices = data.get("choices", [])
                if not choices:
                    raise GenerationError("LLM response contained empty choices array.")

                raw_content = choices[0].get("message", {}).get("content", "")

                # 5. Parse and validate structured output
                try:
                    parsed_payload = target_schema.model_validate_json(raw_content)
                except ValidationError:
                    # Fallback attempt: strip markdown json fences if model wrapped the JSON
                    cleaned = re.sub(r"^```json\s*", "", raw_content.strip())
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                    parsed_payload = target_schema.model_validate_json(cleaned)

                # Extract domain fields
                payload_dict = parsed_payload.model_dump()
                answer = str(payload_dict.get("answer", ""))
                citations = list(payload_dict.get("citations", []))
                confidence = float(payload_dict.get("confidence_score", 1.0))
                grounded = bool(payload_dict.get("grounded", True))
                insufficient = bool(payload_dict.get("insufficient_context", False))

                usage = data.get("usage", {})

                return GeneratedResponse(
                    answer=answer,
                    citations=citations,
                    confidence_score=confidence,
                    grounded=grounded,
                    insufficient_context=insufficient,
                    referenced_documents=[],
                    model=data.get("model", self.model),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    latency_seconds=round(latency, 4),
                    metadata={"raw_payload": payload_dict},
                )

            except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                is_timeout = isinstance(net_err, httpx.TimeoutException)
                if attempt >= self.max_retries:
                    if is_timeout:
                        raise GenerationTimeoutError(
                            provider="openai",
                            timeout_seconds=self.timeout_seconds,
                        ) from net_err
                    raise GenerationError(
                        f"Network error during LLM generation request: {net_err}"
                    ) from net_err

                base = self.backoff_factor * (2**attempt)
                jitter = random.uniform(0.0, 0.3)
                delay = min(self.max_backoff_seconds, base + jitter)

                logger.warning(
                    f"Network error calling LLM provider ({net_err}). Retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        parameters: GenerationParameters | None = None,
    ) -> AsyncIterator[str]:
        """Stream token deltas via SSE."""
        if not self._api_key:
            raise GenerationAuthenticationError(
                provider="openai",
                reason="API key is missing or empty for streaming.",
            )

        params = parameters or GenerationParameters()
        endpoint = f"{self.api_base}/chat/completions"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "top_p": params.top_p,
            "stream": True,
        }
        if params.stop:
            payload["stop"] = params.stop

        try:
            async with self._client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=self._build_headers(),
            ) as response:
                if response.status_code in (401, 403):
                    raise GenerationAuthenticationError(
                        provider="openai",
                        reason=f"HTTP {response.status_code}: Unauthorized stream",
                    )
                if response.status_code == 429:
                    raise GenerationRateLimitError(provider="openai")
                if response.is_error:
                    error_text = await response.aread()
                    raise GenerationError(
                        f"Streaming error HTTP {response.status_code}: {error_text.decode('utf-8', errors='ignore')}"
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_obj = json.loads(data_str)
                        choices = chunk_obj.get("choices", [])
                        if choices:
                            delta_text = choices[0].get("delta", {}).get("content", "")
                            if delta_text:
                                yield delta_text
                    except json.JSONDecodeError:
                        continue

        except httpx.TimeoutException as timeout_err:
            raise GenerationTimeoutError(
                provider="openai",
                timeout_seconds=self.timeout_seconds,
            ) from timeout_err
