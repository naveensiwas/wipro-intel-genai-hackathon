"""
LLM endpoint clients.

Two implementations share the same .invoke(prompt) → EndpointResponse interface:

  SageMakerEndpointClient  — calls AWS SageMaker via boto3 + SSO profile (local dev)
  LlamaEndpointClient      — calls a remote Llama HTTP endpoint directly (server deploy)

The factory function get_endpoint_client() reads cfg.llm_mode and returns the right one.
"""
from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from typing import Any, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from logger_config import get_logger, log_success, log_error, log_warning


# Module-level loggers — one per client class
_sm_logger    = get_logger("llm.SageMakerClient")
_llama_logger = get_logger("llm.LlamaClient")


# ── Shared response wrapper ────────────────────────────────────────────────────

@dataclass
class EndpointResponse:
    """EndpointResponse is a small data container class (a @dataclass) used as the standard return type from
    both endpoint clients.
    """
    content: str
    raw: Optional[Any] = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _coerce_prompt(prompt: Any) -> str:
    """Coerce prompt-like inputs into a plain string."""
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return str(prompt)


def _extract_content(data: Any, logger_obj, client_name: str) -> str:
    """Extract generated text from common OpenAI-compatible and TGI response shapes."""

    # OpenAI-compatible response: dict with choices -> message -> content
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    i.get("text", "") if isinstance(i, dict) else str(i)
                    for i in content
                )

        # Some endpoints return {"generated_text": "..."}
        text = data.get("generated_text")
        if isinstance(text, str):
            return text

    # TGI-style response: list[0].generated_text
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            text = item.get("generated_text")
            if isinstance(text, str):
                return text

    err = f"Could not extract text from response: {data}"
    log_error(logger_obj, err)
    raise ValueError(f"[{client_name}] {err}")


# ── SageMaker client (llm_mode = "sagemaker") ─────────────────────────────────

class SageMakerEndpointClient:
    """
    Calls an AWS SageMaker real-time endpoint using a named boto3 SSO profile.

    Payload (TGI-style):
        {"inputs": "<prompt>", "parameters": {"temperature": ..., "max_new_tokens": ...}}

    Handles both TGI list/dict and OpenAI-compatible choices response shapes.
    """

    def __init__(
        self,
        endpoint_name: str,
        aws_profile: str,
        aws_region: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        verbose: bool = False,
    ):
        self.endpoint_name = endpoint_name
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.timeout       = timeout
        self.verbose       = verbose

        _sm_logger.info(
            f"Creating boto3 session — profile='{aws_profile}' "
            f"region='{aws_region}' endpoint='{endpoint_name}'"
        )
        try:
            session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
            self._client = session.client("sagemaker-runtime", region_name=aws_region)
            log_success(_sm_logger, f"boto3 SageMaker runtime client created for '{endpoint_name}'")
        except Exception as exc:
            log_error(_sm_logger, f"Failed to create boto3 session with profile '{aws_profile}'", exc)
            raise RuntimeError(
                f"[SageMakerClient] Failed to create boto3 session "
                f"with profile '{aws_profile}': {exc}"
            ) from exc

    def invoke(self, prompt: Any) -> EndpointResponse:
        """Invoke the SageMaker endpoint with the given prompt and return the response content."""
        text_prompt = _coerce_prompt(prompt)
        _sm_logger.debug(f"[TRACE] SageMakerEndpointClient.invoke() — sending to endpoint '{self.endpoint_name}'")
        payload = {
            "inputs": text_prompt,
            "parameters": {
                "temperature":      self.temperature,
                "max_new_tokens":   self.max_tokens,
                "return_full_text": False,
            },
        }
        if self.verbose:
            _sm_logger.debug(f"Payload: {json.dumps(payload)}")

        try:
            _sm_logger.info(f"Invoking SageMaker endpoint '{self.endpoint_name}'")
            response = self._client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )
        except ClientError as exc:
            log_error(_sm_logger, f"ClientError on '{self.endpoint_name}'", exc)
            raise RuntimeError(
                f"[SageMakerClient] ClientError on '{self.endpoint_name}': {exc}"
            ) from exc
        except BotoCoreError as exc:
            log_error(_sm_logger, f"BotoCoreError on '{self.endpoint_name}'", exc)
            raise RuntimeError(
                f"[SageMakerClient] BotoCoreError on '{self.endpoint_name}': {exc}"
            ) from exc

        raw_body = response["Body"].read().decode("utf-8")
        data = json.loads(raw_body)
        if self.verbose:
            _sm_logger.debug(f"Raw response: {raw_body}")

        content = _extract_content(data, _sm_logger, "SageMakerClient")
        _sm_logger.info(f"Response received ({len(content)} chars)")
        return EndpointResponse(content=content, raw=data)


# ── Llama HTTP client (llm_mode = "llama") ────────────────────────────────────

class LlamaEndpointClient:
    """
    Calls a remote Llama HTTP endpoint directly using requests.

    Payload (OpenAI chat completions compatible):
        {"model": ..., "messages": [{"role": "user", "content": ...}], ...}

    Used when deployed on a server that has direct access to the Llama endpoint.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        verbose: bool = False,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.api_key     = api_key
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.verbose     = verbose

        _llama_logger.info(
            f"Configured Llama HTTP client — url='{self.base_url}' model='{self.model}'"
        )

    def invoke(self, prompt: Any) -> EndpointResponse:
        """Invoke the Llama HTTP endpoint with the given prompt and return the response content."""
        text_prompt = _coerce_prompt(prompt)
        _llama_logger.debug(f"[TRACE] LlamaEndpointClient.invoke() — sending to endpoint '{self.base_url}'")
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": text_prompt}],
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "dummy":
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.verbose:
            _llama_logger.debug(f"Payload: {json.dumps(payload)}")

        _llama_logger.info(f"POST {self.base_url}/chat/completions")
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            log_error(_llama_logger, f"HTTP error from endpoint: {exc}", exc)
            raise
        except requests.exceptions.ConnectionError as exc:
            log_error(_llama_logger, f"Connection error — is the endpoint reachable? {self.base_url}", exc)
            raise
        except requests.exceptions.Timeout as exc:
            log_warning(_llama_logger, f"Request timed out after {self.timeout}s")
            raise

        data = response.json()
        if self.verbose:
            _llama_logger.debug(f"Raw response: {data}")

        content = _extract_content(data, _llama_logger, "LlamaClient")
        _llama_logger.info(f"Response received ({len(content)} chars)")
        return EndpointResponse(content=content, raw=data)


# ── Factory function ───────────────────────────────────────────────────────────
__all__ = [
    "EndpointResponse",
    "SageMakerEndpointClient",
    "LlamaEndpointClient",
]
