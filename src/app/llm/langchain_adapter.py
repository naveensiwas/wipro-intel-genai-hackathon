"""
LangChain-compatible adapter for custom endpoint clients.

This lets RetrievalQA and other LangChain chains treat the custom SageMaker/HTTP
clients as proper LLMs (Runnable-compatible).
"""
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import ConfigDict


class EndpointLLM(LLM):
    """Thin LangChain LLM wrapper over a custom endpoint transport client."""

    client: Any
    llm_mode: str
    llm_name: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        """Return a string identifier for this LLM type."""
        return f"endpoint_{self.llm_mode}"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Invoke the underlying client with the prompt and return the response text."""
        response = self.client.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)

        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]
                    break

        return text

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return identifying parameters for this LLM (used in caching, logging, etc.)."""
        return {
            "llm_mode": self.llm_mode,
            "llm_name": self.llm_name,
        }
