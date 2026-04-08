"""
LLM client factory.
Reads cfg.llm_mode and returns a cached LangChain-compatible LLM wrapper:
  "sagemaker" → EndpointLLM over SageMakerEndpointClient
  "llama"     → EndpointLLM over LlamaEndpointClient
"""
from functools import lru_cache
from app.config.settings import cfg
from app.llm.endpoint_client import SageMakerEndpointClient, LlamaEndpointClient
from app.llm.langchain_adapter import EndpointLLM
from app.core.logging import get_logger, log_success, log_error

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> EndpointLLM:
    """
    Return a cached LangChain-compatible LLM based on configured backend.

    Reads LLM_MODE environment variable to determine which endpoint client to use:
    - "sagemaker": AWS SageMaker real-time endpoint (for local dev with SSO)
    - "llama": Direct HTTP endpoint supporting OpenAI chat completions API

    The returned EndpointLLM is a thin LangChain wrapper that makes the endpoint
    compatible with RetrievalQA chains and other LangChain components.

    Returns:
        EndpointLLM: Cached LangChain-compatible LLM instance wrapping the configured endpoint

    Raises:
        ValueError: If LLM_MODE is not "sagemaker" or "llama"
        RuntimeError: If endpoint initialization fails (boto3 setup, network connection, etc.)

    Note:
        Result is cached via @lru_cache(maxsize=1), so repeated calls return the same instance.
        This is critical for performance since LLM initialization involves network setup.

    Examples:
        llm = get_llm()  # Returns cached EndpointLLM instance
        result = llm.invoke("What is a fever?")
    """
    mode = cfg.llm_mode.strip().lower()
    logger.info(f"LLM_MODE = '{mode}'")

    # For SageMaker, we use a custom SageMakerEndpointClient wrapped in an EndpointLLM.
    if mode == "sagemaker":
        logger.info(
            f"Initialising SageMaker endpoint '{cfg.sagemaker_endpoint}' "
            f"(profile={cfg.aws_profile}, region={cfg.aws_region})"
        )
        try:
            client = SageMakerEndpointClient(
                endpoint_name=cfg.sagemaker_endpoint,
                aws_profile=cfg.aws_profile,
                aws_region=cfg.aws_region,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.llm_timeout,
                verbose=cfg.verbose,
            )
            llm = EndpointLLM(
                client=client,
                llm_mode="sagemaker",
                llm_name=cfg.sagemaker_endpoint,
            )
            log_success(logger, f"SageMaker LangChain LLM ready — endpoint='{cfg.sagemaker_endpoint}'")
            return llm
        except Exception as exc:
            log_error(logger, "Failed to initialise SageMaker client", exc)
            raise

    # For HTTP-based LLMs, we assume a generic EndpointLLM that can wrap any HTTP client.
    if mode == "llama":
        logger.info(
            f"Initialising Llama HTTP endpoint '{cfg.llm_base_url}' "
            f"(model={cfg.llm_model})"
        )
        try:
            client = LlamaEndpointClient(
                base_url=cfg.llm_base_url,
                model=cfg.llm_model,
                api_key=cfg.llm_api_key,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.llm_timeout,
                verbose=cfg.verbose,
            )
            llm = EndpointLLM(
                client=client,
                llm_mode="llama",
                llm_name=cfg.llm_model,
            )
            log_success(logger, f"Llama LangChain LLM ready — url='{cfg.llm_base_url}'")
            return llm
        except Exception as exc:
            log_error(logger, "Failed to initialise Llama client", exc)
            raise

    msg = f"Unknown LLM_MODE='{mode}'. Valid values: 'sagemaker' | 'llama'"
    log_error(logger, msg)
    raise ValueError(f"[ModelLoader] {msg}")
