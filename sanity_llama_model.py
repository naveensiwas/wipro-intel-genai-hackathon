"""
Sanity Check – LLM Endpoint Inference Validation

Direct endpoint test (non-Streamlit).
Validates endpoint connectivity and inference capability.
"""
import time
from config import cfg
from llm.model_loader import get_llm
from logger_config import get_logger, log_section, log_step, log_success, log_error

logger = get_logger(__name__)

# ── Display endpoint config ──
log_section(logger, "Endpoint Configuration")
logger.info(f"LLM_MODE = '{cfg.llm_mode}'")

if cfg.llm_mode == "sagemaker":
    logger.info(f"Target → AWS SageMaker endpoint: {cfg.sagemaker_endpoint}")
    logger.info(f"         AWS profile:  {cfg.aws_profile}")
    logger.info(f"         AWS region:   {cfg.aws_region}")
else:
    logger.info(f"Target → Llama HTTP endpoint: {cfg.llm_base_url}")
    logger.info(f"         Model: {cfg.llm_model}")

logger.info(f"Temperature: {cfg.temperature} | Max tokens: {cfg.max_tokens} | Timeout: {cfg.llm_timeout}s")

# ── Initialize and test ──
log_step(logger, 1, "Initializing LLM client")
try:
    llm = get_llm()
    log_success(logger, "LLM client initialized")
except Exception as exc:
    log_error(logger, "Failed to initialize LLM client", exc)
    raise SystemExit(1)

# ── Run test inference ──
log_step(logger, 2, "Running test inference")
test_prompt = "What is Machine Learning?"
try:
    start_time = time.time()
    response = llm.invoke(test_prompt)
    end_time = time.time()

    elapsed_seconds = end_time - start_time
    content = response.content if hasattr(response, "content") else str(response)

    log_success(
        logger,
        f"Inference completed successfully in {elapsed_seconds:.2f} seconds. Response length: {len(content)} characters."
    )

    logger.debug("LLM Prompt: %s", test_prompt)
    log_success(logger, f"📝 LLM Response: {content}")
except Exception as exc:
    log_error(logger, f"Inference failed on prompt: '{test_prompt}'", exc)
    raise SystemExit(1)

log_success(logger, "Sanity check PASSED")
