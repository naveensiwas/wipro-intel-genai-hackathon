"""
Sanity Check – LLM Endpoint Inference Validation

Tests the full LLM call path with the configured endpoint.
Displays endpoint config and logs all stages of the inference.
"""
import sys
import os
# Ensure the project root is on sys.path so config and other modules can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from config import cfg
from llm.model_loader import get_llm
from logger_config import get_logger, log_section, log_step, log_success, log_error
logger = get_logger(__name__)

st.set_page_config(page_title="LLM Endpoint Sanity Check", page_icon="🧪", layout="wide")

# Center the full app content
_left, center, _right = st.columns([1, 2, 1])

with center:
    # ── Page header ──
    st.title("🧪 LLM Endpoint Sanity Check")
    st.markdown("Validates connectivity and inference on the configured LLM endpoint.")
    st.divider()

    # ── Display endpoint config ──
    log_section(logger, "Endpoint Configuration")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write(f"**Mode:** `{cfg.llm_mode}`")
        if cfg.llm_mode == "sagemaker":
            st.write(f"**Endpoint:** `{cfg.sagemaker_endpoint}`")
            st.write(f"**Region:** `{cfg.aws_region}`")
            st.write(f"**Profile:** `{cfg.aws_profile}`")
        else:
            st.write(f"**Base URL:** `{cfg.llm_base_url}`")
            st.write(f"**Model:** `{cfg.llm_model}`")

    with col2:
        st.write(f"**Temperature:** `{cfg.temperature}`")
        st.write(f"**Max Tokens:** `{cfg.max_tokens}`")
        st.write(f"**Timeout:** `{cfg.llm_timeout}s`")

    st.divider()

    # ── Initialize LLM client ──
    log_step(logger, 1, "Initializing LLM client")
    try:
        with st.spinner("⏳ Connecting to LLM endpoint..."):
            llm = get_llm()
        log_success(logger, "LLM client initialised")
        st.success("✅ LLM client connected")
    except Exception as exc:
        log_error(logger, "Failed to initialize LLM client", exc)
        st.error(f"❌ Failed to connect: {exc}")
        st.stop()

    # ── Test prompt input ──
    st.subheader("🧠 Test Inference")
    prompt = st.text_input("Enter a test prompt:", value="What is Gen AI in simple words ?", key="test_prompt")

    if st.button("▶️ Run Inference", type="primary"):
        log_step(logger, 2, f"Running inference with prompt: '{prompt}...'")
        try:
            with st.spinner("🔄 Calling endpoint..."):
                response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            log_success(logger, f"Inference complete ({len(content)} chars)")

            st.success("✅ Endpoint responded")
            st.markdown("### Response")
            st.write(content)

        except Exception as exc:
            log_error(logger, f"Inference failed: {exc}", exc)
            st.error(f"❌ Endpoint error: {exc}")

    st.divider()
    st.caption("✅ Use this tool to validate your LLM endpoint before running the main app.")
