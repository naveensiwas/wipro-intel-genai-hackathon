import sys
print("Starting test...", flush=True)

try:
    from rag.data_loader import load_symptom_documents, load_condition_documents, load_preventive_documents
    s = load_symptom_documents()
    c = load_condition_documents()
    p = load_preventive_documents()
    print(f"Symptoms: {len(s)}, Conditions: {len(c)}, Preventive: {len(p)}, Total: {len(s)+len(c)+len(p)}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback; traceback.print_exc()

try:
    from rag.embedder import get_embeddings
    print("Embedder import OK", flush=True)
except Exception as e:
    print(f"Embedder ERROR: {e}", flush=True)

try:
    from rag.vector_store import build_or_load_vector_store
    print("VectorStore import OK", flush=True)
except Exception as e:
    print(f"VectorStore ERROR: {e}", flush=True)

try:
    from rag.retriever import build_rag_chain, retrieve_sources
    print("Retriever import OK", flush=True)
except Exception as e:
    print(f"Retriever ERROR: {e}", flush=True)

try:
    from llm.model_loader import get_llm
    llm = get_llm()
    print(f"ModelLoader import OK ({llm.__class__.__name__})", flush=True)
except Exception as e:
    print(f"ModelLoader ERROR: {e}", flush=True)

try:
    from ui.sidebar import render_sidebar
    from ui.chat_interface import init_chat_state, render_chat_history, add_message
    print("UI imports OK", flush=True)
except Exception as e:
    print(f"UI ERROR: {e}", flush=True)

print("Test complete.", flush=True)
