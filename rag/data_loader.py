"""
Loads the three healthcare JSON files and converts them into
LangChain Document objects suitable for embedding and retrieval.
"""
import json
from langchain_core.documents import Document
from config import cfg
from logger_config import get_logger, log_success, log_error

logger = get_logger(__name__)


def _load_json(path: str) -> list:
    logger.debug(f"Reading JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_symptom_documents() -> list[Document]:
    """Convert symptoms_conditions.json into Documents."""
    data = _load_json(cfg.symptoms_conditions_path)
    docs = []
    for item in data:
        symptom = item["symptom"]
        conditions = ", ".join(item.get("related_conditions", []))
        body_system = item.get("body_system", "General")
        description = item.get("description", "")
        source = item.get("source", "")
        content = (
            f"Symptom: {symptom}\n"
            f"Body System: {body_system}\n"
            f"Description: {description}\n"
            f"Commonly associated conditions: {conditions}\n"
            f"Source: {source}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source_file": "symptoms_conditions", "symptom": symptom, "body_system": body_system}
        ))
    logger.debug(f"Loaded {len(docs)} symptom documents")
    return docs


def load_condition_documents() -> list[Document]:
    """Convert conditions_info.json into Documents."""
    data = _load_json(cfg.conditions_info_path)
    docs = []
    for item in data:
        name = item["name"]
        common_name = item.get("common_name", name)
        description = item.get("description", "")
        causes = "; ".join(item.get("common_causes", []))
        symptoms = ", ".join(item.get("typical_symptoms", []))
        care_tips = " | ".join(item.get("general_care_tips", []))
        when_to_seek = item.get("when_to_seek_care", "")
        specialist = item.get("specialist_type", "")
        prevention = ", ".join(item.get("prevention", []))
        source = item.get("source", "")
        content = (
            f"Condition: {common_name} ({name})\n"
            f"Description: {description}\n"
            f"Common Causes: {causes}\n"
            f"Typical Symptoms: {symptoms}\n"
            f"General Care Tips: {care_tips}\n"
            f"When to Seek Care: {when_to_seek}\n"
            f"Recommended Specialist: {specialist}\n"
            f"Prevention: {prevention}\n"
            f"Source: {source}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source_file": "conditions_info", "condition": name, "specialist": specialist}
        ))
    logger.debug(f"Loaded {len(docs)} condition documents")
    return docs


def load_preventive_documents() -> list[Document]:
    """Convert preventive_tips.json into Documents."""
    data = _load_json(cfg.preventive_tips_path)
    docs = []
    for item in data:
        category = item["category"]
        tips = "\n- ".join(item.get("tips", []))
        source = item.get("source", "")
        content = (
            f"Preventive Health Category: {category}\n"
            f"Tips:\n- {tips}\n"
            f"Source: {source}"
        )
        docs.append(Document(
            page_content=content,
            metadata={"source_file": "preventive_tips", "category": category}
        ))
    logger.debug(f"Loaded {len(docs)} preventive documents")
    return docs


def load_all_documents() -> list[Document]:
    """Load and return all healthcare documents."""
    logger.info("Loading all healthcare documents...")
    docs = []
    try:
        docs.extend(load_symptom_documents())
        docs.extend(load_condition_documents())
        docs.extend(load_preventive_documents())
        log_success(logger, f"Loaded {len(docs)} documents total")
    except Exception as exc:
        log_error(logger, "Failed to load healthcare documents", exc)
        raise
    return docs

