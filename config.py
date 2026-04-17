import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# ==========================================
# Path configuration
# ==========================================
# vLLM endpoint
# Your vLLM server address (use localhost if on the same machine)
VLLM_API_BASE = "http://localhost:8000/v1"
# Must match the name passed to vLLM via --served-model-name
VLLM_MODEL_NAME = "my-local-model"

# ChromaDB persistence path
DB_PATH = "./chroma_db_med"
COLLECTION_NAME = "med_rare_diseases"

# PDF input directory
INPUT_DIR = "./data/pdfs"

# Model cache directory
MODEL_CACHE_DIR = "./model_cache"

# ==========================================
# Model configuration
# ==========================================
# Local LLM model path (can be a HuggingFace model ID or an absolute local path)
# Recommended: "Qwen/Qwen2-7B-Instruct" or "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"

# Embedding model path (must match the one used by the indexer)
EMBED_MODEL_PATH = "BAAI/bge-m3"

# ==========================================
# LLM generation configuration
# ==========================================
LLM_CONTEXT_WINDOW = 32000
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1
LLM_DO_SAMPLE = True


# ==========================================
# Model loader helpers (merged from the original models.py / models2.py)
# ==========================================

_llm = None
_embed_model = None


def get_embed_model(device: str = "cuda"):
    """
    Load and cache the embedding model (initialize only once), and set Settings.embed_model.
    """
    global _embed_model
    if _embed_model is None:
        print("🔄 Loading Embedding Model...")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_PATH,
            cache_folder=MODEL_CACHE_DIR,
            device=device,
            embed_batch_size=64
        )
        Settings.embed_model = _embed_model
    return _embed_model


def get_generation_model():
    """
    Load and cache the LLM client (initialize only once), and set Settings.llm.
    """
    global _llm
    if _llm is None:
        print("🚀 Initializing LLM via LlamaIndex ...")
        _llm = OpenAILike(
            model=VLLM_MODEL_NAME,
            api_base=VLLM_API_BASE,
            api_key="fake_key",  # vLLM does not require a real key, but this cannot be empty
            is_chat_model=True,
            context_window=32000,
            timeout=60.0
        )
        Settings.llm = _llm
        print("✅ Successfully connected to the vLLM inference engine.")
    return _llm

# ==========================================


# ==========================================
# Prompt templates
# ==========================================
# Intent extraction prompt
INTENT_EXTRACTION_PROMPT = """You are a medical retrieval assistant. Analyze the user's input and extract the core retrieval conditions.
User Input: "{query}"

Please output JSON format, do not include Markdown tags, and include the following fields:
- disease_name: (string) The most specific disease name or gene name, if none then null
- keywords: (string) 3-5 medical keywords for searching, separated by spaces
- is_social: (boolean) Whether the user is primarily looking for a support group/social connection
"""

# System prompt for answer generation
GENERATION_SYSTEM_TEMPLATE = """You are a professional medical assistant. Answer the user's questions based on the following medical guideline content.
If the guidelines do not contain relevant information, state so clearly.

Medical Guideline Content:
{context}

Answer Requirements:
- The answer should be in English
- Answer accurately based on the guideline content
- If there is no relevant information in the guidelines, say "No relevant information found in the guidelines"
- Answers should be professional, accurate, and easy to understand
- You may refer to prior conversation history to provide more coherent answers
- At the end of your answer, you must list all reference sources (format: [Source: \"filename\", Page X])
"""

# User prompt for answer generation
GENERATION_USER_TEMPLATE = "{question}"

# Response for social/support-group intent
SOCIAL_RESPONSE = "I detected that you want to find a support group. Please scan the QR code below and follow the 'WandouSir' public account to get community information..."

# ==========================================
# Disease keyword dictionary (for quick matching)
# ==========================================
# A list of common rare disease names (can be loaded from the database; hard-coded here as a placeholder)
# In practice, this can be dynamically derived from indexed file names


# Rejection response
REJECTION_RESPONSE = "Sorry, I did not find any relevant specific information in the existing authoritative medical guideline database. For safety, please consult a professional doctor."

