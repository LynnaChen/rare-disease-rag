import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# 💡 改用 OpenAI 兼容接口，因为 vLLM 模拟了 OpenAI 接口
from llama_index.llms.openai_like import OpenAILike

# ==========================================
# 路径配置
# ==========================================
#vllm 路径
# 你的 vLLM 服务器地址（如果是同一台机器，用 localhost）
VLLM_API_BASE = "http://localhost:8000/v1"
# 必须与你启动 vLLM 时 --served-model-name 指定的名字一致
VLLM_MODEL_NAME = "my-local-model"

# ChromaDB 数据库路径
DB_PATH = "./chroma_db_med"
COLLECTION_NAME = "med_rare_diseases"

# PDF 输入目录
INPUT_DIR = "/fs/scratch/users/chenla/RareDisease_rag/nur pdf"

# 模型缓存目录
MODEL_CACHE_DIR = "./model_cache"

# ==========================================
# 模型配置
# ==========================================
# 本地 LLM 模型路径 (可以是 HuggingFace ID 或本地绝对路径)
# 推荐: "Qwen/Qwen2-7B-Instruct" 或 "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_MODEL_PATH = "Qwen/Qwen2-7B-Instruct"

# Embedding 模型路径 (必须与 Indexer 一致)
# 👇 改成这个长路径
EMBED_MODEL_PATH = "/fs/scratch/users/chenla/RareDisease_rag/model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

# ==========================================
# LLM 生成配置
# ==========================================
LLM_CONTEXT_WINDOW = 32000
LLM_MAX_NEW_TOKENS = 1024
LLM_TEMPERATURE = 0.1
LLM_DO_SAMPLE = True


# ==========================================
# 模型加载函数（原 models.py / models2.py 逻辑合并到这里）
# ==========================================

_llm = None
_embed_model = None


def get_embed_model(device: str = "cuda"):
    """
    加载并缓存 Embedding 模型（只初始化一次），同时写入 Settings.embed_model。
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
    加载并缓存本地 LLM（只初始化一次），同时写入 Settings.llm。
    """
    global _llm
    if _llm is None:
        print("🚀 正在通过 LlamaIndex 加载本地 LLM ...")
        _llm = OpenAILike(
            model=VLLM_MODEL_NAME,
            api_base=VLLM_API_BASE,
            api_key="fake_key", # vLLM 本地运行不需要真实 Key，但这里不能为空
            is_chat_model=True,
            context_window=32000,
            timeout=60.0
        )
        Settings.llm = _llm
        print("✅ 成功连接至 vLLM 推理引擎。")
    return _llm

# ==========================================


# ==========================================
# Prompt 模板
# ==========================================
# 意图识别 Prompt
INTENT_EXTRACTION_PROMPT = """You are a medical retrieval assistant. Analyze the user's input and extract the core retrieval conditions.
User Input: "{query}"

Please output JSON format, do not include Markdown tags, and include the following fields:
- disease_name: (string) The most specific disease name or gene name, if none then null
- keywords: (string) 3-5 medical keywords for searching, separated by spaces
- is_social: (boolean) Whether the user is primarily looking for a support group/social connection
"""

# 生成回答的系统提示词
GENERATION_SYSTEM_TEMPLATE = """You are a professional medical assistant. Answer the user's questions based on the following medical guideline content.
If the guidelines do not contain relevant information, state so clearly.

【Medical Guideline Content】:
{context}

【Answer Requirements】:
- The answer should be in English
- Answer accurately based on the guideline content
- If there is no relevant information in the guidelines, say "No relevant information found in the guidelines"
- Answers should be professional, accurate, and easy to understand
- You may refer to prior conversation history to provide more coherent answers
- At the end of your answer, you must list all reference sources (format: [Source: 《filename》, Page X])
"""

# 生成回答的用户提示词
GENERATION_USER_TEMPLATE = "{question}"

# 社交需求回复
SOCIAL_RESPONSE = "I detected that you want to find a support group. Please scan the QR code below to follow '豌豆Sir'公众号 to get community information..."

# ==========================================
# 疾病名关键词字典（用于快速匹配）
# ==========================================
# 常见罕见病名称列表（可以从数据库加载，这里先硬编码示例）
# 实际使用时可以从已索引的文件名中动态加载


# 拒答话术
REJECTION_RESPONSE = "Sorry, I did not find any relevant specific information in the existing authoritative medical guideline database. For safety, please consult a professional doctor."

