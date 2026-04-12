import os
from collections import defaultdict
from typing import List

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever  
import chromadb

from config import DB_PATH, COLLECTION_NAME, get_embed_model

# ==========================================
# Globals 
# ==========================================
_index = None
_storage_context = None
_vector_retriever = None
_bm25_retriever = None

# ==========================================
# 1. Setup & Loader 
# ==========================================
def load_retrievers():
    global _index, _storage_context, _vector_retriever, _bm25_retriever
    
    if _vector_retriever is not None and _bm25_retriever is not None:
        return _vector_retriever, _bm25_retriever

    print(f"🔄 初始化检索系统...")
    Settings.embed_model = get_embed_model()

    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    _storage_context = StorageContext.from_defaults(
        persist_dir=DB_PATH,
        vector_store=vector_store
    )

    _index = load_index_from_storage(_storage_context)
    
    # 初始化向量检索器 (Top 20 海选子块)
    _vector_retriever = _index.as_retriever(similarity_top_k=20)

    # 初始化 BM25 检索器
    print("🔨 Building BM25 Index...")
    nodes = list(_index.docstore.docs.values())
    _bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, 
        similarity_top_k=20 
    )
    
    print("✅ 双路检索器加载完成。")
    return _vector_retriever, _bm25_retriever


# ==========================================
# 2. RRF Fusion Algorithm (保持标准实现)
# ==========================================
def reciprocal_rank_fusion(results_list: List[List[NodeWithScore]], k: int = 60) -> List[NodeWithScore]:
    fused_scores = defaultdict(float)
    node_map = {}

    for results in results_list:
        for rank, node_ws in enumerate(results):
            node_id = node_ws.node.node_id
            node_map[node_id] = node_ws.node
            fused_scores[node_id] += 1.0 / (k + rank + 1)

    sorted_node_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    fused_results = [NodeWithScore(node=node_map[node_id], score=score) for node_id, score in sorted_node_ids]
    return fused_results


# ==========================================
# 3. 进阶检索逻辑：父节点积分聚合排序
# ==========================================

def retrieve(query: str, top_k: int = 3):
    """
    1. 双路检索子块 (Child)
    2. RRF 融合打分 (Child)
    3. 父节点 (Parent) 继承并累加子块分数
    4. 对父节点进行总分重排，取 Top K
    """
    vector_retriever, bm25_retriever = load_retrievers()
    global _index

    print(f"\n🔍 Query: {query}")
    
    # --- Step 1: 检索子块 ---
    vector_nodes = vector_retriever.retrieve(query)
    bm25_nodes = bm25_retriever.retrieve(query)
    
    # --- Step 2: RRF 融合 (针对子块) ---
    fused_child_nodes = reciprocal_rank_fusion([vector_nodes, bm25_nodes])
    
    if not fused_child_nodes:
        return []

    # --- Step 3: 父节点积分累加 ---
    # 我们拿前 20 个子块作为“评分员”
    parent_scores = defaultdict(float)
    parent_node_map = {}

    for child_ws in fused_child_nodes[:20]:
        child_node = child_ws.node
        child_score = child_ws.score # 子块的 RRF 分数

        # 找到父节点 ID (从 metadata 或属性中提取)
        parent_id = getattr(child_node, 'index_id', child_node.metadata.get('index_id'))
        
        if not parent_id:
            continue

        # 核心：将子块分数累加给父节点
        parent_scores[parent_id] += child_score

        # 缓存父节点对象，避免重复查询库
        if parent_id not in parent_node_map:
            try:
                parent_node_map[parent_id] = _index.docstore.get_node(parent_id)
            except Exception:
                continue

    # --- Step 4: 根据累加后的总分重新对父节点排序 ---
    # 这一步解决了“去重”和“加权”两个问题
    sorted_parent_ids = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

    # --- Step 5: 取最终 Top K ---
    final_parent_nodes = []
    for pid, score in sorted_parent_ids[:top_k]:
        p_node = parent_node_map[pid]
        # 把文件名打出来看看，方便 Debug
        source_name = p_node.metadata.get('disease_name', 'Unknown')
        print(f"  -> 🔗 父节点积分排名: {score:.4f} (Source: {source_name})")
        final_parent_nodes.append(p_node)

    print(f"📦 最终返回 {len(final_parent_nodes)} 个综合权重最高的父节点文档。")
    return final_parent_nodes