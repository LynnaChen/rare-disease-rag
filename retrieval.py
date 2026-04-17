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

    print("🔄 Initializing retrieval system...")
    Settings.embed_model = get_embed_model()

    db = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    _storage_context = StorageContext.from_defaults(
        persist_dir=DB_PATH,
        vector_store=vector_store
    )

    _index = load_index_from_storage(_storage_context)
    
    # Initialize vector retriever (top-20 candidate child chunks)
    _vector_retriever = _index.as_retriever(similarity_top_k=20)

    # Initialize BM25 retriever
    print("🔨 Building BM25 Index...")
    nodes = list(_index.docstore.docs.values())
    _bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, 
        similarity_top_k=20 
    )
    
    print("✅ Dual retrievers are ready.")
    return _vector_retriever, _bm25_retriever


# ==========================================
# 2. RRF Fusion Algorithm (standard implementation)
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
# 3. Advanced retrieval: parent-score aggregation and reranking
# ==========================================

def retrieve(query: str, top_k: int = 3):
    """
    1. Retrieve child chunks via two retrievers (Child)
    2. Fuse child results with RRF scoring (Child)
    3. Aggregate child scores onto their corresponding parents (Parent)
    4. Rerank parents by aggregated score and return Top-K
    """
    vector_retriever, bm25_retriever = load_retrievers()
    global _index

    print(f"\n🔍 Query: {query}")
    
    # --- Step 1: retrieve child chunks ---
    vector_nodes = vector_retriever.retrieve(query)
    bm25_nodes = bm25_retriever.retrieve(query)
    
    # --- Step 2: RRF fusion (on child chunks) ---
    fused_child_nodes = reciprocal_rank_fusion([vector_nodes, bm25_nodes])
    
    if not fused_child_nodes:
        return []

    # --- Step 3: aggregate scores onto parents ---
    # Use the top-20 fused child chunks as "voters"
    parent_scores = defaultdict(float)
    parent_node_map = {}

    for child_ws in fused_child_nodes[:20]:
        child_node = child_ws.node
        child_score = child_ws.score  # RRF score for the child chunk

        # Resolve parent ID (from attribute or metadata)
        parent_id = getattr(child_node, 'index_id', child_node.metadata.get('index_id'))
        
        if not parent_id:
            continue

        # Core: accumulate child score into its parent
        parent_scores[parent_id] += child_score

        # Cache parent node objects to avoid repeated docstore lookups
        if parent_id not in parent_node_map:
            try:
                parent_node_map[parent_id] = _index.docstore.get_node(parent_id)
            except Exception:
                continue

    # --- Step 4: rerank parents by aggregated score ---
    # This naturally de-duplicates and rewards parents hit by multiple children
    sorted_parent_ids = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)

    # --- Step 5: return final Top-K parents ---
    final_parent_nodes = []
    for pid, score in sorted_parent_ids[:top_k]:
        p_node = parent_node_map[pid]
        # Print source name for debugging
        source_name = p_node.metadata.get('disease_name', 'Unknown')
        print(f"  -> 🔗 Parent aggregated score: {score:.4f} (Source: {source_name})")
        final_parent_nodes.append(p_node)

    print(f"📦 Returning {len(final_parent_nodes)} parent documents with the highest aggregated scores.")
    return final_parent_nodes