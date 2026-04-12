# RareDisease RAG

基于医学指南 PDF 的罕见病问答 RAG（Retrieval-Augmented Generation）系统。文档经解析、分块、向量化与混合检索后，由大模型基于检索结果生成回答。

---

## 项目结构

```
RareDisease_rag/
├── config.py       # 路径、模型、Prompt 等配置
├── indexer.py      # 离线建库：PDF → 解析 → 分块 → 向量化 → ChromaDB
├── retrieval.py    # 检索：双路检索 + RRF 融合 + 父节点积分聚合排序
├── generation.py   # 生成：上下文拼接 + LLM 生成
├── main.py         # 交互式 CLI 入口
├── eva.py          # RAGAS 评估脚本
├── models.py       # 模型加载兼容层（代理到 config）
├── groundtruth.json # 评估用标注数据
└── chroma_db_med/  # ChromaDB 持久化目录（建库后生成）
```

---

## RAG 各步骤与方法说明

### 步骤 1：文档加载（Document Loading）

| 项目 | 方法 / 实现 |
|------|-------------|
| **所用组件** | **LlamaIndex** `SimpleDirectoryReader` + **Docling** `DoclingReader` |
| **文档格式** | PDF（通过 `file_extractor={".pdf": docling_reader}` 指定） |
| **PDF 解析** | **Docling**：`PdfFormatOption` + `PdfPipelineOptions`，支持表格结构解析（`do_table_structure=True`），可选 OCR（本项目 `do_ocr=False`），GPU 加速（`AcceleratorDevice.CUDA`） |
| **元数据** | 从文件路径提取：`get_file_metadata()` → `disease_name`（文件名无扩展名）、`file_name`（含扩展名） |

代码位置：`indexer.py` — `SimpleDirectoryReader`、`DoclingReader`、`get_file_metadata`。

---

### 步骤 2：分块 / 节点解析（Chunking & Node Parsing）

| 项目 | 方法 / 实现 |
|------|-------------|
| **策略** | **层级 Parent-Child 分块**（先粗后细，便于检索细粒度、返回粗粒度） |
| **父节点 (Parent)** | **LlamaIndex** `SentenceSplitter`：`chunk_size=1024`，`chunk_overlap=200`，按句切分 |
| **子节点 (Child)** | **LlamaIndex** `SentenceSplitter`：`chunk_size=128`，`chunk_overlap=20`，在父节点基础上再切 |
| **关联方式** | 子节点通过 `IndexNode.from_text_node(child_node, index_id=parent_node.node_id)` 指向父节点；父节点仅入 Docstore，子节点入向量库并带 `index_id` 用于回溯 |

代码位置：`indexer.py` — `parent_splitter` / `child_splitter`、循环中 `get_nodes_from_documents` 与 `IndexNode.from_text_node`。

---

### 步骤 3：向量化与索引存储（Embedding & Indexing）

| 项目 | 方法 / 实现 |
|------|-------------|
| **向量化模型** | **HuggingFace BGE-M3**：`llama_index.embeddings.huggingface.HuggingFaceEmbedding`，本地路径由 `config.EMBED_MODEL_PATH` 指定，`embed_batch_size=64` |
| **向量库** | **ChromaDB**：`chromadb.PersistentClient(path=DB_PATH)`，集合名 `COLLECTION_NAME`，通过 LlamaIndex 的 `ChromaVectorStore` 写入 |
| **写入内容** | **仅子节点**：对 Child Nodes 做 Embedding 并写入 Chroma；Parent Nodes 只写入 LlamaIndex 的 Docstore，不向量化 |
| **索引构建** | `VectorStoreIndex(nodes_to_index, storage_context=storage_context)`，后续批次用 `index.insert_nodes(nodes_to_index)`；同时 `index.docstore.add_documents([p_node])` 写入父节点 |

代码位置：`indexer.py` — `get_embed_model`（在 config）、`ChromaVectorStore`、`VectorStoreIndex`；Embedding 配置在 `config.py`。

---

### 步骤 4：检索（Retrieval）

| 项目 | 方法 / 实现 |
|------|-------------|
| **检索对象** | 对 **Child Nodes** 做双路检索，经 RRF 融合后按 **父节点积分聚合** 重排，最终返回 **Parent Nodes** |
| **路径 A — 向量检索** | LlamaIndex `VectorStoreIndex.as_retriever(similarity_top_k=20)`，基于 BGE-M3 的向量相似度（默认余弦等，由 Chroma 实现） |
| **路径 B — 关键词检索** | **BM25**：`llama_index.retrievers.bm25.BM25Retriever.from_defaults(nodes=..., similarity_top_k=20)`，对 Docstore 中的节点建 BM25 倒排索引 |
| **融合算法** | **RRF（Reciprocal Rank Fusion）**：`Score = Σ 1/(k + rank + 1)`，`k=60`，对两路 Top 20 子块按 node_id 融合得到 `fused_child_nodes` |
| **父节点积分累加** | 取 RRF 融合后前 20 个子块作为“评分员”：每个子块将其 RRF 分数累加到对应父节点（`parent_scores[parent_id] += child_score`），父节点 ID 来自子节点的 `index_id` |
| **取 Top K** | 按父节点 **累加总分** 降序排序，取 Top K（默认 `top_k=3`）个 **父节点**；同一父节点若被多个子块命中则总分更高，自然实现去重与加权 |
| **返回** | 从 Docstore 取回上述 Top K 父节点，作为 `retrieve()` 的返回列表（综合权重最高的父节点文档） |

代码位置：`retrieval.py` — `load_retrievers()`、`reciprocal_rank_fusion()`、`retrieve()`（含父节点积分聚合逻辑）。

---

### 步骤 5：生成（Generation）

| 项目 | 方法 / 实现 |
|------|-------------|
| **上下文拼接** | `build_context_from_parents(parent_nodes, max_chars=6000)`：按 `[Document i: file_name, Page X]\n{text}` 拼接，带文件名与页码，总长度上限 6000 字符 |
| **LLM 接口** | **OpenAI 兼容 API**：`llama_index.llms.openai_like.OpenAILike`，对接 **vLLM** 本地服务（`api_base`、`model` 在 config），`is_chat_model=True` |
| **Prompt** | 系统提示：`GENERATION_SYSTEM_TEMPLATE`（含 `{context}`）；用户提示：`GENERATION_USER_TEMPLATE`（含 `{question}`）；无检索结果时直接返回 `REJECTION_RESPONSE` |
| **调用方式** | `ChatMessage(role=SYSTEM/USER, content=...)` → `llm.chat(messages)`，取 `response.message.content` |

代码位置：`generation.py` — `build_context_from_parents`、`generate_answer`；`config.py` — `GENERATION_SYSTEM_TEMPLATE`、`GENERATION_USER_TEMPLATE`、`REJECTION_RESPONSE`、vLLM/OpenAILike 配置。

---

### 步骤 6：评估（Evaluation，可选）

| 项目 | 方法 / 实现 |
|------|-------------|
| **框架** | **Ragas**：`ragas.evaluate` |
| **流程** | 从 `groundtruth*.json` 读 question / reference；对每条 question 调用 `retrieve()` + `generate_answer()` 得到 answer 与 contexts；构造 `Dataset` 后送入 `evaluate()` |
| **指标** | `faithfulness`、`answer_relevancy`、`context_precision`、`context_recall` |
| **评估用模型** | LLM：OpenAI `gpt-4o-mini`；Embedding：OpenAI `text-embedding-3-small`（由 `OPENAI_API_KEY` 指定） |

代码位置：`eva.py`。

---

## 依赖与运行

- **Python**：建议 3.10+  
- **主要依赖**：`llama-index`、`llama-index-embeddings-huggingface`、`llama-index-vector-stores-chroma`、`llama-index-readers-file`、`docling`、`chromadb`、`sentence-transformers`（或 HuggingFace Transformers）、vLLM 服务端（生成阶段）。评估需 `ragas`、`openai`、`datasets`。
- **建库**：配置 `config.py` 中 `INPUT_DIR`（PDF 目录）、`DB_PATH`、`EMBED_MODEL_PATH` 等，然后运行  
  `python indexer.py`
- **问答**：先启动 vLLM（模型名与 `config.VLLM_MODEL_NAME`、`VLLM_API_BASE` 一致），再运行  
  `python main.py`
- **评估**：设置 `OPENAI_API_KEY`，修改 `eva.py` 中 `EVAL_DATA_PATH` 等，运行  
  `python eva.py`

---

## 配置要点（config.py）

- **INPUT_DIR**：PDF 医学指南所在目录  
- **DB_PATH** / **COLLECTION_NAME**：ChromaDB 路径与集合名  
- **EMBED_MODEL_PATH**：BGE-M3 本地路径（与 indexer 一致）  
- **VLLM_API_BASE** / **VLLM_MODEL_NAME**：vLLM 服务地址与模型名  
- **GENERATION_SYSTEM_TEMPLATE** / **GENERATION_USER_TEMPLATE**：生成用系统/用户提示词  

按上述步骤与方法即可复现或修改本 RAG 流程。
