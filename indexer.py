"""
Indexer: offline script for building the index only.
"""
from config import DB_PATH, COLLECTION_NAME, INPUT_DIR, MODEL_CACHE_DIR, EMBED_MODEL_PATH
import os
import gc
import torch
from llama_index.core import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    Settings
)
from llama_index.core.schema import IndexNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.readers.docling import DoclingReader
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice

from config import (
    DB_PATH, COLLECTION_NAME, INPUT_DIR, MODEL_CACHE_DIR,
    EMBED_MODEL_PATH, get_embed_model)
import torch
import onnxruntime as ort

print(f"🔦 PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"🔦 ONNX Runtime Devices: {ort.get_device()}")
# ==========================================
# 1. Setup
# ==========================================
# GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Current device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU model: {torch.cuda.get_device_name(0)}")

# Embedding model
get_embed_model(device=DEVICE)

# ==========================================
# 2. ChromaDB
# ==========================================
print(f"🔄 Connecting to ChromaDB at {DB_PATH}...")
db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ==========================================
# 3. Docling & Node Parsers
# ==========================================
# 3.1. Docling for reading PDFs
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8, 
    device=AcceleratorDevice.CUDA 
)
docling_reader = DoclingReader(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
file_extractor = {".pdf": docling_reader}

# 3.2. Splitters
# parent splitter
parent_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=200
)

# child splitter
child_splitter = SentenceSplitter(
    chunk_size=128,
    chunk_overlap=20
)


# ==========================================
# 4. Input Files
# ==========================================
if not os.path.exists(INPUT_DIR):
    raise ValueError(f"❌ Input directory not found: {INPUT_DIR}")

all_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".pdf")]
all_files.sort()
print(f"📂 Found {len(all_files)} PDF files in total.")

# ==========================================
# 5. metadata extraction function
# ==========================================
def get_file_metadata(file_path: str) -> dict:
    """
    Extract metadata from a file path.
    Example: "SMA.pdf" -> {"disease_name": "SMA", "file_name": "SMA.pdf"}
    """
    file_name = os.path.basename(file_path)
    
    disease_name = os.path.splitext(file_name)[0]
    return {
        "disease_name": disease_name,
        "file_name": file_name
    }

# ==========================================
# 6. The Loop
# ==========================================
index = None
print("\n🏁 Start Indexing Process...")

for i in range(0, len(all_files), 50):
    batch_files = all_files[i : i + 50]
    batch_num = i // 50 + 1
    print(f"\n🚀 Processing Batch {batch_num} ({len(batch_files)} files)...")

    try:
        # -----------------------------------
        # A. Loading
        # -----------------------------------
        print("   - [1/4] Reading PDFs & OCR...")
        dir_reader = SimpleDirectoryReader(
            input_files=batch_files,
            file_extractor=file_extractor,
            file_metadata=get_file_metadata
        )
        documents = dir_reader.load_data()
        
        # -----------------------------------
        # B. Hierarchical Nodes
        # -----------------------------------
        print("   - [2/4] Building Semantic Parent-Child Nodes...")

        nodes_to_index = []      # child nodes (embedded)
        nodes_to_docstore = []  # parent nodes 

        # 1. parent nodes
        parent_nodes = parent_splitter.get_nodes_from_documents(documents)

        for p_node in parent_nodes:
            parent_metadata = p_node.metadata.copy() if p_node.metadata else {}

            parent_file_name = parent_metadata.get("file_name", "")
            parent_disease_name = parent_metadata.get("disease_name", "")
            parent_page_number = (
                parent_metadata.get("page_number") or 
                parent_metadata.get("page") or 
                parent_metadata.get("page_num") or
                None
            )

            # 2. child nodes
            child_nodes = child_splitter.get_nodes_from_documents([p_node])

            # 3. Linking & metadata inheritance
            for c_node in child_nodes:
                if not c_node.metadata:
                    c_node.metadata = {}

                c_node.metadata["file_name"] = parent_file_name
                c_node.metadata["disease_name"] = parent_disease_name

                if parent_page_number is not None:
                    c_node.metadata["page_number"] = parent_page_number

                index_node = IndexNode.from_text_node(
                    c_node,
                    index_id=p_node.node_id
                )

                nodes_to_index.append(index_node)

            # 4. parent nodes only in docstore
            nodes_to_docstore.append(p_node)

        print(f"     > Created {len(nodes_to_index)} child nodes (embedded)")
        print(f"     > Created {len(nodes_to_docstore)} parent nodes (docstore only)")

        # -----------------------------------
        # C. Embedding & Indexing
        # -----------------------------------
        print("   - [3/4] Embedding & Inserting into ChromaDB...")
        
        # Only save child nodes to the vector store (embeddings will be generated).
        if index is None:
            index = VectorStoreIndex(
                nodes_to_index,  # child nodes only
                storage_context=storage_context,
                show_progress=True
            )
        else:
            index.insert_nodes(nodes_to_index)  # only insert child nodes to vector store
        
        # add parent nodes to docstore
        for p_node in nodes_to_docstore:
            index.docstore.add_documents([p_node])
        
        print(f"     > child nodes saved to vector store: {len(nodes_to_index)}")
        print(f"     > parent nodes saved to docstore: {len(nodes_to_docstore)}")
        
        print(f"   ✅ Batch {batch_num} Done!")

        # -----------------------------------
        # D. Cleanup
        # -----------------------------------
        print("   - [4/4] Cleaning up memory...")
        del documents, parent_nodes, nodes_to_index, dir_reader
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error in Batch {batch_num}: {e}")
        continue

# ==========================================
# Done
# ==========================================
storage_context.persist(persist_dir=DB_PATH) 
print(f"📂 mapping relationship saved to: {DB_PATH}")
print("\n" + "="*50)
print("🎉 All finished!")
print(f"💾 Data stored in: {os.path.abspath(DB_PATH)}")
print("="*50)

