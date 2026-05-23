import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as config

def get_retrival_context(query: str, k: int):
    """Connects to Chroma DB, retrive similar documents and retrun a combined
    single context string  

    Args:
        query (str): Query string
        k (int): How many similar query to pull up
    """
    # Check if Vector DB exists or not
    if not os.path.exists(config.DB_DIR):
        print(f"❌ Vector database not found at '{config.DB_DIR}'. Run ingest.py first!")
        return
    
    # Initiallizing Same Embedding model
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or False else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Loading Existing Chroma Vector Data Base
    vector_db = Chroma(
        persist_directory= config.DB_DIR,
        embedding_function= embeddings
    )

    # Perfrom a Similarity Search
    matched_docs = vector_db.similarity_search(
        query= query,
        k= k
    )

    if not matched_docs:
        return ""
    
    # Clean and merged all context in a single block
    context_segments = []
    for i, doc in enumerate(matched_docs, start= 1):
        clean_text = doc.page_content.strip()
        context_segments.append(f"[Document Snippet {i}]:\n{clean_text}")
    
    # Joining with new line
    return "\n\n".join(context_segments)

# if __name__ == "__main__":
#     query = "History of Language model"
#     res = get_retrival_context(
#         query= query,
#         k = 3
#     )
#     print(res)