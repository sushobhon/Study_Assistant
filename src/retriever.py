import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import src.config as config
# import config

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
        return {"context": "", "sources": []}
    
    # Clean and merged all context in a single block
    context_segments = []
    sources = set()    # Using a set to prevent duplicate file names
    for i, doc in enumerate(matched_docs, start= 1):
        clean_text = doc.page_content.strip()
        context_segments.append(f"[Document Snippet {i}]:\n{clean_text}")

        # Extract the source file path from the metadata
        file_path = doc.metadata.get('source', 'Unknown Source')
        file_name = os.path.basename(file_path)

        # Extracting page number of the document.
        page_num = doc.metadata.get('page')

        # Only for PDF we will have page for other sources we will not have page
        # if page number is not present then extracting chunk number
        if page_num is not None:
            display_source = f"{file_name} (Page {page_num + 1})"
        else:
            display_source = file_name
        
        sources.add(display_source)
    
    # Joining with new line
    return {
        "context": "\n\n".join(context_segments),
        'sources': sorted(list(sources))
    }

# if __name__ == "__main__":
#     query = "History of Language model"
#     res = get_retrival_context(
#         query= query,
#         k = 3
#     )
#     print(res)