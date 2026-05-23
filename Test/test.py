import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Get the path to the root directory (one level up from this file)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. Add the root directory to the Python search path if it isn't there
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 3. Now you can safely import your src modules!
import src.config as config

# Testing Database
def test_database():
    # 1. Check if Database exists or not
    if not os.path.exists(config.DB_DIR):
        print(f"❌ Error: Database directory '{config.DB_DIR}' not found. Please run ingest.py first.")
        return
    
    print("🧠 Connecting to existing vector database...")
    # Configure computing device dynamically (prefer CUDA/GPU if available, else CPU)
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or False else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} # Essential for clean cosine/L2 boundaries
    )

    # Load the Persisting Data base
    db = Chroma(
        persist_directory= config.DB_DIR,
        embedding_function= embeddings
    )

    # Define test query
    test_query = "History of Language model"
    print(f"\n🔍 Querying database for: '{test_query}'\n")

    try:
        # Retrive 3 most similar Documents
        results = db.similarity_search_with_score(
            query= test_query,
            k= 3
        )

        if not results:
            print("⚠️ No matching chunks found. The database might be empty.")
            return
        
        print(f"🎉 Successfully retrieved {len(results)} matching chunks:\n")
        print("=" * 60)

        for i, (doc, score) in enumerate(results, start= 1):
            # print(doc.metadata)
            source_file = doc.metadata.get('source', 'Unknown Source')
            print(f"\n📄 [Match #{i}] | Source: {os.path.basename(source_file)}")
            print(f"📐 Distance Score: {score:.4f} (Lower = closer match)")
            print("-" * 60)
            print(doc.page_content.strip())
            print("=" * 60)
    except Exception as e:
        print(f"❌ An error occurred while searching: {e}")

if __name__ == "__main__":
    test_database()