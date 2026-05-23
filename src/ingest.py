import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config

def get_loader(file_path: str):
    """Detects file extention and return appropriate loader

    Args:
        file_path (str): Document file path
    """
    # extracting extention
    _,ext = os.path.splitext(file_path.lower())

    # Returning Data Loader based on file extention
    if ext == ".txt":
        return TextLoader(file_path= file_path, encoding= 'UTF-8')
    elif ext == ".md":
        return UnstructuredMarkdownLoader(file_path= file_path)
    elif ext == ".pdf":
        return PyPDFLoader(file_path= file_path)
    else:
        print(f"⚠️ Skipping unsupported file type: {file_path}")
        return None

def main():
    # 1. Gather all documents from the document folder
    if not os.path.exists(config.DOC_DIR):
        print(f"❌ Error: The directory '{config.DOC_DIR}' does not exist. Create it and add files.")
        return
    
    raw_documents = []
    print(f"📂 Scanning for documents in: {config.DOC_DIR}...")

    for root, _, files in os.walk(config.DOC_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            loader = get_loader(file_path= file_path)

            if loader:
                try:
                    print(f"📄 Loading: {file}")
                    raw_documents.extend(loader.load())
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
    
    if not raw_documents:
        print(f"🛑 No valid text documents found to ingest.")
        return
    
    # 2. Split the data into chunks based on the specification
    print(f"✂️ Splitting {len(raw_documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= config.CHUNK_SIZE,
        chunk_overlap= config.CHUNK_OVERLAP,
        length_function= len,
        is_separator_regex= False
    )

    chunks = text_splitter.split_documents(raw_documents)
    # Add this print check to debug
    if len(chunks) == 0:
        print("🛑 Stop! No text chunks were generated. Check if your documents contain readable text.")
        return
    print(f"✅ Created {len(chunks)} total text chunks.")

    # 3. initialize Ollama Embedding and Create Vector Database
    print(f"🧠 Initializing embedding model: '{config.EMBED_MODEL}' via Ollama...")
    embeddings = OllamaEmbeddings(model= config.EMBED_MODEL)

    print(f"💾 Creating Chroma DB and saving embeddings to local directory: '{config.DB_DIR}'...")

    # Chroma Persist automatically to the specified directory
    vector_db = Chroma.from_documents(
        documents= chunks,
        embedding= embeddings,
        persist_directory= config.DB_DIR
    )

    print(f"🎉 Ingestion Complete! Vector database safely stored at '{config.DB_DIR}'.")

if __name__ == "__main__":
    main()