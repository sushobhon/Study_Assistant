import sys
from src.retriever import get_retrival_context
from src.generator import generate_answer
import src.config as config

def main():
    print("🚀 Starting Phase 1 Gemma-RAG Pipeline...")

    while True:
        # 1. Accept User Query
        user_query = input("\nAsk your question about the document:\n")
        if not user_query.strip():
            print("Empty Query. Exiting ...\n")
            break

        try:
            # 2. Retrive Relevent chunks
            context = get_retrival_context(
                query= user_query,
                k= config.NUMBER_OF_CHUNKS_TO_RETURN
            )

            if not context:
                print("⚠️ Warning: No matching documents found in the database. Proceeding with empty context.")
            
            # 3. Generate Responce using LLM
            answer = generate_answer(
                query= user_query,
                context= context
            )

            # 4. Display the clean output
            print("\n✨ LLM's Context:")
            print("=" * 60)
            print(context)
            print("=" * 60)
            print("\n✨ LLM's Answer:")
            print("=" * 60)
            print(answer)
            print("=" * 60)

        except Exception as e:
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()