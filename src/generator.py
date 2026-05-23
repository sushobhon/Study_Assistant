from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import src.config as config

def generate_answer(query: str, context: str) -> str:
    """Combine Query text and Retrived Context as Chat Template.
    Then use LLM model to create Grounded context.

    Args:
        query (str): User Query
        context (str): Retrived Context from Vector Database

    Returns:
        str: Returns a Grounded answer using LLM model
    """
    # 1. initiallize the LLM model
    llm = OllamaLLM(model= config.LLM_MODEL)

    # 2. Define a clean defensive system prompt
    prompt_template = ChatPromptTemplate.from_messages([
        (
            'system',
            """You are a helpful assistant. Answer the user's question using ONLY the provided text snippets.
            If the answer cannot be found in the text, say 'I cannot find the answer in the provided documents.'
            Do not make things up.\n\n
            Provided Text Snippets:\n{context}
            """
        ),
        (
            'human', '{query}'
        )
    ])

    # 3. Chain the prompt template with LLM model
    chain = prompt_template | llm

    # 4. Invoke Chain to execute information
    print(f"🤖 Invoking local '{config.LLM_MODEL}' model via Ollama...")
    response = chain.invoke(
        {
            'context': context,
            'query': query
        }
    )

    return response

# if __name__ == "__main__":

#     from retriever import get_retrival_context
#     query = "Explain Tokenization method in details with example"
#     context = get_retrival_context(query= query, k= 3)
#     result = generate_answer(
#         query= query,
#         context= context
#     )
#     print("+"*60)
#     print(result)