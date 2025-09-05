# chatbot.py

import os
from dotenv import load_dotenv
from retrieval.hybrid_search import HybridSearch
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(model_name='gemini-2.5-flash')



# Initialize HybridSearch
hybrid_search = HybridSearch()

def chatbot(query: str, collection: str = "unified_collection", top_k: int = 5):
    """
    Chatbot pipeline:
    1. Run hybrid search
    2. Format retrieved data as context
    3. Send query + context to Gemini LLM
    4. Return response
    """

    # Step 1: Retrieve from Qdrant
    results = hybrid_search.search(
        query=query,
        collection_name=collection,
        filters={},
        top_k=top_k,
        result_type="hybrid_dense"
    )

    # Step 2: Process results
    context_docs = hybrid_search.process_results(results)
    
    context = "\n\n".join(context_docs)

    # Step 3: Prepare prompt for Gemini
    prompt = f"""
        You are a helpful assistant with access to retrieved knowledge.
        User query: {query}

        Relevant context from documents:
        {context}

        Answer the query concisely and accurately using the context. If the context is not enough, say you are unsure.
        """

    # Step 4: Send to Gemini
    response = model.generate_content(
        contents=prompt,
        stream=True
    )

    for chunk in response:
        print(chunk.text, end='', flush=True)

    return response.text

if __name__ == "__main__":
    print("🤖 Chatbot ready! Type your questions below.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        
        answer = chatbot(query)
        print(f"AI: {answer}")