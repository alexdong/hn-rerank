import os
import openai
import numpy as np

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a given text using OpenAI API."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    # Convert the list to a numpy array
    return np.array(embedding_response.data[0].embedding, dtype=np.float32)

if __name__ == "__main__":
    # Test the embedding generation
    import sys
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    test_text = "This is a test sentence for embedding generation"
    embedding = generate_embedding(test_text)
    
    print(f"Generated embedding for: '{test_text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding type: {type(embedding)}")
