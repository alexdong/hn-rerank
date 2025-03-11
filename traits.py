import json
import openai
import numpy as np
from typing import Dict, Any, List, Tuple

SYSTEM_PROMPT = """
You are a specialized content analyzer for a Hacker News personalization system. Your task is to identify technical interests and domain expertise from user bios, then assign rarity weights to help rank content. You understand the Hacker News ecosystem well - which topics are common (programming, startups, AI) versus rare (specialized scientific domains, niche technologies). Extract concepts a
 the right granularity level that would match with article titles and topics. Be precise, consistent, and focus on technical/professional interests rather than personal traits."""


USER_PROMPT = """
 Analyze the following user bio text and extract up to 25 areas the user might be interested to read on Hacker News. Return a CSV file where each line contains a concept and its rarity weight, separated by a comma. 

 1. The first value in each row is a distinct concept or interest area (e.g.,
     "machine learning", "rust programming", "quantum computing")
 2. The second value is an estimated rarity score from 0.01 to 1.00,
     representing how rare this concept is general internet content. 1.0 is the
     rarest and 0.01 is the most common. Make sure the weights are distributed
     linearly across the range.

 Guidelines:
 - Prefer more interest areas over fewer, even if the weights are lower
 - Use consistent terminology that would appear in article titles
 - Extract concepts at an appropriate level of granularity (not too broad, not too specific)
 - If the bio is very short, extract fewer concepts but maintain quality
 - If the bio is long, extract more concepts but ensure they are distinct
 - Ensure weights are evenly distributed across the 0-1 range

 **Example User Bio**:
I am a theoretical biologist, interested in disease ecology. My tools are R, Clojure, compartmentalism disease modeling, and statistical GAM models, using data layers like geophysical, climate, biodiversity, and land use. I also work on tech applied to agriculture, conservation, forecasting, third-world development, AI, and large language models.

 **Example Output**:
theoretical biology,0.96
disease ecology,0.81
compartmental disease modeling,0.94
statistical GAM models,0.89
geospatial data analysis,0.72
R programming,0.47
Clojure,0.68
agricultural technology,0.55
conservation technology,0.62
forecasting systems,0.52
third-world development,0.79
large language models,0.29
"""

async def extract_key_concepts(text: str) -> Dict[str, float]:
    """
    Takes a user's bio text and extracts key concepts with their weights.
    
    Args:
        text: The user's bio or preference text
        
    Returns:
        A dictionary mapping concepts to their weights (0-1)
    """
    client = openai.AsyncOpenAI()
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PROMPT}\n\nInput:\n{text}"}
        ],
        temperature=0.2, # Low temperature for more deterministic results
    )
    
    # Extract the concepts from the CSV response
    try:
        concepts_text = response.choices[0].message.content
        concepts = {}
        
        # Handle case where the response might include markdown code blocks
        if "```" in concepts_text:
            # Extract content from code block (whether it's labeled as csv or not)
            csv_part = concepts_text.split("```")[1].split("```")[0].strip()
            lines = csv_part.strip().split('\n')
        else:
            # Use the whole response
            lines = concepts_text.strip().split('\n')
        
        # Process each line as CSV (concept,weight)
        for line in lines:
            if ',' in line:
                concept, weight_str = line.split(',', 1)
                concept = concept.strip()
                try:
                    weight = float(weight_str.strip())
                    concepts[concept] = weight
                except ValueError:
                    print(f"[WARNING] Could not parse weight from line: {line}")
                    
    except (IndexError, ValueError) as e:
        print(f"[ERROR] Failed to parse concepts from response: {concepts_text}")
        print(f"[ERROR] Exception details: {str(e)}")
        concepts = {}
    
    print(f"[INFO] Extracted {len(concepts)} key concepts from user text")
    return concepts


async def get_weighted_embeddings(concepts: Dict[str, float]) -> List[Tuple[np.ndarray, float]]:
    """
    Takes a dictionary of concepts and their weights, calculates embeddings for each concept,
    and returns a list of (embedding, weight) tuples.
    
    Args:
        concepts: A dictionary mapping concepts to their weights (0-1)
        
    Returns:
        A list of tuples containing (embedding, weight)
    """
    from embedding import generate_embedding
    
    weighted_embeddings = []
    
    for concept, weight in concepts.items():
        embedding = generate_embedding(concept)
        weighted_embeddings.append((embedding, weight))
    
    print(f"[INFO] Generated {len(weighted_embeddings)} weighted embeddings")
    return weighted_embeddings


if __name__ == "__main__":
    # Test the function
    import asyncio
    import os
    
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        exit(1)
    
    async def test_extract():
        test_text = "I'm an AI scientist/engineer. I love to tinker with technology. My preferred language is Python. I'm interested in tracking latest development in AI research and its applications. I am also interested in economics, geopolitics, gardening, cooking and design."  
        concepts = await extract_key_concepts(test_text)
        print(json.dumps(concepts, indent=2))
        
        # Test get_weighted_embeddings
        weighted_embeddings = await get_weighted_embeddings(concepts)
        print(f"\n[INFO] Sample of weighted embeddings:")
        # Print first embedding with limited dimensions for readability
        if weighted_embeddings:
            first_concept = list(concepts.keys())[0]
            first_embedding, first_weight = weighted_embeddings[0]
            print(f"Concept: {first_concept}")
            print(f"Weight: {first_weight}")
            print(f"Embedding (first 5 dimensions): {first_embedding[:5]}")
            print(f"Embedding shape: {first_embedding.shape}")
    
    asyncio.run(test_extract())
