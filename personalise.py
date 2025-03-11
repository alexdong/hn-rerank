import json
import openai
from typing import Dict, Any

PROMPT = """
Analyze the following text and extract key concepts. Return a JSON where each
concept is assigned a weight (0-1) based **solely on its rarity on Hacker
News**, with 1.0 = rarest and 0.01 = most common. Normalize linearly.  

**Example Input**:  
"I am a theoretical biologist, interested in disease ecology. My tools are R,
Clojure, compartmentalism disease modeling, and statistical GAM models, using
data layers like geophysical, climate, biodiversity, and land use. I also work
on tech applied to agriculture, conservation, forecasting, third-world
development, AI, and large language models."  

**Example Output**:  
{
    "theoretical biology": 0.95,  
    "compartmentalism disease modeling": 0.85,  
    "statistical GAM_models": 0.80,  
    "geophysical data layers": 0.70,  
    "biodiversity data layers": 0.65,  
    "tech applied to conservation": 0.55,  
    "third world development": 0.35,  
    "AI applications": 0.25,  
    "large language models": 0.15
}  
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
            {"role": "system", "content": "You extract key concepts from text and assign weights based on rarity."},
            {"role": "user", "content": f"{PROMPT}\n\nInput:\n{text}"}
        ],
        temperature=0.2,
    )
    
    # Extract the JSON from the response
    try:
        concepts_text = response.choices[0].message.content
        # Handle case where the response might include markdown code blocks or explanatory text
        if "```json" in concepts_text:
            json_part = concepts_text.split("```json")[1].split("```")[0].strip()
            concepts = json.loads(json_part)
        elif "```" in concepts_text:
            json_part = concepts_text.split("```")[1].split("```")[0].strip()
            concepts = json.loads(json_part)
        else:
            # Try to find and parse the JSON part
            concepts = json.loads(concepts_text)
    except (json.JSONDecodeError, IndexError):
        print(f"[ERROR] Failed to parse concepts from response: {concepts_text}")
        concepts = {}
    
    print(f"[INFO] Extracted {len(concepts)} key concepts from user text")
    return concepts

if __name__ == "__main__":
    # Test the function
    import asyncio
    import os
    
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        exit(1)
    
    async def test_extract():
        test_text = "I am a theoretical biologist, interested in disease ecology. My tools are R, Clojure, compartmentalism disease modeling, and statistical GAM models."
        concepts = await extract_key_concepts(test_text)
        print(json.dumps(concepts, indent=2))
    
    asyncio.run(test_extract())
