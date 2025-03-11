import json
import openai
from typing import Dict, Any

SYSTEM_PROMPT = """
You are a specialized content analyzer for a Hacker News personalization system. Your task is to identify technical interests and domain expertise from user bios, then assign rarity weights to help rank content. You understand the Hacker News ecosystem well - which topics are common (programming, startups, AI) versus rare (specialized scientific domains, niche technologies). Extract concepts a
 the right granularity level that would match with article titles and topics. Be precise, consistent, and focus on technical/professional interests rather than personal traits."""


USER_PROMPT = """
 Analyze the following user bio text and extract up to 25 areas the user might be interested to read on Hacker News. Return a JSON object where:

 1. Each key is a distinct concept or interest area (e.g., "machine learning",
 "rust programming", "quantum computing")
 2. Each value is a weight from 0.0 to 1.0 representing how rare this concept
 is on Hacker News. 1.0 is the rarest and 0.1 is the most common. Make sure the
 weights are distributed linearly across the range.

 Guidelines:
 - Use consistent terminology that would appear in article titles
 - Extract concepts at an appropriate level of granularity (not too broad, not too specific)
 - If the bio is very short, extract fewer concepts but maintain quality
 - If the bio is long, extract more concepts but ensure they are distinct
 - Ensure weights are evenly distributed across the 0-1 range

 **Example Input**:
 "I am a theoretical biologist, interested in disease ecology. My tools are R, Clojure, compartmentalism disease modeling, and statistical GAM models, using data layers like geophysical, climate, biodiversity, and land use. I also work on tech applied to agriculture, conservation, forecasting, third-world development, AI, and large language models."

 **Example Output**:
 {
   "theoretical biology": 0.96,
   "disease ecology": 0.81,
   "compartmental disease modeling": 0.94,
   "statistical GAM models": 0.89,
   "geospatial data analysis": 0.72,
   "R programming": 0.47,
   "Clojure": 0.68,
   "agricultural technology": 0.55,
   "conservation technology": 0.62,
   "forecasting systems": 0.52,
   "third-world development": 0.79,
   "large language models": 0.29
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{USER_PROMPT}\n\nInput:\n{text}"}
        ],
        temperature=0.2, # Low temperature for more deterministic results
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
        # print out the error, ai!
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
        test_text = "I'm an AI scientist/engineer. I love to tinker with technology. My preferred language is Python. I'm interested in tracking latest development in AI research and its applications. I am also interested in economics, geopolitics, gardening, cooking and design."  
        concepts = await extract_key_concepts(test_text)
        print(json.dumps(concepts, indent=2))
    
    asyncio.run(test_extract())
