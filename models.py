from typing import List, Optional, Dict, Any
import numpy as np
import os
import json
import pathlib
import asyncio
import httpx
import openai
from pydantic import BaseModel, Field


# Constants needed for the function
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item"
LOCAL_CACHE = "./cache"


class Post(BaseModel):
    id: int
    title: str
    url: Optional[str] = None
    score: int
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True  # To allow numpy arrays

# rename the function to fetch_post and it returns Post, ai!
async def fetch_post_details(client: httpx.AsyncClient, post_id: int) -> Optional[Dict[str, Any]]:
    """Fetch details for a single post."""
    try:
        # Check if the post is already cached
        cache_dir = pathlib.Path(LOCAL_CACHE)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{post_id}.json"
        
        if cache_file.exists():
            with open(cache_file, "r") as f:
                print(f"[INFO] Loading post {post_id} from cache")
                return json.load(f)
        
        # If not cached, fetch from API
        response = await client.get(f"{ITEM_URL}/{post_id}.json")
        response.raise_for_status()
        
        post_data = response.json()
        
        # Generate embedding for the post title if it has one
        if post_data and 'title' in post_data:
            # Move the following into a separate function in a new file `embedding.py`, ai!
            try:
                client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                embedding_response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[post_data['title']]
                )
                post_data['embedding'] = embedding_response.data[0].embedding
                print(f"[INFO] Generated embedding for post {post_id}")
            except Exception as e:
                print(f"[ERROR] Failed to generate embedding for post {post_id}: {e}")
                post_data['embedding'] = None
        
        # Cache the response to a file
        with open(cache_file, "w") as f:
            json.dump(post_data, f)
            
        return post_data
    except (httpx.HTTPError, httpx.ReadTimeout) as e:
        print(f"[ERROR] Failed to fetch post {post_id}: {e}")
        return None



if __name__ == "__main__":
    # Test the Post dataclass
    post = Post(id=123, title="Test Post", url="https://example.com", score=100)
    print(f"Created post: {post}")
    
    # Test with embedding
    post.embedding = np.array([0.1, 0.2, 0.3])
    print(f"Post with embedding: {post}")
    
    # Test fetch_post_details with a real post ID
    import sys
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    async def test_fetch():
        test_post_id = 43332658
        async with httpx.AsyncClient(timeout=30.0) as client:
            post_data = await fetch_post_details(client, test_post_id)
            if post_data:
                post = Post(
                    id=post_data['id'],
                    title=post_data['title'],
                    url=post_data.get('url'),
                    score=post_data.get('score', 0),
                    embedding=post_data.get('embedding')
                )
                print(f"\nFetched post: {post}")
                print(f"Embedding length: {len(post.embedding) if post.embedding else None}")
            else:
                print(f"Failed to fetch post {test_post_id}")
    
    asyncio.run(test_fetch())
