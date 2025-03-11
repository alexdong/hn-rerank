from typing import List, Optional
import numpy as np
import os
import json
import pathlib
import asyncio
import httpx
from pydantic import BaseModel, Field

from config import ITEM_URL, LOCAL_CACHE
from embedding import generate_embedding


# Constants needed for the function
cache_dir: pathlib.Path = pathlib.Path(LOCAL_CACHE)
cache_dir.mkdir(exist_ok=True)


class Post(BaseModel):
    id: int
    title: str
    url: Optional[str] = None
    score: int
    embedding: np.ndarray 
    
    class Config:
        arbitrary_types_allowed = True  # To allow numpy arrays
        
    def dict(self, *args, **kwargs):
        """Override dict method to convert numpy array to list for JSON serialization."""
        d = super().dict(*args, **kwargs)
        d["embedding"] = self.embedding.tolist()
        return d
    
    def save_to_cache(self):
        """Save the post to a cache file."""
        cache_file = cache_dir / f"{self.id}.json"
        
        # Convert to dict and ensure embedding is a list for JSON serialization
        json_data = self.dict()
        
        with open(cache_file, "w") as f:
            json.dump(json_data, f)
    
    @staticmethod
    def from_cache(cache_file: pathlib.Path) -> Optional['Post']:
        """Load a Post object from a cache file."""
        if not cache_file.exists(): return None
            
        with open(cache_file, "r") as f:
            post_data = json.load(f)
            post_data['embedding'] = np.array(post_data['embedding'], dtype=np.float32)
            return Post(**post_data)


async def fetch_post(client: httpx.AsyncClient, post_id: int) -> Post:
    """Fetch details for a single post."""
    # Try to load from cache first
    cache_file = cache_dir / f"{post_id}.json"
    post = Post.from_cache(cache_file)
    if post:
        print(f"[INFO] Loading post {post_id} from cache")
        return post
        
    # If not cached, fetch from API
    response = await client.get(f"{ITEM_URL}/{post_id}.json")
    response.raise_for_status()
    post_data = response.json()

    post_data['embedding'] = generate_embedding(post_data['title'])
    print(f"[INFO] Generated embedding for post {post_id}: {post_data}")

    post = Post(**post_data)
    post.save_to_cache()
    return post


if __name__ == "__main__":
    # Test fetch_post with a real post ID
    import sys
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    async def test_fetch():
        test_post_id = 43334644 # 43332658
        async with httpx.AsyncClient(timeout=30.0) as client:
            post = await fetch_post(client, test_post_id)
            if post:
                print(f"\nFetched post: {post}")
                print(f"Embedding shape: {post.embedding.shape}")
            else:
                print(f"Failed to fetch post {test_post_id}")
    
    asyncio.run(test_fetch())
