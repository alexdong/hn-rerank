from typing import List, Optional
import numpy as np
import os
import json
import pathlib
import asyncio
import httpx
from pydantic import BaseModel, Field
from embedding import generate_embedding


# Constants needed for the function
ITEM_URL = "https://hacker-news.firebaseio.com/v0/item"
LOCAL_CACHE = "./cache"


class Post(BaseModel):
    id: int
    title: str
    url: str = ""  # Default empty string instead of Optional
    score: int
    embedding: np.ndarray  # Not Optional
    
    class Config:
        arbitrary_types_allowed = True  # To allow numpy arrays
        
    def dict(self, *args, **kwargs):
        """Override dict method to convert numpy array to list for JSON serialization."""
        d = super().dict(*args, **kwargs)
        d["embedding"] = self.embedding.tolist()
        return d
    
    def save_to_cache(self, cache_dir: pathlib.Path = pathlib.Path(LOCAL_CACHE)):
        """Save the post to a cache file."""
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{self.id}.json"
        
        # Convert to dict and ensure embedding is a list for JSON serialization
        json_data = self.dict()
        
        with open(cache_file, "w") as f:
            json.dump(json_data, f)
        
        return cache_file
    
    @staticmethod
    def from_cache(cache_file: pathlib.Path) -> Optional['Post']:
        """Load a Post object from a cache file."""
        if not cache_file.exists():
            return None
            
        with open(cache_file, "r") as f:
            post_data = json.load(f)
            
        if post_data.get('embedding') is not None and 'title' in post_data:
            return Post(
                id=post_data['id'],
                title=post_data['title'],
                url=post_data.get('url', ""),
                score=post_data.get('score', 0),
                embedding=np.array(post_data['embedding'], dtype=np.float32)
            )
        return None


async def fetch_post(client: httpx.AsyncClient, post_id: int) -> Optional[Post]:
    """Fetch details for a single post."""
    cache_dir = pathlib.Path(LOCAL_CACHE)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{post_id}.json"
    
    # Try to load from cache first
    post = Post.from_cache(cache_file)
    if post:
        print(f"[INFO] Loading post {post_id} from cache")
        return post
        
    # If not cached, fetch from API
    response = await client.get(f"{ITEM_URL}/{post_id}.json")
    response.raise_for_status()
    post_data = response.json()
    
    # Generate embedding for the post title if it has one
    if post_data and 'title' in post_data:
        post_data['embedding'] = generate_embedding(post_data['title'])
        print(f"[INFO] Generated embedding for post {post_id}")
        
        # Build a Post object from post_data
        post = Post(
            id=post_data['id'],
            title=post_data['title'],
            url=post_data.get('url', ""),
            score=post_data.get('score', 0),
            embedding=post_data['embedding']
        )
        
        # Save to cache
        post.save_to_cache(cache_dir)
        
        return post
    
    return None


if __name__ == "__main__":
    # Test fetch_post with a real post ID
    import sys
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    async def test_fetch():
        test_post_id = 43332658
        async with httpx.AsyncClient(timeout=30.0) as client:
            post = await fetch_post(client, test_post_id)
            if post:
                print(f"\nFetched post: {post}")
                print(f"Embedding shape: {post.embedding.shape}")
            else:
                print(f"Failed to fetch post {test_post_id}")
    
    asyncio.run(test_fetch())
