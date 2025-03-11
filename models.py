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

# rename the function to fetch_post and it returns Post, ai!
async def fetch_post(client: httpx.AsyncClient, post_id: int) -> Optional[Post]:
    """Fetch details for a single post."""
    # Check if the post is already cached
    cache_dir = pathlib.Path(LOCAL_CACHE)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{post_id}.json"
    
    if cache_file.exists():
        with open(cache_file, "r") as f:
            print(f"[INFO] Loading post {post_id} from cache")
            post_data = json.load(f)
            # Convert embedding from list to numpy array
            if post_data.get('embedding'):
                post_data['embedding'] = np.array(post_data['embedding'], dtype=np.float32)
    else:
        # If not cached, fetch from API
        response = await client.get(f"{ITEM_URL}/{post_id}.json")
        response.raise_for_status()
        
        post_data = response.json()
        
        # Generate embedding for the post title if it has one
        if post_data and 'title' in post_data:
            post_data['embedding'] = generate_embedding(post_data['title'])
            print(f"[INFO] Generated embedding for post {post_id}")
        
        # Cache the response to a file - convert numpy array to list for JSON serialization
        json_data = post_data.copy()
        if json_data.get('embedding'):
            json_data['embedding'] = json_data['embedding'].tolist()
            
        with open(cache_file, "w") as f:
            json.dump(json_data, f)
    
    # Convert to Post object
    if post_data and 'title' in post_data and post_data.get('embedding'):
        return Post(
            id=post_data['id'],
            title=post_data['title'],
            url=post_data.get('url', ""),  # Default to empty string
            score=post_data.get('score', 0),
            embedding=post_data['embedding']
        )
    return None



if __name__ == "__main__":
    # Test the Post dataclass
    post = Post(id=123, title="Test Post", url="https://example.com", score=100, 
                embedding=np.array([0.1, 0.2, 0.3]))
    print(f"Created post: {post}")
    
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
