import os
import time
import asyncio
import httpx
import numpy as np
import json
import pathlib
from typing import List, Dict, Any, Optional
import openai
from models import Post


HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
TOP_STORIES_URL = f"{HN_API_BASE}/topstories.json"
ITEM_URL = f"{HN_API_BASE}/item"
MAX_POSTS = 500
BATCH_SIZE = 20  # Number of concurrent requests
LOCAL_CACHE = "./cache"


async def fetch_top_story_ids() -> List[int]:
    """Fetch IDs of top stories from Hacker News API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(TOP_STORIES_URL)
        response.raise_for_status()
        return response.json()[:MAX_POSTS]


# Move this function to models.py and update its __main__ for a real test with postid 43332658, ai!
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


async def fetch_all_posts(story_ids: List[int]) -> List[Post]:
    """Fetch details for all top stories in batches."""
    posts = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(story_ids), BATCH_SIZE):
            batch_ids = story_ids[i:i+BATCH_SIZE]
            batch_tasks = [fetch_post_details(client, post_id) for post_id in batch_ids]
            batch_results = await asyncio.gather(*batch_tasks)
            
            for result in batch_results:
                if result and 'title' in result:
                    post = Post(
                        id=result['id'],
                        title=result['title'],
                        url=result.get('url'),
                        score=result.get('score', 0),
                        embedding=result.get('embedding')
                    )
                    posts.append(post)
            
            print(f"[INFO] Fetched {len(posts)}/{len(story_ids)} posts")
            
    return posts


async def collect_hn_data() -> List[Post]:
    """Main function to collect HN data and generate embeddings."""
    print("[INFO] Starting HN data collection")
    
    # Fetch top story IDs
    story_ids = await fetch_top_story_ids()
    print(f"[INFO] Fetched {len(story_ids)} top story IDs")
    
    # Fetch post details
    posts = await fetch_all_posts(story_ids)
    print(f"[INFO] Fetched details for {len(posts)} posts")
    
    # Embeddings are already generated during post fetching
    print(f"[INFO] Posts with embeddings: {sum(1 for p in posts if p.embedding is not None)}")
    
    return posts


if __name__ == "__main__":
    import sys
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        sys.exit(1)
        
    posts = asyncio.run(collect_hn_data())
    
    # Print some sample data to verify
    if posts:
        print(f"\nSample post: {posts[0]}")
        print(f"Embedding shape: {posts[0].embedding.shape if posts[0].embedding is not None else None}")
        print(f"\nTotal posts collected: {len(posts)}")
