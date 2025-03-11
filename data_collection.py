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


async def fetch_post_details(client: httpx.AsyncClient, post_id: int) -> Optional[Dict[str, Any]]:
    """Fetch details for a single post."""
    try:
        response = await client.get(f"{ITEM_URL}/{post_id}.json")
        response.raise_for_status()
        
        # Cache the response to a file
        cache_dir = pathlib.Path(LOCAL_CACHE)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"{post_id}.json"
        with open(cache_file, "w") as f:
            json.dump(response.json(), f)
            
        return response.json()
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
                    )
                    posts.append(post)
            
            print(f"[INFO] Fetched {len(posts)}/{len(story_ids)} posts")
            
    return posts


async def generate_embeddings(posts: List[Post]) -> List[Post]:
    """Generate embeddings for post titles using OpenAI API."""
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Process in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i+batch_size]
        titles = [post.title for post in batch]
        
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=titles
            )
            
            for j, post in enumerate(batch):
                post.embedding = np.array(response.data[j].embedding)
                
            print(f"[INFO] Generated embeddings for posts {i+1}-{min(i+batch_size, len(posts))}")
            
            # Sleep to avoid rate limits
            if i + batch_size < len(posts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[ERROR] Failed to generate embeddings for batch {i//batch_size+1}: {e}")
    
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
    
    # Generate embeddings
    posts_with_embeddings = await generate_embeddings(posts)
    print(f"[INFO] Generated embeddings for {len(posts_with_embeddings)} posts")
    
    return posts_with_embeddings


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
