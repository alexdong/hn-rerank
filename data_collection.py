import os
import time
import asyncio
import httpx
import numpy as np
import json
import pathlib
from typing import List, Dict, Any, Optional
from posts import Post, fetch_post


HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
TOP_STORIES_URL = f"{HN_API_BASE}/topstories.json"
ITEM_URL = f"{HN_API_BASE}/item"
BATCH_SIZE = 20  # Number of concurrent requests
LOCAL_CACHE = "./cache"


async def fetch_top_story_ids() -> List[int]:
    """Fetch IDs of top stories from Hacker News API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(TOP_STORIES_URL)
        response.raise_for_status()
        return response.json()


async def fetch_all_posts(story_ids: List[int]) -> List[Post]:
    """Fetch details for all top stories in batches."""
    posts = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(story_ids), BATCH_SIZE):
            batch_ids = story_ids[i:i+BATCH_SIZE]
            batch_tasks = [fetch_post(client, post_id) for post_id in batch_ids]
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Filter out None results
            posts.extend([post for post in batch_results if post])
            
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
    print(f"[INFO] Posts with embeddings: {len(posts)}")
    
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
        print(f"Embedding shape: {posts[0].embedding.shape}")
        print(f"\nTotal posts collected: {len(posts)}")
