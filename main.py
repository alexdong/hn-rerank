import asyncio
import os
from data_collection import collect_hn_data
from models import Post
from typing import List


async def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        return
        
    posts = await collect_hn_data()
    print(f"[INFO] Collected {len(posts)} posts with embeddings")
    
    # Example: Print top 5 posts by score
    top_posts = sorted(posts, key=lambda p: p.score, reverse=True)[:5]
    print("\nTop 5 posts by HN score:")
    for i, post in enumerate(top_posts, 1):
        print(f"{i}. {post.title} (Score: {post.score}) - {post.url}")


if __name__ == "__main__":
    asyncio.run(main())