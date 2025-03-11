from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Post:
    id: int
    title: str
    url: Optional[str]
    score: int
    embedding: Optional[np.ndarray] = None


if __name__ == "__main__":
    # Test the Post dataclass
    post = Post(id=123, title="Test Post", url="https://example.com", score=100)
    print(f"Created post: {post}")
    
    # Test with embedding
    post.embedding = np.array([0.1, 0.2, 0.3])
    print(f"Post with embedding: {post}")