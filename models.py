from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field


class Post(BaseModel):
    id: int
    title: str
    url: Optional[str] = None
    score: int
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True  # To allow numpy arrays


if __name__ == "__main__":
    # Test the Post dataclass
    post = Post(id=123, title="Test Post", url="https://example.com", score=100)
    print(f"Created post: {post}")
    
    # Test with embedding
    post.embedding = np.array([0.1, 0.2, 0.3])
    print(f"Post with embedding: {post}")
