import numpy as np
from typing import List, Tuple, Dict, Any
from post import Post

def rank_posts(posts: List[Post], weighted_embeddings: List[Tuple[np.ndarray, float]]) -> List[Post]:
    """
    Ranks posts based on similarity to user interests using weighted embeddings.
    
    Args:
        posts: List of Post objects to rank
        weighted_embeddings: List of (embedding, weight) tuples representing user interests
        
    Returns:
        List of Post objects sorted by relevance to user interests
    """
    if not posts or not weighted_embeddings:
        print("[WARNING] No posts or weighted embeddings provided for ranking")
        return posts
    
    # Calculate relevance scores for each post
    scored_posts = []
    
    for post in posts:
        # Skip posts without embeddings
        if post.embedding is None or not any(post.embedding):
            continue
            
        # Calculate similarity to each weighted concept
        total_score = 0.0
        total_weight = 0.0
        
        for concept_embedding, weight in weighted_embeddings:
            # Calculate cosine similarity between post and concept
            similarity = cosine_similarity(post.embedding, concept_embedding)
            
            # Apply weight to similarity
            total_score += similarity * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
            
        # Combine with original HN score for balanced ranking
        # Convert HN score to 0-1 range (assuming max score around 1000)
        hn_score_normalized = min(post.score / 1000.0, 1.0) if post.score else 0.0
        
        # Weighted combination (0.7 for relevance, 0.3 for original ranking)
        combined_score = (0.7 * final_score) + (0.3 * hn_score_normalized)
        
        scored_posts.append((post, combined_score))
    
    print(f"[INFO] Calculated relevance scores for {len(scored_posts)} posts")
    
    # Sort posts by combined score in descending order
    ranked_posts = [post for post, score in sorted(scored_posts, key=lambda x: x[1], reverse=True)]
    
    return ranked_posts


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (between -1 and 1)
    """
    # Ensure vectors are normalized for proper cosine similarity
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


if __name__ == "__main__":
    # Test the ranking function
    import asyncio
    from fetch import collect_hn_data
    from traits import extract_key_concepts, get_weighted_embeddings
    
    async def test_ranking():
        # Get sample posts
        posts = await collect_hn_data()
        
        # Get sample user interests
        test_bio = "I'm a software engineer interested in AI, machine learning, and distributed systems."
        concepts = await extract_key_concepts(test_bio)
        weighted_embeddings = await get_weighted_embeddings(concepts)
        
        # Rank posts
        ranked_posts = rank_posts(posts, weighted_embeddings)
        
        # Print top 10 ranked posts
        print("\n[INFO] Top 10 ranked posts:")
        for i, post in enumerate(ranked_posts[:10]):
            print(f"{i+1}. {post.title} (Score: {post.score})")
    
    asyncio.run(test_ranking())
