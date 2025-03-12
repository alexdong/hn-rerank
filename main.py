from fasthtml.common import *
import asyncio
import os
from typing import List, Optional

# Import our existing modules
from fetch import collect_hn_data
from traits import extract_traits
from ranking import rank_posts
from post import Post

# Initialize FastHTML app with WebSocket support
app, rt = fast_app(exts='ws')

# Global state to store posts
posts: List[Post] = []

# Main route for the application
@rt('/')
def get():
    return Titled("HN Re-ranking",
        # Main container with split view
        Div(
            # Left side: Bio input
            Div(
                H2("Your Interests"),
                P("Enter your bio to personalize HN posts based on your interests:"),
                Form(
                    Textarea(
                        id="bio", 
                        name="bio", 
                        placeholder="I am a theoretical biologist, interested in disease ecology...",
                        rows=10
                    ),
                    Div(
                        Button("Submit", type="submit", id="submit-btn"),
                        Button("Refresh HN Data", id="refresh-btn", type="button"),
                        cls="button-group"
                    ),
                    id="bio-form",
                    hx_post="/rank",
                    hx_target="#posts-list",
                    hx_indicator="#status"
                ),
                Div(id="status", cls="status-bar htmx-indicator"),
                cls="left-panel"
            ),
            
            # Right side: Posts list
            Div(
                H2("Ranked Posts"),
                Div(id="posts-list", cls="posts-list"),
                cls="right-panel"
            ),
            
            cls="split-view"
        ),
        
        # Add CSS for the layout
        Style("""
            .split-view {
                display: flex;
                gap: 2rem;
            }
            .left-panel {
                flex: 1;
                min-width: 300px;
            }
            .right-panel {
                flex: 2;
            }
            .button-group {
                display: flex;
                gap: 1rem;
                margin-top: 1rem;
            }
            .status-bar {
                margin-top: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
                min-height: 1.5rem;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            .htmx-indicator {
                opacity: 0;
            }
            .htmx-request .htmx-indicator {
                opacity: 1;
                background-color: #f0f0f0;
                color: #333;
            }
            .htmx-request.status-bar::before {
                content: "Processing...";
            }
            .posts-list {
                margin-top: 1rem;
            }
            .post-item {
                margin-bottom: 1rem;
                padding: 1rem;
                border-radius: 4px;
                background-color: var(--card-background-color);
                border: 1px solid var(--muted-border-color);
            }
            .post-title {
                margin-top: 0;
                margin-bottom: 0.5rem;
            }
            .post-score {
                font-size: 0.9rem;
                color: var(--muted-color);
            }
            
            /* Mobile responsive */
            @media (max-width: 768px) {
                .split-view {
                    flex-direction: column;
                }
            }
        """),
        
        # Add JavaScript for the refresh button
        Script("""
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('refresh-btn').addEventListener('click', function() {
                const statusEl = document.getElementById('status');
                statusEl.textContent = "Refreshing HN data...";
                statusEl.style.opacity = 1;
                
                fetch('/refresh', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'HX-Request': 'true'
                    }
                })
                .then(response => response.text())
                .then(data => {
                    statusEl.textContent = "HN data refreshed! Submit your bio to rank posts.";
                    setTimeout(() => {
                        statusEl.style.opacity = 0;
                    }, 3000);
                })
                .catch(error => {
                    statusEl.textContent = "Error refreshing data: " + error;
                });
            });
        });
        """)
    )

# Route to handle the ranking request
@rt('/rank')
async def post(bio: str):
    # Check if bio is empty
    if not bio or bio.strip() == "":
        return Div(P("Please enter your interests to rank posts."), cls="post-item")
    
    # Extract user traits/interests
    try:
        weighted_embeddings = await extract_traits(bio)
    except Exception as e:
        return Div(P(f"Error analyzing interests: {str(e)}"), cls="post-item")
    
    # Fetch posts if not already fetched
    global posts
    if not posts:
        try:
            posts = await collect_hn_data()
        except Exception as e:
            return Div(P(f"Error fetching posts: {str(e)}"), cls="post-item")
    
    # Rank posts based on user interests
    try:
        ranked_posts = rank_posts(posts, weighted_embeddings)
    except Exception as e:
        return Div(P(f"Error ranking posts: {str(e)}"), cls="post-item")
    
    # Generate HTML for ranked posts
    posts_html = []
    for i, post in enumerate(ranked_posts[:500]):
        url = post.url if post.url else f"https://news.ycombinator.com/item?id={post.id}"
        post_html = Article(
            H3(A(post.title, href=url, target="_blank"), cls="post-title"),
            P(f"Score: {post.score}", cls="post-score"),
            cls="post-item"
        )
        posts_html.append(post_html)
    
    # Return the ranked posts
    return Div(*posts_html)

# Route to refresh HN data
@rt('/refresh')
async def post():
    try:
        # Fetch fresh posts
        global posts
        posts = await collect_hn_data()
        return "HN data refreshed successfully"
    except Exception as e:
        return f"Error refreshing data: {str(e)}"

# Initial data loading on startup
@app.on_startup
async def startup():
    global posts
    try:
        posts = await collect_hn_data()
        print("[INFO] Initial HN data loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load initial HN data: {str(e)}")
        posts = []

# Start the server
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("[ERROR] OPENAI_API_KEY environment variable not set")
        exit(1)
    serve()
