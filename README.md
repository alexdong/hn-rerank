# Hakcer News Re-ranking According to Personal Preferences

## 1. Problem Statement

The project is to re-ranks the Hacker News posts according to the user's
personal preferences. Here is the functional spec. 

> Build an API that given user submitted bio, returns top 500 stories from
> Hacker News front page (use their API for this), ranked in the order of
> relevancy to user's interests.
>
> Example input: I am a theoretical biologist, interested in disease ecology.
> My tools are R, clojure , compartmentalism disease modeling, and statistical
> GAM models, using a variety of data layers (geophysical, reconstructions,
> climate, biodiversity, land use). Besides that I am interested in tech
> applied to the a subset of the current problems of the world (agriculture /
> biodiversity / conservation / forecasting), development of third world
> countries and AI, large language models.
>
> Output: Top 500 items from Hacker News, ranked in the order of relevance to
> the user.
> 
> The solution should optimize for development time - what is the
> simplest/quickest way to do this that you can think of? Spend no more than 4
> hours on this.

## 2. Core Design

I'd like to use latent concept similiarity score between each post and the user's 
stated perference to augment the existing score as voted by the HN community. 

The project is divided into three parts:

1. HN data collection: every time the app starts, it'll fetch the top 500 posts
   and then fetch each individual `item` and store them in `posts: List[Post]`.
   For each new `post: [title, embedding, url]`, calculate an embedding vector
   for the title [3]. 

2. Vectorise user preference: the user's preference is a text string. I'd
   prepare a prompt to generate a weighted interest vector 
   `interests: List[text, embedding, weight]` 

3. Rank the posts: for each post, calculate the cosine similarity between the
    post's title embedding and the user's interest embedding. The final score
    would be a weighted sum of the HN score and the cosine similarity score.
    Let's use numpy's `@` for this.

## 4. Website Design

To make it more fun, I'd like to use `fasthtml` to build a
single page website for this. Here is a description of the web page:

1. The web page has left-right split view (make it mobile friendly so it
   becomes top-bottom for mobile screen)
2. The left side is the user's bio input. The user can type in the bio and
   click a submit button. I also want a status bar to show the progress of
   backend operations (from websocket output). Also show a refresh button to
   re-fetch the HN data to the right of status bar.
3. The right is the list of 500 posts. Each post is a link
   [post.title](post.url). When the user clicked the "submit" button, the list
   will be re-ordered based the ranking algorithm described above.


## 5. Implementation Details

- DO NOT use any database. Just store the data in memory.
- DO NOT try ... except ... for error handling. Just let the program crash.
- DO NOT attempt to make the code production ready. Just make it work.
- DO NOT generate documentation or comments unless it's necessary.
- DO break the code into small .py files and use `__main__` block to test each
  one individually.
- DO produce `Makefile` to run `uv run main.py` by default and `make deploy` will deploy to railway[5].



[1]: https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty
[2]: https://hacker-news.firebaseio.com/v0/item/8863.json?print=pretty
[3]: https://platform.openai.com/docs/guides/embeddings
[4]: https://www.fastht.ml/ and `fasthtml-ctx.txt` is provided in the repo.
[5]: The following is how to deploy the app to railway.

    > ```bash
    > railway up -c
    > ```
