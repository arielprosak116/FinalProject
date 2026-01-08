GOAL - getting high scores on hard queries.

- Initial retrieval - top 1000 docs using rm3, take top 100
- Splitting 100 docs to paragraphs
- (Optional) create a custom BM25 class using the GLOBAL df values but the LOCAL tf values to rank passages
- Or just take all passages
- cross-encode them and rank the docs based on a softmaxexp or just smoothing of max and mean.
- Then you can just use this score as the reranked result or smooth the score with the rm3.