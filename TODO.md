GOAL - getting high scores on hard queries.
- RRF w/originaL
- Hard queries
- Copy ariel code
- doc.id unified structure
- Write parameter-testing code to maximize performance?
- Upgrade to the newest qwen reranker? nvembed?

- Do LLM query enrichment (one with openai, one with a quantized big model, see the difference in quality)
- Sentence transformer benchmarking
- Hard queries analysis → maximize recall for the cross encoder to do its job

- Initial retrieval - top 1000 docs using rm3, take top 100
- Splitting 100 docs to paragraphs
- (Optional) create a custom BM25 class using the GLOBAL df values but the LOCAL tf values to rank passages
- Or just take all passages
- cross-encode them and rank the docs based on a softmaxexp or just smoothing of max and mean.
- Then you can just use this score as the reranked result or smooth the score with the rm3.