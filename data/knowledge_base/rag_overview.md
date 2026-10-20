# Retrieval-Augmented Generation (RAG)

## Definition
Retrieval-Augmented Generation combines a large language model with an external knowledge source. Instead of relying only on what the model memorised during training, the system retrieves relevant documents at query time and adds them to the prompt. The technique was introduced by Lewis et al. (Facebook AI Research) in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks".

## Pipeline
1. Indexing: documents are loaded, split into chunks, converted to embeddings and stored in a vector database.
2. Retrieval: the user query is embedded and the most similar chunks are found with similarity search (top-k).
3. Generation: the retrieved chunks are inserted into the prompt and the LLM writes an answer grounded in them.

## Benefits
RAG reduces hallucinations by grounding answers in retrieved text. It gives access to up-to-date or private knowledge without retraining the model. It also allows source attribution, so users can verify where a statement came from.

## Limitations
RAG does not eliminate hallucinations: the model can still ignore or misread the retrieved context. Answer quality depends on retrieval quality, and poor chunking or embeddings return irrelevant passages. Context-window limits restrict how much evidence can be included, and retrieval adds latency and infrastructure cost.

## Common improvements
Hybrid search (BM25 plus dense vectors), re-ranking of retrieved chunks, query rewriting, and maximal marginal relevance (MMR) for diverse results. RAG systems are commonly evaluated on faithfulness (groundedness), answer relevance, and context precision and recall.
