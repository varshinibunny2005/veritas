# The Transformer Architecture

The Transformer was introduced by Vaswani et al. at Google in the 2017 paper "Attention Is All You Need". It replaced recurrence with self-attention, allowing all positions in a sequence to be processed in parallel.

## Key components
Self-attention lets each token weigh every other token when building its representation. Multi-head attention runs several attention functions in parallel so the model can capture different relationships. Because the architecture has no recurrence, positional encodings are added to the embeddings to give the model information about token order. Each layer also contains a feed-forward network, residual connections and layer normalisation.

## Variants
The original model is an encoder-decoder used for machine translation. BERT (2018) uses only the encoder and is trained with masked language modelling. The GPT family uses only the decoder and is trained to predict the next token, which is the basis of most modern chat LLMs.

## Cost
The compute and memory of standard self-attention grow quadratically with sequence length, which motivates techniques such as FlashAttention and sparse attention.
