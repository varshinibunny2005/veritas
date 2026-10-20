# Evaluating LLM Outputs

## Hallucination
A hallucination is generated content that is fluent but false or unsupported. Intrinsic hallucinations contradict the source material given to the model; extrinsic hallucinations add information that cannot be verified from the source.

## Groundedness
Groundedness (also called faithfulness) measures whether every claim in an answer is supported by the provided reference documents. A common method extracts atomic claims from the answer and checks each one against retrieved evidence.

## Self-consistency and sampling
Self-consistency (Wang et al., 2022) samples several reasoning paths and keeps the answer they agree on. SelfCheckGPT (Manakul et al., 2023) applies a similar idea to hallucination detection: if a fact is real, independently sampled answers will tend to agree with it, whereas hallucinated facts vary between samples.

## LLM-as-a-judge
Using one LLM to grade another is cheap and scalable but has known biases: position bias, verbosity bias (preferring longer answers) and self-preference bias (favouring its own style). Using a panel of several independent judges, each with a narrow rubric, generally gives more reliable scores than a single judge; the "Replacing Judges with Juries" paper (Verga et al., 2024) reports this for a panel of smaller models.

## Completeness
A complete answer addresses every part of the question, including implicit expectations such as limitations when benefits are requested. Completeness is often checked by decomposing the question into sub-questions and verifying that each one is answered.
