"""Ready-made test cases for the UI (they match the sample knowledge base)."""

EXAMPLES = [
    {
        "id": "eiffel",
        "title": "Eiffel Tower (factual errors)",
        "question": "Tell me about the Eiffel Tower: when was it built, how tall is it, and what is it made of?",
        "answer": (
            "The Eiffel Tower was completed in 1899 for the World's Fair in Paris. "
            "It stands about 330 metres tall and is made of steel. "
            "It was drawn personally by Gustave Eiffel and was always intended to stay in Paris permanently."
        ),
    },
    {
        "id": "rag",
        "title": "RAG (incomplete + overclaim)",
        "question": "What is Retrieval-Augmented Generation and what are its main benefits and limitations?",
        "answer": (
            "Retrieval-Augmented Generation combines a large language model with an external knowledge source "
            "by retrieving relevant documents at query time and adding them to the prompt. "
            "It gives access to up-to-date or private knowledge without retraining the model. "
            "RAG completely eliminates hallucinations because the model only uses retrieved text."
        ),
    },
    {
        "id": "photosynthesis",
        "title": "Photosynthesis (good answer)",
        "question": "Explain photosynthesis, including where each stage happens and its overall equation.",
        "answer": (
            "Photosynthesis converts light energy into chemical energy in the chloroplasts of plants. "
            "In the light-dependent reactions, in the thylakoid membranes, chlorophyll absorbs light and splits water, "
            "releasing oxygen and producing ATP and NADPH. "
            "In the Calvin cycle, in the stroma, the enzyme RuBisCO fixes carbon dioxide and the plant builds glucose. "
            "The overall equation is 6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2."
        ),
    },
    {
        "id": "python",
        "title": "Python (self-contradiction)",
        "question": "Who created Python and when was it first released?",
        "answer": (
            "Python was created by Guido van Rossum and first released in 1991. "
            "It is a dynamically typed, garbage-collected language. "
            "Python was first released in 1994 and became popular quickly."
        ),
    },
]
