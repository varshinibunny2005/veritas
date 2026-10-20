"""Small text helpers shared by the offline heuristics and the hashing embeddings."""
import re

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with about as into like through over
    after before between from up down out off is are was were be been being am do does did doing have
    has had having i you he she it we they me him her us them my your his its our their this that
    these those there here what which who whom whose when where why how not no nor so than too very
    can will just should would could may might must also both each few more most other some such only
    own same s t don now""".split()
)


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def token_set(text: str) -> set:
    return set(tokens(text))


def numbers(text: str) -> set:
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    return {n.replace(",", "").rstrip(".") for n in found}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]
