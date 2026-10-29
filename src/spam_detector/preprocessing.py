"""Versioned, deterministic English text preparation; no label-dependent state."""

import re
import unicodedata

from bs4 import BeautifulSoup

PREPROCESSING_VERSION = 1


def canonical_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Email text must be a string")
    text = unicodedata.normalize("NFKC", text)
    # Plain CSV email bodies need not be MIME-parsed. Strip markup if present.
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        for element in soup(["script", "style"]):
            element.decompose()
        text = soup.get_text(" ")
    text = text.lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " specialurl ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " specialemail ", text)
    text = re.sub(r"[$£€]\s*\d[\d,.]*", " specialmoney ", text)
    text = re.sub(r"\d+(?:[,.]\d+)*", " specialnumber ", text)
    return " ".join(re.findall(r"[a-z]{2,}", text))
