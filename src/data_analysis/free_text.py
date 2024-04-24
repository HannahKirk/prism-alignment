"""
Some free-text helper functions to preprocess text data.
"""

import string
import logging
import nltk
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)

# Download stopwords from NLTK
nltk.download("stopwords")

# Set up text cleaning
stop_words = set(stopwords.words("english"))


def preprocess(text, lowercase=True, remove_punct=True):
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.replace("-", " ")
        text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def remove_stopwords_and_questionwords(text, question):
    """Remove stopwords and words found in the question from the text."""
    question_words = [w for w in question.split(" ")]

    # Preprocess the text
    cleaned_text = preprocess(text)

    # Remove stopwords and question words
    return " ".join(
        word
        for word in cleaned_text.split()
        if word not in stop_words and word not in question_words
    )
