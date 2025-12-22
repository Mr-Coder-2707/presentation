"""
Text Preprocessor Module
Handles text preprocessing including tokenization, stemming, stopword removal
"""

import re
import string
from typing import List, Set
from collections import Counter


class TextPreprocessor:
    """Preprocesses text for information retrieval"""

    # Common English stopwords
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'then', 'once', 'if', 'because', 'about', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between',
        'under', 'again', 'further', 'while', 'any', 'being', 'having', 'doing'
    }

    def __init__(self, remove_stopwords: bool = True, use_stemming: bool = True,
                 lowercase: bool = True, remove_punctuation: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.stemmer = None

        if use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                print("NLTK not installed. Stemming disabled.")
                self.use_stemming = False

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens"""
        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            # Replace punctuation with spaces
            text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)

        # Split on whitespace and filter empty strings
        tokens = text.split()
        tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [t for t in tokens if t.lower() not in self.STOPWORDS]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        if self.stemmer:
            return [self.stemmer.stem(t) for t in tokens]
        return tokens

    def preprocess(self, text: str) -> List[str]:
        """Full preprocessing pipeline"""
        # Tokenize
        tokens = self.tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)

        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = self.stem_tokens(tokens)

        return tokens

    def preprocess_to_string(self, text: str) -> str:
        """Preprocess and return as joined string"""
        tokens = self.preprocess(text)
        return ' '.join(tokens)

    def get_term_frequencies(self, tokens: List[str]) -> Counter:
        """Get term frequency counts"""
        return Counter(tokens)

    def get_vocabulary(self, documents_tokens: List[List[str]]) -> Set[str]:
        """Get unique vocabulary from all documents"""
        vocab = set()
        for tokens in documents_tokens:
            vocab.update(tokens)
        return vocab

