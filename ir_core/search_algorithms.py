"""
Search Algorithms Module
Contains various retrieval algorithms: Boolean, TF-IDF, BM25, Vector Space Model
"""

import math
import time
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict, Counter
from abc import ABC, abstractmethod

from .document_loader import Document
from .preprocessor import TextPreprocessor


class SearchResult:
    """Represents a search result with score and timing"""

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def __repr__(self):
        return f"Result(rank={self.rank}, score={self.score:.4f}, doc='{self.document.title}')"


class BaseSearchAlgorithm(ABC):
    """Abstract base class for search algorithms"""

    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.documents: List[Document] = []
        self.index_time = 0.0
        self.search_time = 0.0
        self.algorithm_name = "Base"

    @abstractmethod
    def build_index(self, documents: List[Document]):
        """Build the search index"""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for documents matching the query"""
        pass

    def get_timing_info(self) -> Dict[str, float]:
        """Return timing information"""
        return {
            "index_time": self.index_time,
            "search_time": self.search_time
        }


class BooleanSearch(BaseSearchAlgorithm):
    """Boolean retrieval model with AND, OR, NOT operators"""

    def __init__(self, preprocessor: TextPreprocessor = None):
        super().__init__(preprocessor)
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self.algorithm_name = "Boolean Search"

    def build_index(self, documents: List[Document]):
        """Build inverted index for boolean retrieval"""
        start_time = time.time()
        self.documents = documents
        self.inverted_index.clear()

        for doc in documents:
            tokens = self.preprocessor.preprocess(doc.content)
            doc.preprocessed_content = ' '.join(tokens)

            for token in set(tokens):  # Use set to avoid duplicates
                self.inverted_index[token].add(doc.doc_id)

        self.index_time = time.time() - start_time

    def _parse_query(self, query: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Parse query into AND, OR, NOT terms"""
        # Simple parsing: default is AND, use OR and NOT keywords
        query = query.upper()

        and_terms = set()
        or_terms = set()
        not_terms = set()

        # Split by OR first
        or_parts = query.split(' OR ')

        for part in or_parts:
            part = part.strip()
            if ' NOT ' in part:
                main_part, not_part = part.split(' NOT ', 1)
                not_terms.update(self.preprocessor.preprocess(not_part.lower()))
                part = main_part

            terms = self.preprocessor.preprocess(part.lower())
            if len(or_parts) > 1:
                or_terms.update(terms)
            else:
                and_terms.update(terms)

        return and_terms, or_terms, not_terms

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform boolean search"""
        start_time = time.time()

        # Simple approach: treat all terms as AND
        tokens = self.preprocessor.preprocess(query)

        if not tokens:
            self.search_time = time.time() - start_time
            return []

        # Start with all documents for first term
        result_docs = None

        for token in tokens:
            if token in self.inverted_index:
                if result_docs is None:
                    result_docs = self.inverted_index[token].copy()
                else:
                    result_docs = result_docs.intersection(self.inverted_index[token])
            else:
                result_docs = set()
                break

        if result_docs is None:
            result_docs = set()

        # Create results
        results = []
        doc_map = {doc.doc_id: doc for doc in self.documents}

        for rank, doc_id in enumerate(result_docs, 1):
            if doc_id in doc_map:
                results.append(SearchResult(
                    document=doc_map[doc_id],
                    score=1.0,  # Boolean: either match or not
                    rank=rank
                ))
            if len(results) >= top_k:
                break

        self.search_time = time.time() - start_time
        return results


class TFIDFSearch(BaseSearchAlgorithm):
    """TF-IDF (Term Frequency - Inverse Document Frequency) retrieval"""

    def __init__(self, preprocessor: TextPreprocessor = None):
        super().__init__(preprocessor)
        self.tfidf_matrix: Dict[int, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.vocabulary: Set[str] = set()
        self.algorithm_name = "TF-IDF"

    def build_index(self, documents: List[Document]):
        """Build TF-IDF index"""
        start_time = time.time()
        self.documents = documents
        self.tfidf_matrix.clear()
        self.idf.clear()

        N = len(documents)
        doc_freq: Dict[str, int] = defaultdict(int)

        # Calculate term frequencies and document frequencies
        doc_tokens: Dict[int, List[str]] = {}

        for doc in documents:
            tokens = self.preprocessor.preprocess(doc.content)
            doc.preprocessed_content = ' '.join(tokens)
            doc_tokens[doc.doc_id] = tokens

            unique_tokens = set(tokens)
            self.vocabulary.update(unique_tokens)

            for token in unique_tokens:
                doc_freq[token] += 1

        # Calculate IDF for each term
        for term, df in doc_freq.items():
            self.idf[term] = math.log(N / df) if df > 0 else 0

        # Calculate TF-IDF for each document
        for doc in documents:
            tokens = doc_tokens[doc.doc_id]
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1

            self.tfidf_matrix[doc.doc_id] = {}
            for term, count in tf.items():
                # Normalized TF * IDF
                normalized_tf = count / max_tf
                self.tfidf_matrix[doc.doc_id][term] = normalized_tf * self.idf.get(term, 0)

        self.index_time = time.time() - start_time

    def _get_query_vector(self, query: str) -> Dict[str, float]:
        """Convert query to TF-IDF vector"""
        tokens = self.preprocessor.preprocess(query)
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1

        query_vector = {}
        for term, count in tf.items():
            normalized_tf = count / max_tf
            query_vector[term] = normalized_tf * self.idf.get(term, 0)

        return query_vector

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using TF-IDF cosine similarity"""
        start_time = time.time()

        query_vector = self._get_query_vector(query)

        if not query_vector:
            self.search_time = time.time() - start_time
            return []

        # Calculate similarity with all documents
        scores = []
        doc_map = {doc.doc_id: doc for doc in self.documents}

        for doc_id, doc_vector in self.tfidf_matrix.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                scores.append((doc_id, similarity))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank
            ))

        self.search_time = time.time() - start_time
        return results


class BM25Search(BaseSearchAlgorithm):
    """BM25 (Best Matching 25) retrieval algorithm"""

    def __init__(self, preprocessor: TextPreprocessor = None, k1: float = 1.5, b: float = 0.75):
        super().__init__(preprocessor)
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.doc_lengths: Dict[int, int] = {}
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_term_freqs: Dict[int, Counter] = {}
        self.N: int = 0
        self.algorithm_name = "BM25"

    def build_index(self, documents: List[Document]):
        """Build BM25 index"""
        start_time = time.time()
        self.documents = documents
        self.N = len(documents)

        total_length = 0

        for doc in documents:
            tokens = self.preprocessor.preprocess(doc.content)
            doc.preprocessed_content = ' '.join(tokens)

            self.doc_lengths[doc.doc_id] = len(tokens)
            total_length += len(tokens)

            self.doc_term_freqs[doc.doc_id] = Counter(tokens)

            for token in set(tokens):
                self.doc_freqs[token] += 1

        self.avg_doc_length = total_length / self.N if self.N > 0 else 0

        self.index_time = time.time() - start_time

    def _idf(self, term: str) -> float:
        """Calculate IDF for BM25"""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score_document(self, doc_id: int, query_terms: List[str]) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        term_freqs = self.doc_term_freqs.get(doc_id, Counter())

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using BM25"""
        start_time = time.time()

        query_terms = self.preprocessor.preprocess(query)

        if not query_terms:
            self.search_time = time.time() - start_time
            return []

        # Score all documents
        scores = []
        doc_map = {doc.doc_id: doc for doc in self.documents}

        for doc in self.documents:
            score = self._score_document(doc.doc_id, query_terms)
            if score > 0:
                scores.append((doc.doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create results
        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank
            ))

        self.search_time = time.time() - start_time
        return results


class VectorSpaceModel(BaseSearchAlgorithm):
    """Vector Space Model with different weighting schemes"""

    def __init__(self, preprocessor: TextPreprocessor = None, weighting: str = 'tf'):
        super().__init__(preprocessor)
        self.weighting = weighting  # 'tf', 'binary', 'log'
        self.doc_vectors: Dict[int, Dict[str, float]] = {}
        self.vocabulary: Set[str] = set()
        self.algorithm_name = f"Vector Space Model ({weighting})"

    def build_index(self, documents: List[Document]):
        """Build document vectors"""
        start_time = time.time()
        self.documents = documents

        for doc in documents:
            tokens = self.preprocessor.preprocess(doc.content)
            doc.preprocessed_content = ' '.join(tokens)
            self.vocabulary.update(tokens)

            tf = Counter(tokens)

            if self.weighting == 'binary':
                self.doc_vectors[doc.doc_id] = {t: 1.0 for t in tf}
            elif self.weighting == 'log':
                self.doc_vectors[doc.doc_id] = {t: 1 + math.log(c) for t, c in tf.items()}
            else:  # tf
                self.doc_vectors[doc.doc_id] = {t: float(c) for t, c in tf.items()}

        self.index_time = time.time() - start_time

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity"""
        common_terms = set(vec1.keys()) & set(vec2.keys())

        if not common_terms:
            return 0.0

        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using vector space model"""
        start_time = time.time()

        tokens = self.preprocessor.preprocess(query)

        if not tokens:
            self.search_time = time.time() - start_time
            return []

        # Create query vector
        tf = Counter(tokens)
        if self.weighting == 'binary':
            query_vector = {t: 1.0 for t in tf}
        elif self.weighting == 'log':
            query_vector = {t: 1 + math.log(c) for t, c in tf.items()}
        else:
            query_vector = {t: float(c) for t, c in tf.items()}

        # Calculate similarities
        scores = []
        doc_map = {doc.doc_id: doc for doc in self.documents}

        for doc_id, doc_vector in self.doc_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                scores.append((doc_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank
            ))

        self.search_time = time.time() - start_time
        return results


class JaccardSearch(BaseSearchAlgorithm):
    """Jaccard Similarity based retrieval"""

    def __init__(self, preprocessor: TextPreprocessor = None):
        super().__init__(preprocessor)
        self.doc_token_sets: Dict[int, Set[str]] = {}
        self.algorithm_name = "Jaccard Similarity"

    def build_index(self, documents: List[Document]):
        """Build token sets for each document"""
        start_time = time.time()
        self.documents = documents

        for doc in documents:
            tokens = self.preprocessor.preprocess(doc.content)
            doc.preprocessed_content = ' '.join(tokens)
            self.doc_token_sets[doc.doc_id] = set(tokens)

        self.index_time = time.time() - start_time

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using Jaccard similarity"""
        start_time = time.time()

        query_tokens = set(self.preprocessor.preprocess(query))

        if not query_tokens:
            self.search_time = time.time() - start_time
            return []

        scores = []
        doc_map = {doc.doc_id: doc for doc in self.documents}

        for doc_id, doc_tokens in self.doc_token_sets.items():
            similarity = self._jaccard_similarity(query_tokens, doc_tokens)
            if similarity > 0:
                scores.append((doc_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (doc_id, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=score,
                rank=rank
            ))

        self.search_time = time.time() - start_time
        return results


def get_available_algorithms() -> Dict[str, Any]:
    """Return dictionary of available search algorithms"""
    return {
        "1": ("Boolean Search", BooleanSearch),
        "2": ("TF-IDF", TFIDFSearch),
        "3": ("BM25", BM25Search),
        "4": ("Vector Space Model (TF)", lambda p: VectorSpaceModel(p, 'tf')),
        "5": ("Vector Space Model (Binary)", lambda p: VectorSpaceModel(p, 'binary')),
        "6": ("Vector Space Model (Log)", lambda p: VectorSpaceModel(p, 'log')),
        "7": ("Jaccard Similarity", JaccardSearch),
    }
