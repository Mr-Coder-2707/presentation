"""
Evaluation Metrics Module
Calculates Precision, Recall, F1-Score, Accuracy, and other IR metrics
"""

from typing import List, Dict, Set
from dataclasses import dataclass

from .search_algorithms import SearchResult


@dataclass
class EvaluationResult:
    """Stores evaluation metrics for a search query"""
    query: str
    algorithm: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    retrieved_count: int
    relevant_count: int
    relevant_retrieved_count: int
    total_documents: int
    index_time: float
    search_time: float

    def __repr__(self):
        return (f"Evaluation(algorithm='{self.algorithm}', "
                f"P={self.precision:.4f}, R={self.recall:.4f}, "
                f"F1={self.f1_score:.4f}, Acc={self.accuracy:.4f})")


class Evaluator:
    """Evaluates search results using standard IR metrics"""

    def __init__(self):
        self.evaluation_history: List[EvaluationResult] = []

    def calculate_precision(self, retrieved: Set[int], relevant: Set[int]) -> float:
        """
        Precision = |Relevant ∩ Retrieved| / |Retrieved|
        Measures the fraction of retrieved documents that are relevant
        """
        if not retrieved:
            return 0.0

        relevant_retrieved = retrieved & relevant
        return len(relevant_retrieved) / len(retrieved)

    def calculate_recall(self, retrieved: Set[int], relevant: Set[int]) -> float:
        """
        Recall = |Relevant ∩ Retrieved| / |Relevant|
        Measures the fraction of relevant documents that are retrieved
        """
        if not relevant:
            return 0.0

        relevant_retrieved = retrieved & relevant
        return len(relevant_retrieved) / len(relevant)

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        Harmonic mean of precision and recall
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_accuracy(self, retrieved: Set[int], relevant: Set[int],
                          total_documents: int) -> float:
        """
        Accuracy = (TP + TN) / Total
        Where:
        - TP (True Positives) = Relevant documents that were retrieved
        - TN (True Negatives) = Non-relevant documents that were not retrieved
        """
        if total_documents == 0:
            return 0.0

        all_docs = set(range(1, total_documents + 1))

        # True Positives: relevant AND retrieved
        tp = len(retrieved & relevant)

        # True Negatives: NOT relevant AND NOT retrieved
        not_relevant = all_docs - relevant
        not_retrieved = all_docs - retrieved
        tn = len(not_relevant & not_retrieved)

        return (tp + tn) / total_documents

    def calculate_precision_at_k(self, results: List[SearchResult],
                                  relevant: Set[int], k: int) -> float:
        """
        Precision@K: Precision considering only top K results
        """
        if k <= 0 or not results:
            return 0.0

        top_k_ids = {r.document.doc_id for r in results[:k]}
        return self.calculate_precision(top_k_ids, relevant)

    def calculate_average_precision(self, results: List[SearchResult],
                                     relevant: Set[int]) -> float:
        """
        Average Precision (AP): Average of precision values at each relevant document
        """
        if not relevant or not results:
            return 0.0

        precisions = []
        relevant_count = 0

        for i, result in enumerate(results, 1):
            if result.document.doc_id in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precisions.append(precision_at_i)

        return sum(precisions) / len(relevant) if precisions else 0.0

    def calculate_dcg(self, results: List[SearchResult],
                      relevant: Set[int], k: int = None) -> float:
        """
        Discounted Cumulative Gain (DCG)
        Gives higher weight to relevant documents appearing earlier
        """
        import math

        if k is None:
            k = len(results)

        dcg = 0.0
        for i, result in enumerate(results[:k], 1):
            relevance = 1 if result.document.doc_id in relevant else 0
            dcg += relevance / math.log2(i + 1)

        return dcg

    def calculate_ndcg(self, results: List[SearchResult],
                       relevant: Set[int], k: int = None) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG)
        DCG normalized by ideal DCG
        """
        import math

        if k is None:
            k = len(results)

        dcg = self.calculate_dcg(results, relevant, k)

        # Calculate ideal DCG (all relevant documents at top)
        ideal_dcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            ideal_dcg += 1 / math.log2(i + 1)

        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def calculate_mrr(self, results: List[SearchResult], relevant: Set[int]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        1 / rank of first relevant document
        """
        for i, result in enumerate(results, 1):
            if result.document.doc_id in relevant:
                return 1.0 / i
        return 0.0

    def evaluate(self, query: str, results: List[SearchResult],
                 relevant_doc_ids: Set[int], total_documents: int,
                 algorithm_name: str, index_time: float,
                 search_time: float) -> EvaluationResult:
        """
        Perform full evaluation of search results
        """
        retrieved_ids = {r.document.doc_id for r in results}

        precision = self.calculate_precision(retrieved_ids, relevant_doc_ids)
        recall = self.calculate_recall(retrieved_ids, relevant_doc_ids)
        f1_score = self.calculate_f1_score(precision, recall)
        accuracy = self.calculate_accuracy(retrieved_ids, relevant_doc_ids, total_documents)

        eval_result = EvaluationResult(
            query=query,
            algorithm=algorithm_name,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            retrieved_count=len(retrieved_ids),
            relevant_count=len(relevant_doc_ids),
            relevant_retrieved_count=len(retrieved_ids & relevant_doc_ids),
            total_documents=total_documents,
            index_time=index_time,
            search_time=search_time
        )

        self.evaluation_history.append(eval_result)
        return eval_result

    def evaluate_with_advanced_metrics(self, query: str, results: List[SearchResult],
                                        relevant_doc_ids: Set[int], total_documents: int,
                                        algorithm_name: str, index_time: float,
                                        search_time: float) -> Dict:
        """
        Perform evaluation with advanced metrics
        """
        basic_eval = self.evaluate(query, results, relevant_doc_ids,
                                   total_documents, algorithm_name,
                                   index_time, search_time)

        advanced_metrics = {
            "basic": basic_eval,
            "precision_at_5": self.calculate_precision_at_k(results, relevant_doc_ids, 5),
            "precision_at_10": self.calculate_precision_at_k(results, relevant_doc_ids, 10),
            "average_precision": self.calculate_average_precision(results, relevant_doc_ids),
            "ndcg": self.calculate_ndcg(results, relevant_doc_ids),
            "ndcg_at_5": self.calculate_ndcg(results, relevant_doc_ids, 5),
            "mrr": self.calculate_mrr(results, relevant_doc_ids)
        }

        return advanced_metrics

    def compare_algorithms(self, evaluations: List[EvaluationResult]) -> str:
        """
        Generate comparison report for multiple algorithm evaluations
        """
        if not evaluations:
            return "No evaluations to compare."

        report = "\n" + "=" * 80 + "\n"
        report += "ALGORITHM COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"

        # Header
        report += f"{'Algorithm':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12} {'Time (ms)':<12}\n"
        report += "-" * 90 + "\n"

        # Data rows
        for eval_result in evaluations:
            total_time = (eval_result.index_time + eval_result.search_time) * 1000
            report += (f"{eval_result.algorithm:<30} "
                      f"{eval_result.precision:<12.4f} "
                      f"{eval_result.recall:<12.4f} "
                      f"{eval_result.f1_score:<12.4f} "
                      f"{eval_result.accuracy:<12.4f} "
                      f"{total_time:<12.2f}\n")

        report += "-" * 90 + "\n"

        # Find best algorithm for each metric
        best_precision = max(evaluations, key=lambda x: x.precision)
        best_recall = max(evaluations, key=lambda x: x.recall)
        best_f1 = max(evaluations, key=lambda x: x.f1_score)
        best_accuracy = max(evaluations, key=lambda x: x.accuracy)
        fastest = min(evaluations, key=lambda x: x.index_time + x.search_time)

        report += "\nBest Performance:\n"
        report += f"  - Highest Precision: {best_precision.algorithm} ({best_precision.precision:.4f})\n"
        report += f"  - Highest Recall: {best_recall.algorithm} ({best_recall.recall:.4f})\n"
        report += f"  - Highest F1-Score: {best_f1.algorithm} ({best_f1.f1_score:.4f})\n"
        report += f"  - Highest Accuracy: {best_accuracy.algorithm} ({best_accuracy.accuracy:.4f})\n"
        report += f"  - Fastest: {fastest.algorithm} ({(fastest.index_time + fastest.search_time)*1000:.2f} ms)\n"

        return report

    def get_history(self) -> List[EvaluationResult]:
        """Return evaluation history"""
        return self.evaluation_history

    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history = []


def format_evaluation_result(eval_result: EvaluationResult) -> str:
    """Format evaluation result for display"""
    output = "\n" + "-" * 60 + "\n"
    output += f"EVALUATION RESULTS - {eval_result.algorithm}\n"
    output += "-" * 60 + "\n"
    output += f"Query: '{eval_result.query}'\n\n"

    output += "Retrieval Statistics:\n"
    output += f"  - Total Documents: {eval_result.total_documents}\n"
    output += f"  - Retrieved Documents: {eval_result.retrieved_count}\n"
    output += f"  - Relevant Documents: {eval_result.relevant_count}\n"
    output += f"  - Relevant & Retrieved: {eval_result.relevant_retrieved_count}\n\n"

    output += "Performance Metrics:\n"
    output += f"  - Precision:  {eval_result.precision:.4f} ({eval_result.precision*100:.2f}%)\n"
    output += f"  - Recall:     {eval_result.recall:.4f} ({eval_result.recall*100:.2f}%)\n"
    output += f"  - F1-Score:   {eval_result.f1_score:.4f} ({eval_result.f1_score*100:.2f}%)\n"
    output += f"  - Accuracy:   {eval_result.accuracy:.4f} ({eval_result.accuracy*100:.2f}%)\n\n"

    output += "Timing:\n"
    output += f"  - Index Time:  {eval_result.index_time*1000:.4f} ms\n"
    output += f"  - Search Time: {eval_result.search_time*1000:.4f} ms\n"
    output += f"  - Total Time:  {(eval_result.index_time + eval_result.search_time)*1000:.4f} ms\n"
    output += "-" * 60 + "\n"

    return output
