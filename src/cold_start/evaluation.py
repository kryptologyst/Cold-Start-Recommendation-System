"""Evaluation metrics and model comparison for cold-start recommendations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """Collection of recommendation evaluation metrics."""
    
    @staticmethod
    def precision_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
        """Calculate Precision@K.
        
        Args:
            y_true: List of relevant items.
            y_pred: List of predicted items.
            k: Number of top items to consider.
            
        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0
        
        y_pred_k = y_pred[:k]
        if not y_pred_k:
            return 0.0
        
        relevant_items = set(y_true)
        recommended_items = set(y_pred_k)
        
        intersection = len(relevant_items & recommended_items)
        return intersection / len(recommended_items)
    
    @staticmethod
    def recall_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
        """Calculate Recall@K.
        
        Args:
            y_true: List of relevant items.
            y_pred: List of predicted items.
            k: Number of top items to consider.
            
        Returns:
            Recall@K score.
        """
        if k == 0 or not y_true:
            return 0.0
        
        y_pred_k = y_pred[:k]
        if not y_pred_k:
            return 0.0
        
        relevant_items = set(y_true)
        recommended_items = set(y_pred_k)
        
        intersection = len(relevant_items & recommended_items)
        return intersection / len(relevant_items)
    
    @staticmethod
    def hit_rate_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            y_true: List of relevant items.
            y_pred: List of predicted items.
            k: Number of top items to consider.
            
        Returns:
            Hit Rate@K score.
        """
        if k == 0 or not y_true:
            return 0.0
        
        y_pred_k = y_pred[:k]
        if not y_pred_k:
            return 0.0
        
        relevant_items = set(y_true)
        recommended_items = set(y_pred_k)
        
        return 1.0 if len(relevant_items & recommended_items) > 0 else 0.0
    
    @staticmethod
    def ndcg_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
        """Calculate NDCG@K.
        
        Args:
            y_true: List of relevant items.
            y_pred: List of predicted items.
            k: Number of top items to consider.
            
        Returns:
            NDCG@K score.
        """
        if k == 0 or not y_true:
            return 0.0
        
        y_pred_k = y_pred[:k]
        if not y_pred_k:
            return 0.0
        
        relevant_items = set(y_true)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(y_pred_k):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def map_at_k(y_true: List[str], y_pred: List[str], k: int = 10) -> float:
        """Calculate MAP@K.
        
        Args:
            y_true: List of relevant items.
            y_pred: List of predicted items.
            k: Number of top items to consider.
            
        Returns:
            MAP@K score.
        """
        if k == 0 or not y_true:
            return 0.0
        
        y_pred_k = y_pred[:k]
        if not y_pred_k:
            return 0.0
        
        relevant_items = set(y_true)
        
        # Calculate average precision
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(y_pred_k):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items) if relevant_items else 0.0
    
    @staticmethod
    def coverage_at_k(y_pred_list: List[List[str]], all_items: List[str], k: int = 10) -> float:
        """Calculate Coverage@K.
        
        Args:
            y_pred_list: List of recommendation lists.
            all_items: List of all available items.
            k: Number of top items to consider.
            
        Returns:
            Coverage@K score.
        """
        if not y_pred_list or not all_items:
            return 0.0
        
        all_items_set = set(all_items)
        recommended_items = set()
        
        for y_pred in y_pred_list:
            y_pred_k = y_pred[:k]
            recommended_items.update(y_pred_k)
        
        return len(recommended_items & all_items_set) / len(all_items_set)
    
    @staticmethod
    def diversity_at_k(y_pred_list: List[List[str]], k: int = 10) -> float:
        """Calculate Diversity@K using intra-list diversity.
        
        Args:
            y_pred_list: List of recommendation lists.
            k: Number of top items to consider.
            
        Returns:
            Diversity@K score.
        """
        if not y_pred_list:
            return 0.0
        
        diversity_scores = []
        
        for y_pred in y_pred_list:
            y_pred_k = y_pred[:k]
            if len(y_pred_k) < 2:
                diversity_scores.append(0.0)
                continue
            
            # Calculate pairwise diversity (simplified - assumes items are different)
            unique_items = len(set(y_pred_k))
            diversity_scores.append(unique_items / len(y_pred_k))
        
        return np.mean(diversity_scores)
    
    @staticmethod
    def novelty_at_k(y_pred_list: List[List[str]], item_popularity: Dict[str, int], k: int = 10) -> float:
        """Calculate Novelty@K.
        
        Args:
            y_pred_list: List of recommendation lists.
            item_popularity: Dictionary mapping item IDs to popularity counts.
            k: Number of top items to consider.
            
        Returns:
            Novelty@K score.
        """
        if not y_pred_list or not item_popularity:
            return 0.0
        
        novelty_scores = []
        max_popularity = max(item_popularity.values()) if item_popularity else 1
        
        for y_pred in y_pred_list:
            y_pred_k = y_pred[:k]
            if not y_pred_k:
                novelty_scores.append(0.0)
                continue
            
            # Calculate average novelty (inverse of popularity)
            item_novelties = []
            for item in y_pred_k:
                popularity = item_popularity.get(item, 0)
                novelty = 1.0 - (popularity / max_popularity)
                item_novelties.append(novelty)
            
            novelty_scores.append(np.mean(item_novelties))
        
        return np.mean(novelty_scores)


class Evaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute. If None, uses default metrics.
        """
        self.metrics = metrics or ['precision', 'recall', 'hit_rate', 'ndcg', 'map']
        self.metric_functions = {
            'precision': Metrics.precision_at_k,
            'recall': Metrics.recall_at_k,
            'hit_rate': Metrics.hit_rate_at_k,
            'ndcg': Metrics.ndcg_at_k,
            'map': Metrics.map_at_k,
        }
    
    def evaluate_model(self, model, test_data: pd.DataFrame, 
                      items_df: pd.DataFrame, k_values: List[int] = None) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Recommendation model to evaluate.
            test_data: Test interactions DataFrame.
            items_df: Items DataFrame.
            k_values: List of k values to evaluate.
            
        Returns:
            Dictionary of metric scores.
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        results = {}
        
        # Group test data by user
        user_groups = test_data.groupby('user_id')
        
        for k in k_values:
            metric_scores = {metric: [] for metric in self.metrics}
            
            for user_id, user_data in user_groups:
                # Get ground truth items for this user
                ground_truth = user_data['item_id'].tolist()
                
                # Get recommendations from model
                try:
                    recommendations = model.recommend(user_id, n_recommendations=k)
                except Exception as e:
                    logger.warning(f"Error getting recommendations for user {user_id}: {e}")
                    recommendations = []
                
                # Calculate metrics
                for metric in self.metrics:
                    if metric in self.metric_functions:
                        score = self.metric_functions[metric](ground_truth, recommendations, k)
                        metric_scores[metric].append(score)
            
            # Average metrics across users
            for metric in self.metrics:
                if metric_scores[metric]:
                    avg_score = np.mean(metric_scores[metric])
                    results[f"{metric}@{k}"] = avg_score
                else:
                    results[f"{metric}@{k}"] = 0.0
        
        return results
    
    def evaluate_cold_start(self, model, cold_users: List[str], cold_items: List[str],
                           test_data: pd.DataFrame, items_df: pd.DataFrame,
                           k_values: List[int] = None) -> Dict[str, float]:
        """Evaluate model performance on cold-start scenarios.
        
        Args:
            model: Recommendation model to evaluate.
            cold_users: List of cold-start user IDs.
            cold_items: List of cold-start item IDs.
            test_data: Test interactions DataFrame.
            items_df: Items DataFrame.
            k_values: List of k values to evaluate.
            
        Returns:
            Dictionary of cold-start metric scores.
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        results = {}
        
        # Evaluate cold-start users
        cold_user_results = {}
        for k in k_values:
            metric_scores = {metric: [] for metric in self.metrics}
            
            for user_id in cold_users:
                user_test_data = test_data[test_data['user_id'] == user_id]
                if user_test_data.empty:
                    continue
                
                ground_truth = user_test_data['item_id'].tolist()
                
                try:
                    recommendations = model.recommend_for_cold_user(n_recommendations=k)
                except Exception as e:
                    logger.warning(f"Error getting cold-start recommendations for user {user_id}: {e}")
                    recommendations = []
                
                for metric in self.metrics:
                    if metric in self.metric_functions:
                        score = self.metric_functions[metric](ground_truth, recommendations, k)
                        metric_scores[metric].append(score)
            
            for metric in self.metrics:
                if metric_scores[metric]:
                    avg_score = np.mean(metric_scores[metric])
                    cold_user_results[f"{metric}@{k}"] = avg_score
                else:
                    cold_user_results[f"{metric}@{k}"] = 0.0
        
        # Add prefix to distinguish cold-start results
        for key, value in cold_user_results.items():
            results[f"cold_user_{key}"] = value
        
        return results
    
    def compare_models(self, models: Dict[str, Any], test_data: pd.DataFrame,
                      items_df: pd.DataFrame, k_values: List[int] = None) -> pd.DataFrame:
        """Compare multiple models and return results as DataFrame.
        
        Args:
            models: Dictionary mapping model names to model instances.
            test_data: Test interactions DataFrame.
            items_df: Items DataFrame.
            k_values: List of k values to evaluate.
            
        Returns:
            DataFrame with model comparison results.
        """
        if k_values is None:
            k_values = [5, 10, 20]
        
        all_results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            results = self.evaluate_model(model, test_data, items_df, k_values)
            results['model'] = model_name
            all_results.append(results)
        
        return pd.DataFrame(all_results)
    
    def create_leaderboard(self, results_df: pd.DataFrame, 
                         primary_metric: str = 'ndcg@10') -> pd.DataFrame:
        """Create a leaderboard sorted by primary metric.
        
        Args:
            results_df: Results DataFrame from compare_models.
            primary_metric: Primary metric to sort by.
            
        Returns:
            Sorted leaderboard DataFrame.
        """
        if primary_metric not in results_df.columns:
            logger.warning(f"Primary metric {primary_metric} not found in results")
            return results_df
        
        leaderboard = results_df.sort_values(primary_metric, ascending=False)
        return leaderboard
