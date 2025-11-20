"""Unit tests for cold-start recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cold_start.data import DataGenerator, DataLoader
from cold_start.models import ContentBasedRecommender, HybridRecommender, ColdStartRecommender
from cold_start.evaluation import Metrics, Evaluator
from cold_start.utils import set_seed


class TestDataGenerator:
    """Test cases for DataGenerator."""
    
    def test_generate_items(self):
        """Test item generation."""
        generator = DataGenerator(seed=42)
        items_df = generator.generate_items(n_items=10)
        
        assert len(items_df) == 10
        assert 'item_id' in items_df.columns
        assert 'title' in items_df.columns
        assert 'category' in items_df.columns
        assert 'brand' in items_df.columns
        assert 'price' in items_df.columns
    
    def test_generate_users(self):
        """Test user generation."""
        generator = DataGenerator(seed=42)
        users_df = generator.generate_users(n_users=5)
        
        assert len(users_df) == 5
        assert 'user_id' in users_df.columns
        assert 'age_group' in users_df.columns
        assert 'location' in users_df.columns
    
    def test_generate_interactions(self):
        """Test interaction generation."""
        generator = DataGenerator(seed=42)
        items_df = generator.generate_items(n_items=5)
        users_df = generator.generate_users(n_users=3)
        
        interactions_df = generator.generate_interactions(users_df, items_df, n_interactions=10)
        
        assert len(interactions_df) == 10
        assert 'user_id' in interactions_df.columns
        assert 'item_id' in interactions_df.columns
        assert 'rating' in interactions_df.columns
        assert 'timestamp' in interactions_df.columns
    
    def test_create_cold_start_scenarios(self):
        """Test cold-start scenario creation."""
        generator = DataGenerator(seed=42)
        items_df = generator.generate_items(n_items=10)
        users_df = generator.generate_users(n_users=5)
        interactions_df = generator.generate_interactions(users_df, items_df, n_interactions=20)
        
        filtered_interactions, cold_users, cold_items = generator.create_cold_start_scenarios(
            interactions_df, users_df, items_df, cold_user_ratio=0.2, cold_item_ratio=0.2
        )
        
        assert len(filtered_interactions) <= len(interactions_df)
        assert len(cold_users) > 0
        assert len(cold_items) > 0


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.generator = DataGenerator(seed=42)
        self.items_df = self.generator.generate_items(n_items=5)
        self.users_df = self.generator.generate_users(n_users=3)
        self.interactions_df = self.generator.generate_interactions(
            self.users_df, self.items_df, n_interactions=10
        )
        
        # Add dummy features
        for i in range(5):
            self.items_df[f'feature_{i}'] = np.random.random(len(self.items_df))
    
    def test_fit(self):
        """Test model fitting."""
        model = ContentBasedRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        assert model.item_features is not None
        assert model.item_ids is not None
        assert len(model.user_profiles) > 0
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = ContentBasedRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        user_id = self.users_df['user_id'].iloc[0]
        recommendations = model.recommend(user_id, n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(item_id in model.item_ids for item_id in recommendations)
    
    def test_recommend_for_cold_user(self):
        """Test cold-user recommendations."""
        model = ContentBasedRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend_for_cold_user(n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(item_id in model.item_ids for item_id in recommendations)
    
    def test_recommend_for_cold_item(self):
        """Test cold-item recommendations."""
        model = ContentBasedRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        item_id = self.items_df['item_id'].iloc[0]
        recommendations = model.recommend_for_cold_item(item_id, n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(user_id in model.user_profiles for user_id in recommendations)


class TestHybridRecommender:
    """Test cases for HybridRecommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.generator = DataGenerator(seed=42)
        self.items_df = self.generator.generate_items(n_items=5)
        self.users_df = self.generator.generate_users(n_users=3)
        self.interactions_df = self.generator.generate_interactions(
            self.users_df, self.items_df, n_interactions=10
        )
        
        # Add dummy features
        for i in range(5):
            self.items_df[f'feature_{i}'] = np.random.random(len(self.items_df))
    
    def test_fit(self):
        """Test model fitting."""
        model = HybridRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        assert model.content_model is not None
        assert model.user_item_matrix is not None
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = HybridRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        user_id = self.users_df['user_id'].iloc[0]
        recommendations = model.recommend(user_id, n_recommendations=3)
        
        assert len(recommendations) <= 3


class TestColdStartRecommender:
    """Test cases for ColdStartRecommender."""
    
    def setup_method(self):
        """Set up test data."""
        self.generator = DataGenerator(seed=42)
        self.items_df = self.generator.generate_items(n_items=5)
        self.users_df = self.generator.generate_users(n_users=3)
        self.interactions_df = self.generator.generate_interactions(
            self.users_df, self.items_df, n_interactions=10
        )
        
        # Add dummy features
        for i in range(5):
            self.items_df[f'feature_{i}'] = np.random.random(len(self.items_df))
    
    def test_fit(self):
        """Test model fitting."""
        model = ColdStartRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        assert model.item_features is not None
        assert model.item_ids is not None
    
    def test_recommend_for_cold_user(self):
        """Test cold-user recommendations."""
        model = ColdStartRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        recommendations = model.recommend_for_cold_user(n_recommendations=3)
        
        assert len(recommendations) <= 3
        assert all(item_id in model.item_ids for item_id in recommendations)


class TestMetrics:
    """Test cases for Metrics."""
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        y_true = ['item1', 'item2', 'item3']
        y_pred = ['item1', 'item4', 'item5']
        
        precision = Metrics.precision_at_k(y_true, y_pred, k=3)
        assert precision == 1/3  # 1 relevant item out of 3 predicted
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        y_true = ['item1', 'item2', 'item3']
        y_pred = ['item1', 'item4', 'item5']
        
        recall = Metrics.recall_at_k(y_true, y_pred, k=3)
        assert recall == 1/3  # 1 relevant item out of 3 total relevant
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate@K calculation."""
        y_true = ['item1', 'item2', 'item3']
        y_pred = ['item1', 'item4', 'item5']
        
        hit_rate = Metrics.hit_rate_at_k(y_true, y_pred, k=3)
        assert hit_rate == 1.0  # At least one relevant item found
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        y_true = ['item1', 'item2']
        y_pred = ['item1', 'item3', 'item2']
        
        ndcg = Metrics.ndcg_at_k(y_true, y_pred, k=3)
        assert 0 <= ndcg <= 1
    
    def test_map_at_k(self):
        """Test MAP@K calculation."""
        y_true = ['item1', 'item2']
        y_pred = ['item1', 'item3', 'item2']
        
        map_score = Metrics.map_at_k(y_true, y_pred, k=3)
        assert 0 <= map_score <= 1
    
    def test_coverage_at_k(self):
        """Test Coverage@K calculation."""
        y_pred_list = [['item1', 'item2'], ['item2', 'item3']]
        all_items = ['item1', 'item2', 'item3', 'item4']
        
        coverage = Metrics.coverage_at_k(y_pred_list, all_items, k=2)
        assert coverage == 3/4  # 3 unique items covered out of 4 total
    
    def test_diversity_at_k(self):
        """Test Diversity@K calculation."""
        y_pred_list = [['item1', 'item2', 'item3'], ['item1', 'item1', 'item2']]
        
        diversity = Metrics.diversity_at_k(y_pred_list, k=3)
        assert 0 <= diversity <= 1
    
    def test_novelty_at_k(self):
        """Test Novelty@K calculation."""
        y_pred_list = [['item1', 'item2'], ['item2', 'item3']]
        item_popularity = {'item1': 10, 'item2': 5, 'item3': 1}
        
        novelty = Metrics.novelty_at_k(y_pred_list, item_popularity, k=2)
        assert 0 <= novelty <= 1


class TestEvaluator:
    """Test cases for Evaluator."""
    
    def setup_method(self):
        """Set up test data."""
        self.generator = DataGenerator(seed=42)
        self.items_df = self.generator.generate_items(n_items=5)
        self.users_df = self.generator.generate_users(n_users=3)
        self.interactions_df = self.generator.generate_interactions(
            self.users_df, self.items_df, n_interactions=10
        )
        
        # Add dummy features
        for i in range(5):
            self.items_df[f'feature_{i}'] = np.random.random(len(self.items_df))
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        model = ContentBasedRecommender()
        model.fit(self.interactions_df, self.items_df, self.users_df)
        
        evaluator = Evaluator()
        results = evaluator.evaluate_model(model, self.interactions_df, self.items_df)
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_compare_models(self):
        """Test model comparison."""
        models = {
            "Content-Based": ContentBasedRecommender(),
            "Cold-Start": ColdStartRecommender()
        }
        
        for model in models.values():
            model.fit(self.interactions_df, self.items_df, self.users_df)
        
        evaluator = Evaluator()
        results_df = evaluator.compare_models(models, self.interactions_df, self.items_df)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(models)
        assert 'model' in results_df.columns


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy seed
        np.random.seed(42)
        random_array1 = np.random.random(5)
        
        set_seed(42)
        random_array2 = np.random.random(5)
        
        np.testing.assert_array_equal(random_array1, random_array2)


if __name__ == "__main__":
    pytest.main([__file__])
