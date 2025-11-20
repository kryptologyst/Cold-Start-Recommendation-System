"""Cold-start recommendation models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation models."""
    
    @abstractmethod
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, 
            users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the recommendation model.
        
        Args:
            interactions_df: User-item interactions.
            items_df: Item features.
            users_df: User features (optional).
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        pass
    
    @abstractmethod
    def recommend_for_cold_user(self, user_features: Optional[np.ndarray] = None,
                               n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a cold-start user.
        
        Args:
            user_features: User feature vector (optional).
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        pass
    
    @abstractmethod
    def recommend_for_cold_item(self, item_id: str, n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for users who might like a cold-start item.
        
        Args:
            item_id: Item identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended user IDs.
        """
        pass


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation model for cold-start scenarios."""
    
    def __init__(self, similarity_threshold: float = 0.1):
        """Initialize content-based recommender.
        
        Args:
            similarity_threshold: Minimum similarity threshold for recommendations.
        """
        self.similarity_threshold = similarity_threshold
        self.item_features = None
        self.item_ids = None
        self.user_profiles = {}
        self.item_similarity_matrix = None
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, 
            users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the content-based model.
        
        Args:
            interactions_df: User-item interactions.
            items_df: Item features.
            users_df: User features (optional).
        """
        self.item_ids = items_df['item_id'].tolist()
        self.item_features = items_df.drop('item_id', axis=1).values
        
        # Compute item similarity matrix
        self.item_similarity_matrix = cosine_similarity(self.item_features)
        
        # Build user profiles based on their interactions
        for user_id in interactions_df['user_id'].unique():
            user_items = interactions_df[interactions_df['user_id'] == user_id]
            user_profile = self._build_user_profile(user_items, items_df)
            self.user_profiles[user_id] = user_profile
            
        logger.info(f"Fitted content-based model for {len(self.user_profiles)} users and {len(self.item_ids)} items")
    
    def _build_user_profile(self, user_items: pd.DataFrame, items_df: pd.DataFrame) -> np.ndarray:
        """Build user profile from their item interactions.
        
        Args:
            user_items: User's item interactions.
            items_df: Item features.
            
        Returns:
            User profile vector.
        """
        profile = np.zeros(self.item_features.shape[1])
        
        for _, interaction in user_items.iterrows():
            item_idx = self.item_ids.index(interaction['item_id'])
            # Weight by rating
            weight = interaction.get('rating', 1.0) / 5.0
            profile += weight * self.item_features[item_idx]
        
        # Normalize
        if np.sum(profile) > 0:
            profile = profile / np.sum(profile)
            
        return profile
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a known user.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        if user_id not in self.user_profiles:
            return self.recommend_for_cold_user(n_recommendations=n_recommendations)
        
        user_profile = self.user_profiles[user_id]
        scores = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [self.item_ids[i] for i in top_indices if scores[i] > self.similarity_threshold]
        
        return recommendations
    
    def recommend_for_cold_user(self, user_features: Optional[np.ndarray] = None,
                               n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a cold-start user.
        
        Args:
            user_features: User feature vector (optional).
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        if user_features is not None:
            # Use user features to find similar users and their preferences
            scores = cosine_similarity([user_features], self.item_features)[0]
        else:
            # Fallback: recommend popular items
            scores = np.random.random(len(self.item_ids))
        
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        return [self.item_ids[i] for i in top_indices]
    
    def recommend_for_cold_item(self, item_id: str, n_recommendations: int = 10) -> List[str]:
        """Find users who might like a cold-start item.
        
        Args:
            item_id: Item identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended user IDs.
        """
        if item_id not in self.item_ids:
            return []
        
        item_idx = self.item_ids.index(item_id)
        item_features = self.item_features[item_idx]
        
        # Find users with similar preferences
        user_scores = {}
        for user_id, user_profile in self.user_profiles.items():
            similarity = cosine_similarity([user_profile], [item_features])[0][0]
            user_scores[user_id] = similarity
        
        # Sort by similarity and return top users
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in sorted_users[:n_recommendations]]


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation model combining content-based and collaborative filtering."""
    
    def __init__(self, content_weight: float = 0.7, collaborative_weight: float = 0.3):
        """Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based component.
            collaborative_weight: Weight for collaborative filtering component.
        """
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.content_model = ContentBasedRecommender()
        self.collaborative_model = None
        self.user_item_matrix = None
        self.item_ids = None
        self.user_ids = None
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, 
            users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the hybrid model.
        
        Args:
            interactions_df: User-item interactions.
            items_df: Item features.
            users_df: User features (optional).
        """
        # Fit content-based component
        self.content_model.fit(interactions_df, items_df, users_df)
        
        # Build collaborative filtering component
        self._build_collaborative_model(interactions_df)
        
        logger.info("Fitted hybrid recommendation model")
    
    def _build_collaborative_model(self, interactions_df: pd.DataFrame) -> None:
        """Build collaborative filtering model.
        
        Args:
            interactions_df: User-item interactions.
        """
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_ids = self.user_item_matrix.columns.tolist()
        
        # Simple collaborative filtering using item-based similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_matrix = pd.DataFrame(
            item_similarity, 
            index=self.item_ids, 
            columns=self.item_ids
        )
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Generate hybrid recommendations for a user.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        # Get content-based recommendations
        content_recs = self.content_model.recommend(user_id, n_recommendations)
        
        # Get collaborative filtering recommendations
        collab_recs = self._get_collaborative_recommendations(user_id, n_recommendations)
        
        # Combine recommendations
        combined_scores = {}
        
        # Add content-based scores
        for item_id in content_recs:
            combined_scores[item_id] = self.content_weight
        
        # Add collaborative scores
        for item_id in collab_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += self.collaborative_weight
            else:
                combined_scores[item_id] = self.collaborative_weight
        
        # Sort by combined score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]
    
    def _get_collaborative_recommendations(self, user_id: str, n_recommendations: int) -> List[str]:
        """Get collaborative filtering recommendations.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        if user_id not in self.user_ids:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        user_items = user_ratings[user_ratings > 0].index.tolist()
        
        if not user_items:
            return []
        
        # Find similar items to user's rated items
        item_scores = {}
        for rated_item in user_items:
            if rated_item in self.item_similarity_matrix.index:
                similarities = self.item_similarity_matrix.loc[rated_item]
                for item_id, similarity in similarities.items():
                    if item_id not in user_items:  # Don't recommend already rated items
                        if item_id in item_scores:
                            item_scores[item_id] = max(item_scores[item_id], similarity)
                        else:
                            item_scores[item_id] = similarity
        
        # Sort by similarity score
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:n_recommendations]]
    
    def recommend_for_cold_user(self, user_features: Optional[np.ndarray] = None,
                               n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a cold-start user.
        
        Args:
            user_features: User feature vector (optional).
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        # For cold users, rely more on content-based recommendations
        return self.content_model.recommend_for_cold_user(user_features, n_recommendations)
    
    def recommend_for_cold_item(self, item_id: str, n_recommendations: int = 10) -> List[str]:
        """Find users who might like a cold-start item.
        
        Args:
            item_id: Item identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended user IDs.
        """
        return self.content_model.recommend_for_cold_item(item_id, n_recommendations)


class ColdStartRecommender(BaseRecommender):
    """Advanced cold-start recommendation model using multiple strategies."""
    
    def __init__(self, use_popularity: bool = True, use_diversity: bool = True):
        """Initialize cold-start recommender.
        
        Args:
            use_popularity: Whether to use popularity-based recommendations.
            use_diversity: Whether to ensure diversity in recommendations.
        """
        self.use_popularity = use_popularity
        self.use_diversity = use_diversity
        self.item_popularity = {}
        self.item_features = None
        self.item_ids = None
        self.user_profiles = {}
        
    def fit(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame, 
            users_df: Optional[pd.DataFrame] = None) -> None:
        """Fit the cold-start model.
        
        Args:
            interactions_df: User-item interactions.
            items_df: Item features.
            users_df: User features (optional).
        """
        self.item_ids = items_df['item_id'].tolist()
        self.item_features = items_df.drop('item_id', axis=1).values
        
        # Calculate item popularity
        if self.use_popularity:
            self.item_popularity = interactions_df['item_id'].value_counts().to_dict()
        
        # Build user profiles for warm users
        for user_id in interactions_df['user_id'].unique():
            user_items = interactions_df[interactions_df['user_id'] == user_id]
            user_profile = self._build_user_profile(user_items, items_df)
            self.user_profiles[user_id] = user_profile
        
        logger.info(f"Fitted cold-start model for {len(self.user_profiles)} users and {len(self.item_ids)} items")
    
    def _build_user_profile(self, user_items: pd.DataFrame, items_df: pd.DataFrame) -> np.ndarray:
        """Build user profile from interactions.
        
        Args:
            user_items: User's item interactions.
            items_df: Item features.
            
        Returns:
            User profile vector.
        """
        profile = np.zeros(self.item_features.shape[1])
        
        for _, interaction in user_items.iterrows():
            item_idx = self.item_ids.index(interaction['item_id'])
            weight = interaction.get('rating', 1.0) / 5.0
            profile += weight * self.item_features[item_idx]
        
        if np.sum(profile) > 0:
            profile = profile / np.sum(profile)
            
        return profile
    
    def recommend(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        if user_id not in self.user_profiles:
            return self.recommend_for_cold_user(n_recommendations=n_recommendations)
        
        user_profile = self.user_profiles[user_id]
        scores = cosine_similarity([user_profile], self.item_features)[0]
        
        # Apply diversity if enabled
        if self.use_diversity:
            recommendations = self._diversify_recommendations(scores, n_recommendations)
        else:
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            recommendations = [self.item_ids[i] for i in top_indices]
        
        return recommendations
    
    def recommend_for_cold_user(self, user_features: Optional[np.ndarray] = None,
                               n_recommendations: int = 10) -> List[str]:
        """Generate recommendations for a cold-start user.
        
        Args:
            user_features: User feature vector (optional).
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended item IDs.
        """
        if self.use_popularity:
            # Use popularity-based recommendations
            popular_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
            recommendations = [item_id for item_id, _ in popular_items[:n_recommendations]]
        else:
            # Use random recommendations
            recommendations = np.random.choice(self.item_ids, n_recommendations, replace=False).tolist()
        
        return recommendations
    
    def recommend_for_cold_item(self, item_id: str, n_recommendations: int = 10) -> List[str]:
        """Find users who might like a cold-start item.
        
        Args:
            item_id: Item identifier.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of recommended user IDs.
        """
        if item_id not in self.item_ids:
            return []
        
        item_idx = self.item_ids.index(item_id)
        item_features = self.item_features[item_idx]
        
        # Find users with similar preferences
        user_scores = {}
        for user_id, user_profile in self.user_profiles.items():
            similarity = cosine_similarity([user_profile], [item_features])[0][0]
            user_scores[user_id] = similarity
        
        # Sort by similarity
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in sorted_users[:n_recommendations]]
    
    def _diversify_recommendations(self, scores: np.ndarray, n_recommendations: int) -> List[str]:
        """Apply diversity to recommendations using MMR (Maximal Marginal Relevance).
        
        Args:
            scores: Item similarity scores.
            n_recommendations: Number of recommendations to generate.
            
        Returns:
            List of diverse recommended item IDs.
        """
        lambda_param = 0.7  # Balance between relevance and diversity
        
        recommendations = []
        remaining_items = list(range(len(self.item_ids)))
        
        # Start with the most relevant item
        best_idx = np.argmax(scores)
        recommendations.append(self.item_ids[best_idx])
        remaining_items.remove(best_idx)
        
        # Add items using MMR
        while len(recommendations) < n_recommendations and remaining_items:
            best_score = -np.inf
            best_idx = None
            
            for idx in remaining_items:
                # Relevance score
                relevance = scores[idx]
                
                # Diversity score (max similarity to already selected items)
                diversity = 0
                if recommendations:
                    selected_features = [self.item_features[self.item_ids.index(item_id)] 
                                      for item_id in recommendations]
                    similarities = cosine_similarity([self.item_features[idx]], selected_features)[0]
                    diversity = np.max(similarities)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                recommendations.append(self.item_ids[best_idx])
                remaining_items.remove(best_idx)
            else:
                break
        
        return recommendations
