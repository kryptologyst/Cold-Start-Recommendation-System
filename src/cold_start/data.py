"""Data loading and generation utilities for cold-start recommendations."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sentence_transformers import SentenceTransformer

from .utils import set_seed, ensure_dir

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate realistic synthetic data for cold-start recommendation evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        set_seed(seed)
        self.seed = seed
        
    def generate_items(self, n_items: int = 1000) -> pd.DataFrame:
        """Generate synthetic item data with features.
        
        Args:
            n_items: Number of items to generate.
            
        Returns:
            DataFrame with item features.
        """
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF']
        
        items = []
        for i in range(n_items):
            item = {
                'item_id': f'item_{i:04d}',
                'title': f'Product {i}',
                'category': np.random.choice(categories),
                'brand': np.random.choice(brands),
                'price': np.random.lognormal(4, 1),  # Log-normal price distribution
                'rating_avg': np.random.uniform(2.5, 5.0),
                'rating_count': np.random.poisson(50),
                'description': f'High-quality {np.random.choice(categories).lower()} product with excellent features.',
                'tags': ','.join(np.random.choice(['premium', 'popular', 'new', 'sale', 'featured'], 
                                                 size=np.random.randint(1, 4), replace=False))
            }
            items.append(item)
            
        return pd.DataFrame(items)
    
    def generate_users(self, n_users: int = 500) -> pd.DataFrame:
        """Generate synthetic user data.
        
        Args:
            n_users: Number of users to generate.
            
        Returns:
            DataFrame with user features.
        """
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR']
        
        users = []
        for i in range(n_users):
            user = {
                'user_id': f'user_{i:04d}',
                'age_group': np.random.choice(age_groups),
                'location': np.random.choice(locations),
                'signup_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365)),
                'preferred_categories': ','.join(np.random.choice(
                    ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], 
                    size=np.random.randint(1, 3), replace=False
                ))
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_interactions(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
                            n_interactions: int = 10000) -> pd.DataFrame:
        """Generate realistic user-item interactions with cold-start scenarios.
        
        Args:
            users_df: User DataFrame.
            items_df: Item DataFrame.
            n_interactions: Number of interactions to generate.
            
        Returns:
            DataFrame with user-item interactions.
        """
        interactions = []
        
        # Generate interactions with popularity bias and user preferences
        for _ in range(n_interactions):
            user_id = np.random.choice(users_df['user_id'])
            item_id = np.random.choice(items_df['item_id'])
            
            # Get user and item info
            user = users_df[users_df['user_id'] == user_id].iloc[0]
            item = items_df[items_df['item_id'] == item_id].iloc[0]
            
            # Simulate rating based on item popularity and user preferences
            base_rating = item['rating_avg']
            
            # Add preference bonus if item category matches user preferences
            if item['category'] in user['preferred_categories']:
                base_rating += 0.5
            
            # Add some noise
            rating = max(1, min(5, np.random.normal(base_rating, 0.8)))
            rating = int(round(rating))
            
            # Generate timestamp with recency bias
            timestamp = pd.Timestamp.now() - pd.Timedelta(
                days=np.random.exponential(30)  # Exponential decay for recency
            )
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp,
                'weight': 1.0
            })
            
        return pd.DataFrame(interactions)
    
    def create_cold_start_scenarios(self, interactions_df: pd.DataFrame, 
                                 users_df: pd.DataFrame, items_df: pd.DataFrame,
                                 cold_user_ratio: float = 0.1,
                                 cold_item_ratio: float = 0.1) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Create cold-start scenarios by removing interactions for some users/items.
        
        Args:
            interactions_df: Full interactions DataFrame.
            users_df: User DataFrame.
            items_df: Item DataFrame.
            cold_user_ratio: Fraction of users to make cold-start.
            cold_item_ratio: Fraction of items to make cold-start.
            
        Returns:
            Tuple of (filtered interactions, cold users, cold items).
        """
        # Select cold-start users (users with few interactions)
        user_interaction_counts = interactions_df['user_id'].value_counts()
        cold_users = user_interaction_counts[
            user_interaction_counts <= user_interaction_counts.quantile(cold_user_ratio)
        ].index.tolist()
        
        # Select cold-start items (items with few interactions)
        item_interaction_counts = interactions_df['item_id'].value_counts()
        cold_items = item_interaction_counts[
            item_interaction_counts <= item_interaction_counts.quantile(cold_item_ratio)
        ].index.tolist()
        
        # Remove interactions for cold-start users and items
        filtered_interactions = interactions_df[
            (~interactions_df['user_id'].isin(cold_users)) & 
            (~interactions_df['item_id'].isin(cold_items))
        ].copy()
        
        logger.info(f"Created cold-start scenario: {len(cold_users)} cold users, {len(cold_items)} cold items")
        logger.info(f"Reduced interactions from {len(interactions_df)} to {len(filtered_interactions)}")
        
        return filtered_interactions, cold_users, cold_items


class DataLoader:
    """Load and preprocess data for cold-start recommendations."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files.
        """
        self.data_dir = Path(data_dir)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load interaction, item, and user data.
        
        Returns:
            Tuple of (interactions, items, users) DataFrames.
        """
        interactions_path = self.data_dir / "interactions.csv"
        items_path = self.data_dir / "items.csv"
        users_path = self.data_dir / "users.csv"
        
        if not all(p.exists() for p in [interactions_path, items_path, users_path]):
            raise FileNotFoundError("Required data files not found. Run data generation first.")
        
        interactions = pd.read_csv(interactions_path)
        items = pd.read_csv(items_path)
        users = pd.read_csv(users_path)
        
        # Convert timestamps
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
        users['signup_date'] = pd.to_datetime(users['signup_date'])
        
        return interactions, items, users
    
    def prepare_item_features(self, items_df: pd.DataFrame) -> np.ndarray:
        """Prepare item features for content-based recommendations.
        
        Args:
            items_df: Items DataFrame.
            
        Returns:
            Feature matrix for items.
        """
        # Text features from description and tags
        text_features = self.tfidf_vectorizer.fit_transform(
            items_df['description'] + ' ' + items_df['tags'].fillna('')
        ).toarray()
        
        # Categorical features
        category_encoded = self._encode_categorical(items_df['category'])
        brand_encoded = self._encode_categorical(items_df['brand'])
        
        # Numerical features
        numerical_features = items_df[['price', 'rating_avg', 'rating_count']].values
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine all features
        features = np.hstack([
            text_features,
            category_encoded,
            brand_encoded,
            numerical_features
        ])
        
        return features
    
    def prepare_user_features(self, users_df: pd.DataFrame) -> np.ndarray:
        """Prepare user features for recommendations.
        
        Args:
            users_df: Users DataFrame.
            
        Returns:
            Feature matrix for users.
        """
        # Categorical features
        age_encoded = self._encode_categorical(users_df['age_group'])
        location_encoded = self._encode_categorical(users_df['location'])
        
        # Time-based features
        days_since_signup = (pd.Timestamp.now() - users_df['signup_date']).dt.days.values.reshape(-1, 1)
        days_since_signup = self.scaler.fit_transform(days_since_signup)
        
        # Combine features
        features = np.hstack([
            age_encoded,
            location_encoded,
            days_since_signup
        ])
        
        return features
    
    def _encode_categorical(self, series: pd.Series) -> np.ndarray:
        """Encode categorical features.
        
        Args:
            series: Categorical series to encode.
            
        Returns:
            One-hot encoded features.
        """
        if series.name not in self.label_encoders:
            self.label_encoders[series.name] = LabelEncoder()
        
        encoded = self.label_encoders[series.name].fit_transform(series)
        
        # Convert to one-hot encoding
        n_classes = len(self.label_encoders[series.name].classes_)
        one_hot = np.zeros((len(encoded), n_classes))
        one_hot[np.arange(len(encoded)), encoded] = 1
        
        return one_hot
    
    def get_item_embeddings(self, items_df: pd.DataFrame) -> np.ndarray:
        """Get semantic embeddings for items using sentence transformers.
        
        Args:
            items_df: Items DataFrame.
            
        Returns:
            Item embeddings matrix.
        """
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            texts = items_df['title'] + ' ' + items_df['description'] + ' ' + items_df['tags'].fillna('')
            embeddings = model.encode(texts)
            return embeddings
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            # Fallback to TF-IDF
            return self.tfidf_vectorizer.fit_transform(
                items_df['title'] + ' ' + items_df['description']
            ).toarray()
    
    def split_data(self, interactions_df: pd.DataFrame, 
                  test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions into train and test sets chronologically.
        
        Args:
            interactions_df: Interactions DataFrame.
            test_ratio: Fraction of data to use for testing.
            
        Returns:
            Tuple of (train, test) DataFrames.
        """
        # Sort by timestamp
        interactions_sorted = interactions_df.sort_values('timestamp')
        
        # Split chronologically
        split_idx = int(len(interactions_sorted) * (1 - test_ratio))
        train_df = interactions_sorted.iloc[:split_idx]
        test_df = interactions_sorted.iloc[split_idx:]
        
        return train_df, test_df
