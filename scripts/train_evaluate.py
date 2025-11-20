#!/usr/bin/env python3
"""Main training and evaluation script for cold-start recommendation system."""

import sys
import logging
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cold_start.data import DataLoader
from cold_start.models import ContentBasedRecommender, HybridRecommender, ColdStartRecommender
from cold_start.evaluation import Evaluator
from cold_start.utils import set_seed, load_config, ensure_dir

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train and evaluate cold-start recommendation models."""
    # Load configuration
    config = load_config("configs/config.yaml")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create output directory
    output_dir = ensure_dir("models")
    
    logger.info("Loading data...")
    
    # Load data
    data_loader = DataLoader("data")
    interactions_df, items_df, users_df = data_loader.load_data()
    
    # Load cold-start information
    with open("data/cold_start_info.json", "r") as f:
        cold_start_info = json.load(f)
    
    cold_users = cold_start_info["cold_users"]
    cold_items = cold_start_info["cold_items"]
    
    # Split data
    train_df, test_df = data_loader.split_data(interactions_df, config["data"]["test_ratio"])
    
    logger.info(f"Training data: {len(train_df)} interactions")
    logger.info(f"Test data: {len(test_df)} interactions")
    
    # Prepare features
    logger.info("Preparing features...")
    item_features = data_loader.prepare_item_features(items_df)
    user_features = data_loader.prepare_user_features(users_df)
    
    # Add features to DataFrames
    items_df_with_features = items_df.copy()
    for i, feature_name in enumerate([f"feature_{j}" for j in range(item_features.shape[1])]):
        items_df_with_features[feature_name] = item_features[:, i]
    
    users_df_with_features = users_df.copy()
    for i, feature_name in enumerate([f"user_feature_{j}" for j in range(user_features.shape[1])]):
        users_df_with_features[feature_name] = user_features[:, i]
    
    # Initialize models
    logger.info("Initializing models...")
    models = {
        "Content-Based": ContentBasedRecommender(
            similarity_threshold=config["models"]["content_based"]["similarity_threshold"]
        ),
        "Hybrid": HybridRecommender(
            content_weight=config["models"]["hybrid"]["content_weight"],
            collaborative_weight=config["models"]["hybrid"]["collaborative_weight"]
        ),
        "Cold-Start": ColdStartRecommender(
            use_popularity=config["models"]["cold_start"]["use_popularity"],
            use_diversity=config["models"]["cold_start"]["use_diversity"]
        )
    }
    
    # Train models
    logger.info("Training models...")
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        model.fit(train_df, items_df_with_features, users_df_with_features)
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = Evaluator(config["evaluation"]["metrics"])
    
    # Regular evaluation
    results_df = evaluator.compare_models(
        models, test_df, items_df_with_features, config["evaluation"]["k_values"]
    )
    
    # Cold-start evaluation
    logger.info("Evaluating cold-start performance...")
    cold_start_results = {}
    for model_name, model in models.items():
        logger.info(f"Evaluating cold-start performance for {model_name}...")
        cold_results = evaluator.evaluate_cold_start(
            model, cold_users, cold_items, test_df, items_df_with_features,
            config["evaluation"]["k_values"]
        )
        cold_start_results[model_name] = cold_results
    
    # Create leaderboard
    leaderboard = evaluator.create_leaderboard(
        results_df, config["evaluation"]["primary_metric"]
    )
    
    # Save results
    logger.info("Saving results...")
    results_df.to_csv(output_dir / "model_results.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    
    # Save cold-start results
    with open(output_dir / "cold_start_results.json", "w") as f:
        json.dump(cold_start_results, f, indent=2)
    
    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(leaderboard.to_string(index=False))
    
    print("\n" + "="*80)
    print("COLD-START PERFORMANCE SUMMARY")
    print("="*80)
    
    primary_metric = config["evaluation"]["primary_metric"]
    for model_name, results in cold_start_results.items():
        cold_user_metric = f"cold_user_{primary_metric}"
        if cold_user_metric in results:
            print(f"{model_name}: {cold_user_metric} = {results[cold_user_metric]:.4f}")
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()
