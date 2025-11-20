#!/usr/bin/env python3
"""Data generation script for cold-start recommendation system."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cold_start.data import DataGenerator
from cold_start.utils import set_seed, load_config, ensure_dir

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate synthetic data for cold-start recommendation evaluation."""
    # Load configuration
    config = load_config("configs/config.yaml")
    data_config = config["data"]
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create data directory
    data_dir = ensure_dir("data")
    
    logger.info("Generating synthetic data...")
    
    # Initialize data generator
    generator = DataGenerator(seed=42)
    
    # Generate data
    logger.info(f"Generating {data_config['n_items']} items...")
    items_df = generator.generate_items(data_config["n_items"])
    
    logger.info(f"Generating {data_config['n_users']} users...")
    users_df = generator.generate_users(data_config["n_users"])
    
    logger.info(f"Generating {data_config['n_interactions']} interactions...")
    interactions_df = generator.generate_interactions(
        users_df, items_df, data_config["n_interactions"]
    )
    
    # Create cold-start scenarios
    logger.info("Creating cold-start scenarios...")
    filtered_interactions, cold_users, cold_items = generator.create_cold_start_scenarios(
        interactions_df, users_df, items_df,
        data_config["cold_user_ratio"], data_config["cold_item_ratio"]
    )
    
    # Save data
    logger.info("Saving data files...")
    items_df.to_csv(data_dir / "items.csv", index=False)
    users_df.to_csv(data_dir / "users.csv", index=False)
    interactions_df.to_csv(data_dir / "interactions.csv", index=False)
    filtered_interactions.to_csv(data_dir / "interactions_filtered.csv", index=False)
    
    # Save cold-start metadata
    import json
    cold_start_info = {
        "cold_users": cold_users,
        "cold_items": cold_items,
        "n_cold_users": len(cold_users),
        "n_cold_items": len(cold_items),
        "n_total_interactions": len(interactions_df),
        "n_filtered_interactions": len(filtered_interactions)
    }
    
    with open(data_dir / "cold_start_info.json", "w") as f:
        json.dump(cold_start_info, f, indent=2)
    
    logger.info("Data generation completed!")
    logger.info(f"Generated {len(items_df)} items, {len(users_df)} users, {len(interactions_df)} interactions")
    logger.info(f"Cold-start scenario: {len(cold_users)} cold users, {len(cold_items)} cold items")


if __name__ == "__main__":
    main()
