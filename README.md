# Cold-Start Recommendation System

A comprehensive solution for addressing the cold-start problem in recommendation systems, featuring multiple approaches including content-based filtering, hybrid methods, and advanced cold-start techniques.

## Overview

The cold-start problem occurs when there is insufficient data for making recommendations, commonly happening with:
- **New users** with no interaction history (user-based cold start)
- **New items** with no user interactions (item-based cold start)

This project implements and compares multiple strategies to mitigate these challenges:

1. **Content-Based Filtering**: Recommends items based on their features and characteristics
2. **Hybrid Approaches**: Combines content-based and collaborative filtering methods
3. **Advanced Cold-Start Techniques**: Uses popularity, diversity, and sophisticated feature engineering

## Features

- **Multiple Model Architectures**: Content-based, hybrid, and specialized cold-start models
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Diversity, Novelty
- **Interactive Demo**: Streamlit-based web application for exploring recommendations
- **Realistic Data Generation**: Synthetic data with realistic user behavior patterns
- **Production-Ready Code**: Type hints, comprehensive testing, CI/CD pipeline
- **Cold-Start Scenarios**: Built-in support for evaluating cold-start performance

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Cold-Start-Recommendation-System.git
cd Cold-Start-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate synthetic data:
```bash
python scripts/generate_data.py
```

4. Train and evaluate models:
```bash
python scripts/train_evaluate.py
```

5. Launch the interactive demo:
```bash
streamlit run scripts/demo.py
```

### Basic Usage

```python
from cold_start.data import DataLoader
from cold_start.models import ContentBasedRecommender, HybridRecommender
from cold_start.evaluation import Evaluator

# Load data
data_loader = DataLoader("data")
interactions_df, items_df, users_df = data_loader.load_data()

# Initialize models
content_model = ContentBasedRecommender()
hybrid_model = HybridRecommender()

# Train models
content_model.fit(interactions_df, items_df, users_df)
hybrid_model.fit(interactions_df, items_df, users_df)

# Generate recommendations
recommendations = content_model.recommend("user_001", n_recommendations=10)
cold_start_recs = content_model.recommend_for_cold_user(n_recommendations=10)

# Evaluate performance
evaluator = Evaluator()
results = evaluator.evaluate_model(content_model, test_data, items_df)
```

## Project Structure

```
0335_Cold-start_problem_solutions/
├── src/
│   └── cold_start/
│       ├── __init__.py
│       ├── data.py              # Data loading and generation
│       ├── models.py            # Recommendation models
│       ├── evaluation.py       # Evaluation metrics and comparison
│       └── utils.py            # Utility functions
├── data/                       # Generated data files
├── models/                     # Trained model outputs
├── configs/
│   └── config.yaml            # Configuration settings
├── scripts/
│   ├── generate_data.py       # Data generation script
│   ├── train_evaluate.py      # Training and evaluation
│   └── demo.py               # Streamlit demo application
├── tests/
│   └── test_cold_start.py     # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Data Schema

### Interactions Data (`interactions.csv`)
- `user_id`: User identifier
- `item_id`: Item identifier  
- `rating`: User rating (1-5)
- `timestamp`: Interaction timestamp
- `weight`: Interaction weight

### Items Data (`items.csv`)
- `item_id`: Item identifier
- `title`: Item title
- `category`: Item category
- `brand`: Item brand
- `price`: Item price
- `rating_avg`: Average rating
- `rating_count`: Number of ratings
- `description`: Item description
- `tags`: Item tags

### Users Data (`users.csv`)
- `user_id`: User identifier
- `age_group`: User age group
- `location`: User location
- `signup_date`: User signup date
- `preferred_categories`: User's preferred categories

## Models

### Content-Based Recommender
- Uses item features to build user profiles
- Recommends items similar to user's historical preferences
- Effective for cold-start users with some interaction history

### Hybrid Recommender
- Combines content-based and collaborative filtering
- Balances between content similarity and user behavior patterns
- Provides robust recommendations for both warm and cold users

### Cold-Start Recommender
- Specialized for extreme cold-start scenarios
- Uses popularity-based recommendations for new users
- Implements diversity techniques (MMR) for better recommendation quality
- Supports both user and item cold-start scenarios

## Evaluation Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation
- **Coverage@K**: Fraction of items that can be recommended
- **Diversity@K**: Intra-list diversity of recommendations
- **Novelty@K**: Average novelty of recommended items

## Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Data settings
data:
  n_users: 500
  n_items: 1000
  n_interactions: 10000
  cold_user_ratio: 0.1
  cold_item_ratio: 0.1
  test_ratio: 0.2

# Model settings
models:
  content_based:
    similarity_threshold: 0.1
  hybrid:
    content_weight: 0.7
    collaborative_weight: 0.3
  cold_start:
    use_popularity: true
    use_diversity: true

# Evaluation settings
evaluation:
  metrics: ['precision', 'recall', 'hit_rate', 'ndcg', 'map']
  k_values: [5, 10, 20]
  primary_metric: 'ndcg@10'
```

## Interactive Demo

The Streamlit demo provides:

1. **User Recommendations**: Select users and get personalized recommendations
2. **Model Comparison**: Compare performance across different models
3. **Cold-Start Analysis**: Analyze cold-start scenarios and performance
4. **Data Overview**: Explore dataset statistics and patterns

Launch with:
```bash
streamlit run scripts/demo.py
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src/ --cov-report=html
```

## Development

### Code Quality

The project uses several tools for code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pytest**: Testing framework

Run all quality checks:
```bash
black src/ scripts/ tests/
ruff check src/ scripts/ tests/
mypy src/
pytest tests/
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Performance

The system is designed for efficiency:

- **Caching**: Streamlit demo uses caching for fast data loading
- **Vectorized Operations**: NumPy-based computations for speed
- **Memory Efficient**: Handles large datasets without excessive memory usage
- **Scalable Architecture**: Modular design allows for easy scaling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cold_start_recommendation_system,
  title={Cold-Start Recommendation System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Cold-Start-Recommendation-System}
}
```

## Acknowledgments

- Built with modern Python libraries: scikit-learn, pandas, numpy, streamlit
- Inspired by research in cold-start recommendation systems
- Designed for educational and research purposes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Data Not Found**: Run `python scripts/generate_data.py` first
3. **Memory Issues**: Reduce dataset size in `configs/config.yaml`
4. **Streamlit Issues**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Getting Help

- Check the test files for usage examples
- Review the configuration file for parameter tuning
- Examine the demo application for interactive exploration
- Run the evaluation script to understand model performance
# Cold-Start-Recommendation-System
