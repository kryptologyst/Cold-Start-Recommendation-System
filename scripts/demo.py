"""Streamlit demo for cold-start recommendation system."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cold_start.data import DataLoader
from cold_start.models import ContentBasedRecommender, HybridRecommender, ColdStartRecommender
from cold_start.utils import set_seed, load_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Cold-Start Recommendation System",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed
set_seed(42)

@st.cache_data
def load_data():
    """Load and cache data."""
    try:
        data_loader = DataLoader("data")
        interactions_df, items_df, users_df = data_loader.load_data()
        
        # Load cold-start information
        with open("data/cold_start_info.json", "r") as f:
            cold_start_info = json.load(f)
        
        # Prepare features
        item_features = data_loader.prepare_item_features(items_df)
        user_features = data_loader.prepare_user_features(users_df)
        
        # Add features to DataFrames
        items_df_with_features = items_df.copy()
        for i, feature_name in enumerate([f"feature_{j}" for j in range(item_features.shape[1])]):
            items_df_with_features[feature_name] = item_features[:, i]
        
        users_df_with_features = users_df.copy()
        for i, feature_name in enumerate([f"user_feature_{j}" for j in range(user_features.shape[1])]):
            users_df_with_features[feature_name] = user_features[:, i]
        
        return interactions_df, items_df_with_features, users_df_with_features, cold_start_info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

@st.cache_resource
def load_models():
    """Load and cache trained models."""
    try:
        config = load_config("configs/config.yaml")
        
        # Load data
        data_loader = DataLoader("data")
        interactions_df, items_df, users_df = data_loader.load_data()
        
        # Prepare features
        item_features = data_loader.prepare_item_features(items_df)
        user_features = data_loader.prepare_user_features(users_df)
        
        # Add features to DataFrames
        items_df_with_features = items_df.copy()
        for i, feature_name in enumerate([f"feature_{j}" for j in range(item_features.shape[1])]):
            items_df_with_features[feature_name] = item_features[:, i]
        
        users_df_with_features = users_df.copy()
        for i, feature_name in enumerate([f"user_feature_{j}" for j in range(user_features.shape[1])]):
            users_df_with_features[feature_name] = user_features[:, i]
        
        # Load training data
        train_df, _ = data_loader.split_data(interactions_df, config["data"]["test_ratio"])
        
        # Initialize and train models
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
        for model_name, model in models.items():
            model.fit(train_df, items_df_with_features, users_df_with_features)
        
        return models, items_df_with_features, users_df_with_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    """Main Streamlit application."""
    st.title("❄️ Cold-Start Recommendation System")
    st.markdown("Explore solutions for the cold-start problem in recommendation systems")
    
    # Load data
    with st.spinner("Loading data..."):
        interactions_df, items_df, users_df, cold_start_info = load_data()
    
    if interactions_df is None:
        st.error("Failed to load data. Please run the data generation script first.")
        return
    
    # Load models
    with st.spinner("Loading models..."):
        models, items_df_features, users_df_features = load_models()
    
    if models is None:
        st.error("Failed to load models. Please run the training script first.")
        return
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        help="Choose which recommendation model to use"
    )
    
    # Number of recommendations
    n_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=50,
        value=10,
        help="Number of items to recommend"
    )
    
    # Show explanations
    show_explanations = st.sidebar.checkbox(
        "Show Explanations",
        value=True,
        help="Show why items were recommended"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 User Recommendations", "📊 Model Comparison", "❄️ Cold-Start Analysis", "📈 Data Overview"])
    
    with tab1:
        st.header("User Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select User")
            
            # User selection
            user_options = ["Cold-Start User (New)"] + users_df['user_id'].tolist()
            selected_user = st.selectbox("Choose a user", user_options)
            
            # Cold-start user simulation
            if selected_user == "Cold-Start User (New)":
                st.info("Simulating recommendations for a new user with no interaction history")
                user_id = "cold_start_user"
                user_history = []
            else:
                user_id = selected_user
                user_history = interactions_df[interactions_df['user_id'] == user_id]['item_id'].tolist()
            
            # Display user info
            if user_id != "cold_start_user":
                user_info = users_df[users_df['user_id'] == user_id].iloc[0]
                st.write("**User Information:**")
                st.write(f"- Age Group: {user_info['age_group']}")
                st.write(f"- Location: {user_info['location']}")
                st.write(f"- Preferred Categories: {user_info['preferred_categories']}")
                st.write(f"- Signup Date: {user_info['signup_date'].strftime('%Y-%m-%d')}")
            
            # Display interaction history
            if user_history:
                st.write(f"**Interaction History ({len(user_history)} items):**")
                history_items = items_df[items_df['item_id'].isin(user_history)]
                for _, item in history_items.iterrows():
                    st.write(f"- {item['title']} ({item['category']}) - Rating: {item.get('rating', 'N/A')}")
            else:
                st.write("**No interaction history** (Cold-start scenario)")
        
        with col2:
            st.subheader("Recommendations")
            
            # Get recommendations
            model = models[selected_model]
            
            if user_id == "cold_start_user":
                recommendations = model.recommend_for_cold_user(n_recommendations=n_recommendations)
            else:
                recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
            
            if recommendations:
                st.write(f"**Top {len(recommendations)} recommendations using {selected_model} model:**")
                
                for i, item_id in enumerate(recommendations, 1):
                    item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                    
                    with st.expander(f"{i}. {item_info['title']}", expanded=(i <= 3)):
                        col_a, col_b = st.columns([2, 1])
                        
                        with col_a:
                            st.write(f"**Category:** {item_info['category']}")
                            st.write(f"**Brand:** {item_info['brand']}")
                            st.write(f"**Price:** ${item_info['price']:.2f}")
                            st.write(f"**Average Rating:** {item_info['rating_avg']:.1f}/5.0")
                            st.write(f"**Rating Count:** {item_info['rating_count']}")
                            st.write(f"**Description:** {item_info['description']}")
                        
                        with col_b:
                            if show_explanations:
                                st.write("**Why recommended:**")
                                if user_id == "cold_start_user":
                                    st.write("- Popular item for new users")
                                    st.write("- High average rating")
                                else:
                                    st.write("- Matches your preferences")
                                    st.write("- Similar to your history")
            else:
                st.warning("No recommendations available")
    
    with tab2:
        st.header("Model Comparison")
        
        # Load evaluation results
        try:
            results_df = pd.read_csv("models/model_results.csv")
            leaderboard_df = pd.read_csv("models/leaderboard.csv")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                
                # Select metric to visualize
                metric_cols = [col for col in results_df.columns if '@' in col and col != 'model']
                selected_metric = st.selectbox("Select Metric", metric_cols)
                
                # Create bar chart
                fig = px.bar(
                    results_df, 
                    x='model', 
                    y=selected_metric,
                    title=f"{selected_metric} Comparison",
                    color='model'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Leaderboard")
                st.dataframe(leaderboard_df, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            st.dataframe(results_df, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Evaluation results not found. Please run the training script first.")
    
    with tab3:
        st.header("Cold-Start Analysis")
        
        if cold_start_info:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Cold Users",
                    cold_start_info["n_cold_users"],
                    help="Users with very few interactions"
                )
            
            with col2:
                st.metric(
                    "Cold Items",
                    cold_start_info["n_cold_items"],
                    help="Items with very few interactions"
                )
            
            with col3:
                cold_ratio = (cold_start_info["n_cold_users"] + cold_start_info["n_cold_items"]) / (
                    len(users_df) + len(items_df)
                )
                st.metric(
                    "Cold-Start Ratio",
                    f"{cold_ratio:.1%}",
                    help="Percentage of cold-start entities"
                )
            
            # Cold-start performance
            try:
                with open("models/cold_start_results.json", "r") as f:
                    cold_results = json.load(f)
                
                st.subheader("Cold-Start Performance")
                
                # Create comparison chart
                metrics_data = []
                for model_name, results in cold_results.items():
                    for metric, value in results.items():
                        if metric.startswith("cold_user_"):
                            metrics_data.append({
                                "Model": model_name,
                                "Metric": metric.replace("cold_user_", ""),
                                "Value": value
                            })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Pivot for visualization
                    pivot_df = metrics_df.pivot(index="Model", columns="Metric", values="Value")
                    
                    # Create heatmap
                    fig = px.imshow(
                        pivot_df.values,
                        labels=dict(x="Metric", y="Model", color="Score"),
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        aspect="auto",
                        title="Cold-Start Performance Heatmap"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except FileNotFoundError:
                st.warning("Cold-start evaluation results not found.")
        
        # Cold-start recommendations demo
        st.subheader("Cold-Start Recommendations Demo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cold User Recommendations:**")
            cold_recs = models[selected_model].recommend_for_cold_user(n_recommendations=5)
            
            for i, item_id in enumerate(cold_recs, 1):
                item_info = items_df[items_df['item_id'] == item_id].iloc[0]
                st.write(f"{i}. {item_info['title']} ({item_info['category']})")
        
        with col2:
            if cold_start_info["cold_items"]:
                st.write("**Cold Item Analysis:**")
                cold_item = cold_start_info["cold_items"][0]
                item_info = items_df[items_df['item_id'] == cold_item].iloc[0]
                
                st.write(f"**Item:** {item_info['title']}")
                st.write(f"**Category:** {item_info['category']}")
                st.write(f"**Interactions:** {cold_start_info['n_cold_items']}")
                
                # Find potential users
                potential_users = models[selected_model].recommend_for_cold_item(cold_item, 5)
                st.write(f"**Potential Users:** {len(potential_users)}")
    
    with tab4:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            st.metric("Total Users", len(users_df))
            st.metric("Total Items", len(items_df))
            st.metric("Total Interactions", len(interactions_df))
            st.metric("Average Rating", f"{interactions_df['rating'].mean():.2f}")
        
        with col2:
            st.subheader("Category Distribution")
            category_counts = items_df['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Item Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Interaction patterns
        st.subheader("Interaction Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            rating_counts = interactions_df['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                title="Rating Distribution",
                labels={"x": "Rating", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # User activity
            user_activity = interactions_df['user_id'].value_counts()
            fig = px.histogram(
                x=user_activity.values,
                title="User Activity Distribution",
                labels={"x": "Number of Interactions", "y": "Number of Users"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Data")
        
        tab_items, tab_users, tab_interactions = st.tabs(["Items", "Users", "Interactions"])
        
        with tab_items:
            st.dataframe(items_df.head(10), use_container_width=True)
        
        with tab_users:
            st.dataframe(users_df.head(10), use_container_width=True)
        
        with tab_interactions:
            st.dataframe(interactions_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
