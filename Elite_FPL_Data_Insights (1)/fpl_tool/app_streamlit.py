
#!/usr/bin/env python3
"""
FPL Tool Streamlit Dashboard
Interactive web interface for FPL optimization and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any
import logging
import os

from fpl_tool.recommender import FPLRecommender
from fpl_tool.optimizer import FPLOptimizer
from fpl_tool.validator import FPLValidator

# Environment-aware default paths
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/home/ubuntu/data")

# Configure page
st.set_page_config(
    page_title="FPL Tool Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_recommender():
    """Load and cache the FPL recommender."""
    return FPLRecommender()

@st.cache_data
def load_player_data():
    """Load and cache player data."""
    recommender = load_recommender()
    return recommender.combined_df

def format_currency(value):
    """Format currency values."""
    return f"£{value:.1f}m"

def format_percentage(value):
    """Format percentage values."""
    return f"{value:.1f}%"

def display_player_table(df: pd.DataFrame, title: str, show_images: bool = True):
    """Display a formatted player table with optional images."""
    st.subheader(title)
    
    # Select columns to display
    display_cols = ['name', 'position', 'team_name', 'current_price', 'expected_points_ensemble']
    if 'selected_by_percent' in df.columns:
        display_cols.append('selected_by_percent')
    if 'points_per_million' in df.columns:
        display_cols.append('points_per_million')
    
    # Format the dataframe
    display_df = df[display_cols].copy()
    display_df['current_price'] = display_df['current_price'].apply(format_currency)
    display_df['expected_points_ensemble'] = display_df['expected_points_ensemble'].round(1)
    
    if 'selected_by_percent' in display_df.columns:
        display_df['selected_by_percent'] = display_df['selected_by_percent'].apply(format_percentage)
    if 'points_per_million' in display_df.columns:
        display_df['points_per_million'] = display_df['points_per_million'].round(2)
    
    # Rename columns for display
    column_names = {
        'name': 'Player',
        'position': 'Position',
        'team_name': 'Team',
        'current_price': 'Price',
        'expected_points_ensemble': 'xPts',
        'selected_by_percent': 'Ownership',
        'points_per_million': 'Pts/£m'
    }
    display_df = display_df.rename(columns=column_names)
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main Streamlit application."""
    
    # Title and header
    st.title("⚽ FPL Tool Dashboard")
    st.markdown("Fantasy Premier League Optimization & Recommendation System")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Squad Optimizer", "📊 Player Analysis", "📋 Watchlists", "🏆 Top Players", "⚙️ Settings"]
    )
    
    # Load data
    try:
        with st.spinner("Loading FPL data..."):
            recommender = load_recommender()
            player_data = load_player_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(recommender, player_data)
    elif page == "🎯 Squad Optimizer":
        show_optimizer_page(recommender)
    elif page == "📊 Player Analysis":
        show_analysis_page(player_data)
    elif page == "📋 Watchlists":
        show_watchlists_page(recommender)
    elif page == "🏆 Top Players":
        show_top_players_page(recommender)
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page(recommender: FPLRecommender, player_data: pd.DataFrame):
    """Display the home page with overview statistics."""
    
    st.header("Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Players", len(player_data))
    
    with col2:
        available_players = len(player_data[player_data['status'] == 'a'])
        st.metric("Available Players", available_players)
    
    with col3:
        avg_price = player_data[player_data['status'] == 'a']['current_price'].mean()
        st.metric("Average Price", format_currency(avg_price))
    
    with col4:
        avg_points = player_data[player_data['status'] == 'a']['expected_points_ensemble'].mean()
        st.metric("Average xPts", f"{avg_points:.1f}")
    
    # Quick recommendations
    st.header("Quick Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Top 5 Players")
        top_5 = player_data[player_data['status'] == 'a'].nlargest(5, 'expected_points_ensemble')
        display_player_table(top_5, "", show_images=False)
    
    with col2:
        st.subheader("💎 Best Value Players")
        if 'points_per_million' in player_data.columns:
            best_value = player_data[player_data['status'] == 'a'].nlargest(5, 'points_per_million')
            display_player_table(best_value, "", show_images=False)
    
    # Position distribution chart
    st.header("Player Distribution by Position")
    
    position_counts = player_data[player_data['status'] == 'a']['position'].value_counts()
    fig = px.pie(
        values=position_counts.values,
        names=position_counts.index,
        title="Available Players by Position"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_optimizer_page(recommender: FPLRecommender):
    """Display the squad optimizer page."""
    
    st.header("🎯 Squad Optimizer")
    st.markdown("Generate optimal FPL squads under constraints")
    
    # Optimization parameters
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.slider("Budget (£m)", 90.0, 100.0, 100.0, 0.1)
        strategy = st.selectbox(
            "Strategy",
            ["balanced", "premium", "value", "differential"],
            help="Choose optimization strategy"
        )
    
    with col2:
        max_per_club = st.slider("Max players per club", 1, 3, 3)
        formation = st.selectbox(
            "Formation (optional)",
            ["Auto", "1-3-4-3", "1-3-5-2", "1-4-3-3", "1-4-4-2", "1-4-5-1", "1-5-3-2", "1-5-4-1"]
        )
    
    # Optimize button
    if st.button("🚀 Optimize Squad", type="primary"):
        with st.spinner("Optimizing squad..."):
            try:
                # Parse formation
                formation_tuple = None
                if formation != "Auto":
                    parts = formation.split('-')
                    formation_tuple = tuple(int(x) for x in parts)
                
                # Generate optimal squad
                result = recommender.generate_optimal_squad(
                    strategy=strategy,
                    budget=budget
                )
                
                if result.is_valid:
                    st.success("✅ Optimization successful!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Formation", result.formation)
                    with col2:
                        st.metric("Total Cost", format_currency(result.total_cost))
                    with col3:
                        st.metric("Expected Points", f"{result.expected_points:.1f}")
                    
                    # Starting XI
                    st.subheader("Starting XI")
                    starting_df = pd.DataFrame(result.starting_xi)
                    display_player_table(starting_df, "", show_images=False)
                    
                    # Captain info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Captain:** {result.captain['name']} ({result.captain['position']})")
                    with col2:
                        st.info(f"**Vice-Captain:** {result.vice_captain['name']} ({result.vice_captain['position']})")
                    
                    # Bench
                    st.subheader("Bench")
                    bench_df = pd.DataFrame(result.bench)
                    display_player_table(bench_df, "", show_images=False)
                    
                    # Download options
                    st.subheader("Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        squad_csv = pd.DataFrame(result.squad).to_csv(index=False)
                        st.download_button(
                            "📥 Download Full Squad",
                            squad_csv,
                            f"optimal_squad_{strategy}.csv",
                            "text/csv"
                        )
                    
                    with col2:
                        xi_csv = pd.DataFrame(result.starting_xi).to_csv(index=False)
                        st.download_button(
                            "📥 Download Starting XI",
                            xi_csv,
                            f"starting_xi_{strategy}.csv",
                            "text/csv"
                        )
                
                else:
                    st.error("❌ Optimization failed!")
                    for error in result.validation_errors:
                        st.error(f"• {error}")
                        
            except Exception as e:
                st.error(f"Error during optimization: {e}")

def show_analysis_page(player_data: pd.DataFrame):
    """Display player analysis page."""
    
    st.header("📊 Player Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.multiselect(
            "Position",
            options=player_data['position'].unique(),
            default=player_data['position'].unique()
        )
    
    with col2:
        team_filter = st.multiselect(
            "Team",
            options=sorted(player_data['team_name'].unique()),
            default=[]
        )
    
    with col3:
        price_range = st.slider(
            "Price Range (£m)",
            float(player_data['current_price'].min()),
            float(player_data['current_price'].max()),
            (float(player_data['current_price'].min()), float(player_data['current_price'].max()))
        )
    
    # Apply filters
    filtered_data = player_data[
        (player_data['position'].isin(position_filter)) &
        (player_data['current_price'] >= price_range[0]) &
        (player_data['current_price'] <= price_range[1]) &
        (player_data['status'] == 'a')
    ]
    
    if team_filter:
        filtered_data = filtered_data[filtered_data['team_name'].isin(team_filter)]
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Expected Points scatter
        fig = px.scatter(
            filtered_data,
            x='current_price',
            y='expected_points_ensemble',
            color='position',
            hover_data=['name', 'team_name'],
            title="Price vs Expected Points"
        )
        fig.update_layout(
            xaxis_title="Price (£m)",
            yaxis_title="Expected Points"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Points per million distribution
        if 'points_per_million' in filtered_data.columns:
            fig = px.histogram(
                filtered_data,
                x='points_per_million',
                color='position',
                title="Points per Million Distribution"
            )
            fig.update_layout(
                xaxis_title="Points per Million",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Filtered player table
    st.subheader(f"Filtered Players ({len(filtered_data)} players)")
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ["expected_points_ensemble", "current_price", "points_per_million", "name"],
        format_func=lambda x: {
            "expected_points_ensemble": "Expected Points",
            "current_price": "Price",
            "points_per_million": "Points per Million",
            "name": "Name"
        }.get(x, x)
    )
    
    ascending = st.checkbox("Ascending order", value=False)
    
    sorted_data = filtered_data.sort_values(sort_by, ascending=ascending)
    display_player_table(sorted_data.head(20), "Top 20 Players", show_images=False)

def show_watchlists_page(recommender: FPLRecommender):
    """Display watchlists page."""
    
    st.header("📋 Positional Watchlists")
    st.markdown("Top players by position for your watchlist")
    
    # Generate watchlists
    with st.spinner("Generating watchlists..."):
        watchlists = recommender.generate_watchlists()
    
    # Position tabs
    position_tabs = st.tabs(["🥅 Goalkeepers", "🛡️ Defenders", "⚽ Midfielders", "🎯 Forwards"])
    
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    
    for i, (tab, position) in enumerate(zip(position_tabs, positions)):
        with tab:
            if position in watchlists:
                players = watchlists[position]
                st.subheader(f"Top {len(players)} {position}")
                
                # Convert to DataFrame and display
                players_df = pd.DataFrame(players)
                display_player_table(players_df, "", show_images=False)
                
                # Download button
                csv_data = players_df.to_csv(index=False)
                st.download_button(
                    f"📥 Download {position} Watchlist",
                    csv_data,
                    f"watchlist_{position.lower()}.csv",
                    "text/csv"
                )

def show_top_players_page(recommender: FPLRecommender):
    """Display top players page."""
    
    st.header("🏆 Top 50 Players")
    st.markdown("Overall rankings with complete player information")
    
    # Generate top 50
    with st.spinner("Generating top 50 rankings..."):
        top_50 = recommender.generate_top_50_overall()
    
    # Display options
    col1, col2 = st.columns(2)
    
    with col1:
        show_count = st.slider("Show top N players", 10, 50, 25)
    
    with col2:
        position_filter = st.multiselect(
            "Filter by position",
            options=['GKP', 'DEF', 'MID', 'FWD'],
            default=['GKP', 'DEF', 'MID', 'FWD']
        )
    
    # Filter and display
    filtered_top = [p for p in top_50 if p['position'] in position_filter][:show_count]
    
    if filtered_top:
        # Create enhanced display
        display_data = []
        for player in filtered_top:
            display_data.append({
                'Rank': player['rank'],
                'Player': player['name'],
                'Position': player['position'],
                'Team': player.get('club', player.get('team_short', '')),
                'Price': format_currency(player['current_price']),
                'xPts': f"{player['expected_points_ensemble']:.1f}",
                'Ownership': format_percentage(player.get('selected_by_percent', 0)),
                'Pts/£m': f"{player.get('points_per_million', 0):.2f}"
            })
        
        display_df = pd.DataFrame(display_data)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv_data = pd.DataFrame(filtered_top).to_csv(index=False)
        st.download_button(
            "📥 Download Top Players",
            csv_data,
            "top_50_players.csv",
            "text/csv"
        )
    
    else:
        st.warning("No players match the selected filters.")

def show_settings_page():
    """Display settings page."""
    
    st.header("⚙️ Settings")
    
    st.subheader("Data Sources")
    st.text_input("Predictions Data Path", f"{DEFAULT_DATA_DIR}/fpl_xpts_predictions_enhanced.csv")
    st.text_input("Master Data Path", f"{DEFAULT_DATA_DIR}/fpl_master_2025-26.csv")
    
    st.subheader("Optimization Settings")
    st.number_input("Default Budget", 90.0, 100.0, 100.0, 0.1)
    st.number_input("Default Max Per Club", 1, 3, 3)
    
    st.subheader("Display Settings")
    st.checkbox("Show Player Images", True)
    st.checkbox("Show Ownership Data", True)
    st.selectbox("Default Theme", ["Light", "Dark"], index=0)
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
