
#!/usr/bin/env python3
"""
FPL Recommendation Engine
End-to-end recommendation system that generates optimal squads, watchlists, and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import requests
import json
import os
from .optimizer import FPLOptimizer, OptimizationResult
from .validator import FPLValidator

logger = logging.getLogger(__name__)

# Environment-aware default paths
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/home/ubuntu/data")
DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/ubuntu/output")

class FPLRecommender:
    """
    Comprehensive FPL recommendation system that generates:
    - Optimal squads with multiple strategies
    - Positional watchlists
    - Top 50 rankings with images
    - Differential and budget options
    - Chip strategy recommendations
    """
    
    def __init__(self, 
                 data_path: str = f"{DEFAULT_DATA_DIR}/fpl_xpts_predictions_enhanced.csv",
                 master_data_path: str = f"{DEFAULT_DATA_DIR}/fpl_master_2025-26.csv"):
        """
        Initialize the FPL Recommender.
        
        Args:
            data_path: Path to enhanced predictions CSV
            master_data_path: Path to master dataset CSV
        """
        self.data_path = data_path
        self.master_data_path = master_data_path
        self.optimizer = FPLOptimizer(data_path)
        self.validator = FPLValidator()
        
        # Load data
        self.predictions_df = None
        self.master_df = None
        self.load_data()
        
        # FPL API base URL for images
        self.fpl_api_base = "https://fantasy.premierleague.com/api/"
        self.photo_base_url = "https://resources.premierleague.com/premierleague/photos/players/250x250/"
        
    def load_data(self) -> None:
        """Load and merge prediction and master datasets."""
        try:
            logger.info("Loading recommendation data...")
            
            # Load predictions
            self.predictions_df = pd.read_csv(self.data_path)
            
            # Load master data for additional info
            self.master_df = pd.read_csv(self.master_data_path)
            
            # Merge datasets
            self.combined_df = pd.merge(
                self.predictions_df,
                self.master_df[['player_id', 'web_name', 'photo_url', 'photo_code', 
                               'team_short', 'selected_by_percent', 'transfers_in', 'transfers_out']],
                on='player_id',
                how='left'
            )
            
            logger.info(f"Loaded {len(self.combined_df)} players for recommendations")
            
        except Exception as e:
            logger.error(f"Error loading recommendation data: {e}")
            raise
    
    def get_player_photo_url(self, player_id: int, photo_code: str = None) -> str:
        """
        Get player headshot URL from FPL API or fallback sources.
        
        Args:
            player_id: FPL player ID
            photo_code: Photo code from FPL API
            
        Returns:
            Working photo URL or fallback placeholder
        """
        try:
            # Primary: FPL official photo
            if photo_code:
                fpl_url = f"{self.photo_base_url}p{photo_code}.png"
                return fpl_url
            
            # Fallback: Generic placeholder
            return "https://fantasy.premierleague.com/dist/img/shirts/shirt_0-66.png"
            
        except Exception:
            return "https://fantasy.premierleague.com/dist/img/shirts/shirt_0-66.png"
    
    def generate_optimal_squad(self, 
                              strategy: str = "balanced",
                              budget: float = 100.0) -> OptimizationResult:
        """
        Generate optimal squad based on strategy.
        
        Args:
            strategy: Strategy type ("balanced", "premium", "value", "differential")
            budget: Budget constraint
            
        Returns:
            OptimizationResult with optimal squad
        """
        logger.info(f"Generating {strategy} optimal squad...")
        
        if strategy == "premium":
            # Premium strategy: favor expensive, high-ownership players
            modified_df = self.combined_df.copy()
            modified_df['expected_points_ensemble'] *= (1 + modified_df['current_price'] / 20)
            
        elif strategy == "value":
            # Value strategy: maximize points per million
            modified_df = self.combined_df.copy()
            modified_df['expected_points_ensemble'] *= (modified_df['points_per_million'] / 10)
            
        elif strategy == "differential":
            # Differential strategy: favor low ownership players
            modified_df = self.combined_df.copy()
            ownership_penalty = np.where(modified_df['selected_by_percent'] > 10, 0.8, 1.2)
            modified_df['expected_points_ensemble'] *= ownership_penalty
            
        else:  # balanced
            modified_df = self.combined_df.copy()
        
        # Update optimizer data temporarily
        original_df = self.optimizer.players_df.copy()
        self.optimizer.players_df = modified_df
        
        try:
            result = self.optimizer.optimize_squad(budget=budget)
            return result
        finally:
            # Restore original data
            self.optimizer.players_df = original_df
    
    def generate_watchlists(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate positional watchlists with top players by position.
        
        Returns:
            Dictionary with position-based player lists
        """
        logger.info("Generating positional watchlists...")
        
        watchlists = {}
        position_limits = {
            'GKP': 15,
            'DEF': 25, 
            'MID': 25,
            'FWD': 20
        }
        
        for position, limit in position_limits.items():
            pos_players = self.combined_df[
                (self.combined_df['position'] == position) & 
                (self.combined_df['status'] == 'a')
            ].copy()
            
            # Sort by expected points descending
            pos_players = pos_players.sort_values('expected_points_ensemble', ascending=False)
            
            # Take top N players
            top_players = pos_players.head(limit)
            
            # Add photo URLs
            watchlist = []
            for _, player in top_players.iterrows():
                player_dict = player.to_dict()
                player_dict['photo_url'] = self.get_player_photo_url(
                    player['player_id'], 
                    player.get('photo_code')
                )
                watchlist.append(player_dict)
            
            watchlists[position] = watchlist
        
        return watchlists
    
    def generate_top_50_overall(self) -> List[Dict[str, Any]]:
        """
        Generate top 50 overall players with club and photo information.
        
        Returns:
            List of top 50 players with complete information
        """
        logger.info("Generating top 50 overall rankings...")
        
        # Filter available players and sort by expected points
        available_players = self.combined_df[
            self.combined_df['status'] == 'a'
        ].copy()
        
        top_50 = available_players.nlargest(50, 'expected_points_ensemble')
        
        # Enhance with photo URLs and formatting
        top_50_list = []
        for rank, (_, player) in enumerate(top_50.iterrows(), 1):
            player_dict = player.to_dict()
            player_dict['rank'] = rank
            player_dict['photo_url'] = self.get_player_photo_url(
                player['player_id'], 
                player.get('photo_code')
            )
            player_dict['club'] = player.get('team_short', player.get('team_name', ''))
            top_50_list.append(player_dict)
        
        return top_50_list
    
    def generate_differentials(self, max_ownership: float = 10.0) -> List[Dict[str, Any]]:
        """
        Generate differential players (low ownership, high potential).
        
        Args:
            max_ownership: Maximum ownership percentage for differentials
            
        Returns:
            List of differential player recommendations
        """
        logger.info(f"Generating differentials (≤{max_ownership}% ownership)...")
        
        # Filter for low ownership players
        differentials = self.combined_df[
            (self.combined_df['status'] == 'a') &
            (self.combined_df['selected_by_percent'] <= max_ownership)
        ].copy()
        
        # Sort by expected points descending
        differentials = differentials.sort_values('expected_points_ensemble', ascending=False)
        
        # Take top 20 differentials
        top_differentials = differentials.head(20)
        
        # Add photo URLs
        differential_list = []
        for _, player in top_differentials.iterrows():
            player_dict = player.to_dict()
            player_dict['photo_url'] = self.get_player_photo_url(
                player['player_id'], 
                player.get('photo_code')
            )
            differential_list.append(player_dict)
        
        return differential_list
    
    def generate_budget_enablers(self, 
                                max_price_def: float = 4.5,
                                max_price_mid: float = 5.0) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate budget enabler options.
        
        Args:
            max_price_def: Maximum price for budget defenders
            max_price_mid: Maximum price for budget midfielders
            
        Returns:
            Dictionary with budget options by position
        """
        logger.info("Generating budget enablers...")
        
        budget_options = {}
        
        # Budget defenders (≤£4.5m)
        budget_defs = self.combined_df[
            (self.combined_df['position'] == 'DEF') &
            (self.combined_df['status'] == 'a') &
            (self.combined_df['current_price'] <= max_price_def)
        ].copy()
        
        budget_defs = budget_defs.sort_values('expected_points_ensemble', ascending=False)
        budget_options['DEF'] = budget_defs.head(10).to_dict('records')
        
        # Budget midfielders (≤£5.0m)
        budget_mids = self.combined_df[
            (self.combined_df['position'] == 'MID') &
            (self.combined_df['status'] == 'a') &
            (self.combined_df['current_price'] <= max_price_mid)
        ].copy()
        
        budget_mids = budget_mids.sort_values('expected_points_ensemble', ascending=False)
        budget_options['MID'] = budget_mids.head(10).to_dict('records')
        
        return budget_options
    
    def generate_captaincy_recommendations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate captain and vice-captain recommendations.
        
        Returns:
            Dictionary with premium and safe captaincy options
        """
        logger.info("Generating captaincy recommendations...")
        
        # Premium captains (high expected points, high price)
        premium_captains = self.combined_df[
            (self.combined_df['status'] == 'a') &
            (self.combined_df['current_price'] >= 8.0)
        ].copy()
        
        premium_captains = premium_captains.nlargest(10, 'expected_points_ensemble')
        
        # Safe captains (consistent performers, medium ownership)
        safe_captains = self.combined_df[
            (self.combined_df['status'] == 'a') &
            (self.combined_df['selected_by_percent'] >= 15) &
            (self.combined_df['selected_by_percent'] <= 40)
        ].copy()
        
        safe_captains = safe_captains.nlargest(10, 'expected_points_ensemble')
        
        return {
            'premium': premium_captains.to_dict('records'),
            'safe': safe_captains.to_dict('records')
        }
    
    def generate_chip_strategy(self, gameweek: int = 1) -> Dict[str, Any]:
        """
        Generate chip usage strategy recommendations.
        
        Args:
            gameweek: Current gameweek
            
        Returns:
            Dictionary with chip strategy recommendations
        """
        logger.info("Generating chip strategy recommendations...")
        
        # This would analyze fixture data for optimal chip timing
        # For now, return general recommendations
        
        strategy = {
            "wildcard_1": {
                "recommended_gw": "8-12",
                "reason": "After international break, injury updates available"
            },
            "free_hit": {
                "recommended_gw": "BGW or DGW",
                "reason": "Use during blank gameweeks or strong double gameweeks"
            },
            "bench_boost": {
                "recommended_gw": "DGW with strong bench",
                "reason": "Maximize returns when bench players have doubles"
            },
            "triple_captain": {
                "recommended_gw": "DGW with premium captain",
                "reason": "Use on premium player with favorable double fixtures"
            }
        }
        
        return strategy
    
    def generate_complete_recommendations(self, 
                                        gameweek: int = 1,
                                        budget: float = 100.0) -> Dict[str, Any]:
        """
        Generate complete weekly recommendations package.
        
        Args:
            gameweek: Current gameweek
            budget: Available budget
            
        Returns:
            Complete recommendations dictionary
        """
        logger.info(f"Generating complete recommendations for GW{gameweek}...")
        
        recommendations = {
            "gameweek": gameweek,
            "budget": budget,
            "generated_at": pd.Timestamp.now().isoformat(),
            
            # Optimal squads
            "optimal_squads": {
                "balanced": self.generate_optimal_squad("balanced", budget),
                "premium": self.generate_optimal_squad("premium", budget),
                "value": self.generate_optimal_squad("value", budget),
                "differential": self.generate_optimal_squad("differential", budget)
            },
            
            # Player lists
            "watchlists": self.generate_watchlists(),
            "top_50_overall": self.generate_top_50_overall(),
            "differentials": self.generate_differentials(),
            "budget_enablers": self.generate_budget_enablers(),
            "captaincy": self.generate_captaincy_recommendations(),
            
            # Strategy
            "chip_strategy": self.generate_chip_strategy(gameweek)
        }
        
        # Validate all squads
        for strategy, result in recommendations["optimal_squads"].items():
            if result.is_valid:
                is_valid, errors = self.validator.validate_complete_team(
                    result.squad, result.starting_xi, result.bench,
                    result.captain, result.vice_captain
                )
                result.is_valid = is_valid
                result.validation_errors = errors
        
        logger.info("Complete recommendations generated successfully")
        return recommendations
    
    def export_recommendations_csv(self, 
                                  recommendations: Dict[str, Any], 
                                  output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, str]:
        """
        Export recommendations to CSV files.
        
        Args:
            recommendations: Complete recommendations dictionary
            output_dir: Output directory path
            
        Returns:
            Dictionary mapping export type to file path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export top 50
        top_50_df = pd.DataFrame(recommendations["top_50_overall"])
        top_50_path = f"{output_dir}/top_50_overall.csv"
        top_50_df.to_csv(top_50_path, index=False)
        exported_files["top_50"] = top_50_path
        
        # Export watchlists
        for position, players in recommendations["watchlists"].items():
            watchlist_df = pd.DataFrame(players)
            watchlist_path = f"{output_dir}/watchlist_{position.lower()}.csv"
            watchlist_df.to_csv(watchlist_path, index=False)
            exported_files[f"watchlist_{position}"] = watchlist_path
        
        # Export optimal squads
        for strategy, result in recommendations["optimal_squads"].items():
            if result.squad:
                squad_df = pd.DataFrame(result.squad)
                squad_path = f"{output_dir}/optimal_squad_{strategy}.csv"
                squad_df.to_csv(squad_path, index=False)
                exported_files[f"squad_{strategy}"] = squad_path
        
        logger.info(f"Exported {len(exported_files)} recommendation files to {output_dir}")
        return exported_files
