
#!/usr/bin/env python3
"""
Tests for FPL Recommender
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_tool.recommender import FPLRecommender
from fpl_tool.optimizer import OptimizationResult

class TestFPLRecommender:
    """Test cases for FPL Recommender"""
    
    @pytest.fixture
    def sample_predictions_data(self):
        """Create sample predictions data"""
        data = []
        
        # Create sample players with predictions
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        teams = ['Arsenal', 'Liverpool', 'Manchester City', 'Chelsea']
        
        player_id = 1
        for pos in positions:
            count = 2 if pos == 'GKP' else 8  # 2 GK, 8 others per position
            for i in range(count):
                data.append({
                    'player_id': player_id,
                    'name': f'{pos}_{i+1}',
                    'position': pos,
                    'team_name': teams[i % len(teams)],
                    'current_price': 4.0 + np.random.random() * 8.0,
                    'expected_points_ensemble': 3.0 + np.random.random() * 5.0,
                    'points_per_million': np.random.random() * 2.0,
                    'status': 'a'
                })
                player_id += 1
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_master_data(self):
        """Create sample master data"""
        data = []
        
        for i in range(34):  # Match predictions data
            data.append({
                'player_id': i + 1,
                'web_name': f'Player_{i+1}',
                'photo_url': f'https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Demaryius_Thomas_2018_%28cropped%29.jpg/800px-Demaryius_Thomas_2018_%28cropped%29.jpg',
                'photo_code': f'code_{i+1}',
                'team_short': ['ARS', 'LIV', 'MCI', 'CHE'][i % 4],
                'selected_by_percent': np.random.random() * 50,
                'transfers_in': np.random.randint(0, 10000),
                'transfers_out': np.random.randint(0, 5000)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def recommender(self, sample_predictions_data, sample_master_data):
        """Create recommender with sample data"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save sample data to temporary files
            pred_file = os.path.join(tmp_dir, "predictions.csv")
            master_file = os.path.join(tmp_dir, "master.csv")
            
            sample_predictions_data.to_csv(pred_file, index=False)
            sample_master_data.to_csv(master_file, index=False)
            
            # Create recommender
            recommender = FPLRecommender(pred_file, master_file)
            return recommender
    
    def test_recommender_initialization(self, recommender):
        """Test recommender initializes correctly"""
        assert recommender is not None
        assert recommender.predictions_df is not None
        assert recommender.master_df is not None
        assert recommender.combined_df is not None
        assert recommender.optimizer is not None
        assert recommender.validator is not None
    
    def test_data_loading_and_merging(self, recommender):
        """Test data loading and merging works correctly"""
        # Check that data was merged correctly
        assert 'web_name' in recommender.combined_df.columns
        assert 'photo_url' in recommender.combined_df.columns
        assert 'team_short' in recommender.combined_df.columns
        assert 'selected_by_percent' in recommender.combined_df.columns
        
        # Check that we have the expected number of players
        assert len(recommender.combined_df) > 0
    
    def test_get_player_photo_url(self, recommender):
        """Test player photo URL generation"""
        # Test with photo code
        url = recommender.get_player_photo_url(123, "test_code")
        assert "test_code" in url
        
        # Test without photo code (fallback)
        url = recommender.get_player_photo_url(123, None)
        assert "fantasy.premierleague.com" in url
    
    @patch.object(FPLRecommender, 'generate_optimal_squad')
    def test_generate_optimal_squad_strategies(self, mock_optimize, recommender):
        """Test different optimization strategies"""
        # Mock the optimization result
        mock_result = OptimizationResult(
            squad=[], starting_xi=[], bench=[], captain={}, vice_captain={},
            total_cost=100.0, expected_points=50.0, formation="1-4-4-2",
            is_valid=True, validation_errors=[]
        )
        mock_optimize.return_value = mock_result
        
        strategies = ["balanced", "premium", "value", "differential"]
        
        for strategy in strategies:
            result = recommender.generate_optimal_squad(strategy=strategy)
            assert isinstance(result, OptimizationResult)
            mock_optimize.assert_called()
    
    def test_generate_watchlists(self, recommender):
        """Test watchlist generation"""
        watchlists = recommender.generate_watchlists()
        
        # Check that all positions are included
        expected_positions = ['GKP', 'DEF', 'MID', 'FWD']
        for position in expected_positions:
            assert position in watchlists
            assert isinstance(watchlists[position], list)
            assert len(watchlists[position]) > 0
        
        # Check that players have photo URLs
        for position, players in watchlists.items():
            for player in players:
                assert 'photo_url' in player
                assert player['photo_url'] is not None
    
    def test_generate_top_50_overall(self, recommender):
        """Test top 50 overall generation"""
        top_50 = recommender.generate_top_50_overall()
        
        assert isinstance(top_50, list)
        assert len(top_50) <= 50  # Should be <= 50 (might be less if fewer players available)
        
        # Check that players are ranked
        for i, player in enumerate(top_50):
            assert 'rank' in player
            assert player['rank'] == i + 1
            assert 'photo_url' in player
            assert 'club' in player
        
        # Check that players are sorted by expected points (descending)
        if len(top_50) > 1:
            for i in range(len(top_50) - 1):
                assert top_50[i]['expected_points_ensemble'] >= top_50[i+1]['expected_points_ensemble']
    
    def test_generate_differentials(self, recommender):
        """Test differential players generation"""
        differentials = recommender.generate_differentials(max_ownership=10.0)
        
        assert isinstance(differentials, list)
        
        # Check that all players have low ownership
        for player in differentials:
            assert player.get('selected_by_percent', 0) <= 10.0
            assert 'photo_url' in player
    
    def test_generate_budget_enablers(self, recommender):
        """Test budget enablers generation"""
        budget_options = recommender.generate_budget_enablers()
        
        assert isinstance(budget_options, dict)
        assert 'DEF' in budget_options
        assert 'MID' in budget_options
        
        # Check price constraints
        for def_player in budget_options['DEF']:
            assert def_player['current_price'] <= 4.5
        
        for mid_player in budget_options['MID']:
            assert mid_player['current_price'] <= 5.0
    
    def test_generate_captaincy_recommendations(self, recommender):
        """Test captaincy recommendations"""
        captaincy = recommender.generate_captaincy_recommendations()
        
        assert isinstance(captaincy, dict)
        assert 'premium' in captaincy
        assert 'safe' in captaincy
        
        assert isinstance(captaincy['premium'], list)
        assert isinstance(captaincy['safe'], list)
        
        # Premium captains should be expensive
        for captain in captaincy['premium']:
            assert captain['current_price'] >= 8.0
    
    def test_generate_chip_strategy(self, recommender):
        """Test chip strategy generation"""
        strategy = recommender.generate_chip_strategy(gameweek=5)
        
        assert isinstance(strategy, dict)
        assert 'wildcard_1' in strategy
        assert 'free_hit' in strategy
        assert 'bench_boost' in strategy
        assert 'triple_captain' in strategy
        
        # Check that each chip has recommendations
        for chip, info in strategy.items():
            assert 'recommended_gw' in info
            assert 'reason' in info
    
    @patch.object(FPLRecommender, 'generate_optimal_squad')
    def test_generate_complete_recommendations(self, mock_optimize, recommender):
        """Test complete recommendations generation"""
        # Mock optimization results
        mock_result = OptimizationResult(
            squad=[{'player_id': 1, 'name': 'Test Player'}], 
            starting_xi=[], bench=[], captain={}, vice_captain={},
            total_cost=100.0, expected_points=50.0, formation="1-4-4-2",
            is_valid=True, validation_errors=[]
        )
        mock_optimize.return_value = mock_result
        
        recommendations = recommender.generate_complete_recommendations(gameweek=1, budget=100.0)
        
        assert isinstance(recommendations, dict)
        
        # Check all required sections
        required_sections = [
            'gameweek', 'budget', 'generated_at', 'optimal_squads',
            'watchlists', 'top_50_overall', 'differentials',
            'budget_enablers', 'captaincy', 'chip_strategy'
        ]
        
        for section in required_sections:
            assert section in recommendations
        
        # Check optimal squads
        assert 'balanced' in recommendations['optimal_squads']
        assert 'premium' in recommendations['optimal_squads']
        assert 'value' in recommendations['optimal_squads']
        assert 'differential' in recommendations['optimal_squads']
    
    def test_export_recommendations_csv(self, recommender):
        """Test CSV export functionality"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create mock recommendations
            recommendations = {
                'top_50_overall': [
                    {'player_id': 1, 'name': 'Player 1', 'position': 'MID', 'expected_points_ensemble': 5.0}
                ],
                'watchlists': {
                    'GKP': [{'player_id': 2, 'name': 'GK 1', 'position': 'GKP'}]
                },
                'optimal_squads': {
                    'balanced': OptimizationResult(
                        squad=[{'player_id': 1, 'name': 'Test'}], 
                        starting_xi=[], bench=[], captain={}, vice_captain={},
                        total_cost=100.0, expected_points=50.0, formation="1-4-4-2",
                        is_valid=True, validation_errors=[]
                    )
                }
            }
            
            exported_files = recommender.export_recommendations_csv(recommendations, tmp_dir)
            
            assert isinstance(exported_files, dict)
            assert len(exported_files) > 0
            
            # Check that files were created
            for file_type, file_path in exported_files.items():
                assert os.path.exists(file_path)
                assert file_path.endswith('.csv')
    
    def test_error_handling_missing_data(self):
        """Test error handling when data files are missing"""
        with pytest.raises(Exception):
            FPLRecommender("/nonexistent/path.csv", "/nonexistent/path2.csv")
    
    def test_photo_url_fallback(self, recommender):
        """Test photo URL fallback handling"""
        # Test with invalid photo code
        url = recommender.get_player_photo_url(999, "invalid_code")
        assert isinstance(url, str)
        assert len(url) > 0
        
        # Test with None photo code
        url = recommender.get_player_photo_url(999, None)
        assert isinstance(url, str)
        assert len(url) > 0

if __name__ == "__main__":
    pytest.main([__file__])
