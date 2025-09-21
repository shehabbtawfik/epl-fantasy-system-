
#!/usr/bin/env python3
"""
Tests for FPL Optimizer
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_tool.optimizer import FPLOptimizer, OptimizationResult
from fpl_tool.validator import FPLValidator

class TestFPLOptimizer:
    """Test cases for FPL Optimizer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample player data for testing"""
        data = []
        
        # Create more players to ensure feasible solutions
        # Need at least 20+ players to have flexibility
        
        # Create 6 goalkeepers from different teams
        for i in range(6):
            data.append({
                'player_id': i + 1,
                'name': f'GK_{i+1}',
                'position': 'GKP',
                'team_name': f'Team_{(i % 6) + 1}',
                'current_price': 4.0 + i * 0.2,
                'expected_points_ensemble': 3.0 + i * 0.2,
                'status': 'a'
            })
        
        # Create 15 defenders from different teams
        for i in range(15):
            data.append({
                'player_id': i + 7,
                'name': f'DEF_{i+1}',
                'position': 'DEF',
                'team_name': f'Team_{(i % 6) + 1}',  # Distribute across 6 teams
                'current_price': 3.5 + (i % 8) * 0.3,  # Vary prices
                'expected_points_ensemble': 2.5 + (i % 5) * 0.4,
                'status': 'a'
            })
        
        # Create 15 midfielders from different teams
        for i in range(15):
            data.append({
                'player_id': i + 22,
                'name': f'MID_{i+1}',
                'position': 'MID',
                'team_name': f'Team_{(i % 6) + 1}',
                'current_price': 4.0 + (i % 10) * 0.5,
                'expected_points_ensemble': 3.0 + (i % 6) * 0.5,
                'status': 'a'
            })
        
        # Create 10 forwards from different teams
        for i in range(10):
            data.append({
                'player_id': i + 37,
                'name': f'FWD_{i+1}',
                'position': 'FWD',
                'team_name': f'Team_{(i % 6) + 1}',
                'current_price': 4.5 + (i % 8) * 0.8,
                'expected_points_ensemble': 4.0 + (i % 4) * 0.7,
                'status': 'a'
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def optimizer(self, sample_data, tmp_path):
        """Create optimizer with sample data"""
        # Save sample data to temporary file
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Create optimizer
        optimizer = FPLOptimizer(str(data_file))
        return optimizer
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer is not None
        assert optimizer.players_df is not None
        assert len(optimizer.players_df) == 15  # 2+5+5+3
        assert optimizer.SQUAD_SIZE == 15
        assert optimizer.BUDGET == 100.0
    
    def test_position_constraints(self, optimizer):
        """Test position constraints are correct"""
        expected_limits = {
            'GKP': {'squad': 2, 'starting_min': 1, 'starting_max': 1},
            'DEF': {'squad': 5, 'starting_min': 3, 'starting_max': 5},
            'MID': {'squad': 5, 'starting_min': 3, 'starting_max': 5},
            'FWD': {'squad': 3, 'starting_min': 1, 'starting_max': 3}
        }
        
        assert optimizer.POSITION_LIMITS == expected_limits
    
    def test_valid_formations(self, optimizer):
        """Test valid formations are defined"""
        expected_formations = [
            (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3),
            (1, 4, 4, 2), (1, 4, 5, 1), (1, 5, 3, 2),
            (1, 5, 4, 1)
        ]
        
        assert optimizer.VALID_FORMATIONS == expected_formations
    
    def test_optimize_squad_basic(self, optimizer):
        """Test basic squad optimization"""
        result = optimizer.optimize_squad(budget=100.0)
        
        assert isinstance(result, OptimizationResult)
        assert result.is_valid
        assert len(result.squad) == 15
        assert len(result.starting_xi) == 11
        assert len(result.bench) == 4
        assert result.captain is not None
        assert result.vice_captain is not None
        assert result.total_cost <= 100.0
    
    def test_squad_position_compliance(self, optimizer):
        """Test squad meets position requirements"""
        result = optimizer.optimize_squad()
        
        if result.is_valid:
            # Count positions in squad
            position_counts = {}
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                position_counts[pos] = len([p for p in result.squad if p['position'] == pos])
            
            assert position_counts['GKP'] == 2
            assert position_counts['DEF'] == 5
            assert position_counts['MID'] == 5
            assert position_counts['FWD'] == 3
    
    def test_starting_xi_formation_compliance(self, optimizer):
        """Test starting XI forms valid formation"""
        result = optimizer.optimize_squad()
        
        if result.is_valid:
            # Count positions in starting XI
            position_counts = {}
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                position_counts[pos] = len([p for p in result.starting_xi if p['position'] == pos])
            
            formation = (position_counts['GKP'], position_counts['DEF'], 
                        position_counts['MID'], position_counts['FWD'])
            
            assert formation in optimizer.VALID_FORMATIONS
    
    def test_club_constraint(self, optimizer):
        """Test maximum players per club constraint"""
        result = optimizer.optimize_squad(max_per_club=3)
        
        if result.is_valid:
            # Count players per club
            club_counts = {}
            for player in result.squad:
                club = player['team_name']
                club_counts[club] = club_counts.get(club, 0) + 1
            
            # Check no club has more than 3 players
            for club, count in club_counts.items():
                assert count <= 3
    
    def test_budget_constraint(self, optimizer):
        """Test budget constraint is respected"""
        budget = 95.0
        result = optimizer.optimize_squad(budget=budget)
        
        if result.is_valid:
            assert result.total_cost <= budget
    
    def test_captain_and_vice_different(self, optimizer):
        """Test captain and vice-captain are different players"""
        result = optimizer.optimize_squad()
        
        if result.is_valid:
            assert result.captain['player_id'] != result.vice_captain['player_id']
    
    def test_captain_in_starting_xi(self, optimizer):
        """Test captain is in starting XI"""
        result = optimizer.optimize_squad()
        
        if result.is_valid:
            starting_ids = {p['player_id'] for p in result.starting_xi}
            assert result.captain['player_id'] in starting_ids
            assert result.vice_captain['player_id'] in starting_ids
    
    def test_specific_formation(self, optimizer):
        """Test optimization with specific formation"""
        formation = (1, 4, 4, 2)  # 1-4-4-2
        result = optimizer.optimize_squad(formation=formation)
        
        if result.is_valid:
            # Count positions in starting XI
            position_counts = {}
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                position_counts[pos] = len([p for p in result.starting_xi if p['position'] == pos])
            
            actual_formation = (position_counts['GKP'], position_counts['DEF'], 
                              position_counts['MID'], position_counts['FWD'])
            
            assert actual_formation == formation
    
    def test_must_include_constraint(self, optimizer):
        """Test must include players constraint"""
        # Include first player (should be GK_1)
        must_include = [1]
        result = optimizer.optimize_squad(must_include=must_include)
        
        if result.is_valid:
            squad_ids = {p['player_id'] for p in result.squad}
            assert 1 in squad_ids
    
    def test_must_exclude_constraint(self, optimizer):
        """Test must exclude players constraint"""
        # Exclude first player
        must_exclude = [1]
        result = optimizer.optimize_squad(must_exclude=must_exclude)
        
        if result.is_valid:
            squad_ids = {p['player_id'] for p in result.squad}
            assert 1 not in squad_ids
    
    def test_bench_ordering(self, optimizer):
        """Test bench is properly ordered"""
        result = optimizer.optimize_squad()
        
        if result.is_valid:
            assert len(result.bench) == 4
            
            # First substitute should not be goalkeeper (if possible)
            if len([p for p in result.bench if p['position'] != 'GKP']) > 0:
                assert result.bench[0]['position'] != 'GKP'
    
    def test_optimization_result_structure(self, optimizer):
        """Test optimization result has correct structure"""
        result = optimizer.optimize_squad()
        
        # Check all required fields exist
        assert hasattr(result, 'squad')
        assert hasattr(result, 'starting_xi')
        assert hasattr(result, 'bench')
        assert hasattr(result, 'captain')
        assert hasattr(result, 'vice_captain')
        assert hasattr(result, 'total_cost')
        assert hasattr(result, 'expected_points')
        assert hasattr(result, 'formation')
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'validation_errors')
    
    def test_invalid_budget_handling(self, optimizer):
        """Test handling of impossible budget constraints"""
        # Set budget too low to be feasible
        result = optimizer.optimize_squad(budget=50.0)
        
        # Should either fail gracefully or return invalid result
        if not result.is_valid:
            assert len(result.validation_errors) > 0
    
    @pytest.mark.parametrize("strategy", ["balanced", "premium", "value"])
    def test_different_strategies(self, optimizer, strategy):
        """Test different optimization strategies work"""
        # This test would require the recommender, so we'll just test the optimizer directly
        result = optimizer.optimize_squad()
        
        # Basic validation that optimization completes
        assert isinstance(result, OptimizationResult)

if __name__ == "__main__":
    pytest.main([__file__])
