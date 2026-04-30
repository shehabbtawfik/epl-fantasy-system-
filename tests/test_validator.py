
#!/usr/bin/env python3
"""
Tests for FPL Validator
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_tool.validator import FPLValidator

class TestFPLValidator:
    """Test cases for FPL Validator"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return FPLValidator()
    
    @pytest.fixture
    def valid_squad(self):
        """Create a valid 15-man squad"""
        squad = []
        
        # 2 Goalkeepers
        for i in range(2):
            squad.append({
                'player_id': i + 1,
                'name': f'GK_{i+1}',
                'position': 'GKP',
                'team_name': f'Team_{i+1}',
                'current_price': 4.5,
                'status': 'a'
            })
        
        # 5 Defenders
        for i in range(5):
            squad.append({
                'player_id': i + 3,
                'name': f'DEF_{i+1}',
                'position': 'DEF',
                'team_name': f'Team_{(i % 3) + 1}',
                'current_price': 4.5,
                'status': 'a'
            })
        
        # 5 Midfielders
        for i in range(5):
            squad.append({
                'player_id': i + 8,
                'name': f'MID_{i+1}',
                'position': 'MID',
                'team_name': f'Team_{(i % 3) + 1}',
                'current_price': 5.0,
                'status': 'a'
            })
        
        # 3 Forwards
        for i in range(3):
            squad.append({
                'player_id': i + 13,
                'name': f'FWD_{i+1}',
                'position': 'FWD',
                'team_name': f'Team_{(i % 3) + 1}',
                'current_price': 7.0,
                'status': 'a'
            })
        
        return squad
    
    @pytest.fixture
    def valid_starting_xi(self):
        """Create a valid starting XI (1-4-4-2 formation)"""
        return [
            {'player_id': 1, 'name': 'GK_1', 'position': 'GKP', 'team_name': 'Team_1'},
            {'player_id': 3, 'name': 'DEF_1', 'position': 'DEF', 'team_name': 'Team_1'},
            {'player_id': 4, 'name': 'DEF_2', 'position': 'DEF', 'team_name': 'Team_2'},
            {'player_id': 5, 'name': 'DEF_3', 'position': 'DEF', 'team_name': 'Team_3'},
            {'player_id': 6, 'name': 'DEF_4', 'position': 'DEF', 'team_name': 'Team_1'},
            {'player_id': 8, 'name': 'MID_1', 'position': 'MID', 'team_name': 'Team_2'},
            {'player_id': 9, 'name': 'MID_2', 'position': 'MID', 'team_name': 'Team_3'},
            {'player_id': 10, 'name': 'MID_3', 'position': 'MID', 'team_name': 'Team_1'},
            {'player_id': 11, 'name': 'MID_4', 'position': 'MID', 'team_name': 'Team_2'},
            {'player_id': 13, 'name': 'FWD_1', 'position': 'FWD', 'team_name': 'Team_3'},
            {'player_id': 14, 'name': 'FWD_2', 'position': 'FWD', 'team_name': 'Team_1'}
        ]
    
    def test_validator_initialization(self, validator):
        """Test validator initializes with correct constraints"""
        assert validator.SQUAD_SIZE == 15
        assert validator.STARTING_XI_SIZE == 11
        assert validator.MAX_PER_CLUB == 3
        assert validator.BUDGET == 100.0
        
        # Check position limits
        assert validator.POSITION_LIMITS['GKP']['squad'] == 2
        assert validator.POSITION_LIMITS['DEF']['squad'] == 5
        assert validator.POSITION_LIMITS['MID']['squad'] == 5
        assert validator.POSITION_LIMITS['FWD']['squad'] == 3
    
    def test_valid_squad_passes(self, validator, valid_squad):
        """Test that a valid squad passes validation"""
        is_valid, errors = validator.validate_squad(valid_squad)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_squad_size(self, validator, valid_squad):
        """Test squad size validation"""
        # Test too few players
        short_squad = valid_squad[:10]
        is_valid, errors = validator.validate_squad(short_squad)
        
        assert not is_valid
        assert any("Squad size must be 15" in error for error in errors)
        
        # Test too many players
        long_squad = valid_squad + [valid_squad[0]]  # Duplicate first player
        is_valid, errors = validator.validate_squad(long_squad)
        
        assert not is_valid
        assert any("Squad size must be 15" in error for error in errors)
    
    def test_invalid_position_counts(self, validator, valid_squad):
        """Test position count validation"""
        # Remove a goalkeeper, add a defender
        invalid_squad = valid_squad.copy()
        invalid_squad = [p for p in invalid_squad if not (p['position'] == 'GKP' and p['player_id'] == 2)]
        invalid_squad.append({
            'player_id': 99,
            'name': 'Extra_DEF',
            'position': 'DEF',
            'team_name': 'Team_1',
            'current_price': 4.5,
            'status': 'a'
        })
        
        is_valid, errors = validator.validate_squad(invalid_squad)
        
        assert not is_valid
        assert any("Must have exactly 2 GKP" in error for error in errors)
        assert any("Must have exactly 5 DEF" in error for error in errors)
    
    def test_club_limit_violation(self, validator, valid_squad):
        """Test club limit validation"""
        # Modify squad to have 4 players from same team
        invalid_squad = valid_squad.copy()
        for i, player in enumerate(invalid_squad):
            if i < 4:  # First 4 players from same team
                player['team_name'] = 'Team_1'
        
        is_valid, errors = validator.validate_squad(invalid_squad)
        
        assert not is_valid
        assert any("Too many players from Team_1: 4" in error for error in errors)
    
    def test_budget_constraint(self, validator, valid_squad):
        """Test budget constraint validation"""
        # Make squad too expensive
        expensive_squad = valid_squad.copy()
        for player in expensive_squad:
            player['current_price'] = 15.0  # Make all players very expensive
        
        is_valid, errors = validator.validate_squad(expensive_squad)
        
        assert not is_valid
        assert any("exceeds budget" in error for error in errors)
    
    def test_unavailable_players(self, validator, valid_squad):
        """Test validation of player availability"""
        # Make one player unavailable
        invalid_squad = valid_squad.copy()
        invalid_squad[0]['status'] = 'i'  # Injured
        
        is_valid, errors = validator.validate_squad(invalid_squad)
        
        assert not is_valid
        assert any("is not available" in error for error in errors)
    
    def test_valid_starting_xi(self, validator, valid_starting_xi):
        """Test valid starting XI passes validation"""
        is_valid, errors = validator.validate_starting_xi(valid_starting_xi)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_starting_xi_size(self, validator, valid_starting_xi):
        """Test starting XI size validation"""
        # Test too few players
        short_xi = valid_starting_xi[:10]
        is_valid, errors = validator.validate_starting_xi(short_xi)
        
        assert not is_valid
        assert any("Starting XI must have 11 players" in error for error in errors)
    
    def test_invalid_formation(self, validator):
        """Test invalid formation detection"""
        # Create invalid formation (2 goalkeepers)
        invalid_xi = [
            {'player_id': 1, 'name': 'GK_1', 'position': 'GKP', 'team_name': 'Team_1'},
            {'player_id': 2, 'name': 'GK_2', 'position': 'GKP', 'team_name': 'Team_2'},
            {'player_id': 3, 'name': 'DEF_1', 'position': 'DEF', 'team_name': 'Team_1'},
            {'player_id': 4, 'name': 'DEF_2', 'position': 'DEF', 'team_name': 'Team_2'},
            {'player_id': 5, 'name': 'DEF_3', 'position': 'DEF', 'team_name': 'Team_3'},
            {'player_id': 8, 'name': 'MID_1', 'position': 'MID', 'team_name': 'Team_1'},
            {'player_id': 9, 'name': 'MID_2', 'position': 'MID', 'team_name': 'Team_2'},
            {'player_id': 10, 'name': 'MID_3', 'position': 'MID', 'team_name': 'Team_3'},
            {'player_id': 11, 'name': 'MID_4', 'position': 'MID', 'team_name': 'Team_1'},
            {'player_id': 13, 'name': 'FWD_1', 'position': 'FWD', 'team_name': 'Team_2'},
            {'player_id': 14, 'name': 'FWD_2', 'position': 'FWD', 'team_name': 'Team_3'}
        ]
        
        is_valid, errors = validator.validate_starting_xi(invalid_xi)
        
        assert not is_valid
        assert any("Invalid formation" in error for error in errors)
    
    def test_captaincy_validation(self, validator, valid_starting_xi):
        """Test captain and vice-captain validation"""
        captain = valid_starting_xi[0]  # First player as captain
        vice_captain = valid_starting_xi[1]  # Second player as vice
        
        is_valid, errors = validator.validate_captaincy(valid_starting_xi, captain, vice_captain)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_captain_not_in_starting_xi(self, validator, valid_starting_xi):
        """Test captain must be in starting XI"""
        captain = {'player_id': 99, 'name': 'Not_In_XI', 'position': 'MID', 'team_name': 'Team_1'}
        vice_captain = valid_starting_xi[1]
        
        is_valid, errors = validator.validate_captaincy(valid_starting_xi, captain, vice_captain)
        
        assert not is_valid
        assert any("Captain must be in starting XI" in error for error in errors)
    
    def test_same_captain_and_vice(self, validator, valid_starting_xi):
        """Test captain and vice-captain must be different"""
        captain = valid_starting_xi[0]
        vice_captain = valid_starting_xi[0]  # Same player
        
        is_valid, errors = validator.validate_captaincy(valid_starting_xi, captain, vice_captain)
        
        assert not is_valid
        assert any("Captain and vice-captain must be different" in error for error in errors)
    
    def test_bench_validation(self, validator):
        """Test bench validation"""
        valid_bench = [
            {'player_id': 2, 'name': 'GK_2', 'position': 'GKP', 'team_name': 'Team_2'},
            {'player_id': 7, 'name': 'DEF_5', 'position': 'DEF', 'team_name': 'Team_3'},
            {'player_id': 12, 'name': 'MID_5', 'position': 'MID', 'team_name': 'Team_1'},
            {'player_id': 15, 'name': 'FWD_3', 'position': 'FWD', 'team_name': 'Team_2'}
        ]
        
        is_valid, errors = validator.validate_bench_order(valid_bench)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_bench_size(self, validator):
        """Test bench size validation"""
        invalid_bench = [
            {'player_id': 2, 'name': 'GK_2', 'position': 'GKP', 'team_name': 'Team_2'},
            {'player_id': 7, 'name': 'DEF_5', 'position': 'DEF', 'team_name': 'Team_3'}
        ]
        
        is_valid, errors = validator.validate_bench_order(invalid_bench)
        
        assert not is_valid
        assert any("Bench must have 4 players" in error for error in errors)
    
    def test_goalkeeper_first_substitute(self, validator):
        """Test first substitute should not be goalkeeper"""
        invalid_bench = [
            {'player_id': 2, 'name': 'GK_2', 'position': 'GKP', 'team_name': 'Team_2'},  # GK first
            {'player_id': 7, 'name': 'DEF_5', 'position': 'DEF', 'team_name': 'Team_3'},
            {'player_id': 12, 'name': 'MID_5', 'position': 'MID', 'team_name': 'Team_1'},
            {'player_id': 15, 'name': 'FWD_3', 'position': 'FWD', 'team_name': 'Team_2'}
        ]
        
        is_valid, errors = validator.validate_bench_order(invalid_bench)
        
        assert not is_valid
        assert any("First substitute should not be a goalkeeper" in error for error in errors)
    
    def test_complete_team_validation(self, validator, valid_squad, valid_starting_xi):
        """Test complete team validation"""
        # Create valid bench (remaining players from squad)
        starting_ids = {p['player_id'] for p in valid_starting_xi}
        bench = [p for p in valid_squad if p['player_id'] not in starting_ids]
        
        captain = valid_starting_xi[0]
        vice_captain = valid_starting_xi[1]
        
        is_valid, errors = validator.validate_complete_team(
            valid_squad, valid_starting_xi, bench, captain, vice_captain
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_formation_string_generation(self, validator, valid_starting_xi):
        """Test formation string generation"""
        formation_str = validator.get_formation_string(valid_starting_xi)
        
        # Should be 1-4-4-2 based on our valid_starting_xi fixture
        assert formation_str == "1-4-4-2"
    
    @pytest.mark.parametrize("formation,expected", [
        ([{'position': 'GKP'}] + [{'position': 'DEF'}]*3 + [{'position': 'MID'}]*4 + [{'position': 'FWD'}]*3, "1-3-4-3"),
        ([{'position': 'GKP'}] + [{'position': 'DEF'}]*4 + [{'position': 'MID'}]*3 + [{'position': 'FWD'}]*3, "1-4-3-3"),
        ([{'position': 'GKP'}] + [{'position': 'DEF'}]*5 + [{'position': 'MID'}]*4 + [{'position': 'FWD'}]*1, "1-5-4-1"),
    ])
    def test_various_formations(self, validator, formation, expected):
        """Test various formation string generations"""
        formation_str = validator.get_formation_string(formation)
        assert formation_str == expected

if __name__ == "__main__":
    pytest.main([__file__])
