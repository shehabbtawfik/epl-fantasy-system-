
#!/usr/bin/env python3
"""
FPL Rules Validator
Validates squad compliance with all FPL rules and constraints.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class FPLValidator:
    """
    Validates FPL squads against all official rules and constraints.
    """
    
    def __init__(self):
        """Initialize the FPL Validator with rule definitions."""
        self.SQUAD_SIZE = 15
        self.STARTING_XI_SIZE = 11
        self.MAX_PER_CLUB = 3
        self.BUDGET = 100.0
        
        # Position constraints
        self.POSITION_LIMITS = {
            'GKP': {'squad': 2, 'starting_min': 1, 'starting_max': 1},
            'DEF': {'squad': 5, 'starting_min': 3, 'starting_max': 5},
            'MID': {'squad': 5, 'starting_min': 3, 'starting_max': 5},
            'FWD': {'squad': 3, 'starting_min': 1, 'starting_max': 3}
        }
        
        # Valid formations (GK-DEF-MID-FWD)
        self.VALID_FORMATIONS = [
            (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3),
            (1, 4, 4, 2), (1, 4, 5, 1), (1, 5, 3, 2),
            (1, 5, 4, 1)
        ]
    
    def validate_squad(self, squad: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate complete 15-man squad against FPL rules.
        
        Args:
            squad: List of player dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Rule 1: Squad size must be exactly 15
        if len(squad) != self.SQUAD_SIZE:
            errors.append(f"Squad size must be {self.SQUAD_SIZE}, got {len(squad)}")
        
        # Rule 2: Position limits
        position_counts = {}
        for pos in self.POSITION_LIMITS:
            position_counts[pos] = len([p for p in squad if p.get('position') == pos])
        
        for pos, limits in self.POSITION_LIMITS.items():
            if position_counts[pos] != limits['squad']:
                errors.append(f"Must have exactly {limits['squad']} {pos}, got {position_counts[pos]}")
        
        # Rule 3: Club limits (max 3 per club)
        club_counts = {}
        for player in squad:
            club = player.get('team_name', 'Unknown')
            club_counts[club] = club_counts.get(club, 0) + 1
        
        for club, count in club_counts.items():
            if count > self.MAX_PER_CLUB:
                errors.append(f"Too many players from {club}: {count} (max {self.MAX_PER_CLUB})")
        
        # Rule 4: Budget constraint
        total_cost = sum(player.get('current_price', 0) for player in squad)
        if total_cost > self.BUDGET:
            errors.append(f"Squad cost £{total_cost:.1f}m exceeds budget £{self.BUDGET:.1f}m")
        
        # Rule 5: All players must be available
        for player in squad:
            if player.get('status') != 'a':
                errors.append(f"Player {player.get('name', 'Unknown')} is not available (status: {player.get('status')})")
        
        return len(errors) == 0, errors
    
    def validate_starting_xi(self, starting_xi: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate starting XI against FPL formation rules.
        
        Args:
            starting_xi: List of 11 starting player dictionaries
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Rule 1: Starting XI size must be exactly 11
        if len(starting_xi) != self.STARTING_XI_SIZE:
            errors.append(f"Starting XI must have {self.STARTING_XI_SIZE} players, got {len(starting_xi)}")
            return False, errors
        
        # Rule 2: Position counts for valid formation
        position_counts = {}
        for pos in self.POSITION_LIMITS:
            position_counts[pos] = len([p for p in starting_xi if p.get('position') == pos])
        
        formation = (position_counts['GKP'], position_counts['DEF'], 
                    position_counts['MID'], position_counts['FWD'])
        
        if formation not in self.VALID_FORMATIONS:
            errors.append(f"Invalid formation {formation[0]}-{formation[1]}-{formation[2]}-{formation[3]}")
        
        # Rule 3: Position limits
        for pos, limits in self.POSITION_LIMITS.items():
            count = position_counts[pos]
            if count < limits['starting_min'] or count > limits['starting_max']:
                errors.append(f"Invalid {pos} count: {count} (must be {limits['starting_min']}-{limits['starting_max']})")
        
        return len(errors) == 0, errors
    
    def validate_captaincy(self, 
                          starting_xi: List[Dict[str, Any]], 
                          captain: Dict[str, Any], 
                          vice_captain: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate captain and vice-captain selections.
        
        Args:
            starting_xi: List of starting XI players
            captain: Captain player dictionary
            vice_captain: Vice-captain player dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        starting_ids = {p.get('player_id') for p in starting_xi}
        
        # Rule 1: Captain must be in starting XI
        if captain.get('player_id') not in starting_ids:
            errors.append("Captain must be in starting XI")
        
        # Rule 2: Vice-captain must be in starting XI
        if vice_captain.get('player_id') not in starting_ids:
            errors.append("Vice-captain must be in starting XI")
        
        # Rule 3: Captain and vice-captain must be different players
        if captain.get('player_id') == vice_captain.get('player_id'):
            errors.append("Captain and vice-captain must be different players")
        
        return len(errors) == 0, errors
    
    def validate_bench_order(self, bench: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate bench ordering rules.
        
        Args:
            bench: List of bench players in order (1st, 2nd, 3rd substitute)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Rule 1: Bench must have exactly 4 players
        if len(bench) != 4:
            errors.append(f"Bench must have 4 players, got {len(bench)}")
            return False, errors
        
        # Rule 2: First substitute should be an outfield player (not GK) - but this is just a recommendation
        # We'll make this a warning rather than an error for now
        if bench[0].get('position') == 'GKP':
            # This is actually allowed in FPL, just not optimal
            pass
        
        # Rule 3: Goalkeeper should be on bench
        gk_on_bench = any(p.get('position') == 'GKP' for p in bench)
        if not gk_on_bench:
            errors.append("Must have a goalkeeper on the bench")
        
        return len(errors) == 0, errors
    
    def validate_complete_team(self, 
                              squad: List[Dict[str, Any]],
                              starting_xi: List[Dict[str, Any]],
                              bench: List[Dict[str, Any]],
                              captain: Dict[str, Any],
                              vice_captain: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Perform complete validation of entire team setup.
        
        Args:
            squad: Complete 15-man squad
            starting_xi: Starting XI players
            bench: Bench players
            captain: Captain selection
            vice_captain: Vice-captain selection
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        all_errors = []
        
        # Validate squad
        squad_valid, squad_errors = self.validate_squad(squad)
        all_errors.extend(squad_errors)
        
        # Validate starting XI
        xi_valid, xi_errors = self.validate_starting_xi(starting_xi)
        all_errors.extend(xi_errors)
        
        # Validate bench
        bench_valid, bench_errors = self.validate_bench_order(bench)
        all_errors.extend(bench_errors)
        
        # Validate captaincy
        cap_valid, cap_errors = self.validate_captaincy(starting_xi, captain, vice_captain)
        all_errors.extend(cap_errors)
        
        # Cross-validation: starting XI + bench should equal squad
        all_selected_ids = {p.get('player_id') for p in starting_xi + bench}
        squad_ids = {p.get('player_id') for p in squad}
        
        if all_selected_ids != squad_ids:
            all_errors.append("Starting XI + bench does not match squad")
        
        return len(all_errors) == 0, all_errors
    
    def get_formation_string(self, starting_xi: List[Dict[str, Any]]) -> str:
        """
        Get formation string from starting XI.
        
        Args:
            starting_xi: List of starting XI players
            
        Returns:
            Formation string (e.g., "1-4-4-2")
        """
        position_counts = {}
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            position_counts[pos] = len([p for p in starting_xi if p.get('position') == pos])
        
        return f"{position_counts['GKP']}-{position_counts['DEF']}-{position_counts['MID']}-{position_counts['FWD']}"
