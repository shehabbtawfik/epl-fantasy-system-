
#!/usr/bin/env python3
"""
FPL Optimization Engine
Integer Linear Programming optimizer that enforces all FPL rules and constraints.
Uses PuLP for optimization with fallback to OR-Tools if needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pulp
import logging
import os
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-aware default paths
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/home/ubuntu/data")

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    squad: List[Dict[str, Any]]
    starting_xi: List[Dict[str, Any]]
    bench: List[Dict[str, Any]]
    captain: Dict[str, Any]
    vice_captain: Dict[str, Any]
    total_cost: float
    expected_points: float
    formation: str
    is_valid: bool
    validation_errors: List[str]

class FPLOptimizer:
    """
    FPL Rules-Compliant Optimizer using Integer Linear Programming.
    
    Enforces all FPL constraints:
    - 15-man squad structure (2 GK, 5 DEF, 5 MID, 3 FWD)
    - ≤3 players per club constraint
    - Valid starting XI formations (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
    - Budget constraint ≤£100.0m
    - Captain and vice-captain selection
    """
    
    def __init__(self, data_path: str = f"{DEFAULT_DATA_DIR}/fpl_xpts_predictions_enhanced.csv"):
        """
        Initialize the FPL Optimizer.
        
        Args:
            data_path: Path to the enhanced predictions CSV file
        """
        self.data_path = data_path
        self.players_df = None
        self.load_data()
        
        # FPL constraints
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
        
    def load_data(self) -> None:
        """Load and prepare player data for optimization."""
        try:
            logger.info(f"Loading player data from {self.data_path}")
            self.players_df = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            required_cols = ['player_id', 'name', 'position', 'team_name', 'current_price', 
                           'expected_points_ensemble', 'status']
            missing_cols = [col for col in required_cols if col not in self.players_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Filter available players only
            self.players_df = self.players_df[self.players_df['status'] == 'a'].copy()
            
            # Convert position names to FPL format
            position_map = {'GK': 'GKP', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}
            if 'position' in self.players_df.columns:
                self.players_df['position'] = self.players_df['position'].map(position_map).fillna(self.players_df['position'])
            
            logger.info(f"Loaded {len(self.players_df)} available players")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def optimize_squad(self, 
                      budget: float = 100.0,
                      max_per_club: int = 3,
                      formation: Optional[Tuple[int, int, int, int]] = None,
                      must_include: Optional[List[int]] = None,
                      must_exclude: Optional[List[int]] = None) -> OptimizationResult:
        """
        Optimize FPL squad using Integer Linear Programming.
        
        Args:
            budget: Maximum budget in millions (default 100.0)
            max_per_club: Maximum players per club (default 3)
            formation: Specific formation to use (GK, DEF, MID, FWD)
            must_include: List of player IDs that must be included
            must_exclude: List of player IDs that must be excluded
            
        Returns:
            OptimizationResult containing optimal squad and metadata
        """
        try:
            logger.info("Starting squad optimization...")
            
            # Create optimization problem
            prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
            
            # Decision variables
            players = self.players_df.copy()
            player_vars = {}
            starting_vars = {}
            captain_vars = {}
            vice_vars = {}
            
            for idx, player in players.iterrows():
                pid = player['player_id']
                player_vars[pid] = pulp.LpVariable(f"player_{pid}", cat='Binary')
                starting_vars[pid] = pulp.LpVariable(f"starting_{pid}", cat='Binary')
                captain_vars[pid] = pulp.LpVariable(f"captain_{pid}", cat='Binary')
                vice_vars[pid] = pulp.LpVariable(f"vice_{pid}", cat='Binary')
            
            # Objective: Maximize expected points (with captain bonus)
            prob += pulp.lpSum([
                player_vars[row['player_id']] * row['expected_points_ensemble'] +
                captain_vars[row['player_id']] * row['expected_points_ensemble'] +  # Captain gets double
                vice_vars[row['player_id']] * 0  # Vice gets no bonus unless captain doesn't play
                for _, row in players.iterrows()
            ])
            
            # Constraint 1: Squad size (15 players)
            prob += pulp.lpSum([player_vars[pid] for pid in player_vars]) == self.SQUAD_SIZE
            
            # Constraint 2: Starting XI size (11 players)
            prob += pulp.lpSum([starting_vars[pid] for pid in starting_vars]) == self.STARTING_XI_SIZE
            
            # Constraint 3: Budget constraint
            prob += pulp.lpSum([
                player_vars[row['player_id']] * row['current_price']
                for _, row in players.iterrows()
            ]) <= budget
            
            # Constraint 4: Position limits for squad
            for pos, limits in self.POSITION_LIMITS.items():
                pos_players = players[players['position'] == pos]
                prob += pulp.lpSum([
                    player_vars[row['player_id']] for _, row in pos_players.iterrows()
                ]) == limits['squad']
            
            # Constraint 5: Position limits for starting XI
            for pos, limits in self.POSITION_LIMITS.items():
                pos_players = players[players['position'] == pos]
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] for _, row in pos_players.iterrows()
                ]) >= limits['starting_min']
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] for _, row in pos_players.iterrows()
                ]) <= limits['starting_max']
            
            # Constraint 6: Club limits
            for team in players['team_name'].unique():
                team_players = players[players['team_name'] == team]
                prob += pulp.lpSum([
                    player_vars[row['player_id']] for _, row in team_players.iterrows()
                ]) <= max_per_club
            
            # Constraint 7: Starting players must be in squad
            for pid in player_vars:
                prob += starting_vars[pid] <= player_vars[pid]
            
            # Constraint 8: Captain and vice-captain constraints
            prob += pulp.lpSum([captain_vars[pid] for pid in captain_vars]) == 1
            prob += pulp.lpSum([vice_vars[pid] for pid in vice_vars]) == 1
            
            # Captain and vice must be in starting XI
            for pid in player_vars:
                prob += captain_vars[pid] <= starting_vars[pid]
                prob += vice_vars[pid] <= starting_vars[pid]
            
            # Captain and vice must be different players
            for pid in player_vars:
                prob += captain_vars[pid] + vice_vars[pid] <= 1
            
            # Constraint 9: Must include/exclude players
            if must_include:
                for pid in must_include:
                    if pid in player_vars:
                        prob += player_vars[pid] == 1
            
            if must_exclude:
                for pid in must_exclude:
                    if pid in player_vars:
                        prob += player_vars[pid] == 0
            
            # Constraint 10: Specific formation if provided
            if formation:
                gk_count, def_count, mid_count, fwd_count = formation
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] 
                    for _, row in players[players['position'] == 'GKP'].iterrows()
                ]) == gk_count
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] 
                    for _, row in players[players['position'] == 'DEF'].iterrows()
                ]) == def_count
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] 
                    for _, row in players[players['position'] == 'MID'].iterrows()
                ]) == mid_count
                prob += pulp.lpSum([
                    starting_vars[row['player_id']] 
                    for _, row in players[players['position'] == 'FWD'].iterrows()
                ]) == fwd_count
            
            # Solve the problem
            logger.info("Solving optimization problem...")
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Check if solution is optimal
            if prob.status != pulp.LpStatusOptimal:
                logger.error(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
                return OptimizationResult(
                    squad=[], starting_xi=[], bench=[], captain={}, vice_captain={},
                    total_cost=0.0, expected_points=0.0, formation="", 
                    is_valid=False, validation_errors=[f"Optimization failed: {pulp.LpStatus[prob.status]}"]
                )
            
            # Extract results
            selected_players = []
            starting_players = []
            captain = None
            vice_captain = None
            
            for _, player in players.iterrows():
                pid = player['player_id']
                if player_vars[pid].varValue == 1:
                    player_dict = player.to_dict()
                    selected_players.append(player_dict)
                    
                    if starting_vars[pid].varValue == 1:
                        starting_players.append(player_dict)
                        
                    if captain_vars[pid].varValue == 1:
                        captain = player_dict
                        
                    if vice_vars[pid].varValue == 1:
                        vice_captain = player_dict
            
            # Calculate bench (squad - starting XI)
            starting_ids = {p['player_id'] for p in starting_players}
            bench_players = [p for p in selected_players if p['player_id'] not in starting_ids]
            
            # Sort bench by position priority (GK, DEF, MID, FWD)
            position_priority = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            bench_players.sort(key=lambda x: (position_priority.get(x['position'], 5), -x['expected_points_ensemble']))
            
            # Calculate totals
            total_cost = sum(p['current_price'] for p in selected_players)
            expected_points = sum(p['expected_points_ensemble'] for p in starting_players)
            if captain:
                expected_points += captain['expected_points_ensemble']  # Captain bonus
            
            # Determine formation
            formation_counts = {}
            for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                formation_counts[pos] = len([p for p in starting_players if p['position'] == pos])
            
            formation_str = f"{formation_counts['GKP']}-{formation_counts['DEF']}-{formation_counts['MID']}-{formation_counts['FWD']}"
            
            logger.info(f"Optimization completed successfully. Formation: {formation_str}, Cost: £{total_cost:.1f}m")
            
            return OptimizationResult(
                squad=selected_players,
                starting_xi=starting_players,
                bench=bench_players,
                captain=captain or {},
                vice_captain=vice_captain or {},
                total_cost=total_cost,
                expected_points=expected_points,
                formation=formation_str,
                is_valid=True,
                validation_errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            return OptimizationResult(
                squad=[], starting_xi=[], bench=[], captain={}, vice_captain={},
                total_cost=0.0, expected_points=0.0, formation="", 
                is_valid=False, validation_errors=[str(e)]
            )
    
    def optimize_transfers(self, 
                          current_squad: List[int],
                          free_transfers: int = 1,
                          budget: float = 0.0) -> Dict[str, Any]:
        """
        Optimize transfer decisions given current squad.
        
        Args:
            current_squad: List of current player IDs
            free_transfers: Number of free transfers available
            budget: Additional budget available (ITB)
            
        Returns:
            Dictionary containing transfer recommendations
        """
        # This would implement transfer optimization logic
        # For now, return placeholder
        return {
            "transfers_out": [],
            "transfers_in": [],
            "cost": 0.0,
            "points_gain": 0.0
        }
