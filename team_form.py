#!/usr/bin/env python3
"""
FPL Team Form Analysis Module
Calculates team-level performance metrics and form indicators
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

class FPLTeamFormAnalyzer:
    def __init__(self):
        self.raw_data_dir = "/home/ubuntu/data/raw"
        self.output_dir = "/home/ubuntu/data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """Load bootstrap and fixtures data"""
        try:
            with open(os.path.join(self.raw_data_dir, "bootstrap_static.json"), 'r') as f:
                bootstrap = json.load(f)
            
            with open(os.path.join(self.raw_data_dir, "fixtures_all.json"), 'r') as f:
                fixtures = json.load(f)
            
            # Try to load live data for current gameweek
            try:
                with open(os.path.join(self.raw_data_dir, "live_gw1.json"), 'r') as f:
                    live_data = json.load(f)
            except FileNotFoundError:
                live_data = None
            
            return bootstrap, fixtures, live_data
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def calculate_team_stats_from_players(self, bootstrap):
        """Calculate team stats by aggregating player stats"""
        print("Calculating team stats from player data...")
        
        players_df = pd.DataFrame(bootstrap['elements'])
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        # Convert numeric columns to proper types
        numeric_columns = ['total_points', 'goals_scored', 'assists', 'clean_sheets', 
                          'goals_conceded', 'minutes', 'saves', 'bonus', 'bps', 
                          'influence', 'creativity', 'threat', 'ict_index', 'now_cost']
        
        for col in numeric_columns:
            players_df[col] = pd.to_numeric(players_df[col], errors='coerce').fillna(0)
        
        # Convert percentage columns
        players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0)
        players_df['selected_by_percent'] = pd.to_numeric(players_df['selected_by_percent'], errors='coerce').fillna(0)
        
        # Aggregate player stats by team
        team_stats = players_df.groupby('team').agg({
            'total_points': 'sum',
            'goals_scored': 'sum',
            'assists': 'sum',
            'clean_sheets': 'sum',
            'goals_conceded': 'sum',
            'minutes': 'sum',
            'saves': 'sum',
            'bonus': 'sum',
            'bps': 'sum',
            'influence': 'sum',
            'creativity': 'sum',
            'threat': 'sum',
            'ict_index': 'sum',
            'form': 'mean',
            'selected_by_percent': 'mean',
            'now_cost': 'sum'
        }).reset_index()
        
        # Merge with team info
        team_stats = team_stats.merge(teams_df[['id', 'name', 'short_name']], 
                                     left_on='team', right_on='id', how='left')
        
        # Calculate derived metrics
        team_stats['goals_per_game'] = team_stats['goals_scored'] / 1  # Assuming 1 game played so far
        team_stats['goals_conceded_per_game'] = team_stats['goals_conceded'] / 1
        team_stats['clean_sheet_percentage'] = team_stats['clean_sheets'] / 1 * 100
        team_stats['avg_player_form'] = team_stats['form']
        team_stats['total_team_value'] = team_stats['now_cost'] / 10  # Convert from tenths
        
        return team_stats
    
    def calculate_fixture_based_form(self, fixtures, teams_df):
        """Calculate form based on completed fixtures"""
        print("Calculating fixture-based form...")
        
        fixtures_df = pd.DataFrame(fixtures)
        completed_fixtures = fixtures_df[fixtures_df['finished'] == True].copy()
        
        if len(completed_fixtures) == 0:
            print("No completed fixtures found - using default form metrics")
            # Return default form data
            default_form = []
            for _, team in teams_df.iterrows():
                default_form.append({
                    'team_id': team['id'],
                    'team_name': team['name'],
                    'games_played': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'goal_difference': 0,
                    'points': 0,
                    'form_last_5': 0,
                    'attacking_strength': team.get('strength_attack_home', 1000) + team.get('strength_attack_away', 1000),
                    'defensive_strength': team.get('strength_defence_home', 1000) + team.get('strength_defence_away', 1000),
                    'home_form': 0,
                    'away_form': 0
                })
            return pd.DataFrame(default_form)
        
        # Process completed fixtures
        team_form = {}
        
        for _, fixture in completed_fixtures.iterrows():
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            home_score = fixture['team_h_score']
            away_score = fixture['team_a_score']
            
            # Initialize teams if not exists
            for team_id in [home_team, away_team]:
                if team_id not in team_form:
                    team_form[team_id] = {
                        'games_played': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'goals_for': 0,
                        'goals_against': 0,
                        'home_games': 0,
                        'away_games': 0,
                        'home_wins': 0,
                        'away_wins': 0,
                        'home_goals_for': 0,
                        'away_goals_for': 0,
                        'home_goals_against': 0,
                        'away_goals_against': 0,
                        'recent_results': []
                    }
            
            # Update home team stats
            team_form[home_team]['games_played'] += 1
            team_form[home_team]['home_games'] += 1
            team_form[home_team]['goals_for'] += home_score
            team_form[home_team]['goals_against'] += away_score
            team_form[home_team]['home_goals_for'] += home_score
            team_form[home_team]['home_goals_against'] += away_score
            
            # Update away team stats
            team_form[away_team]['games_played'] += 1
            team_form[away_team]['away_games'] += 1
            team_form[away_team]['goals_for'] += away_score
            team_form[away_team]['goals_against'] += home_score
            team_form[away_team]['away_goals_for'] += away_score
            team_form[away_team]['away_goals_against'] += home_score
            
            # Determine result
            if home_score > away_score:
                # Home win
                team_form[home_team]['wins'] += 1
                team_form[home_team]['home_wins'] += 1
                team_form[away_team]['losses'] += 1
                team_form[home_team]['recent_results'].append('W')
                team_form[away_team]['recent_results'].append('L')
            elif home_score < away_score:
                # Away win
                team_form[away_team]['wins'] += 1
                team_form[away_team]['away_wins'] += 1
                team_form[home_team]['losses'] += 1
                team_form[home_team]['recent_results'].append('L')
                team_form[away_team]['recent_results'].append('W')
            else:
                # Draw
                team_form[home_team]['draws'] += 1
                team_form[away_team]['draws'] += 1
                team_form[home_team]['recent_results'].append('D')
                team_form[away_team]['recent_results'].append('D')
        
        # Convert to DataFrame
        form_data = []
        for team_id, stats in team_form.items():
            # Get team name
            team_info = teams_df[teams_df['id'] == team_id]
            team_name = team_info['name'].iloc[0] if len(team_info) > 0 else f"Team {team_id}"
            
            # Calculate derived metrics
            games = stats['games_played']
            points = stats['wins'] * 3 + stats['draws']
            
            # Form calculation (last 5 games)
            recent_results = stats['recent_results'][-5:]
            form_points = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in recent_results])
            
            # Home/away form
            home_points = stats['home_wins'] * 3 + (stats['home_games'] - stats['home_wins'] - 
                         (stats['games_played'] - stats['wins'] - stats['draws'] - 
                          (stats['away_games'] - (stats['games_played'] - stats['home_games']))))
            away_points = stats['away_wins'] * 3 + (stats['away_games'] - stats['away_wins'] - 
                         (stats['losses'] - (stats['games_played'] - stats['home_games'] - stats['away_games'])))
            
            form_data.append({
                'team_id': team_id,
                'team_name': team_name,
                'games_played': games,
                'wins': stats['wins'],
                'draws': stats['draws'],
                'losses': stats['losses'],
                'goals_for': stats['goals_for'],
                'goals_against': stats['goals_against'],
                'goal_difference': stats['goals_for'] - stats['goals_against'],
                'points': points,
                'form_last_5': form_points,
                'attacking_strength': stats['goals_for'] / max(games, 1) * 38,  # Season projection
                'defensive_strength': stats['goals_against'] / max(games, 1) * 38,
                'home_form': home_points / max(stats['home_games'], 1) * 3,
                'away_form': away_points / max(stats['away_games'], 1) * 3
            })
        
        return pd.DataFrame(form_data)
    
    def calculate_clean_sheet_probability(self, team_form_df, fixtures):
        """Calculate clean sheet probability for each team"""
        print("Calculating clean sheet probabilities...")
        
        fixtures_df = pd.DataFrame(fixtures)
        upcoming_fixtures = fixtures_df[fixtures_df['finished'] == False].head(20)  # Next few GWs
        
        cs_probabilities = []
        
        for _, team_row in team_form_df.iterrows():
            # Use 'team' column if 'team_id' doesn't exist
            team_id = team_row.get('team_id', team_row.get('team'))
            
            # Base clean sheet probability from defensive strength
            # Use available defensive metric
            defensive_metric = team_row.get('defensive_strength', 
                                          team_row.get('goals_conceded_per_game', 1.5))
            base_cs_prob = max(0.1, min(0.8, 1 - (defensive_metric / 38) / 2))
            
            # Adjust for home/away
            team_fixtures = upcoming_fixtures[
                (upcoming_fixtures['team_h'] == team_id) | 
                (upcoming_fixtures['team_a'] == team_id)
            ].head(6)  # Next 6 fixtures
            
            fixture_cs_probs = []
            
            for _, fixture in team_fixtures.iterrows():
                is_home = fixture['team_h'] == team_id
                opponent_id = fixture['team_a'] if is_home else fixture['team_h']
                
                # Get opponent attacking strength
                # Use flexible column lookup
                team_col = 'team_id' if 'team_id' in team_form_df.columns else 'team'
                opponent_info = team_form_df[team_form_df[team_col] == opponent_id]
                if len(opponent_info) > 0:
                    if 'attacking_strength' in opponent_info.columns:
                        opponent_attack = opponent_info['attacking_strength'].iloc[0]
                    elif 'goals_per_game' in opponent_info.columns:
                        opponent_attack = opponent_info['goals_per_game'].iloc[0] * 38
                    else:
                        opponent_attack = 50
                else:
                    opponent_attack = 50  # Average
                
                # Adjust probability based on opponent and venue
                venue_multiplier = 1.2 if is_home else 0.8
                opponent_multiplier = max(0.3, min(1.7, 2 - (opponent_attack / 38)))
                
                fixture_cs_prob = base_cs_prob * venue_multiplier * opponent_multiplier
                fixture_cs_prob = max(0.05, min(0.85, fixture_cs_prob))
                
                fixture_cs_probs.append({
                    'team_id': team_id,
                    'gameweek': fixture['event'],
                    'is_home': is_home,
                    'opponent_id': opponent_id,
                    'cs_probability': fixture_cs_prob
                })
            
            # Average CS probability for team
            avg_cs_prob = np.mean([f['cs_probability'] for f in fixture_cs_probs]) if fixture_cs_probs else base_cs_prob
            
            cs_probabilities.append({
                'team_id': team_id,
                'team_name': team_row.get('team_name', team_row.get('name', f'Team {team_id}')),
                'avg_cs_probability': avg_cs_prob,
                'base_cs_probability': base_cs_prob,
                'next_6_fixtures': fixture_cs_probs
            })
        
        return cs_probabilities
    
    def generate_team_form_dataset(self):
        """Generate comprehensive team form dataset"""
        print("Generating team form analysis...")
        
        bootstrap, fixtures, live_data = self.load_data()
        if not bootstrap or not fixtures:
            return None
        
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        # Calculate team stats from player aggregation
        team_stats = self.calculate_team_stats_from_players(bootstrap)
        
        # Calculate fixture-based form
        fixture_form = self.calculate_fixture_based_form(fixtures, teams_df)
        
        # Merge team stats with fixture form
        team_form_df = team_stats.merge(
            fixture_form[['team_id', 'games_played', 'wins', 'draws', 'losses', 
                         'goals_for', 'goals_against', 'goal_difference', 'points',
                         'form_last_5', 'home_form', 'away_form']],
            left_on='team', right_on='team_id', how='left'
        )
        
        # Ensure we have team_id column for later use
        if 'team_id' not in team_form_df.columns:
            team_form_df['team_id'] = team_form_df['team']
        
        # Fill missing values for teams without fixture data
        form_columns = ['games_played', 'wins', 'draws', 'losses', 'goals_for', 
                       'goals_against', 'goal_difference', 'points', 'form_last_5',
                       'home_form', 'away_form']
        team_form_df[form_columns] = team_form_df[form_columns].fillna(0)
        
        # Add strength ratings from bootstrap
        strength_data = []
        for _, team in teams_df.iterrows():
            strength_data.append({
                'team_id': team['id'],
                'strength_overall_home': team.get('strength_overall_home', 1000),
                'strength_overall_away': team.get('strength_overall_away', 1000),
                'strength_attack_home': team.get('strength_attack_home', 1000),
                'strength_attack_away': team.get('strength_attack_away', 1000),
                'strength_defence_home': team.get('strength_defence_home', 1000),
                'strength_defence_away': team.get('strength_defence_away', 1000)
            })
        
        strength_df = pd.DataFrame(strength_data)
        team_form_df = team_form_df.merge(strength_df, left_on='team', right_on='team_id', how='left')
        
        # Calculate clean sheet probabilities
        cs_probabilities = self.calculate_clean_sheet_probability(team_form_df, fixtures)
        cs_df = pd.DataFrame([{
            'team_id': cs['team_id'],
            'avg_cs_probability': cs['avg_cs_probability'],
            'base_cs_probability': cs['base_cs_probability']
        } for cs in cs_probabilities])
        
        team_form_df = team_form_df.merge(cs_df, left_on='team', right_on='team_id', how='left')
        
        # Calculate additional metrics
        team_form_df['ppg'] = team_form_df['points'] / team_form_df['games_played'].replace(0, 1)
        team_form_df['goals_per_game'] = team_form_df['goals_for'] / team_form_df['games_played'].replace(0, 1)
        team_form_df['goals_conceded_per_game'] = team_form_df['goals_against'] / team_form_df['games_played'].replace(0, 1)
        team_form_df['attack_rating'] = (team_form_df['strength_attack_home'] + team_form_df['strength_attack_away']) / 2
        team_form_df['defence_rating'] = (team_form_df['strength_defence_home'] + team_form_df['strength_defence_away']) / 2
        
        # Save results
        output_path = os.path.join(self.output_dir, "team_form.csv")
        team_form_df.to_csv(output_path, index=False)
        
        # Save clean sheet details
        cs_details_path = os.path.join(self.output_dir, "clean_sheet_probabilities.csv")
        cs_details = []
        for cs in cs_probabilities:
            for fixture in cs['next_6_fixtures']:
                cs_details.append(fixture)
        
        if cs_details:
            cs_details_df = pd.DataFrame(cs_details)
            cs_details_df.to_csv(cs_details_path, index=False)
        
        print(f"Team form analysis saved: {output_path}")
        print(f"Clean sheet probabilities saved: {cs_details_path}")
        
        # Print summary
        print(f"\nTeam Form Summary:")
        print(f"- Teams analyzed: {len(team_form_df)}")
        print(f"- Average goals per game: {team_form_df['goals_per_game'].mean():.2f}")
        print(f"- Average goals conceded per game: {team_form_df['goals_conceded_per_game'].mean():.2f}")
        print(f"- Average clean sheet probability: {team_form_df['avg_cs_probability'].mean():.2%}")
        
        # Top attacking teams
        top_attack = team_form_df.nlargest(5, 'attack_rating')[['name', 'attack_rating', 'goals_per_game']]
        print(f"\nTop 5 Attacking Teams:")
        for _, team in top_attack.iterrows():
            print(f"- {team['name']}: {team['attack_rating']:.0f} rating, {team['goals_per_game']:.2f} goals/game")
        
        # Top defensive teams
        top_defence = team_form_df.nsmallest(5, 'defence_rating')[['name', 'defence_rating', 'avg_cs_probability']]
        print(f"\nTop 5 Defensive Teams:")
        for _, team in top_defence.iterrows():
            print(f"- {team['name']}: {team['defence_rating']:.0f} rating, {team['avg_cs_probability']:.1%} CS prob")
        
        return team_form_df

def main():
    analyzer = FPLTeamFormAnalyzer()
    team_form = analyzer.generate_team_form_dataset()
    
    if team_form is not None:
        print("\n=== TEAM FORM ANALYSIS COMPLETE ===")
    else:
        print("Team form analysis failed")

if __name__ == "__main__":
    main()
