#!/usr/bin/env python3
"""
FPL Fixture Analysis Module
Detects double gameweeks, blank gameweeks, and calculates fixture congestion
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class FPLFixtureAnalyzer:
    def __init__(self):
        self.raw_data_dir = "/home/ubuntu/data/raw"
        self.output_dir = "/home/ubuntu/data"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """Load fixtures and bootstrap data"""
        try:
            with open(os.path.join(self.raw_data_dir, "fixtures_all.json"), 'r') as f:
                fixtures = json.load(f)
            
            with open(os.path.join(self.raw_data_dir, "bootstrap_static.json"), 'r') as f:
                bootstrap = json.load(f)
            
            return fixtures, bootstrap
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def detect_double_blank_gameweeks(self, fixtures, bootstrap):
        """Detect double and blank gameweeks for each team"""
        print("Analyzing fixture patterns...")
        
        # Convert to DataFrame
        fixtures_df = pd.DataFrame(fixtures)
        teams_df = pd.DataFrame(bootstrap['teams'])
        events_df = pd.DataFrame(bootstrap['events'])
        
        # Create team mapping
        team_id_to_name = dict(zip(teams_df['id'], teams_df['name']))
        
        # Filter for future fixtures only
        future_fixtures = fixtures_df[fixtures_df['finished'] == False].copy()
        
        # Count fixtures per team per gameweek
        fixture_counts = {}
        
        for _, fixture in future_fixtures.iterrows():
            gw = fixture['event']
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            
            # Initialize if not exists
            if gw not in fixture_counts:
                fixture_counts[gw] = {}
            
            # Count fixtures for each team
            for team in [home_team, away_team]:
                if team not in fixture_counts[gw]:
                    fixture_counts[gw][team] = 0
                fixture_counts[gw][team] += 1
        
        # Analyze patterns
        dgw_bgw_analysis = []
        
        for gw in sorted(fixture_counts.keys()):
            gw_data = fixture_counts[gw]
            
            # Find teams with double gameweeks (2+ fixtures)
            dgw_teams = [team for team, count in gw_data.items() if count >= 2]
            
            # Find teams with blank gameweeks (0 fixtures)
            all_teams = set(teams_df['id'])
            teams_with_fixtures = set(gw_data.keys())
            bgw_teams = list(all_teams - teams_with_fixtures)
            
            dgw_bgw_analysis.append({
                'gameweek': gw,
                'total_fixtures': len(future_fixtures[future_fixtures['event'] == gw]),
                'dgw_teams': dgw_teams,
                'dgw_count': len(dgw_teams),
                'bgw_teams': bgw_teams,
                'bgw_count': len(bgw_teams),
                'is_dgw': len(dgw_teams) > 0,
                'is_bgw': len(bgw_teams) > 0
            })
        
        return dgw_bgw_analysis, fixture_counts
    
    def calculate_fixture_congestion(self, fixtures, bootstrap, window_days=14):
        """Calculate fixture congestion index for each team"""
        print("Calculating fixture congestion...")
        
        fixtures_df = pd.DataFrame(fixtures)
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        # Convert kickoff times to datetime
        fixtures_df['kickoff_time'] = pd.to_datetime(fixtures_df['kickoff_time'])
        
        # Filter for future fixtures
        future_fixtures = fixtures_df[
            (fixtures_df['finished'] == False) & 
            (fixtures_df['kickoff_time'].notna())
        ].copy()
        
        congestion_data = []
        
        for team_id in teams_df['id']:
            team_fixtures = future_fixtures[
                (future_fixtures['team_h'] == team_id) | 
                (future_fixtures['team_a'] == team_id)
            ].copy()
            
            if len(team_fixtures) == 0:
                continue
            
            team_fixtures = team_fixtures.sort_values('kickoff_time')
            
            # Calculate congestion for next few fixtures
            for i, (_, fixture) in enumerate(team_fixtures.head(10).iterrows()):
                fixture_date = fixture['kickoff_time']
                
                # Count fixtures in window around this fixture
                window_start = fixture_date - timedelta(days=window_days//2)
                window_end = fixture_date + timedelta(days=window_days//2)
                
                fixtures_in_window = team_fixtures[
                    (team_fixtures['kickoff_time'] >= window_start) &
                    (team_fixtures['kickoff_time'] <= window_end)
                ]
                
                # Calculate days between consecutive fixtures
                if i > 0:
                    prev_fixture_date = team_fixtures.iloc[i-1]['kickoff_time']
                    days_since_last = (fixture_date - prev_fixture_date).days
                else:
                    days_since_last = None
                
                if i < len(team_fixtures) - 1:
                    next_fixture_date = team_fixtures.iloc[i+1]['kickoff_time']
                    days_to_next = (next_fixture_date - fixture_date).days
                else:
                    days_to_next = None
                
                # Congestion index (higher = more congested)
                congestion_index = len(fixtures_in_window) / (window_days / 7)  # fixtures per week
                
                congestion_data.append({
                    'team_id': team_id,
                    'gameweek': fixture['event'],
                    'fixture_date': fixture_date,
                    'fixtures_in_window': len(fixtures_in_window),
                    'congestion_index': congestion_index,
                    'days_since_last': days_since_last,
                    'days_to_next': days_to_next,
                    'is_home': fixture['team_h'] == team_id,
                    'opponent': fixture['team_a'] if fixture['team_h'] == team_id else fixture['team_h']
                })
        
        return pd.DataFrame(congestion_data)
    
    def calculate_rotation_risk(self, congestion_df, dgw_bgw_analysis):
        """Calculate rotation risk based on fixture congestion"""
        print("Calculating rotation risk...")
        
        rotation_risk = congestion_df.copy()
        
        # Base rotation risk from congestion
        rotation_risk['base_rotation_risk'] = np.where(
            rotation_risk['congestion_index'] > 1.5, 0.3,  # High congestion
            np.where(rotation_risk['congestion_index'] > 1.0, 0.15, 0.05)  # Medium/Low
        )
        
        # Additional risk for short turnarounds
        rotation_risk['turnaround_risk'] = np.where(
            rotation_risk['days_since_last'] <= 3, 0.2,
            np.where(rotation_risk['days_since_last'] <= 5, 0.1, 0.0)
        )
        
        # DGW rotation penalty
        dgw_gameweeks = [gw['gameweek'] for gw in dgw_bgw_analysis if gw['is_dgw']]
        rotation_risk['dgw_risk'] = np.where(
            rotation_risk['gameweek'].isin(dgw_gameweeks), 0.25, 0.0
        )
        
        # Combined rotation risk
        rotation_risk['total_rotation_risk'] = (
            rotation_risk['base_rotation_risk'] + 
            rotation_risk['turnaround_risk'] + 
            rotation_risk['dgw_risk']
        ).clip(0, 0.8)  # Cap at 80%
        
        return rotation_risk
    
    def generate_fixture_features(self):
        """Generate comprehensive fixture features dataset"""
        print("Generating fixture features...")
        
        fixtures, bootstrap = self.load_data()
        if not fixtures or not bootstrap:
            return None
        
        # Analyze double/blank gameweeks
        dgw_bgw_analysis, fixture_counts = self.detect_double_blank_gameweeks(fixtures, bootstrap)
        
        # Calculate congestion
        congestion_df = self.calculate_fixture_congestion(fixtures, bootstrap)
        
        # Calculate rotation risk
        rotation_df = self.calculate_rotation_risk(congestion_df, dgw_bgw_analysis)
        
        # Create summary by team and gameweek
        fixture_summary = []
        teams_df = pd.DataFrame(bootstrap['teams'])
        
        for team_id in teams_df['id']:
            team_name = teams_df[teams_df['id'] == team_id]['name'].iloc[0]
            
            for gw_data in dgw_bgw_analysis:
                gw = gw_data['gameweek']
                
                # Get congestion data for this team/gw
                team_gw_congestion = rotation_df[
                    (rotation_df['team_id'] == team_id) & 
                    (rotation_df['gameweek'] == gw)
                ]
                
                if len(team_gw_congestion) > 0:
                    congestion_info = team_gw_congestion.iloc[0]
                    congestion_index = congestion_info['congestion_index']
                    rotation_risk = congestion_info['total_rotation_risk']
                    is_home = congestion_info['is_home']
                    opponent = congestion_info['opponent']
                else:
                    congestion_index = 0
                    rotation_risk = 0
                    is_home = None
                    opponent = None
                
                fixture_summary.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'gameweek': gw,
                    'fixture_count': fixture_counts.get(gw, {}).get(team_id, 0),
                    'is_dgw': team_id in gw_data['dgw_teams'],
                    'is_bgw': team_id in gw_data['bgw_teams'],
                    'congestion_index': congestion_index,
                    'rotation_risk': rotation_risk,
                    'is_home': is_home,
                    'opponent_id': opponent
                })
        
        fixture_features_df = pd.DataFrame(fixture_summary)
        
        # Save results
        output_path = os.path.join(self.output_dir, "fixture_features.csv")
        fixture_features_df.to_csv(output_path, index=False)
        
        # Save DGW/BGW analysis
        dgw_bgw_df = pd.DataFrame(dgw_bgw_analysis)
        dgw_bgw_path = os.path.join(self.output_dir, "dgw_bgw_analysis.csv")
        dgw_bgw_df.to_csv(dgw_bgw_path, index=False)
        
        # Save detailed rotation data
        rotation_path = os.path.join(self.output_dir, "rotation_analysis.csv")
        rotation_df.to_csv(rotation_path, index=False)
        
        print(f"Fixture features saved: {output_path}")
        print(f"DGW/BGW analysis saved: {dgw_bgw_path}")
        print(f"Rotation analysis saved: {rotation_path}")
        
        # Print summary
        dgw_count = len([gw for gw in dgw_bgw_analysis if gw['is_dgw']])
        bgw_count = len([gw for gw in dgw_bgw_analysis if gw['is_bgw']])
        
        print(f"\nFixture Analysis Summary:")
        print(f"- Double Gameweeks detected: {dgw_count}")
        print(f"- Blank Gameweeks detected: {bgw_count}")
        print(f"- Teams analyzed: {len(teams_df)}")
        print(f"- Gameweeks analyzed: {len(dgw_bgw_analysis)}")
        
        return fixture_features_df

def main():
    analyzer = FPLFixtureAnalyzer()
    fixture_features = analyzer.generate_fixture_features()
    
    if fixture_features is not None:
        print("\n=== FIXTURE ANALYSIS COMPLETE ===")
        print(f"Generated features for {len(fixture_features)} team-gameweek combinations")
    else:
        print("Fixture analysis failed")

if __name__ == "__main__":
    main()
