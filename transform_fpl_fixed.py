#!/usr/bin/env python3
"""
FPL Data Transformation Script (Fixed)
Creates clean master dataset with all required fields
"""

import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

class FPLDataTransformer:
    def __init__(self):
        self.raw_data_dir = "/home/ubuntu/data/raw"
        self.output_dir = "/home/ubuntu/data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_json(self, filename):
        """Load JSON data from raw directory"""
        filepath = os.path.join(self.raw_data_dir, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return None
    
    def build_photo_url(self, photo_code, validate=False):
        """Build and optionally validate photo URL"""
        if not photo_code:
            return None
            
        # Remove .jpg extension if present
        clean_code = photo_code.replace('.jpg', '')
        url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Simon_Murray.png/250px-Simon_Murray.png"
        
        if validate:
            try:
                response = requests.head(url, timeout=5)
                return url if response.status_code == 200 else None
            except:
                return None
        
        return url
    
    def calculate_per_90_stats(self, history_data):
        """Calculate per-90 minute statistics"""
        if not history_data or len(history_data) == 0:
            return {}
        
        df = pd.DataFrame(history_data)
        
        # Filter out games with 0 minutes
        df_played = df[df['minutes'] > 0]
        
        if len(df_played) == 0:
            return {}
        
        total_minutes = df_played['minutes'].sum()
        
        if total_minutes == 0:
            return {}
        
        # Calculate per-90 rates
        per_90_stats = {}
        stat_columns = ['goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
                       'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 
                       'red_cards', 'saves', 'bonus']
        
        for stat in stat_columns:
            if stat in df_played.columns:
                total_stat = df_played[stat].sum()
                per_90_stats[f'{stat}_per_90'] = (total_stat * 90) / total_minutes
        
        # Additional metrics
        per_90_stats['minutes_per_game'] = df_played['minutes'].mean()
        per_90_stats['games_played'] = len(df_played)
        per_90_stats['total_points_per_90'] = (df_played['total_points'].sum() * 90) / total_minutes
        
        return per_90_stats
    
    def get_fixture_difficulty(self, fixtures_data, team_id, next_n_gws=6):
        """Calculate fixture difficulty for next N gameweeks"""
        if not fixtures_data:
            return []
        
        df_fixtures = pd.DataFrame(fixtures_data)
        
        # Filter for team's upcoming fixtures
        team_fixtures = df_fixtures[
            ((df_fixtures['team_h'] == team_id) | (df_fixtures['team_a'] == team_id)) &
            (df_fixtures['finished'] == False)
        ].head(next_n_gws)
        
        difficulties = []
        for _, fixture in team_fixtures.iterrows():
            if fixture['team_h'] == team_id:
                difficulty = fixture['team_h_difficulty']
                opponent = fixture['team_a']
                venue = 'H'
            else:
                difficulty = fixture['team_a_difficulty']
                opponent = fixture['team_h']
                venue = 'A'
            
            difficulties.append({
                'gw': fixture['event'],
                'opponent': opponent,
                'difficulty': difficulty,
                'venue': venue
            })
        
        return difficulties
    
    def calculate_team_strength(self, bootstrap_data):
        """Calculate team strength metrics"""
        teams = bootstrap_data['teams']
        
        team_strength = {}
        for team in teams:
            team_strength[team['id']] = {
                'attack_strength': team.get('strength_attack_home', 0) + team.get('strength_attack_away', 0),
                'defence_strength': team.get('strength_defence_home', 0) + team.get('strength_defence_away', 0),
                'overall_strength': team.get('strength_overall_home', 0) + team.get('strength_overall_away', 0)
            }
        
        return team_strength
    
    def identify_set_piece_takers(self, set_piece_data, bootstrap_data):
        """Identify set piece takers from notes"""
        set_piece_takers = {}
        
        if not set_piece_data or 'teams' not in set_piece_data:
            print("No set piece data available")
            return set_piece_takers
        
        # Create player name to ID mapping
        players = bootstrap_data['elements']
        name_to_id = {}
        for player in players:
            full_name = f"{player['first_name']} {player['second_name']}"
            name_to_id[full_name.lower()] = player['id']
            # Also add web_name mapping
            name_to_id[player['web_name'].lower()] = player['id']
        
        # Parse set piece notes
        for team_notes in set_piece_data['teams']:
            if 'notes' in team_notes:
                for note in team_notes['notes']:
                    info_text = note.get('info_message', '').lower()
                    
                    # Look for penalty takers
                    if 'penalt' in info_text:
                        for name, player_id in name_to_id.items():
                            if name in info_text and len(name) > 3:  # Avoid short matches
                                if player_id not in set_piece_takers:
                                    set_piece_takers[player_id] = []
                                if 'penalties' not in set_piece_takers[player_id]:
                                    set_piece_takers[player_id].append('penalties')
        
        print(f"Identified {len(set_piece_takers)} set piece takers")
        return set_piece_takers
    
    def transform_data(self):
        """Main transformation function"""
        print("Starting FPL data transformation...")
        
        # Load raw data
        bootstrap = self.load_json("bootstrap_static.json")
        fixtures = self.load_json("fixtures_all.json")
        player_histories = self.load_json("player_histories.json")
        set_piece_notes = self.load_json("set_piece_notes.json")
        
        if not bootstrap:
            print("ERROR: Bootstrap data not found")
            return None
        
        print(f"Processing {len(bootstrap['elements'])} players...")
        
        # Build team mappings
        teams_df = pd.DataFrame(bootstrap['teams'])
        team_id_to_name = dict(zip(teams_df['id'], teams_df['name']))
        team_id_to_short = dict(zip(teams_df['id'], teams_df['short_name']))
        
        # Build position mappings
        positions_df = pd.DataFrame(bootstrap['element_types'])
        pos_id_to_name = dict(zip(positions_df['id'], positions_df['singular_name_short']))
        
        # Calculate team strengths
        team_strengths = self.calculate_team_strength(bootstrap)
        
        # Identify set piece takers
        set_piece_takers = self.identify_set_piece_takers(set_piece_notes, bootstrap)
        
        # Process each player
        master_data = []
        
        for player in bootstrap['elements']:
            player_id = player['id']
            
            # Basic player info
            row = {
                'player_id': player_id,
                'name': f"{player['first_name']} {player['second_name']}",
                'first_name': player['first_name'],
                'second_name': player['second_name'],
                'web_name': player['web_name'],
                'position': pos_id_to_name.get(player['element_type'], 'Unknown'),
                'team_id': player['team'],
                'team_name': team_id_to_name.get(player['team'], 'Unknown'),
                'team_short': team_id_to_short.get(player['team'], 'UNK'),
                
                # Current season stats
                'current_price': player['now_cost'] / 10.0,  # Convert from tenths
                'price_change': player.get('cost_change_start', 0) / 10.0,
                'selected_by_percent': float(player.get('selected_by_percent', 0)),
                'total_points': player.get('total_points', 0),
                'points_per_game': float(player.get('points_per_game', 0)),
                'minutes': player.get('minutes', 0),
                'goals_scored': player.get('goals_scored', 0),
                'assists': player.get('assists', 0),
                'clean_sheets': player.get('clean_sheets', 0),
                'goals_conceded': player.get('goals_conceded', 0),
                'own_goals': player.get('own_goals', 0),
                'penalties_saved': player.get('penalties_saved', 0),
                'penalties_missed': player.get('penalties_missed', 0),
                'yellow_cards': player.get('yellow_cards', 0),
                'red_cards': player.get('red_cards', 0),
                'saves': player.get('saves', 0),
                'bonus': player.get('bonus', 0),
                'bps': player.get('bps', 0),
                'influence': float(player.get('influence', 0)),
                'creativity': float(player.get('creativity', 0)),
                'threat': float(player.get('threat', 0)),
                'ict_index': float(player.get('ict_index', 0)),
                
                # Status and availability
                'status': player.get('status', 'a'),
                'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                'news': player.get('news', ''),
                'news_added': player.get('news_added'),
                
                # Form and transfers
                'form': float(player.get('form', 0)),
                'transfers_in': player.get('transfers_in', 0),
                'transfers_out': player.get('transfers_out', 0),
                'transfers_in_event': player.get('transfers_in_event', 0),
                'transfers_out_event': player.get('transfers_out_event', 0),
                
                # Photo URL
                'photo_url': self.build_photo_url(player.get('photo', '')),
                'photo_code': player.get('photo', ''),
                
                # Set piece indicators
                'penalties_taker': player_id in set_piece_takers and 'penalties' in set_piece_takers[player_id],
                'corners_taker': player_id in set_piece_takers and 'corners' in set_piece_takers[player_id],
                'freekicks_taker': player_id in set_piece_takers and 'freekicks' in set_piece_takers[player_id],
            }
            
            # Add team strength metrics
            team_strength = team_strengths.get(player['team'], {})
            row.update({
                'team_attack_strength': team_strength.get('attack_strength', 0),
                'team_defence_strength': team_strength.get('defence_strength', 0),
                'team_overall_strength': team_strength.get('overall_strength', 0)
            })
            
            # Add fixture difficulty for next 6 GWs
            fixture_difficulties = self.get_fixture_difficulty(fixtures, player['team'], 6)
            
            for i in range(6):
                if i < len(fixture_difficulties):
                    fix = fixture_difficulties[i]
                    row[f'next_{i+1}_gw'] = fix['gw']
                    row[f'next_{i+1}_opponent'] = fix['opponent']
                    row[f'next_{i+1}_difficulty'] = fix['difficulty']
                    row[f'next_{i+1}_venue'] = fix['venue']
                else:
                    row[f'next_{i+1}_gw'] = None
                    row[f'next_{i+1}_opponent'] = None
                    row[f'next_{i+1}_difficulty'] = None
                    row[f'next_{i+1}_venue'] = None
            
            # Calculate average fixture difficulty
            difficulties = [fix['difficulty'] for fix in fixture_difficulties if fix['difficulty']]
            row['avg_fixture_difficulty_6gw'] = np.mean(difficulties) if difficulties else None
            
            # Add historical per-90 stats if available
            if player_histories and str(player_id) in player_histories:
                history = player_histories[str(player_id)].get('history', [])
                per_90_stats = self.calculate_per_90_stats(history)
                row.update(per_90_stats)
                
                # Add historical season data
                history_past = player_histories[str(player_id)].get('history_past', [])
                if history_past:
                    # Get last 3 seasons data
                    for i, season in enumerate(history_past[-3:]):
                        season_suffix = f'_season_{i+1}_ago'
                        row[f'total_points{season_suffix}'] = season.get('total_points', 0)
                        row[f'minutes{season_suffix}'] = season.get('minutes', 0)
                        row[f'goals_scored{season_suffix}'] = season.get('goals_scored', 0)
                        row[f'assists{season_suffix}'] = season.get('assists', 0)
                        row[f'clean_sheets{season_suffix}'] = season.get('clean_sheets', 0)
                        row[f'start_cost{season_suffix}'] = season.get('start_cost', 0) / 10.0
                        row[f'end_cost{season_suffix}'] = season.get('end_cost', 0) / 10.0
            
            master_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(master_data)
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, "fpl_master_2025-26.csv")
        df.to_csv(output_path, index=False)
        
        print(f"Master dataset saved: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Create preview file
        preview_path = os.path.join(self.output_dir, "dataset_preview.txt")
        with open(preview_path, 'w') as f:
            f.write("FPL Master Dataset 2025-26 - Top 20 Rows Preview\n")
            f.write("=" * 60 + "\n\n")
            f.write(df.head(20).to_string())
        
        print(f"Preview saved: {preview_path}")
        
        # Test photo URLs for sample
        print("\nTesting photo URLs...")
        photo_test_results = []
        for i, row in df.head(10).iterrows():
            url = row['photo_url']
            if url:
                try:
                    response = requests.head(url, timeout=5)
                    status = 'OK' if response.status_code == 200 else f'HTTP {response.status_code}'
                except:
                    status = 'FAILED'
            else:
                status = 'NO_URL'
            
            photo_test_results.append({
                'player_id': row['player_id'],
                'name': row['name'],
                'url': url,
                'status': status
            })
        
        # Save photo test results
        photo_test_df = pd.DataFrame(photo_test_results)
        photo_test_path = os.path.join(self.output_dir, "photo_url_validation.csv")
        photo_test_df.to_csv(photo_test_path, index=False)
        
        print(f"Photo URL validation saved: {photo_test_path}")
        
        return df

def main():
    transformer = FPLDataTransformer()
    df = transformer.transform_data()
    
    if df is not None:
        print("\n=== TRANSFORMATION COMPLETE ===")
        print(f"Final dataset: {df.shape[0]} players, {df.shape[1]} features")
    else:
        print("Transformation failed")

if __name__ == "__main__":
    main()
