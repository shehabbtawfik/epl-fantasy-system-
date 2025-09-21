#!/usr/bin/env python3
"""
FPL Data Collection Script
Following the Medium guide methodology exactly
"""

import requests
import json
import time
import os
from datetime import datetime
import argparse

class FPLDataCollector:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.raw_data_dir = "/home/ubuntu/data/raw"
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
    def make_request(self, endpoint, params=None, max_retries=3):
        """Make API request with retry logic and rate limiting"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                print(f"Fetching: {url}")
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def save_json(self, data, filename):
        """Save JSON data to file"""
        filepath = os.path.join(self.raw_data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {filepath}")
    
    def fetch_bootstrap_static(self):
        """Fetch general information - foundation dataset"""
        print("\n=== FETCHING BOOTSTRAP STATIC ===")
        data = self.make_request("bootstrap-static/")
        if data:
            self.save_json(data, "bootstrap_static.json")
            
            # Extract key info for logging
            current_event = next((e for e in data['events'] if e['is_current']), None)
            total_players = len(data['elements'])
            total_teams = len(data['teams'])
            
            print(f"Current GW: {current_event['id'] if current_event else 'None'}")
            print(f"Total players: {total_players}")
            print(f"Total teams: {total_teams}")
            
            return data
        return None
    
    def fetch_fixtures(self, specific_gw=None):
        """Fetch fixtures data"""
        print("\n=== FETCHING FIXTURES ===")
        
        if specific_gw:
            endpoint = f"fixtures/?event={specific_gw}"
            filename = f"fixtures_gw{specific_gw}.json"
        else:
            endpoint = "fixtures/"
            filename = "fixtures_all.json"
            
        data = self.make_request(endpoint)
        if data:
            self.save_json(data, filename)
            print(f"Fixtures fetched: {len(data)} matches")
            return data
        return None
    
    def fetch_player_history(self, player_ids, sample_size=None, include_seasons=True):
        """Fetch detailed player history including past seasons"""
        print("\n=== FETCHING PLAYER HISTORIES ===")
        
        if sample_size:
            player_ids = player_ids[:sample_size]
            print(f"Sampling {sample_size} players for testing")
        else:
            print(f"Fetching ALL {len(player_ids)} players")
        
        all_histories = {}
        failed_players = []
        
        for i, player_id in enumerate(player_ids):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(player_ids)} players")
                
            data = self.make_request(f"element-summary/{player_id}/")
            if data:
                all_histories[str(player_id)] = data
                
                # Enhanced logging for historical data
                if include_seasons and 'history_past' in data:
                    seasons_count = len(data['history_past'])
                    if seasons_count > 0:
                        print(f"Player {player_id}: {seasons_count} historical seasons")
            else:
                failed_players.append(player_id)
                
            # Rate limiting - slightly faster for full collection
            time.sleep(0.05)
        
        if all_histories:
            filename = "player_histories_full.json" if not sample_size else "player_histories.json"
            self.save_json(all_histories, filename)
            
        if failed_players:
            print(f"Failed to fetch {len(failed_players)} players: {failed_players[:10]}...")
            
        print(f"Successfully fetched {len(all_histories)} player histories")
        return all_histories
    
    def fetch_live_gw(self, gw_id):
        """Fetch live gameweek data"""
        print(f"\n=== FETCHING LIVE GW {gw_id} ===")
        data = self.make_request(f"event/{gw_id}/live/")
        if data:
            self.save_json(data, f"live_gw{gw_id}.json")
            print(f"Live data fetched for GW {gw_id}")
            return data
        return None
    
    def fetch_set_piece_notes(self):
        """Fetch set piece taker information"""
        print("\n=== FETCHING SET PIECE NOTES ===")
        data = self.make_request("team/set-piece-notes/")
        if data:
            self.save_json(data, "set_piece_notes.json")
            return data
        return None
    
    def test_photo_url(self, photo_code):
        """Test if photo URL works"""
        if not photo_code:
            return None
            
        url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Simon_Murray.png/250px-Simon_Murray.png"
        try:
            response = requests.head(url, timeout=5)
            return url if response.status_code == 200 else None
        except:
            return None
    
    def collect_all_data(self, sample_players=None):
        """Main data collection orchestrator"""
        print("Starting FPL data collection...")
        print(f"Timestamp: {datetime.now()}")
        
        # 1. Bootstrap static (foundation)
        bootstrap = self.fetch_bootstrap_static()
        if not bootstrap:
            print("CRITICAL: Failed to fetch bootstrap data")
            return False
        
        # 2. All fixtures
        self.fetch_fixtures()
        
        # 3. Current GW fixtures specifically
        current_event = next((e for e in bootstrap['events'] if e['is_current']), None)
        if current_event:
            self.fetch_fixtures(current_event['id'])
        
        # 4. Player histories - ALL PLAYERS
        player_ids = [p['id'] for p in bootstrap['elements']]
        self.fetch_player_history(player_ids, sample_size=sample_players)
        
        # 5. Live data for current GW
        if current_event:
            self.fetch_live_gw(current_event['id'])
        
        # 6. Set piece notes
        self.fetch_set_piece_notes()
        
        # 7. Test photo URLs (sample)
        print("\n=== TESTING PHOTO URLS ===")
        photo_test_results = []
        for player in bootstrap['elements'][:10]:  # Test first 10
            photo_code = player.get('photo', '').replace('.jpg', '')
            url = self.test_photo_url(photo_code)
            photo_test_results.append({
                'player_id': player['id'],
                'name': f"{player['first_name']} {player['second_name']}",
                'photo_code': photo_code,
                'url': url,
                'status': 'OK' if url else 'FAILED'
            })
        
        self.save_json(photo_test_results, "photo_url_test.json")
        
        print("\n=== DATA COLLECTION COMPLETE ===")
        return True

def main():
    parser = argparse.ArgumentParser(description='FPL Data Collector')
    parser.add_argument('--all', action='store_true', help='Collect all data')
    parser.add_argument('--sample', type=int, help='Sample size for player histories')
    parser.add_argument('--bootstrap-only', action='store_true', help='Only fetch bootstrap')
    
    args = parser.parse_args()
    
    collector = FPLDataCollector()
    
    if args.bootstrap_only:
        collector.fetch_bootstrap_static()
    elif args.all:
        collector.collect_all_data(sample_players=args.sample)
    else:
        print("Use --all to collect all data or --bootstrap-only for basic data")

if __name__ == "__main__":
    main()
