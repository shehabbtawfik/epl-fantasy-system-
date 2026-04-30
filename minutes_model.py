#!/usr/bin/env python3
"""
FPL Minutes Prediction Model
Sophisticated minutes prediction using historical patterns, rotation, and fixture congestion
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class FPLMinutesPredictor:
    def __init__(self):
        self.data_dir = "/home/ubuntu/data"
        self.raw_data_dir = "/home/ubuntu/data/raw"
        self.models_dir = "/home/ubuntu/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.rf_model = None
        self.linear_model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load master dataset and fixture features"""
        try:
            # Load master dataset
            master_path = os.path.join(self.data_dir, "fpl_master_2025-26.csv")
            df = pd.read_csv(master_path)
            
            # Load fixture features
            fixture_path = os.path.join(self.data_dir, "fixture_features.csv")
            fixture_df = pd.read_csv(fixture_path)
            
            print(f"Loaded master dataset: {df.shape}")
            print(f"Loaded fixture features: {fixture_df.shape}")
            
            return df, fixture_df
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def load_historical_data(self):
        """Load historical player data for training"""
        try:
            # Check if full historical data is available
            full_history_path = os.path.join(self.raw_data_dir, "player_histories_full.json")
            if os.path.exists(full_history_path):
                with open(full_history_path, 'r') as f:
                    player_histories = json.load(f)
                print(f"Loaded full historical data for {len(player_histories)} players")
            else:
                # Fall back to partial data
                history_path = os.path.join(self.raw_data_dir, "player_histories.json")
                with open(history_path, 'r') as f:
                    player_histories = json.load(f)
                print(f"Loaded partial historical data for {len(player_histories)} players")
            
            return player_histories
            
        except FileNotFoundError as e:
            print(f"Error loading historical data: {e}")
            return None
    
    def create_training_dataset(self, df, fixture_df, player_histories):
        """Create training dataset from historical match data"""
        print("Creating training dataset from historical matches...")
        
        training_data = []
        
        for player_id, history_data in player_histories.items():
            if 'history' not in history_data:
                continue
                
            player_matches = history_data['history']
            if len(player_matches) == 0:
                continue
            
            # Get player info from master dataset
            player_info = df[df['player_id'] == int(player_id)]
            if len(player_info) == 0:
                continue
            
            player_info = player_info.iloc[0]
            
            # Process each match
            for i, match in enumerate(player_matches):
                # Skip if no minutes data
                if 'minutes' not in match:
                    continue
                
                # Calculate rolling averages (last 5 games)
                recent_matches = player_matches[max(0, i-5):i]
                if len(recent_matches) > 0:
                    avg_minutes_last_5 = np.mean([m.get('minutes', 0) for m in recent_matches])
                    starts_last_5 = sum([1 for m in recent_matches if m.get('minutes', 0) >= 60])
                    sub_appearances_last_5 = sum([1 for m in recent_matches if 0 < m.get('minutes', 0) < 60])
                else:
                    avg_minutes_last_5 = 0
                    starts_last_5 = 0
                    sub_appearances_last_5 = 0
                
                # Calculate season averages up to this point
                season_matches = player_matches[:i+1]
                season_minutes = [m.get('minutes', 0) for m in season_matches]
                season_avg_minutes = np.mean(season_minutes) if season_minutes else 0
                season_starts = sum([1 for m in season_matches if m.get('minutes', 0) >= 60])
                season_games = len(season_matches)
                start_percentage = season_starts / season_games if season_games > 0 else 0
                
                # Get fixture info (simplified - using gameweek)
                gw = match.get('round', 1)
                team_id = player_info['team_id']
                
                # Get fixture features for this team/gameweek
                fixture_info = fixture_df[
                    (fixture_df['team_id'] == team_id) & 
                    (fixture_df['gameweek'] == gw)
                ]
                
                if len(fixture_info) > 0:
                    fixture_info = fixture_info.iloc[0]
                    is_dgw = fixture_info['is_dgw']
                    rotation_risk = fixture_info['rotation_risk']
                    congestion_index = fixture_info['congestion_index']
                    is_home = fixture_info['is_home']
                else:
                    is_dgw = False
                    rotation_risk = 0.1
                    congestion_index = 1.0
                    is_home = True
                
                # Create training row
                training_row = {
                    'player_id': int(player_id),
                    'gameweek': gw,
                    'position': player_info['position'],
                    'team_id': team_id,
                    
                    # Target variable
                    'actual_minutes': match.get('minutes', 0),
                    
                    # Historical features
                    'avg_minutes_last_5': avg_minutes_last_5,
                    'starts_last_5': starts_last_5,
                    'sub_appearances_last_5': sub_appearances_last_5,
                    'season_avg_minutes': season_avg_minutes,
                    'start_percentage': start_percentage,
                    'games_played': season_games,
                    
                    # Player characteristics
                    'current_price': player_info['current_price'],
                    'total_points_season': player_info['total_points'],
                    'form': player_info['form'],
                    'selected_by_percent': player_info['selected_by_percent'],
                    
                    # Fixture features
                    'is_dgw': is_dgw,
                    'rotation_risk': rotation_risk,
                    'congestion_index': congestion_index,
                    'is_home': is_home if is_home is not None else True,
                    
                    # Match context
                    'was_home': match.get('was_home', True),
                    'opponent_team': match.get('opponent_team', 1),
                    
                    # Performance in match
                    'total_points': match.get('total_points', 0),
                    'goals_scored': match.get('goals_scored', 0),
                    'assists': match.get('assists', 0),
                }
                
                training_data.append(training_row)
        
        training_df = pd.DataFrame(training_data)
        print(f"Created training dataset: {training_df.shape}")
        
        return training_df
    
    def engineer_features(self, df):
        """Engineer additional features for minutes prediction"""
        df = df.copy()
        
        # Position encoding
        position_encoding = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_encoded'] = df['position'].map(position_encoding).fillna(3)
        
        # Price tier (higher price = more likely to start)
        df['price_tier'] = pd.cut(df['current_price'], bins=5, labels=[1,2,3,4,5]).astype(float)
        
        # Form tier
        df['form_tier'] = pd.cut(df['form'], bins=5, labels=[1,2,3,4,5]).astype(float)
        
        # Interaction features
        df['price_form_interaction'] = df['current_price'] * df['form']
        df['start_pct_price_interaction'] = df['start_percentage'] * df['current_price']
        
        # Rotation penalty
        df['rotation_penalty'] = df['rotation_risk'] * df['congestion_index']
        
        # Home advantage
        df['home_advantage'] = df['is_home'].fillna(True).astype(int) * 0.1
        
        return df
    
    def train_models(self, training_df):
        """Train Random Forest and Linear models"""
        print("Training minutes prediction models...")
        
        # Engineer features
        training_df = self.engineer_features(training_df)
        
        # Select features
        feature_columns = [
            'avg_minutes_last_5', 'starts_last_5', 'sub_appearances_last_5',
            'season_avg_minutes', 'start_percentage', 'games_played',
            'current_price', 'form', 'selected_by_percent',
            'position_encoded', 'price_tier', 'form_tier',
            'is_dgw', 'rotation_risk', 'congestion_index', 'is_home',
            'price_form_interaction', 'start_pct_price_interaction',
            'rotation_penalty', 'home_advantage'
        ]
        
        # Prepare data
        X = training_df[feature_columns].fillna(0)
        y = training_df['actual_minutes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Linear Model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred_train = self.rf_model.predict(X_train)
        rf_pred_test = self.rf_model.predict(X_test)
        
        linear_pred_train = self.linear_model.predict(X_train_scaled)
        linear_pred_test = self.linear_model.predict(X_test_scaled)
        
        # Calculate metrics
        rf_mae_train = mean_absolute_error(y_train, rf_pred_train)
        rf_mae_test = mean_absolute_error(y_test, rf_pred_test)
        
        linear_mae_train = mean_absolute_error(y_train, linear_pred_train)
        linear_mae_test = mean_absolute_error(y_test, linear_pred_test)
        
        print(f"\nModel Performance:")
        print(f"Random Forest - Train MAE: {rf_mae_train:.2f}, Test MAE: {rf_mae_test:.2f}")
        print(f"Linear Model - Train MAE: {linear_mae_train:.2f}, Test MAE: {linear_mae_test:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save models
        joblib.dump(self.rf_model, os.path.join(self.models_dir, "minutes_rf_model.pkl"))
        joblib.dump(self.linear_model, os.path.join(self.models_dir, "minutes_linear_model.pkl"))
        joblib.dump(self.scaler, os.path.join(self.models_dir, "minutes_scaler.pkl"))
        
        return feature_columns, feature_importance
    
    def predict_minutes(self, df, fixture_df, feature_columns):
        """Predict minutes for current players"""
        print("Predicting minutes for current season...")
        
        # Merge with fixture data for next gameweek
        current_gw = 1  # This should be dynamic based on current gameweek
        
        # Get next gameweek fixtures for each team
        next_gw_fixtures = fixture_df[fixture_df['gameweek'] == current_gw + 1]
        
        # Merge player data with fixture data
        prediction_df = df.merge(
            next_gw_fixtures[['team_id', 'is_dgw', 'rotation_risk', 'congestion_index', 'is_home']],
            on='team_id',
            how='left'
        )
        
        # Fill missing fixture data
        prediction_df['is_dgw'] = prediction_df['is_dgw'].fillna(False)
        prediction_df['rotation_risk'] = prediction_df['rotation_risk'].fillna(0.1)
        prediction_df['congestion_index'] = prediction_df['congestion_index'].fillna(1.0)
        prediction_df['is_home'] = prediction_df['is_home'].fillna(True)
        
        # Add historical features (simplified for current prediction)
        prediction_df['avg_minutes_last_5'] = prediction_df['minutes_per_game'].fillna(0)
        prediction_df['starts_last_5'] = np.where(prediction_df['minutes'] >= 300, 5, 2)  # Estimate
        prediction_df['sub_appearances_last_5'] = np.where(prediction_df['minutes'] < 300, 3, 1)
        prediction_df['season_avg_minutes'] = prediction_df['minutes_per_game'].fillna(0)
        prediction_df['start_percentage'] = np.where(prediction_df['minutes'] > 0, 
                                                   prediction_df['minutes'] / (prediction_df['minutes'] + 270), 0.5)
        prediction_df['games_played'] = np.where(prediction_df['minutes'] > 0, 
                                                prediction_df['minutes'] / prediction_df['minutes_per_game'].fillna(90), 1)
        
        # Engineer features
        prediction_df = self.engineer_features(prediction_df)
        
        # Prepare features
        X_pred = prediction_df[feature_columns].fillna(0)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        rf_predictions = self.rf_model.predict(X_pred)
        linear_predictions = self.linear_model.predict(X_pred_scaled)
        
        # Ensemble prediction (weighted average)
        ensemble_predictions = 0.7 * rf_predictions + 0.3 * linear_predictions
        
        # Apply constraints
        ensemble_predictions = np.clip(ensemble_predictions, 0, 90)
        
        # Adjust for player status
        status_multipliers = {
            'a': 1.0,  # Available
            'd': 0.75, # Doubtful  
            'i': 0.0,  # Injured
            's': 0.0,  # Suspended
            'u': 0.5   # Unavailable
        }
        
        for status, multiplier in status_multipliers.items():
            mask = prediction_df['status'] == status
            ensemble_predictions[mask] *= multiplier
        
        # Adjust for chance of playing
        chance_mask = prediction_df['chance_of_playing_this_round'].notna()
        ensemble_predictions[chance_mask] *= (prediction_df.loc[chance_mask, 'chance_of_playing_this_round'] / 100)
        
        # Create results dataframe
        results_df = prediction_df[['player_id', 'name', 'position', 'team_name', 'status']].copy()
        results_df['predicted_minutes_rf'] = rf_predictions
        results_df['predicted_minutes_linear'] = linear_predictions
        results_df['predicted_minutes_ensemble'] = ensemble_predictions
        results_df['rotation_risk'] = prediction_df['rotation_risk']
        results_df['is_dgw'] = prediction_df['is_dgw']
        results_df['congestion_index'] = prediction_df['congestion_index']
        
        return results_df
    
    def run_full_pipeline(self):
        """Run the complete minutes prediction pipeline"""
        print("Starting FPL Minutes Prediction Pipeline...")
        
        # Load data
        df, fixture_df = self.load_data()
        if df is None or fixture_df is None:
            print("Failed to load data")
            return None
        
        # Load historical data
        player_histories = self.load_historical_data()
        if player_histories is None:
            print("Failed to load historical data")
            return None
        
        # Create training dataset
        training_df = self.create_training_dataset(df, fixture_df, player_histories)
        if len(training_df) == 0:
            print("Failed to create training dataset")
            return None
        
        # Train models
        feature_columns, feature_importance = self.train_models(training_df)
        
        # Make predictions
        predictions_df = self.predict_minutes(df, fixture_df, feature_columns)
        
        # Save results
        output_path = os.path.join(self.data_dir, "minutes_predictions.csv")
        predictions_df.to_csv(output_path, index=False)
        
        # Save feature importance
        importance_path = os.path.join(self.data_dir, "minutes_feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        
        print(f"\nMinutes predictions saved: {output_path}")
        print(f"Feature importance saved: {importance_path}")
        
        # Summary statistics
        print(f"\nPrediction Summary:")
        print(f"- Players analyzed: {len(predictions_df)}")
        print(f"- Average predicted minutes: {predictions_df['predicted_minutes_ensemble'].mean():.1f}")
        print(f"- Players likely to start (>60 min): {sum(predictions_df['predicted_minutes_ensemble'] > 60)}")
        print(f"- Players at rotation risk (>20%): {sum(predictions_df['rotation_risk'] > 0.2)}")
        
        return predictions_df

def main():
    predictor = FPLMinutesPredictor()
    predictions = predictor.run_full_pipeline()
    
    if predictions is not None:
        print("\n=== MINUTES PREDICTION COMPLETE ===")
    else:
        print("Minutes prediction failed")

if __name__ == "__main__":
    main()
