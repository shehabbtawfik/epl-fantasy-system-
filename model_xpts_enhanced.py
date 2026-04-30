#!/usr/bin/env python3
"""
FPL Enhanced Expected Points (xPts) Modeling Script
Builds sophisticated predictive models using ML ensemble methods
Incorporates minutes prediction, fixture analysis, team form, and injury risk
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import json
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class FPLEnhancedxPtsModeler:
    def __init__(self):
        self.data_dir = "/home/ubuntu/data"
        self.raw_data_dir = "/home/ubuntu/data/raw"
        self.models_dir = "/home/ubuntu/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        }
        self.scaler = StandardScaler()
        self.ensemble_weights = {'linear': 0.2, 'ridge': 0.2, 'rf': 0.4, 'gb': 0.2}
        
        # Position-specific scoring rates
        self.position_multipliers = {
            'GK': {'clean_sheet': 4, 'save': 0.33, 'penalty_save': 5, 'goal': 6},
            'DEF': {'clean_sheet': 4, 'goal': 6, 'assist': 3, 'card': -1},
            'MID': {'goal': 5, 'assist': 3, 'clean_sheet': 1, 'card': -1},
            'FWD': {'goal': 4, 'assist': 3, 'card': -1}
        }
    
    def load_all_data(self):
        """Load all datasets including enhanced features"""
        try:
            # Load master dataset
            master_path = os.path.join(self.data_dir, "fpl_master_2025-26.csv")
            df = pd.read_csv(master_path)
            print(f"Loaded master dataset: {df.shape}")
            
            # Load minutes predictions
            minutes_path = os.path.join(self.data_dir, "minutes_predictions.csv")
            if os.path.exists(minutes_path):
                minutes_df = pd.read_csv(minutes_path)
                print(f"Loaded minutes predictions: {minutes_df.shape}")
            else:
                print("Minutes predictions not found - will use basic prediction")
                minutes_df = None
            
            # Load team form
            team_form_path = os.path.join(self.data_dir, "team_form.csv")
            if os.path.exists(team_form_path):
                team_form_df = pd.read_csv(team_form_path)
                print(f"Loaded team form: {team_form_df.shape}")
            else:
                print("Team form not found")
                team_form_df = None
            
            return df, minutes_df, team_form_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def calculate_injury_risk(self, df):
        """Calculate injury risk for each player"""
        df = df.copy()
        
        # Base injury risk from status
        status_risk = {
            'a': 0.05,  # Available
            'd': 0.25,  # Doubtful
            'i': 0.95,  # Injured
            's': 0.10,  # Suspended
            'u': 0.30   # Unavailable
        }
        
        df['injury_risk'] = df['status'].map(status_risk).fillna(0.05)
        
        # Adjust for recent injury news
        df['recent_injury_flag'] = df['news'].str.contains(
            'injury|injured|knock|strain|fitness', case=False, na=False
        ).astype(int)
        
        df['injury_risk'] += df['recent_injury_flag'] * 0.2
        df['injury_risk'] = df['injury_risk'].clip(0, 0.95)
        
        return df['injury_risk']
    
    def merge_enhanced_features(self, df, minutes_df, team_form_df):
        """Merge all enhanced features into master dataset"""
        print("Merging enhanced features...")
        
        enhanced_df = df.copy()
        
        # Merge minutes predictions
        if minutes_df is not None:
            minutes_features = minutes_df[['player_id', 'predicted_minutes_ensemble', 
                                         'rotation_risk']].copy()
            enhanced_df = enhanced_df.merge(minutes_features, on='player_id', how='left')
            enhanced_df['predicted_minutes'] = enhanced_df['predicted_minutes_ensemble'].fillna(
                enhanced_df['minutes_per_game'].fillna(45)
            )
            enhanced_df['rotation_risk'] = enhanced_df['rotation_risk'].fillna(0.1)
        else:
            # Use basic minutes prediction
            enhanced_df['predicted_minutes'] = enhanced_df['minutes_per_game'].fillna(45)
            enhanced_df['rotation_risk'] = 0.1
        
        # Merge team form features
        if team_form_df is not None:
            # Use 'team' column from team_form_df
            team_features = team_form_df[['team', 'goals_per_game', 'goals_conceded_per_game',
                                        'attack_rating', 'defence_rating']].copy()
            enhanced_df = enhanced_df.merge(team_features, left_on='team_id', right_on='team', how='left')
            
            # Fill missing team form data
            enhanced_df['team_goals_per_game'] = enhanced_df['goals_per_game'].fillna(1.5)
            enhanced_df['team_goals_conceded_per_game'] = enhanced_df['goals_conceded_per_game'].fillna(1.5)
            enhanced_df['team_attack_rating'] = enhanced_df['attack_rating'].fillna(1000)
            enhanced_df['team_defence_rating'] = enhanced_df['defence_rating'].fillna(1000)
        else:
            # Add default team form features
            enhanced_df['team_goals_per_game'] = 1.5
            enhanced_df['team_goals_conceded_per_game'] = 1.5
            enhanced_df['team_attack_rating'] = 1000
            enhanced_df['team_defence_rating'] = 1000
        
        # Add injury risk
        enhanced_df['injury_risk'] = self.calculate_injury_risk(enhanced_df)
        
        # Clean sheet probability for defenders and goalkeepers
        enhanced_df['cs_probability'] = 0.3  # Default
        if 'avg_cs_probability' in enhanced_df.columns:
            enhanced_df['cs_probability'] = enhanced_df['avg_cs_probability'].fillna(0.3)
        
        print(f"Enhanced dataset shape: {enhanced_df.shape}")
        return enhanced_df
    
    def engineer_advanced_features(self, df):
        """Engineer advanced features for ML models"""
        df = df.copy()
        
        # Position encoding
        position_encoding = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        df['position_encoded'] = df['position'].map(position_encoding).fillna(3)
        
        # Price tiers
        df['price_tier'] = pd.cut(df['current_price'], bins=5, labels=[1,2,3,4,5]).astype(float)
        
        # Form metrics
        df['form_numeric'] = pd.to_numeric(df['form'], errors='coerce').fillna(0)
        df['form_tier'] = pd.cut(df['form_numeric'], bins=5, labels=[1,2,3,4,5]).astype(float)
        
        # Interaction features
        df['price_form_interaction'] = df['current_price'] * df['form_numeric']
        df['minutes_form_interaction'] = df['predicted_minutes'] * df['form_numeric']
        df['team_attack_player_threat'] = df['team_attack_rating'] * pd.to_numeric(df['threat'], errors='coerce').fillna(0)
        
        # Availability adjustment
        df['availability_multiplier'] = 1.0
        df.loc[df['status'] != 'a', 'availability_multiplier'] = 0.5
        
        # Adjust for chance of playing
        chance_cols = ['chance_of_playing_this_round', 'chance_of_playing_next_round']
        for col in chance_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(100)
                df['availability_multiplier'] *= (df[col] / 100)
        
        # Rotation penalty
        df['rotation_penalty'] = df['rotation_risk'] * df['predicted_minutes'] / 90
        
        return df
    
    def calculate_expected_points(self, df):
        """Calculate expected points using enhanced methodology"""
        df = df.copy()
        
        # Base points from historical performance
        df['base_points'] = pd.to_numeric(df['points_per_game'], errors='coerce').fillna(0)
        
        # Minutes adjustment
        df['minutes_multiplier'] = df['predicted_minutes'] / 90
        
        # Position-specific calculations
        df['expected_goals'] = 0
        df['expected_assists'] = 0
        df['expected_clean_sheets'] = 0
        
        # Goals and assists (per 90 stats)
        goals_per_90 = pd.to_numeric(df.get('goals_scored_per_90', 0), errors='coerce').fillna(0)
        assists_per_90 = pd.to_numeric(df.get('assists_per_90', 0), errors='coerce').fillna(0)
        
        df['expected_goals'] = goals_per_90 * df['minutes_multiplier']
        df['expected_assists'] = assists_per_90 * df['minutes_multiplier']
        
        # Clean sheets for defenders and goalkeepers
        df['expected_clean_sheets'] = 0
        def_gk_mask = df['position'].isin(['DEF', 'GK'])
        df.loc[def_gk_mask, 'expected_clean_sheets'] = df.loc[def_gk_mask, 'cs_probability']
        
        # Calculate points from each component
        df['points_from_goals'] = df['expected_goals'] * df['position'].map({
            'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4
        }).fillna(4)
        
        df['points_from_assists'] = df['expected_assists'] * 3
        df['points_from_cs'] = df['expected_clean_sheets'] * df['position'].map({
            'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0
        }).fillna(0)
        
        # Bonus points (simplified)
        df['expected_bonus'] = pd.to_numeric(df.get('bonus_per_90', 0), errors='coerce').fillna(0) * df['minutes_multiplier']
        
        # Base 2 points for playing
        df['points_from_playing'] = np.where(df['predicted_minutes'] > 0, 2, 0)
        
        # Combine all components
        df['expected_points_base'] = (
            df['points_from_playing'] +
            df['points_from_goals'] +
            df['points_from_assists'] +
            df['points_from_cs'] +
            df['expected_bonus']
        )
        
        # Apply adjustments
        df['expected_points_adjusted'] = (
            df['expected_points_base'] * 
            df['availability_multiplier'] * 
            (1 - df['rotation_penalty']) *
            (1 - df['injury_risk'])
        )
        
        return df['expected_points_adjusted']
    
    def prepare_ml_features(self, df):
        """Prepare features for ML models"""
        feature_columns = [
            # Basic stats
            'current_price', 'form_numeric', 'total_points', 'minutes',
            'goals_scored', 'assists', 'clean_sheets', 'bonus',
            
            # Per-90 stats
            'goals_scored_per_90', 'assists_per_90', 'total_points_per_90',
            
            # Predictions and risk
            'predicted_minutes', 'rotation_risk', 'injury_risk',
            
            # Team metrics
            'team_goals_per_game', 'team_goals_conceded_per_game',
            'team_attack_rating', 'team_defence_rating',
            
            # Engineered features
            'position_encoded', 'price_tier', 'form_tier',
            'price_form_interaction', 'minutes_form_interaction',
            'availability_multiplier', 'cs_probability'
        ]
        
        # Select available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        
        # Fill missing values
        X = df[available_columns].fillna(0)
        
        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, available_columns
    
    def train_ensemble_models(self, X, y):
        """Train ensemble of ML models"""
        print("Training ensemble models...")
        
        # Split data (simple train/test for now)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features for linear models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_predictions = {}
        model_scores = {}
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name in ['linear', 'ridge']:
                model.fit(X_train_scaled, y_train)
                pred_train = model.predict(X_train_scaled)
                pred_test = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, pred_train)
            test_mae = mean_absolute_error(y_test, pred_test)
            test_r2 = r2_score(y_test, pred_test)
            
            model_scores[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            print(f"{name} - Train MAE: {train_mae:.3f}, Test MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(self.models_dir, f"xpts_{name}_model.pkl"))
        
        joblib.dump(self.scaler, os.path.join(self.models_dir, "xpts_scaler.pkl"))
        
        return model_scores
    
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        # Scale features for linear models
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name in ['linear', 'ridge']:
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.ensemble_weights.items():
            ensemble_pred += weight * predictions[name]
        
        return ensemble_pred, predictions
    
    def run_complete_pipeline(self):
        """Run the complete enhanced xPts modeling pipeline"""
        print("Starting Enhanced FPL xPts Modeling Pipeline...")
        
        # Load all data
        df, minutes_df, team_form_df = self.load_all_data()
        if df is None:
            print("Failed to load data")
            return None
        
        # Merge enhanced features
        enhanced_df = self.merge_enhanced_features(df, minutes_df, team_form_df)
        
        # Engineer advanced features
        enhanced_df = self.engineer_advanced_features(enhanced_df)
        
        # Calculate expected points (baseline)
        enhanced_df['expected_points_baseline'] = self.calculate_expected_points(enhanced_df)
        
        # Prepare ML features
        X, feature_columns = self.prepare_ml_features(enhanced_df)
        y = enhanced_df['expected_points_baseline']
        
        print(f"Training data shape: {X.shape}")
        print(f"Features: {feature_columns}")
        
        # Train ensemble models
        model_scores = self.train_ensemble_models(X, y)
        
        # Make ensemble predictions
        ensemble_pred, individual_preds = self.predict_ensemble(X)
        
        # Create results dataframe
        results_df = enhanced_df[['player_id', 'name', 'position', 'team_name', 'current_price', 
                                'status', 'predicted_minutes']].copy()
        
        results_df['expected_points_baseline'] = enhanced_df['expected_points_baseline']
        results_df['expected_points_ensemble'] = ensemble_pred
        
        # Add individual model predictions
        for name, pred in individual_preds.items():
            results_df[f'xpts_{name}'] = pred
        
        # Add risk factors
        results_df['injury_risk'] = enhanced_df['injury_risk']
        results_df['rotation_risk'] = enhanced_df['rotation_risk']
        
        # Calculate value metrics
        results_df['points_per_million'] = results_df['expected_points_ensemble'] / results_df['current_price']
        results_df['value_rank'] = results_df['points_per_million'].rank(ascending=False)
        
        # Sort by expected points
        results_df = results_df.sort_values('expected_points_ensemble', ascending=False)
        
        # Save results
        output_path = os.path.join(self.data_dir, "fpl_xpts_predictions_enhanced.csv")
        results_df.to_csv(output_path, index=False)
        
        # Save model performance
        performance_path = os.path.join(self.data_dir, "model_performance_enhanced.json")
        with open(performance_path, 'w') as f:
            json.dump(model_scores, f, indent=2)
        
        print(f"\nEnhanced xPts predictions saved: {output_path}")
        print(f"Model performance saved: {performance_path}")
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"- Players analyzed: {len(results_df)}")
        print(f"- Average expected points: {results_df['expected_points_ensemble'].mean():.2f}")
        print(f"- Top predicted scorer: {results_df.iloc[0]['name']} ({results_df.iloc[0]['expected_points_ensemble']:.2f} pts)")
        print(f"- Best value: {results_df.loc[results_df['value_rank'] == 1, 'name'].iloc[0]} ({results_df.loc[results_df['value_rank'] == 1, 'points_per_million'].iloc[0]:.2f} pts/£m)")
        
        # Model performance summary
        print(f"\nModel Performance:")
        for name, scores in model_scores.items():
            print(f"- {name}: Test MAE = {scores['test_mae']:.3f}, R² = {scores['test_r2']:.3f}")
        
        return results_df

def main():
    modeler = FPLEnhancedxPtsModeler()
    predictions = modeler.run_complete_pipeline()
    
    if predictions is not None:
        print("\n=== ENHANCED XPTS MODELING COMPLETE ===")
    else:
        print("Enhanced xPts modeling failed")

if __name__ == "__main__":
    main()
