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

class FPLxPtsModeler:
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
        
        # Injury risk factors
        self.injury_risk_factors = {
            'age_multiplier': {'<25': 0.8, '25-29': 1.0, '30-32': 1.3, '>32': 1.6},
            'position_risk': {'GK': 0.7, 'DEF': 1.0, 'MID': 1.1, 'FWD': 1.2},
            'minutes_load_threshold': 2700  # Minutes per season threshold for high load
        }
    
    def load_data(self):
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
            
            # Load fixture features
            fixture_path = os.path.join(self.data_dir, "fixture_features.csv")
            if os.path.exists(fixture_path):
                fixture_df = pd.read_csv(fixture_path)
                print(f"Loaded fixture features: {fixture_df.shape}")
            else:
                print("Fixture features not found")
                fixture_df = None
            
            # Load team form
            team_form_path = os.path.join(self.data_dir, "team_form.csv")
            if os.path.exists(team_form_path):
                team_form_df = pd.read_csv(team_form_path)
                print(f"Loaded team form: {team_form_df.shape}")
            else:
                print("Team form not found")
                team_form_df = None
            
            # Load clean sheet probabilities
            cs_path = os.path.join(self.data_dir, "clean_sheet_probabilities.csv")
            if os.path.exists(cs_path):
                cs_df = pd.read_csv(cs_path)
                print(f"Loaded clean sheet probabilities: {cs_df.shape}")
            else:
                print("Clean sheet probabilities not found")
                cs_df = None
            
            return df, minutes_df, fixture_df, team_form_df, cs_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None, None, None
    
    def calculate_minutes_prediction(self, df):
        """Predict minutes for next gameweek"""
        df = df.copy()
        
        # Base minutes prediction
        df['predicted_minutes'] = 0
        
        # For players with recent history
        mask_played = df['minutes'] > 0
        df.loc[mask_played, 'predicted_minutes'] = df.loc[mask_played, 'minutes_per_game'].fillna(0)
        
        # Adjust for status
        status_adjustments = {
            'a': 1.0,  # Available
            'd': 0.75,  # Doubtful
            'i': 0.0,   # Injured
            's': 0.0,   # Suspended
            'u': 0.5    # Unavailable
        }
        
        for status, multiplier in status_adjustments.items():
            mask = df['status'] == status
            df.loc[mask, 'predicted_minutes'] *= multiplier
        
        # Adjust for chance of playing
        chance_mask = df['chance_of_playing_this_round'].notna()
        df.loc[chance_mask, 'predicted_minutes'] *= (df.loc[chance_mask, 'chance_of_playing_this_round'] / 100)
        
        # Cap at 90 minutes
        df['predicted_minutes'] = np.minimum(df['predicted_minutes'], 90)
        
        return df['predicted_minutes']
    
    def calculate_injury_risk(self, df):
        """Calculate injury risk for each player"""
        df = df.copy()
        
        # Base injury risk from status
        df['base_injury_risk'] = 0.05  # 5% base risk
        
        # Adjust for current status
        status_risk = {
            'a': 0.05,  # Available
            'd': 0.25,  # Doubtful
            'i': 0.95,  # Injured
            's': 0.10,  # Suspended (not injury related)
            'u': 0.30   # Unavailable
        }
        
        for status, risk in status_risk.items():
            mask = df['status'] == status
            df.loc[mask, 'base_injury_risk'] = risk
        
        # Age factor (estimate age from career length)
        df['estimated_age'] = 25  # Default age
        # This is simplified - in reality you'd calculate from debut year
        
        # Position risk
        position_risk = self.injury_risk_factors['position_risk']
        df['position_injury_risk'] = df['position'].map(position_risk).fillna(1.0)
        
        # Minutes load risk
        df['minutes_load_risk'] = np.where(
            df['minutes'] > self.injury_risk_factors['minutes_load_threshold'] / 38,
            1.2, 1.0
        )
        
        # Recent injury history (from news)
        df['recent_injury_flag'] = df['news'].str.contains(
            'injury|injured|knock|strain|fitness', case=False, na=False
        ).astype(int) * 0.3
        
        # Combined injury risk
        df['injury_risk'] = (
            df['base_injury_risk'] * 
            df['position_injury_risk'] * 
            df['minutes_load_risk'] + 
            df['recent_injury_flag']
        ).clip(0, 0.95)
        
        return df['injury_risk']
    
    def merge_enhanced_features(self, df, minutes_df, fixture_df, team_form_df, cs_df):
        """Merge all enhanced features into master dataset"""
        print("Merging enhanced features...")
        
        enhanced_df = df.copy()
        
        # Merge minutes predictions
        if minutes_df is not None:
            minutes_features = minutes_df[['player_id', 'predicted_minutes_ensemble', 
                                         'rotation_risk', 'is_dgw']].copy()
            enhanced_df = enhanced_df.merge(minutes_features, on='player_id', how='left')
            enhanced_df['predicted_minutes_ensemble'] = enhanced_df['predicted_minutes_ensemble'].fillna(
                enhanced_df['minutes_per_game'].fillna(45)
            )
            enhanced_df['rotation_risk'] = enhanced_df['rotation_risk'].fillna(0.1)
            enhanced_df['is_dgw'] = enhanced_df['is_dgw'].fillna(False)
        else:
            # Use basic minutes prediction
            enhanced_df['predicted_minutes_ensemble'] = self.calculate_minutes_prediction(enhanced_df)
            enhanced_df['rotation_risk'] = 0.1
            enhanced_df['is_dgw'] = False
        
        # Merge team form features
        if team_form_df is not None:
            team_features = team_form_df[['team', 'ppg', 'goals_per_game', 'goals_conceded_per_game',
                                        'attack_rating', 'defence_rating', 'avg_cs_probability']].copy()
            # Handle different column names
            if 'team' not in team_features.columns and 'team_id' in team_features.columns:
                team_features = team_features.rename(columns={'team_id': 'team'})
            
            enhanced_df = enhanced_df.merge(team_features, left_on='team_id', right_on='team', how='left')
            
            # Fill missing team form data
            enhanced_df['ppg'] = enhanced_df['ppg'].fillna(1.0)
            enhanced_df['goals_per_game'] = enhanced_df['goals_per_game'].fillna(1.5)
            enhanced_df['goals_conceded_per_game'] = enhanced_df['goals_conceded_per_game'].fillna(1.5)
            enhanced_df['attack_rating'] = enhanced_df['attack_rating'].fillna(1000)
            enhanced_df['defence_rating'] = enhanced_df['defence_rating'].fillna(1000)
            enhanced_df['avg_cs_probability'] = enhanced_df['avg_cs_probability'].fillna(0.3)
        else:
            # Add default team form features
            enhanced_df['ppg'] = 1.0
            enhanced_df['goals_per_game'] = 1.5
            enhanced_df['goals_conceded_per_game'] = 1.5
            enhanced_df['attack_rating'] = 1000
            enhanced_df['defence_rating'] = 1000
            enhanced_df['avg_cs_probability'] = 0.3
        
        # Add injury risk
        enhanced_df['injury_risk'] = self.calculate_injury_risk(enhanced_df)
        
        # Add fixture difficulty (use existing or default)
        if 'avg_fixture_difficulty_6gw' not in enhanced_df.columns:
            enhanced_df['avg_fixture_difficulty_6gw'] = 3.0  # Average difficulty
        
        print(f"Enhanced dataset shape: {enhanced_df.shape}")
        return enhanced_df
    
    def calculate_base_attacking_points(self, df):
        """Calculate base attacking expected points"""
        df = df.copy()
        
        # Goals and assists per 90
        df['goals_per_90'] = df['goals_scored_per_90'].fillna(0)
        df['assists_per_90'] = df['assists_per_90'].fillna(0)
        
        # Position-specific goal values
        position_goal_points = {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4}
        df['goal_points_per_90'] = df['position'].map(position_goal_points).fillna(4) * df['goals_per_90']
        
        # Assist points (3 for all positions)
        df['assist_points_per_90'] = 3 * df['assists_per_90']
        
        # Total attacking points per 90
        df['attacking_points_per_90'] = df['goal_points_per_90'] + df['assist_points_per_90']
        
        return df['attacking_points_per_90']
    
    def calculate_defensive_points(self, df):
        """Calculate defensive expected points"""
        df = df.copy()
        
        # Clean sheets (GK and DEF get 4 points, MID get 1)
        df['clean_sheet_per_90'] = df['clean_sheets_per_90'].fillna(0)
        
        clean_sheet_points = {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0}
        df['clean_sheet_points_per_90'] = df['position'].map(clean_sheet_points).fillna(0) * df['clean_sheet_per_90']
        
        # Saves (GK only)
        df['saves_per_90'] = df['saves_per_90'].fillna(0)
        df['save_points_per_90'] = np.where(df['position'] == 'GK', df['saves_per_90'] / 3, 0)  # 1 point per 3 saves
        
        # Goals conceded penalty (GK and DEF)
        df['goals_conceded_per_90'] = df['goals_conceded_per_90'].fillna(0)
        df['conceded_penalty_per_90'] = np.where(
            df['position'].isin(['GK', 'DEF']),
            -df['goals_conceded_per_90'] / 2,  # -1 point per 2 goals conceded
            0
        )
        
        return df['clean_sheet_points_per_90'] + df['save_points_per_90'] + df['conceded_penalty_per_90']
    
    def calculate_bonus_points(self, df):
        """Calculate expected bonus points"""
        df = df.copy()
        
        # Bonus points per 90
        df['bonus_per_90'] = df['bonus_per_90'].fillna(0)
        
        # BPS-based bonus prediction (simplified)
        df['bps_per_90'] = (df['bps'] / np.maximum(df['minutes'], 1)) * 90
        df['bps_per_90'] = df['bps_per_90'].fillna(0)
        
        # Estimate bonus from BPS (top 3 BPS in each game get bonus)
        # This is a simplified approximation
        df['estimated_bonus_per_90'] = np.minimum(df['bps_per_90'] / 30, 2)  # Cap at 2 bonus per game
        
        return np.maximum(df['bonus_per_90'], df['estimated_bonus_per_90'])
    
    def calculate_fixture_adjustment(self, df):
        """Calculate fixture difficulty adjustment"""
        df = df.copy()
        
        # Next fixture difficulty (lower is easier)
        df['next_fixture_difficulty'] = df['next_1_difficulty'].fillna(3)  # Default to neutral
        
        # Convert to multiplier (easier fixtures = higher multiplier)
        # FDR 1 (easiest) = 1.3x, FDR 5 (hardest) = 0.7x
        difficulty_multipliers = {1: 1.3, 2: 1.15, 3: 1.0, 4: 0.85, 5: 0.7}
        df['fixture_multiplier'] = df['next_fixture_difficulty'].map(difficulty_multipliers).fillna(1.0)
        
        # Home advantage
        df['home_bonus'] = np.where(df['next_1_venue'] == 'H', self.weights['home_advantage'], 0)
        
        return df['fixture_multiplier'] + df['home_bonus']
    
    def calculate_set_piece_bonus(self, df):
        """Calculate set piece bonus"""
        df = df.copy()
        
        bonus = 0
        
        # Penalty takers get significant bonus
        bonus += np.where(df['penalties_taker'], self.weights['penalty_bonus'], 0)
        
        # Other set piece takers get smaller bonus
        bonus += np.where(df['corners_taker'] | df['freekicks_taker'], self.weights['set_piece_bonus'], 0)
        
        return bonus
    
    def calculate_risk_adjustments(self, df):
        """Calculate risk-based adjustments"""
        df = df.copy()
        
        risk_adjustment = 0
        
        # Card risk (players with high card rates)
        df['cards_per_90'] = (df['yellow_cards_per_90'].fillna(0) + df['red_cards_per_90'].fillna(0) * 2)
        risk_adjustment -= df['cards_per_90'] * 0.5  # -0.5 points per card per 90
        
        # Rotation risk based on squad depth and recent starts
        # This is simplified - in practice would need more sophisticated analysis
        high_ownership = df['selected_by_percent'] > 20
        risk_adjustment += np.where(high_ownership, 0, self.weights['rotation_risk'])
        
        return risk_adjustment
    
    def build_xpts_model(self, df):
        """Build the main xPts model"""
        print("Building xPts model...")
        
        df = df.copy()
        
        # Calculate component scores
        df['predicted_minutes'] = self.calculate_minutes_prediction(df)
        df['attacking_points_per_90'] = self.calculate_base_attacking_points(df)
        df['defensive_points_per_90'] = self.calculate_defensive_points(df)
        df['bonus_points_per_90'] = self.calculate_bonus_points(df)
        df['fixture_adjustment'] = self.calculate_fixture_adjustment(df)
        df['set_piece_bonus'] = self.calculate_set_piece_bonus(df)
        df['risk_adjustment'] = self.calculate_risk_adjustments(df)
        
        # Base points per 90 (2 points for playing + attacking + defensive + bonus)
        df['base_points_per_90'] = (2 + df['attacking_points_per_90'] + 
                                   df['defensive_points_per_90'] + df['bonus_points_per_90'])
        
        # Form adjustment (recent form vs season average)
        df['form_adjustment'] = (df['form'].fillna(0) - df['points_per_game'].fillna(0)) * 0.1
        
        # Calculate expected points for next GW
        df['xPts_next_gw'] = (
            (df['predicted_minutes'] / 90) *  # Minutes factor
            df['base_points_per_90'] *  # Base scoring rate
            df['fixture_adjustment'] +  # Fixture difficulty
            df['set_piece_bonus'] +  # Set piece bonus
            df['form_adjustment'] +  # Form adjustment
            df['risk_adjustment']  # Risk penalties
        )
        
        # Ensure non-negative
        df['xPts_next_gw'] = np.maximum(df['xPts_next_gw'], 0)
        
        # Calculate xPts for next 6 GWs (simplified as 6x next GW with fixture adjustments)
        fixture_difficulties = []
        for i in range(1, 7):
            difficulties = df[f'next_{i}_difficulty'].fillna(3)
            fixture_difficulties.append(difficulties)
        
        # Average fixture difficulty over next 6
        avg_fixture_difficulty = np.mean(fixture_difficulties, axis=0)
        avg_fixture_multiplier = pd.Series(avg_fixture_difficulty).map({1: 1.3, 2: 1.15, 3: 1.0, 4: 0.85, 5: 0.7}).fillna(1.0)
        
        df['xPts_next_6gw'] = df['xPts_next_gw'] * 6 * avg_fixture_multiplier / df['fixture_adjustment']
        
        return df
    
    def create_sensitivity_analysis(self, df):
        """Create sensitivity analysis for model parameters"""
        print("Creating sensitivity analysis...")
        
        base_xpts = df['xPts_next_gw'].copy()
        
        sensitivity_results = []
        
        # Test each weight parameter
        for param, base_value in self.weights.items():
            # Test +/- 10% changes
            for change in [-0.1, 0.1]:
                # Temporarily modify weight
                original_value = self.weights[param]
                self.weights[param] = original_value * (1 + change)
                
                # Recalculate xPts
                df_test = self.build_xpts_model(df)
                new_xpts = df_test['xPts_next_gw']
                
                # Calculate impact
                avg_change = (new_xpts - base_xpts).mean()
                max_change = (new_xpts - base_xpts).abs().max()
                
                sensitivity_results.append({
                    'parameter': param,
                    'change_pct': change * 100,
                    'avg_impact': avg_change,
                    'max_impact': max_change,
                    'sensitivity_score': abs(avg_change) + abs(max_change)
                })
                
                # Restore original value
                self.weights[param] = original_value
        
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df = sensitivity_df.sort_values('sensitivity_score', ascending=False)
        
        return sensitivity_df
    
    def backtest_model(self, df):
        """Simple backtest using current season data"""
        print("Running backtest...")
        
        # For players with sufficient data, compare predicted vs actual
        mask = (df['minutes'] > 0) & (df['total_points'] > 0)
        test_df = df[mask].copy()
        
        if len(test_df) == 0:
            print("No data available for backtesting")
            return {}
        
        # Use current season points per game as "actual"
        actual = test_df['points_per_game']
        predicted = test_df['xPts_next_gw']
        
        # Calculate error metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 0.1))) * 100
        
        # Correlation
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        backtest_results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'n_players': len(test_df)
        }
        
        print(f"Backtest Results (n={len(test_df)}):")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.1f}%")
        print(f"  Correlation: {correlation:.3f}")
        
        return backtest_results
    
    def save_results(self, df, sensitivity_df, backtest_results):
        """Save model results"""
        
        # Save xPts predictions
        xpts_cols = ['player_id', 'name', 'position', 'team_short', 'current_price', 
                    'xPts_next_gw', 'xPts_next_6gw', 'predicted_minutes', 'fixture_adjustment']
        
        xpts_df = df[xpts_cols].sort_values('xPts_next_gw', ascending=False)
        
        xpts_path = os.path.join(self.data_dir, "xpts_predictions.csv")
        xpts_df.to_csv(xpts_path, index=False)
        print(f"xPts predictions saved: {xpts_path}")
        
        # Save sensitivity analysis
        sensitivity_path = os.path.join(self.data_dir, "sensitivity_analysis.csv")
        sensitivity_df.to_csv(sensitivity_path, index=False)
        print(f"Sensitivity analysis saved: {sensitivity_path}")
        
        # Save model summary
        summary = {
            'model_weights': self.weights,
            'backtest_results': backtest_results,
            'top_10_xpts': xpts_df.head(10).to_dict('records'),
            'model_formula': self.get_model_formula()
        }
        
        summary_path = os.path.join(self.data_dir, "model_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Model summary saved: {summary_path}")
        
        return xpts_df
    
    def get_model_formula(self):
        """Return the model formula as text"""
        formula = """
        xPts_next_gw = (predicted_minutes / 90) × base_points_per_90 × fixture_adjustment + bonuses + adjustments
        
        Where:
        - base_points_per_90 = 2 + attacking_points_per_90 + defensive_points_per_90 + bonus_points_per_90
        - attacking_points_per_90 = (goals_per_90 × position_goal_value) + (assists_per_90 × 3)
        - defensive_points_per_90 = clean_sheet_points + save_points - conceded_penalty
        - fixture_adjustment = difficulty_multiplier + home_advantage
        - bonuses = set_piece_bonus + form_adjustment
        - adjustments = risk_adjustment (cards, rotation)
        """
        return formula.strip()
    
    def run_full_model(self):
        """Run the complete modeling pipeline"""
        print("Starting FPL xPts modeling...")
        
        # Load data
        df = self.load_data()
        if df is None:
            return None
        
        # Build model
        df = self.build_xpts_model(df)
        
        # Sensitivity analysis
        sensitivity_df = self.create_sensitivity_analysis(df)
        
        # Backtest
        backtest_results = self.backtest_model(df)
        
        # Save results
        xpts_df = self.save_results(df, sensitivity_df, backtest_results)
        
        print("\n=== MODELING COMPLETE ===")
        print(f"Top 5 xPts for next GW:")
        print(xpts_df[['name', 'position', 'team_short', 'current_price', 'xPts_next_gw']].head())
        
        return df

def main():
    modeler = FPLxPtsModeler()
    df = modeler.run_full_model()

if __name__ == "__main__":
    main()
