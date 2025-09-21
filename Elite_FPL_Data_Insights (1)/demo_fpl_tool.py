#!/usr/bin/env python3
"""
FPL Tool Demo Script
Demonstrates the complete Phase 2 functionality
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the fpl_tool to path
sys.path.insert(0, '/home/ubuntu')

from fpl_tool.optimizer import FPLOptimizer
from fpl_tool.recommender import FPLRecommender
from fpl_tool.validator import FPLValidator

def main():
    """Run complete FPL Tool demonstration"""
    
    print("🚀 FPL Tool Phase 2 - Complete Demonstration")
    print("=" * 60)
    
    # Check if data files exist
    data_files = [
        "/home/ubuntu/data/fpl_xpts_predictions_enhanced.csv",
        "/home/ubuntu/data/fpl_master_2025-26.csv"
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("Please ensure Phase 1 data collection is complete.")
        return
    
    print("✅ All required data files found")
    
    # 1. Test Optimizer
    print("\n1️⃣ Testing FPL Optimizer...")
    try:
        optimizer = FPLOptimizer()
        result = optimizer.optimize_squad(budget=100.0)
        
        if result.is_valid:
            print(f"✅ Optimization successful!")
            print(f"   Formation: {result.formation}")
            print(f"   Total cost: £{result.total_cost:.1f}m")
            print(f"   Expected points: {result.expected_points:.1f}")
            print(f"   Captain: {result.captain.get('name', 'N/A')}")
        else:
            print(f"❌ Optimization failed: {result.validation_errors}")
    except Exception as e:
        print(f"❌ Optimizer error: {e}")
    
    # 2. Test Validator
    print("\n2️⃣ Testing FPL Validator...")
    try:
        validator = FPLValidator()
        if result.is_valid:
            is_valid, errors = validator.validate_complete_team(
                result.squad, result.starting_xi, result.bench,
                result.captain, result.vice_captain
            )
            if is_valid:
                print("✅ Squad validation passed!")
            else:
                print(f"❌ Validation errors: {errors}")
        else:
            print("⏭️ Skipping validation (no valid squad)")
    except Exception as e:
        print(f"❌ Validator error: {e}")
    
    # 3. Test Recommender
    print("\n3️⃣ Testing FPL Recommender...")
    try:
        recommender = FPLRecommender()
        
        # Generate watchlists
        watchlists = recommender.generate_watchlists()
        print(f"✅ Generated watchlists:")
        for pos, players in watchlists.items():
            print(f"   {pos}: {len(players)} players")
        
        # Generate top 50
        top_50 = recommender.generate_top_50_overall()
        print(f"✅ Generated top 50 overall rankings")
        print(f"   #1 Player: {top_50[0]['name']} ({top_50[0]['position']}) - {top_50[0]['expected_points_ensemble']:.1f} pts")
        
        # Generate differentials
        differentials = recommender.generate_differentials()
        print(f"✅ Generated {len(differentials)} differential options")
        
    except Exception as e:
        print(f"❌ Recommender error: {e}")
    
    # 4. Test Complete Recommendations
    print("\n4️⃣ Testing Complete Recommendations...")
    try:
        recommendations = recommender.generate_complete_recommendations(gameweek=1)
        
        print("✅ Complete recommendations generated:")
        print(f"   Optimal squads: {len(recommendations['optimal_squads'])} strategies")
        print(f"   Top 50 players: {len(recommendations['top_50_overall'])} players")
        print(f"   Watchlists: {len(recommendations['watchlists'])} positions")
        print(f"   Differentials: {len(recommendations['differentials'])} players")
        
        # Show balanced squad summary
        balanced = recommendations['optimal_squads']['balanced']
        if balanced.is_valid:
            print(f"\n📊 Balanced Strategy Summary:")
            print(f"   Formation: {balanced.formation}")
            print(f"   Cost: £{balanced.total_cost:.1f}m")
            print(f"   Expected Points: {balanced.expected_points:.1f}")
            print(f"   Captain: {balanced.captain.get('name', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Complete recommendations error: {e}")
    
    # 5. Export Test
    print("\n5️⃣ Testing CSV Export...")
    try:
        output_dir = "/home/ubuntu/output"
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = recommender.export_recommendations_csv(recommendations, output_dir)
        print(f"✅ Exported {len(exported_files)} CSV files:")
        for file_type, file_path in exported_files.items():
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            print(f"   {file_type}: {Path(file_path).name} ({file_size} bytes)")
        
    except Exception as e:
        print(f"❌ Export error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 FPL Tool Phase 2 demonstration complete!")
    print("\nNext steps:")
    print("• Use CLI: python -m fpl_tool.cli --help")
    print("• Run Streamlit: streamlit run fpl_tool/app_streamlit.py")
    print("• Run tests: python -m pytest tests/")

if __name__ == "__main__":
    main()
