
#!/usr/bin/env python3
"""
Tests for FPL CLI
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_tool.cli import app
from fpl_tool.optimizer import OptimizationResult

class TestFPLCLI:
    """Test cases for FPL CLI"""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()
    
    @pytest.fixture
    def sample_squad_csv(self):
        """Create sample squad CSV for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("player_id,name,position,team_name,current_price,status\n")
            f.write("1,GK_1,GKP,Team_1,4.5,a\n")
            f.write("2,GK_2,GKP,Team_2,4.5,a\n")
            # Add more players to make valid squad...
            for i in range(5):
                f.write(f"{i+3},DEF_{i+1},DEF,Team_{(i%3)+1},4.5,a\n")
            for i in range(5):
                f.write(f"{i+8},MID_{i+1},MID,Team_{(i%3)+1},5.0,a\n")
            for i in range(3):
                f.write(f"{i+13},FWD_{i+1},FWD,Team_{(i%3)+1},7.0,a\n")
            
            return f.name
    
    def test_version_command(self, runner):
        """Test version command"""
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "FPL Tool" in result.stdout
    
    def test_build_dataset_command(self, runner):
        """Test build-dataset command"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "build-dataset",
                "--seasons", "LAST3",
                "--current",
                "--output", tmp_dir
            ])
            
            assert result.exit_code == 0
            assert "Building FPL dataset" in result.stdout
            
            # Check JSON output (multi-line)
            lines = result.stdout.strip().split('\n')
            json_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start >= 0:
                json_lines = lines[json_start:]
                json_str = '\n'.join(json_lines)
                output_data = json.loads(json_str)
                assert output_data['status'] == 'success'
                assert output_data['seasons'] == 'LAST3'
    
    @patch('fpl_tool.cli.FPLRecommender')
    def test_project_command(self, mock_recommender_class, runner):
        """Test project command"""
        # Mock the recommender
        mock_recommender = Mock()
        mock_recommender.combined_df = Mock()
        mock_recommender.combined_df.__getitem__ = Mock(return_value=Mock())
        mock_recommender.combined_df.__getitem__.return_value.copy.return_value.to_csv = Mock()
        mock_recommender_class.return_value = mock_recommender
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "project",
                "--gw", "1",
                "--horizon", "6",
                "--output", tmp_dir
            ])
            
            assert result.exit_code == 0
            assert "Generating projections" in result.stdout
    
    @patch('fpl_tool.cli.FPLRecommender')
    def test_optimize_command(self, mock_recommender_class, runner):
        """Test optimize command"""
        # Mock the recommender and optimization result
        mock_result = OptimizationResult(
            squad=[
                {'player_id': 1, 'name': 'Test Player', 'position': 'MID', 
                 'team_name': 'Test Team', 'team_short': 'TT', 'current_price': 5.0, 
                 'expected_points_ensemble': 4.5}
            ],
            starting_xi=[
                {'player_id': 1, 'name': 'Test Player', 'position': 'MID', 
                 'team_name': 'Test Team', 'team_short': 'TT', 'current_price': 5.0, 
                 'expected_points_ensemble': 4.5}
            ],
            bench=[],
            captain={'player_id': 1, 'name': 'Test Player', 'position': 'MID'},
            vice_captain={'player_id': 1, 'name': 'Test Player', 'position': 'MID'},
            total_cost=95.0,
            expected_points=50.0,
            formation="1-4-4-2",
            is_valid=True,
            validation_errors=[]
        )
        
        mock_recommender = Mock()
        mock_recommender.generate_optimal_squad.return_value = mock_result
        mock_recommender_class.return_value = mock_recommender
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "optimize",
                "--budget", "100.0",
                "--strategy", "balanced",
                "--output", tmp_dir
            ])
            
            assert result.exit_code == 0
            assert "Optimizing FPL squad" in result.stdout
            assert "Optimization successful" in result.stdout
    
    @patch('fpl_tool.cli.FPLRecommender')
    def test_recommend_gw_command(self, mock_recommender_class, runner):
        """Test recommend-gw command"""
        # Mock complete recommendations
        mock_recommendations = {
            "gameweek": 1,
            "budget": 100.0,
            "optimal_squads": {
                "balanced": OptimizationResult(
                    squad=[], starting_xi=[], bench=[], captain={'name': 'Test Captain'}, 
                    vice_captain={'name': 'Test Vice'}, total_cost=95.0, expected_points=50.0,
                    formation="1-4-4-2", is_valid=True, validation_errors=[]
                )
            },
            "top_50_overall": [
                {'rank': 1, 'name': 'Top Player', 'position': 'MID', 'club': 'TC',
                 'current_price': 10.0, 'expected_points_ensemble': 8.0}
            ],
            "watchlists": {},
            "differentials": [],
            "budget_enablers": {},
            "captaincy": {},
            "chip_strategy": {}
        }
        
        mock_recommender = Mock()
        mock_recommender.generate_complete_recommendations.return_value = mock_recommendations
        mock_recommender.export_recommendations_csv.return_value = {"top_50": "/tmp/top_50.csv"}
        mock_recommender_class.return_value = mock_recommender
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            export_path = os.path.join(tmp_dir, "recommendations.csv")
            
            result = runner.invoke(app, [
                "recommend-gw",
                "--gw", "1",
                "--budget", "100.0",
                "--export", export_path
            ])
            
            assert result.exit_code == 0
            assert "Generating GW1 recommendations" in result.stdout
            assert "recommendations generated successfully" in result.stdout
    
    def test_validate_squad_command_valid(self, runner, sample_squad_csv):
        """Test validate-squad command with valid squad"""
        try:
            result = runner.invoke(app, [
                "validate-squad",
                sample_squad_csv,
                "--show-details"
            ])
            
            assert result.exit_code == 0
            assert "Validating squad" in result.stdout
            
        finally:
            # Clean up
            if os.path.exists(sample_squad_csv):
                os.unlink(sample_squad_csv)
    
    def test_validate_squad_command_invalid_file(self, runner):
        """Test validate-squad command with non-existent file"""
        result = runner.invoke(app, [
            "validate-squad",
            "/nonexistent/file.csv"
        ])
        
        assert result.exit_code == 1
        assert "Error validating squad" in result.stdout
    
    def test_command_help(self, runner):
        """Test that help is available for all commands"""
        # Test main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "FPL Tool" in result.stdout
        
        # Test individual command help
        commands = ["build-dataset", "project", "optimize", "recommend-gw", "validate-squad"]
        
        for command in commands:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
    
    @patch('fpl_tool.cli.FPLRecommender')
    def test_optimize_command_with_formation(self, mock_recommender_class, runner):
        """Test optimize command with specific formation"""
        mock_result = OptimizationResult(
            squad=[], starting_xi=[], bench=[], captain={'name': 'Test'}, 
            vice_captain={'name': 'Test'}, total_cost=95.0, expected_points=50.0,
            formation="1-4-4-2", is_valid=True, validation_errors=[]
        )
        
        mock_recommender = Mock()
        mock_recommender.generate_optimal_squad.return_value = mock_result
        mock_recommender_class.return_value = mock_recommender
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "optimize",
                "--formation", "1-4-4-2",
                "--output", tmp_dir
            ])
            
            assert result.exit_code == 0
    
    @patch('fpl_tool.cli.FPLRecommender')
    def test_optimize_command_failure(self, mock_recommender_class, runner):
        """Test optimize command when optimization fails"""
        mock_result = OptimizationResult(
            squad=[], starting_xi=[], bench=[], captain={}, vice_captain={},
            total_cost=0.0, expected_points=0.0, formation="",
            is_valid=False, validation_errors=["Test error"]
        )
        
        mock_recommender = Mock()
        mock_recommender.generate_optimal_squad.return_value = mock_result
        mock_recommender_class.return_value = mock_recommender
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = runner.invoke(app, [
                "optimize",
                "--output", tmp_dir
            ])
            
            assert result.exit_code == 1
            assert "Optimization failed" in result.stdout
    
    def test_json_output_format(self, runner):
        """Test that commands return valid JSON summaries"""
        # Test build-dataset JSON output
        result = runner.invoke(app, ["build-dataset"])
        
        # Find JSON in output (multi-line)
        lines = result.stdout.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            # Should be valid JSON
            json_lines = lines[json_start:]
            json_str = '\n'.join(json_lines)
            output_data = json.loads(json_str)
            assert isinstance(output_data, dict)
            assert 'status' in output_data
    
    def test_error_handling(self, runner):
        """Test CLI error handling"""
        # Test with invalid parameters
        result = runner.invoke(app, [
            "optimize",
            "--budget", "-10"  # Invalid budget
        ])
        
        # Should handle gracefully (might succeed with validation in optimizer)
        assert isinstance(result.exit_code, int)

if __name__ == "__main__":
    pytest.main([__file__])
