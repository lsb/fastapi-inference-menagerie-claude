"""Unit tests for CLI utilities."""

import pytest
from unittest.mock import patch, MagicMock
import subprocess
from pathlib import Path

from cli.utils import (
    validate_model_name, get_git_sha, check_dependencies,
    run_command, get_user_prefix, check_k8s_connection
)


@pytest.mark.unit
class TestCLIUtils:
    """Test CLI utility functions."""
    
    def test_validate_model_name_valid(self):
        """Test validating valid model names."""
        valid_names = [
            "clip",
            "grounding-sam",
            "qwen_vl",
            "model123",
            "my-model-v2"
        ]
        
        for name in valid_names:
            assert validate_model_name(name) is True
    
    def test_validate_model_name_invalid(self):
        """Test validating invalid model names."""
        invalid_names = [
            "",
            "model with spaces",
            "model@special",
            "model/slash",
            "model.dot"
        ]
        
        for name in invalid_names:
            assert validate_model_name(name) is False
    
    @patch('subprocess.run')
    def test_get_git_sha_success(self, mock_run):
        """Test getting git SHA successfully."""
        mock_result = MagicMock()
        mock_result.stdout = "abc1234\n"
        mock_run.return_value = mock_result
        
        sha = get_git_sha()
        
        assert sha == "abc1234"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_get_git_sha_failure(self, mock_run):
        """Test getting git SHA when git command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        
        sha = get_git_sha()
        
        assert sha == "unknown"
    
    @patch('shutil.which')
    def test_check_dependencies(self, mock_which):
        """Test checking CLI dependencies."""
        # Mock some dependencies as available, others not
        def mock_which_func(cmd):
            available = {"docker", "kubectl"}
            return "/usr/bin/" + cmd if cmd in available else None
        
        mock_which.side_effect = mock_which_func
        
        deps = check_dependencies()
        
        assert deps["docker"] is True
        assert deps["kubectl"] is True
        assert deps["k3d"] is False
        assert deps["git"] is False
    
    @patch('os.getenv')
    def test_get_user_prefix(self, mock_getenv):
        """Test getting user prefix."""
        mock_getenv.return_value = "testuser"
        
        prefix = get_user_prefix()
        
        assert prefix == "testuser"
        mock_getenv.assert_called_once_with("USER", "default")
    
    @patch('os.getenv')
    def test_get_user_prefix_default(self, mock_getenv):
        """Test getting user prefix with default."""
        mock_getenv.return_value = None
        
        prefix = get_user_prefix()
        
        assert prefix == "default"
    
    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test running command successfully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = run_command(["echo", "test"])
        
        assert result == mock_result
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """Test running command that fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        with pytest.raises(subprocess.CalledProcessError):
            run_command(["false"])
    
    @patch('subprocess.run')
    def test_run_command_no_check(self, mock_run):
        """Test running command without checking return code."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        # Should not raise exception when check=False
        result = run_command(["false"], check=False)
        
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_check_k8s_connection_success(self, mock_run):
        """Test successful Kubernetes connection check."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = check_k8s_connection()
        
        assert result is True
        mock_run.assert_called_once_with(
            ["kubectl", "get", "nodes"],
            capture_output=True,
            check=True
        )
    
    @patch('subprocess.run')
    def test_check_k8s_connection_failure(self, mock_run):
        """Test failed Kubernetes connection check."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "kubectl")
        
        result = check_k8s_connection()
        
        assert result is False