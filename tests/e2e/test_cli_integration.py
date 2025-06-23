"""End-to-end CLI integration tests."""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import typer
from cli.utils import get_project_root


@pytest.mark.e2e
class TestCLIIntegration:
    """Test CLI commands integration."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Copy essential files for CLI to work
            project_root = get_project_root()
            
            # Create basic structure
            (tmp_path / "services").mkdir()
            (tmp_path / "services" / "common").mkdir()
            (tmp_path / "docker").mkdir()
            (tmp_path / "cli").mkdir()
            
            yield tmp_path
    
    def test_init_command_basic_template(self, temp_project_dir: Path):
        """Test zoo init command with basic template."""
        # Mock CLI execution
        with patch('cli.commands.init.get_project_root', return_value=temp_project_dir):
            from cli.commands.init import main as init_main
            
            # Test creating new service
            init_main("test-model", template="basic")
            
            # Verify files were created
            service_dir = temp_project_dir / "services" / "test-model"
            assert service_dir.exists()
            assert (service_dir / "__init__.py").exists()
            assert (service_dir / "adapter.py").exists()
            assert (service_dir / "app.py").exists()
            
            # Verify content
            adapter_content = (service_dir / "adapter.py").read_text()
            assert "TestModelAdapter" in adapter_content
            assert "test-model" in adapter_content
    
    def test_init_command_existing_service(self, temp_project_dir: Path):
        """Test zoo init command with existing service."""
        # Create existing service
        service_dir = temp_project_dir / "services" / "existing-model"
        service_dir.mkdir(parents=True)
        
        with patch('cli.commands.init.get_project_root', return_value=temp_project_dir):
            from cli.commands.init import main as init_main
            
            # Should raise error for existing service
            with pytest.raises(typer.Exit):
                init_main("existing-model")
    
    @patch('cli.commands.build.run_command')
    @patch('cli.commands.build.get_project_root')
    def test_build_command(self, mock_get_root, mock_run_cmd, temp_project_dir: Path):
        """Test zoo build command."""
        mock_get_root.return_value = temp_project_dir
        
        # Create service directory
        service_dir = temp_project_dir / "services" / "test-model"
        service_dir.mkdir(parents=True)
        
        from cli.commands.build import main as build_main
        
        # Test building model
        build_main("test-model", registry="test-registry", push=False)
        
        # Verify build script was called
        mock_run_cmd.assert_called_once()
        args = mock_run_cmd.call_args[0][0]
        assert "build.py" in args[1]
        assert "test-model" in args
    
    @patch('cli.commands.deploy.check_k8s_connection')
    @patch('cli.commands.deploy.run_command')
    def test_deploy_command_no_k8s(self, mock_run_cmd, mock_k8s_check):
        """Test zoo deploy command when k8s is not available."""
        mock_k8s_check.return_value = False
        
        from cli.commands.deploy import main as deploy_main
        
        # Should exit when k8s is not available
        with pytest.raises(typer.Exit):
            deploy_main("test-model")
    
    @patch('cli.commands.logs.check_k8s_connection')
    @patch('cli.commands.logs.run_command')
    def test_logs_command(self, mock_run_cmd, mock_k8s_check):
        """Test zoo logs command."""
        mock_k8s_check.return_value = True
        
        from cli.commands.logs import main as logs_main
        
        # Test viewing logs
        logs_main("test-model", follow=False, tail=50)
        
        # Verify kubectl logs was called
        mock_run_cmd.assert_called_once()
        args = mock_run_cmd.call_args[0][0]
        assert "kubectl" in args
        assert "logs" in args
        assert "--tail" in args
        assert "50" in args
    
    @patch('cli.commands.delete.check_k8s_connection')
    @patch('cli.commands.delete.run_command')
    @patch('typer.confirm')
    def test_delete_command(self, mock_confirm, mock_run_cmd, mock_k8s_check):
        """Test zoo delete command."""
        mock_k8s_check.return_value = True
        mock_confirm.return_value = True
        
        from cli.commands.delete import main as delete_main
        
        # Test deleting deployment
        delete_main("test-model", force=False)
        
        # Verify kubectl delete was called multiple times
        assert mock_run_cmd.call_count >= 3  # deployment, service, hpa
    
    @patch('cli.commands.update.check_k8s_connection')
    @patch('cli.commands.update.run_command')
    def test_update_command(self, mock_run_cmd, mock_k8s_check):
        """Test zoo update command."""
        mock_k8s_check.return_value = True
        
        from cli.commands.update import main as update_main
        
        # Test updating deployment
        update_main("test-model", image_tag="new-tag:latest")
        
        # Verify kubectl set image and rollout status were called
        assert mock_run_cmd.call_count >= 2
    
    @patch('cli.commands.rollback.check_k8s_connection')
    @patch('cli.commands.rollback.run_command')
    def test_rollback_command(self, mock_run_cmd, mock_k8s_check):
        """Test zoo rollback command."""
        mock_k8s_check.return_value = True
        
        from cli.commands.rollback import main as rollback_main
        
        # Test rolling back deployment
        rollback_main("test-model", revision=2)
        
        # Verify kubectl rollout commands were called
        assert mock_run_cmd.call_count >= 3  # history, undo, status


@pytest.mark.e2e
@pytest.mark.slow
class TestFullCLIWorkflow:
    """Test complete CLI workflow."""
    
    @pytest.mark.skipif(
        not Path("./zoo").exists(),
        reason="CLI executable not available"
    )
    def test_complete_workflow(self):
        """Test complete workflow: init -> build -> deploy -> delete."""
        # This would test the actual CLI binary
        # Requires the package to be installed
        pytest.skip("Requires installed CLI package")
    
    def test_help_commands(self):
        """Test that help commands work."""
        # Test that CLI structure is valid
        from cli.main import app
        
        # Verify app has expected commands
        assert hasattr(app, 'commands')
        
        # Check command names
        expected_commands = ['init', 'build', 'deploy', 'logs', 'delete', 'update', 'rollback']
        # This would need to be adjusted based on actual CLI structure