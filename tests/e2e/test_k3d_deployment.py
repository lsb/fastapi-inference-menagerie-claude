"""End-to-end tests for k3d deployment."""

import pytest
import subprocess
import time
import requests
from pathlib import Path

from cli.utils import run_command, check_k8s_connection


@pytest.mark.e2e
@pytest.mark.slow
class TestK3DDeployment:
    """Test end-to-end deployment on k3d cluster."""
    
    @classmethod
    def setup_class(cls):
        """Set up k3d cluster for testing."""
        try:
            # Check if k3d is available
            subprocess.run(["k3d", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("k3d not available")
        
        # Create test cluster
        cluster_name = "test-model-zoo"
        cls.cluster_name = cluster_name
        
        try:
            # Delete cluster if it exists
            subprocess.run(
                ["k3d", "cluster", "delete", cluster_name], 
                capture_output=True, 
                check=False
            )
            
            # Create new cluster
            run_command([
                "k3d", "cluster", "create", cluster_name,
                "--agents", "1",
                "--port", "8081:80@loadbalancer",
                "--registry-create", "test-registry:5000"
            ])
            
            # Wait for cluster to be ready
            time.sleep(30)
            
            # Verify connection
            if not check_k8s_connection():
                pytest.skip("Cannot connect to k3d cluster")
                
        except Exception as e:
            pytest.skip(f"Failed to create k3d cluster: {e}")
    
    @classmethod
    def teardown_class(cls):
        """Clean up k3d cluster."""
        try:
            subprocess.run(
                ["k3d", "cluster", "delete", cls.cluster_name],
                capture_output=True,
                check=False
            )
        except Exception:
            pass
    
    def test_cluster_connectivity(self):
        """Test that k3d cluster is accessible."""
        assert check_k8s_connection()
        
        # Check nodes
        result = run_command(
            ["kubectl", "get", "nodes", "-o", "name"],
            capture_output=True
        )
        assert "node/" in result.stdout
    
    def test_namespace_creation(self):
        """Test creating model-zoo namespace."""
        # Create namespace
        run_command([
            "kubectl", "create", "namespace", "test-zoo"
        ], check=False)  # May already exist
        
        # Verify namespace exists
        result = run_command([
            "kubectl", "get", "namespace", "test-zoo"
        ], capture_output=True)
        
        assert "test-zoo" in result.stdout
    
    def test_deploy_test_service(self):
        """Test deploying a simple test service."""
        # Deploy nginx as test service
        nginx_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-nginx
  namespace: test-zoo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test-nginx
  template:
    metadata:
      labels:
        app: test-nginx
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: test-nginx
  namespace: test-zoo
spec:
  selector:
    app: test-nginx
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
"""
        
        # Apply manifest
        process = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=nginx_yaml,
            text=True,
            capture_output=True
        )
        assert process.returncode == 0
        
        # Wait for deployment to be ready
        run_command([
            "kubectl", "wait", "--for=condition=available",
            "deployment/test-nginx", "-n", "test-zoo",
            "--timeout=120s"
        ])
        
        # Verify pod is running
        result = run_command([
            "kubectl", "get", "pods", "-n", "test-zoo",
            "-l", "app=test-nginx", "-o", "jsonpath={.items[0].status.phase}"
        ], capture_output=True)
        
        assert "Running" in result.stdout
    
    @pytest.mark.skipif(
        not Path("./zoo").exists(),
        reason="CLI not available"
    )
    def test_cli_deployment_flow(self):
        """Test full CLI deployment flow."""
        # This test would require the full CLI to be built
        # and would test: init -> build -> deploy -> logs -> delete
        pytest.skip("Full CLI integration test - requires built package")


@pytest.mark.e2e
class TestServiceEndpoints:
    """Test service endpoints in isolation."""
    
    @pytest.fixture(autouse=True)
    def setup_test_service(self):
        """Set up test service for endpoint testing."""
        # This would typically start a test server
        # For now, we'll use a mock
        self.base_url = "http://localhost:8000"
    
    def test_health_endpoint_availability(self):
        """Test that health endpoint is accessible."""
        # This test would make actual HTTP requests
        # when service is running
        pytest.skip("Requires running service")
    
    def test_metrics_endpoint_format(self):
        """Test that metrics endpoint returns proper format."""
        pytest.skip("Requires running service")
    
    def test_api_documentation_access(self):
        """Test that OpenAPI docs are accessible."""
        pytest.skip("Requires running service")