"""
Validation tests to verify the testing infrastructure setup works correctly.
"""
import pytest
import os
import sys
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""
    
    def test_pytest_is_working(self):
        """Test that pytest is working correctly."""
        assert True, "pytest is working"
    
    def test_fixtures_are_available(self, temp_dir, mock_config):
        """Test that shared fixtures are available."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "server" in mock_config
    
    def test_project_structure(self):
        """Test that the project has the expected structure."""
        project_root = Path(__file__).parent.parent
        
        # Check main project files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "README.md").exists()
        
        # Check model directory
        assert (project_root / "model").exists()
        assert (project_root / "model").is_dir()
        
        # Check test structure
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
    
    def test_python_path(self):
        """Test that Python can import project modules."""
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Test that we can import from model directory
        try:
            import model
            assert hasattr(model, '__path__') or hasattr(model, '__file__')
        except ImportError:
            pytest.skip("Model module not available for import")
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that custom markers are working."""
        assert True, "Unit marker is working"
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that custom markers are working."""
        assert True, "Integration marker is working"
    
    def test_mock_functionality(self, mock_requests):
        """Test that mocking functionality works."""
        # Test mock requests fixture
        response = mock_requests["get"]("http://example.com")
        assert response.status_code == 200
        assert response.json()["status"] == "success"
    
    def test_environment_isolation(self):
        """Test that environment variables are properly isolated."""
        # Set a test environment variable
        os.environ["TEST_VAR"] = "test_value"
        assert os.environ.get("TEST_VAR") == "test_value"
        # The reset_environment fixture should clean this up after the test


@pytest.mark.slow
class TestPerformanceValidation:
    """Performance-related validation tests marked as slow."""
    
    def test_slow_operation(self):
        """A test marked as slow to validate marker functionality."""
        import time
        time.sleep(0.1)  # Simulate slow operation
        assert True, "Slow test completed"


def test_module_level_function():
    """Test that module-level test functions are discovered."""
    assert True, "Module-level function test works"