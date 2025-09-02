"""
Shared pytest fixtures for the test suite.
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": {
            "path": "/mock/model/path",
            "device": "cpu",
            "precision": "fp32"
        },
        "server": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False
        }
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing without GPU requirements."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value=MagicMock()) as mock_device:
            yield mock_device


@pytest.fixture
def mock_model():
    """Mock RWKV model for testing."""
    mock_model = MagicMock()
    mock_model.forward.return_value = MagicMock()
    mock_model.eval = MagicMock()
    return mock_model


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "test output"
    return mock_tokenizer


@pytest.fixture
def sample_text():
    """Sample text for testing text processing functions."""
    return "This is a sample text for testing purposes."


@pytest.fixture
def mock_flask_app():
    """Mock Flask application for testing."""
    with patch('flask.Flask') as mock_app:
        app_instance = MagicMock()
        mock_app.return_value = app_instance
        yield app_instance


@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Setup default responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.text = "mock response"
        
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        
        yield {"get": mock_get, "post": mock_post, "response": mock_response}


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = dict(os.environ)
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Capture logs during test execution."""
    with caplog.at_level("DEBUG"):
        yield caplog