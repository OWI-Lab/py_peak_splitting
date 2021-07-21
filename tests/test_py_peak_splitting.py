import pytest
from py_peak_splitting import __version__


def test_version():
    """Test the package version"""
    assert __version__ == "0.0.2"
