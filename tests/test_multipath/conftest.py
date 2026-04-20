"""
Shared fixtures for multipath calculation tests.
"""
import sys
import os
import pytest
import numpy as np

# Setup project paths
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_path, 'src'))
