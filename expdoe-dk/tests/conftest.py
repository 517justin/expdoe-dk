"""Shared pytest fixtures."""
import warnings

import pytest


@pytest.fixture(autouse=True)
def _silence_torch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    yield


def pytest_collection_modifyitems(config, items):
    # Default: skip @pytest.mark.slow tests unless --run-slow is given.
    if config.getoption("--run-slow", default=False):
        return
    skip_slow = pytest.mark.skip(reason="use --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run @pytest.mark.slow integration tests",
    )
