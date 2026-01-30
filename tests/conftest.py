import pytest
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """
    This utility add markers to distinguish between integration, unit, and continuity tests automatically.
    It checks the file paths of the collected tests and assigns the appropriate marker based on their location.
    This allows for easier test selection and organization based on the type of test.

    Run `pytest -m integration` to run only integration tests
    Run `pytest -m unit` to run only unit tests
    Run `pytest -m continuity` to run only continuity tests
    """
    for item in items:
        # Convert nodeid to a path-safe string
        path = Path(str(item.fspath))

        if "tests/integration" in path.as_posix():
            item.add_marker(pytest.mark.integration)
        elif "tests/unit" in path.as_posix():
            item.add_marker(pytest.mark.unit)
        elif "tests/continuity" in path.as_posix():
            item.add_marker(pytest.mark.continuity)
