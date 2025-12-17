import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runner_idx",
        default=0,
        type=int,
        help="Index of the current runner, i.e. container (relevant for test parallelization)",
    )

    parser.addoption(
        "--num_containers",
        default=1,
        type=int,
        help="Number of container (relevant for test parallelization)",
    )

    parser.addoption(
        "--light",
        action="store_true",
        default=False,
        help="Run only light tests (skip tests marked as heavy)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: mark test as heavy")


def pytest_collection_modifyitems(config, items):
    """
    This utility distinguishes between tests marked with the decorator `@pytest.mark.heavy` and standard tests.
    By default, running pytest executes all tests, including heavy ones.
    When using the `--light` mode (i.e., `pytest --light`), only non-heavy tests are collected and executed.

    This allows users to quickly run lightweight tests when making small changes,
    while still supporting a full test suite for more thorough validation.
    """
    if not config.getoption("--light"):
        # default: do nothing, run all tests
        return

    # Skip heavy tests in --light mode
    skip_heavy = pytest.mark.skip(reason="Skipped in light mode (--light)")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
