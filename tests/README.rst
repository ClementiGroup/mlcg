Tests for ``mlcg`` Package
==========================

.. start-intro

The ``mlcg`` package includes a structured test suite to ensure code quality and correctness. Tests are organized by **type**:

- **Unit tests**
  Located in ``tests/unit/``.
  These tests are fast and test individual functions or small components in isolation.

- **Integration tests**
  Located in ``tests/integration/``.
  These tests check the interaction between multiple components and may involve longer-running pipelines or CLI commands.

- **Continuity tests**
  Located in ``tests/continuity/``.
  These tests check loading of old checkpoints to ensure continuity of the code.

----

Running Tests
-------------

Run all tests::

   pytest

Run only integration tests::

   pytest -m integration

Run only unit tests::

   pytest -m unit

Run only continuity tests::

   pytest -m continuity


Adding New Tests
----------------

When contributing new tests:

- Place unit tests in ``tests/unit/``.
- Place integration tests in ``tests/integration/``.
- Place continuity tests in ``tests/continuity/``.
- Markers for integration and unit are automatically added to each test, so it is not necessary to explicitly add them.

This ensures tests are correctly categorized and can be run selectively.

Tests in CI/CD pipeline
-----------------------

Code within the ``mlcg`` package is tested using ``CircleCI`` for continuous integration.
The file ``.test_durations`` is used to split tests across different containers according
to their execution time. When a new, consistent set of tests is added, it is recommended
to update ``.test_durations`` by installing ``pytest-split`` via pip and running::

   pytest --store-durations

This will create a new ``.test_durations`` file automatically.

.. end-intro