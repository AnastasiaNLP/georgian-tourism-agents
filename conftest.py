"""Root pytest configuration.

The default test run is already scoped to ``tests/`` via ``testpaths`` in
``pyproject.toml``. This guard additionally keeps local scratch/diagnostic scripts
in the repo root (which may open network connections on import) from breaking
collection when pytest is pointed at the root explicitly — they are never tests.
"""

collect_ignore_glob = [
    "fast_test.py",
    "diag_qdrant.py",
    "run_test_query.py",
    "*_scratch.py",
]
