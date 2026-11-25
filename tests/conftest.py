"""Shared pytest fixtures.

Unit tests use small in-memory fixtures. Integration tests (marked
``integration``) read the real processed data and ChromaDB in READ-ONLY
mode; they are skipped automatically if the data is not present.
"""

import pytest

from sephra import config


def _real_data_available() -> bool:
    return (config.PRODUCTS_CSV.exists()
            and config.REVIEW_DOCUMENTS_JSONL.exists()
            and (config.CHROMA_DIR / "chroma.sqlite3").exists())


requires_data = pytest.mark.skipif(not _real_data_available(),
                                   reason="real data / ChromaDB not available")


@pytest.fixture(scope="session")
def catalog():
    if not _real_data_available():
        pytest.skip("real data not available")
    from sephra.catalog import get_catalog
    return get_catalog()
