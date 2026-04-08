"""Profil memoire simplifie Etape 8."""

from __future__ import annotations


def test_memory_percent_accessible() -> None:
    """La metrique memoire est recuperable sans crash."""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        assert True
        return

    ram_pct = float(psutil.virtual_memory().percent)
    assert 0.0 <= ram_pct <= 100.0
