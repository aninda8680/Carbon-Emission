"""Smoke tests for the 12-page Streamlit dashboard in carbon_automl.dashboard.app."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from carbon_automl.dashboard import app as dashboard_app

APP_PATH = Path(__file__).resolve().parent.parent / "carbon_automl" / "dashboard" / "app.py"


def test_dashboard_imports_and_defines_12_pages() -> None:
    """Dashboard module imports cleanly and defines all 12 navigation pages."""
    assert len(dashboard_app.PAGES) == 12
    assert dashboard_app.PAGES[0] == "Overview"
    assert "Experiment History" in dashboard_app.PAGES


def test_dashboard_apptest_smoke() -> None:
    """Streamlit AppTest renders Overview and key pages without uncaught exceptions."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()
    assert not at.exception
    assert len(at.sidebar.radio) >= 1

    # Switch through empty-safe navigation pages to verify clean rendering
    for target_page in ("Model Registry", "Experiment History", "Carbon Budget"):
        at.sidebar.radio[0].set_value(target_page).run()
        assert not at.exception
