"""Entry-point regressions run in fresh interpreters outside the repository."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]

STARTUP_CHECK = textwrap.dedent(
    """
    import runpy
    import sys
    import multiprocessing

    from PySide6.QtWidgets import QApplication

    mode, entrypoint = sys.argv[1:]
    sys.argv = [entrypoint]
    checked = []
    startup = []
    multiprocessing.freeze_support = lambda: startup.append("freeze_support")

    def check_workspace(application):
        assert startup == ["freeze_support"]
        from workbench.app import GroupedToolPage, MainWindow, ToolPage

        windows = [widget for widget in application.topLevelWidgets()
                   if isinstance(widget, MainWindow)]
        assert len(windows) == 1
        window = windows[0]
        top_pages = window.project.top_pages()
        assert any(isinstance(page, GroupedToolPage) for page in top_pages)
        assert any(isinstance(page, ToolPage) for page in top_pages)
        leaves = [leaf for page in top_pages
                  for leaf in (page.pages if isinstance(page, GroupedToolPage)
                               else [page])]
        assert window.project.leaf_pages() == leaves
        assert all(isinstance(page, ToolPage) for page in leaves)
        state = window.project.workspace_state()
        assert set(state["pages"]) == {page.tool.id for page in top_pages}
        assert state["selected_tool"] == top_pages[0].tool.id
        assert window.close()
        checked.append(True)
        return 23

    QApplication.exec = check_workspace
    try:
        if mode == "script":
            runpy.run_path(entrypoint, run_name="__main__")
        else:
            runpy.run_module("workbench", run_name="__main__", alter_sys=True)
    except SystemExit as outcome:
        assert outcome.code == 23, outcome.code
    else:
        raise AssertionError("The entry point did not propagate its exit code")
    assert checked == [True]
    """
)

SPAWN_CHECK = textwrap.dedent(
    """
    import runpy
    import sys

    from PySide6.QtWidgets import QApplication

    def unexpected_event_loop(*args):
        raise AssertionError("A spawn import started the GUI")

    QApplication.exec = unexpected_event_loop
    assert QApplication.instance() is None
    runpy.run_path(sys.argv[1], run_name="__mp_main__")
    assert QApplication.instance() is None
    """
)


def _run_isolated(code, arguments, tmp_path, *, package_on_path=False):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    if package_on_path:
        environment["PYTHONPATH"] = str(REPOSITORY)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-c", code, *arguments],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["script", "package"])
def test_entrypoint_builds_a_canonical_workspace_outside_repository(tmp_path, mode):
    _run_isolated(
        STARTUP_CHECK,
        [mode, str(REPOSITORY / "workbench" / "app.py")],
        tmp_path,
        package_on_path=mode == "package",
    )


@pytest.mark.parametrize("entrypoint", ["app.py", "__main__.py"])
def test_spawn_import_does_not_start_a_gui(tmp_path, entrypoint):
    _run_isolated(
        SPAWN_CHECK,
        [str(REPOSITORY / "workbench" / entrypoint)],
        tmp_path,
    )
