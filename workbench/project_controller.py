"""Save/open lifecycle for a complete Workbench workspace."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from workbench.core.project import LoadedProject, load_project, save_project

if TYPE_CHECKING:
    from PySide6.QtWidgets import QStackedWidget
    from workbench.app import GroupedToolPage, MainWindow, ToolPage


PROJECT_FILTER = "Workbench project (*.ecwb)"


class ProjectController(QObject):
    """Manage project identity, unsaved changes and workspace replacement."""

    def __init__(self, window: MainWindow):
        super().__init__(window)
        self.window = window
        self.path: Path | None = None
        self.modified = False
        self._busy = False
        self._restoring = False
        self._loaded: LoadedProject | None = None
        self._build_menu()
        self._connect_pages()
        self._update_title()

    def _build_menu(self) -> None:
        menu = self.file_menu = self.window.menuBar().addMenu("&File")
        # Retain both wrappers to keep PySide's menu ownership stable.
        self.file_menu_action = menu.menuAction()
        self.new_action = self._action(menu, "&New", QKeySequence.StandardKey.New,
                                       self.new_project)
        self.open_action = self._action(menu, "&Open…", QKeySequence.StandardKey.Open,
                                        self.open_dialog)
        menu.addSeparator()
        self.save_action = self._action(menu, "&Save", QKeySequence.StandardKey.Save,
                                        self.save)
        self.save_as_action = self._action(menu, "Save &As…", QKeySequence.StandardKey.SaveAs,
                                           self.save_as)
        self.actions = [self.new_action, self.open_action, self.save_action, self.save_as_action]

    def _action(self, menu, text, shortcut, callback) -> QAction:
        action = QAction(text, self)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: callback())
        menu.addAction(action)
        return action

    def leaf_pages(self, stack: QStackedWidget | None = None) -> list[ToolPage]:
        from workbench.app import GroupedToolPage

        pages = []
        for page in self.top_pages(stack):
            pages.extend(page.pages if isinstance(page, GroupedToolPage) else [page])
        return pages

    def top_pages(self, stack: QStackedWidget | None = None) -> list[ToolPage | GroupedToolPage]:
        from workbench.app import GroupedToolPage, ToolPage

        stack = self.window.stack if stack is None else stack
        pages = []
        for i in range(stack.count()):
            page = stack.widget(i)
            if not isinstance(page, (ToolPage, GroupedToolPage)):
                raise ValueError("The workspace contains an unsupported page.")
            pages.append(page)
        return pages

    def _connect_pages(self) -> None:
        for page in self.leaf_pages():
            page.changed.connect(self.mark_modified)
            page.runningChanged.connect(self._update_actions)
        self._update_actions()

    def _running(self) -> bool:
        return any(page._running for page in self.leaf_pages())

    def _update_actions(self, *_args) -> None:
        enabled = not self._busy and not self._running()
        for action in self.actions:
            action.setEnabled(enabled)

    def _available(self) -> bool:
        if self._busy or self._running():
            self.window._set_status("Wait for the current calculation or project operation to finish.")
            return False
        return True

    def mark_modified(self) -> None:
        if not self._restoring and not self._busy:
            self.modified = True
            self._update_title()

    def _update_title(self) -> None:
        name = self.path.name if self.path is not None else "Untitled"
        self.window.setWindowTitle(f"{name}[*] — Workbench")
        self.window.setWindowModified(self.modified)

    def _commit_editors(self) -> None:
        """Finish focused table/numeric edits before a shortcut saves or closes."""
        focused = QApplication.focusWidget()
        if focused is not None and self.window.isAncestorOf(focused):
            focused.clearFocus()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        central = self.window.centralWidget()
        if central is not None:
            central.setEnabled(not busy)
        if busy:
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QGuiApplication.restoreOverrideCursor()
        self._update_actions()

    def workspace_state(self) -> dict:
        pages = {}
        for page in self.top_pages():
            if page.tool.id in pages:
                raise ValueError(f"Duplicate project page ID: {page.tool.id}")
            pages[page.tool.id] = page.save_state()
        return {
            "pages": pages,
            "selected_tool": self.window._tools[self.window.nav.currentRow()].id,
            "window_size": [self.window.width(), self.window.height()],
        }

    def save(self) -> bool:
        if not self._available():
            return False
        return self.save_as() if self.path is None else self.save_to(self.path)

    def save_as(self) -> bool:
        if not self._available():
            return False
        initial = str(self.path) if self.path is not None else "Untitled.ecwb"
        selected, _ = QFileDialog.getSaveFileName(
            self.window, "Save Workbench project", initial, PROJECT_FILTER)
        if not selected:
            return False
        path = Path(selected)
        if path.suffix.lower() != ".ecwb":
            path = path.with_name(path.name + ".ecwb")
        return self.save_to(path)

    def save_to(self, path: str | Path) -> bool:
        """Update project identity after saving successfully."""
        if not self._available():
            return False
        self._commit_editors()
        path = Path(path).absolute()
        self._set_busy(True)
        try:
            state = self.workspace_state()
            metadata = self._loaded.source_metadata if self._loaded is not None else None
            save_project(path, state, source_metadata=metadata)
        except Exception as exc:
            self.window._set_status(f"Project save failed: {exc}")
            QMessageBox.critical(self.window, "Could not save project", str(exc))
            return False
        finally:
            self._set_busy(False)
        self.path = path
        self.modified = False
        for page in self.leaf_pages():
            page.project_directory = path.parent
            page.export_btn.set_input_dir(str(path.parent))
        self._update_title()
        self.window._set_status(f"Project saved: {path.name}")
        return True

    def _confirm_replace(self) -> bool:
        self._commit_editors()
        if not self.modified:
            return True
        answer = QMessageBox.warning(
            self.window, "Unsaved project",
            "Save changes to the current Workbench project?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def open_dialog(self) -> bool:
        if not self._available():
            return False
        initial = str(self.path.parent) if self.path is not None else ""
        selected, _ = QFileDialog.getOpenFileName(
            self.window, "Open Workbench project", initial, PROJECT_FILTER)
        return bool(selected) and self.open_path(selected)

    def _stage_workspace(self, state: dict, project_dir: Path) -> QStackedWidget:
        if not isinstance(state, dict) or not isinstance(state.get("pages"), dict):
            raise ValueError("The project does not contain a valid workspace.")
        expected = {tool.id for tool in self.window._tools}
        if set(state["pages"]) != expected:
            raise ValueError("The project tools are incompatible with this Workbench version.")
        if state.get("selected_tool") not in expected:
            raise ValueError("The project has an invalid selected page.")
        size = state.get("window_size", [1280, 820])
        if (not isinstance(size, list) or len(size) != 2
                or any(not isinstance(n, int) or not 100 <= n <= 32768 for n in size)):
            raise ValueError("The project has an invalid window size.")
        stack = self.window.create_tool_stack()
        try:
            for page in self.top_pages(stack):
                page.restore_state(state["pages"][page.tool.id], project_dir)
        except Exception:
            stack.deleteLater()
            raise
        return stack

    def open_path(self, path: str | Path, *, prompt: bool = True) -> bool:
        if not self._available():
            return False
        if prompt and not self._confirm_replace():
            return False
        path = Path(path).absolute()
        loaded = None
        self._set_busy(True)
        self._restoring = True
        try:
            loaded = load_project(path)
            stack = self._stage_workspace(loaded.state, path.parent)
            self.window.install_tool_stack(stack)
            old_loaded = self._loaded
            self._loaded = loaded
            self.path = path
            self.modified = False
            for page in self.leaf_pages():
                page.tool.reset_runtime_state()
            self._connect_pages()
            ids = [tool.id for tool in self.window._tools]
            self.window.nav.setCurrentRow(ids.index(loaded.state["selected_tool"]))
            self.window._on_nav(self.window.nav.currentRow())
            self.window.resize(*loaded.state.get("window_size", [1280, 820]))
            if old_loaded is not None:
                old_loaded.close()
        except Exception as exc:
            if loaded is not None and loaded is not self._loaded:
                loaded.close()
            self.window._set_status(f"Project open failed: {exc}")
            QMessageBox.critical(self.window, "Could not open project", str(exc))
            return False
        finally:
            self._restoring = False
            self._set_busy(False)
        self._update_title()
        self.window._set_status(f"Project opened: {path.name}")
        return True

    def new_project(self) -> bool:
        if not self._available() or not self._confirm_replace():
            return False
        self._restoring = True
        try:
            stack = self.window.create_tool_stack()
            self.window.install_tool_stack(stack)
            if self._loaded is not None:
                self._loaded.close()
                self._loaded = None
            self.path = None
            self.modified = False
            for page in self.leaf_pages():
                page.tool.reset_runtime_state()
            self._connect_pages()
            self.window.nav.setCurrentRow(0)
            self.window._on_nav(0)
        except Exception as exc:
            QMessageBox.critical(self.window, "Could not create project", str(exc))
            return False
        finally:
            self._restoring = False
        self._update_title()
        return True

    def request_close(self) -> bool:
        if not self._available() or not self._confirm_replace():
            return False
        if self._loaded is not None:
            self._loaded.close()
            self._loaded = None
        return True
