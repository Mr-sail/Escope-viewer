from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .models import ParsedLog, SignalNode
from .parser import ParseError, parse_log_file
from .plot_panels import BasePlotPanel, Plot2DPanel, Plot3DPanel


class PlotSubWindow(QMdiSubWindow):
    def __init__(self, panel: BasePlotPanel, close_callback: Callable[[BasePlotPanel], None]) -> None:
        super().__init__()
        self._panel = panel
        self._close_callback = close_callback
        self._managed_close = False
        self.setWidget(panel)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandMove, True)
        self.setOption(QMdiSubWindow.SubWindowOption.RubberBandResize, True)
        self.setWindowFlags(
            Qt.WindowType.SubWindow
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._managed_close:
            super().closeEvent(event)
            return
        event.ignore()
        self._close_callback(self._panel)

    def close_from_manager(self) -> None:
        self._managed_close = True
        self.close()


class MainWindow(QMainWindow):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("机器人状态曲线查看器")
        self.resize(1580, 920)

        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.signal_item_map: dict[str, QTreeWidgetItem] = {}
        self.panels: list[BasePlotPanel] = []
        self.panel_windows: dict[BasePlotPanel, PlotSubWindow] = {}
        self.active_panel: BasePlotPanel | None = None
        self._suppress_item_changed = False
        self._initial_path = initial_path

        self._build_ui()
        self.add_panel("xt")

        if self._initial_path is not None:
            self.load_file(self._initial_path)

    def _build_ui(self) -> None:
        open_action = QAction("打开文件", self)
        open_action.triggered.connect(self.open_file_dialog)
        self.menuBar().addAction(open_action)

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        controls_layout = QHBoxLayout()
        self.open_button = QPushButton("打开文件")
        self.clear_button = QPushButton("清空选择")
        self.reset_button = QPushButton("重置视图")
        self.add_xt_button = QPushButton("新增 X-T 图")
        self.add_xy_button = QPushButton("新增 X-Y 图")
        self.add_xyz_button = QPushButton("新增 XYZ 图")
        self.remove_panel_button = QPushButton("关闭当前图")
        self.zoom_x_button = QPushButton("滚轮缩放 X")
        self.zoom_y_button = QPushButton("滚轮缩放 Y")
        self.zoom_xy_button = QPushButton("滚轮缩放 XY")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索字段，例如 J1 / Tcp / ErrCode")
        for button in (self.zoom_x_button, self.zoom_y_button, self.zoom_xy_button):
            button.setCheckable(True)

        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.add_xt_button)
        controls_layout.addWidget(self.add_xy_button)
        controls_layout.addWidget(self.add_xyz_button)
        controls_layout.addWidget(self.remove_panel_button)
        controls_layout.addWidget(self.zoom_x_button)
        controls_layout.addWidget(self.zoom_y_button)
        controls_layout.addWidget(self.zoom_xy_button)
        controls_layout.addWidget(self.search_input, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("信号字段")
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_tree_item_changed)

        self.mdi_area = QMdiArea()
        self.mdi_area.setViewMode(QMdiArea.ViewMode.SubWindowView)
        self.mdi_area.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder)
        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)

        splitter.addWidget(self.tree)
        splitter.addWidget(self.mdi_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1220])

        root_layout.addLayout(controls_layout)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.info_label = QLabel("未加载文件")
        self.cursor_label = QLabel("未选择曲线")
        self.statusBar().addWidget(self.info_label, 1)
        self.statusBar().addPermanentWidget(self.cursor_label, 1)

        self.open_button.clicked.connect(self.open_file_dialog)
        self.clear_button.clicked.connect(self.clear_selection)
        self.reset_button.clicked.connect(self.reset_view)
        self.add_xt_button.clicked.connect(lambda: self.add_panel("xt"))
        self.add_xy_button.clicked.connect(lambda: self.add_panel("xy"))
        self.add_xyz_button.clicked.connect(lambda: self.add_panel("xyz"))
        self.remove_panel_button.clicked.connect(self.remove_active_panel)
        self.zoom_x_button.clicked.connect(lambda: self.set_zoom_mode("x"))
        self.zoom_y_button.clicked.connect(lambda: self.set_zoom_mode("y"))
        self.zoom_xy_button.clicked.connect(lambda: self.set_zoom_mode("xy"))
        self.search_input.textChanged.connect(self.apply_filter)
        self.set_zoom_mode("xy")

    def open_file_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "打开机器人日志",
            "",
            "Log Files (*.txt);;All Files (*)",
        )
        if filename:
            self.load_file(Path(filename))

    def load_file(self, path: Path) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            parsed = parse_log_file(path)
        except ParseError as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.parsed_log = parsed
        self.signal_lookup = {signal.signal_id: signal for signal in parsed.signals}
        self._populate_tree(parsed.signals)
        for panel in self.panels:
            panel.selected_signal_ids = [
                signal_id for signal_id in panel.selected_signal_ids if signal_id in self.signal_lookup
            ]
            panel.update_plot(self.parsed_log, self.signal_lookup)
        if self.active_panel is not None:
            self._sync_tree_from_active_panel()

        self.info_label.setText(
            "文件: "
            f"{parsed.meta.path.name} | 采样点: {parsed.meta.sample_count} | 字段: {parsed.meta.field_count} | "
            f"起始: {parsed.meta.start_time_raw} | 结束: {parsed.meta.end_time_raw} | "
            f"跳过坏行: {parsed.meta.skipped_rows}"
        )
        self.cursor_label.setText("文件已加载，可在左侧勾选信号开始绘图")
        self.setWindowTitle(f"机器人状态曲线查看器 - {parsed.meta.path.name}")
        self.reset_view()

        if parsed.meta.skipped_rows:
            QMessageBox.warning(
                self,
                "部分数据已跳过",
                f"读取过程中跳过了 {parsed.meta.skipped_rows} 行异常数据，其余有效数据仍可正常查看。",
            )

    def _populate_tree(self, signals: list[SignalNode]) -> None:
        self._suppress_item_changed = True
        self.tree.clear()
        self.signal_item_map.clear()

        created_items: dict[tuple[str, ...], QTreeWidgetItem] = {}

        for signal in signals:
            parent = self.tree.invisibleRootItem()
            current_path: list[str] = []

            for depth, part in enumerate(signal.path_parts):
                current_path.append(part)
                key = tuple(current_path)
                item = created_items.get(key)

                if item is None:
                    item = QTreeWidgetItem([part])
                    item.setToolTip(0, " / ".join(current_path))
                    created_items[key] = item
                    parent.addChild(item)

                parent = item

                if depth == len(signal.path_parts) - 1:
                    item.setData(0, Qt.ItemDataRole.UserRole, signal.signal_id)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                    item.setCheckState(0, Qt.CheckState.Unchecked)
                    item.setToolTip(0, signal.full_path)
                    self.signal_item_map[signal.signal_id] = item
                    if not signal.available:
                        item.setDisabled(True)
                        item.setText(0, f"{part} (无数据列)")

        for index in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(index).setExpanded(True)

        self._suppress_item_changed = False
        self.apply_filter(self.search_input.text())
        self._sync_tree_from_active_panel()

    def add_panel(self, mode: str) -> None:
        if mode == "xyz":
            panel: BasePlotPanel = Plot3DPanel(self.set_active_panel, self.remove_panel)
        else:
            panel = Plot2DPanel(mode, self.set_active_panel, self.remove_panel)

        subwindow = PlotSubWindow(panel, self.remove_panel)
        panel.set_status_callback(lambda text, owner=panel: self._update_panel_status(owner, text))
        if isinstance(panel, Plot2DPanel):
            panel.set_cursor_sync_callback(lambda index, owner=panel: self._sync_plot_cursors(owner, index))

        self.mdi_area.addSubWindow(subwindow)
        self.panels.append(panel)
        self.panel_windows[panel] = subwindow
        self._update_panel_titles()
        panel.update_plot(self.parsed_log, self.signal_lookup)
        self._position_new_panel(subwindow, mode)
        subwindow.show()
        self.set_active_panel(panel)

    def remove_panel(self, panel: BasePlotPanel) -> None:
        if panel not in self.panels:
            return

        was_active = self.active_panel is panel
        subwindow = self.panel_windows.pop(panel, None)
        self.panels.remove(panel)

        if subwindow is not None:
            subwindow.close_from_manager()
            self.mdi_area.removeSubWindow(subwindow)
            panel.setParent(None)
            subwindow.deleteLater()

        panel.deleteLater()

        if not self.panels:
            self.active_panel = None
            self.add_panel("xt")
            return

        self._update_panel_titles()
        if was_active:
            self.set_active_panel(self.panels[-1])

    def remove_active_panel(self) -> None:
        if self.active_panel is not None:
            self.remove_panel(self.active_panel)

    def _update_panel_titles(self) -> None:
        mode_names = {"xt": "X-T", "xy": "X-Y", "xyz": "XYZ"}
        for index, panel in enumerate(self.panels, start=1):
            title = f"图 {index} - {mode_names.get(panel.mode, panel.mode.upper())}"
            panel.set_panel_title(title)
            subwindow = self.panel_windows.get(panel)
            if subwindow is not None:
                subwindow.setWindowTitle(title)

    def set_active_panel(self, panel: BasePlotPanel) -> None:
        if panel not in self.panels:
            return

        self.active_panel = panel
        for current in self.panels:
            current.set_active(current is panel)

        subwindow = self.panel_windows.get(panel)
        if subwindow is not None and self.mdi_area.activeSubWindow() is not subwindow:
            self.mdi_area.setActiveSubWindow(subwindow)

        self._sync_tree_from_active_panel()
        self.set_zoom_mode(self._current_zoom_mode())
        self.cursor_label.setText("当前图已切换，可在左侧为该图重新选择信号")

    def _on_subwindow_activated(self, subwindow: QMdiSubWindow | None) -> None:
        if subwindow is None:
            return
        panel = subwindow.widget()
        if isinstance(panel, BasePlotPanel) and panel in self.panels and panel is not self.active_panel:
            self.set_active_panel(panel)

    def _position_new_panel(self, subwindow: PlotSubWindow, mode: str) -> None:
        viewport_size = self.mdi_area.viewport().size()
        viewport_width = max(viewport_size.width(), 900)
        viewport_height = max(viewport_size.height(), 700)
        width = min(max(viewport_width - 80, 640), 1000)
        height = 420 if mode == "xyz" else 320
        height = min(height, viewport_height - 40)
        offset = 36 * (len(self.panels) - 1)
        x = min(offset, max(viewport_width - width - 20, 0))
        y = min(offset, max(viewport_height - height - 20, 0))
        subwindow.resize(width, height)
        subwindow.move(x, y)

    def _sync_tree_from_active_panel(self) -> None:
        if self.active_panel is None:
            return

        self._suppress_item_changed = True
        selected = set(self.active_panel.selected_signal_ids)
        for signal_id, item in self.signal_item_map.items():
            if item.isDisabled():
                continue
            item.setCheckState(0, Qt.CheckState.Checked if signal_id in selected else Qt.CheckState.Unchecked)
        self._suppress_item_changed = False

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._suppress_item_changed:
            return
        signal_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not signal_id or self.active_panel is None:
            return

        checked = item.checkState(0) == Qt.CheckState.Checked
        selection = list(self.active_panel.selected_signal_ids)
        limit = self.active_panel.selection_limit

        if checked:
            if signal_id not in selection:
                if limit is not None and len(selection) >= limit:
                    removed_signal_id = selection.pop(0)
                    removed_item = self.signal_item_map.get(removed_signal_id)
                    if removed_item is not None:
                        self._suppress_item_changed = True
                        removed_item.setCheckState(0, Qt.CheckState.Unchecked)
                        self._suppress_item_changed = False
                selection.append(signal_id)
        else:
            selection = [current for current in selection if current != signal_id]

        self.active_panel.selected_signal_ids = selection
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.active_panel.reset_view()

    def clear_selection(self) -> None:
        if self.active_panel is None:
            return
        self.active_panel.selected_signal_ids = []
        self._suppress_item_changed = True

        for item in self.signal_item_map.values():
            if not item.isDisabled():
                item.setCheckState(0, Qt.CheckState.Unchecked)

        self._suppress_item_changed = False
        self.active_panel.update_plot(self.parsed_log, self.signal_lookup)
        self.cursor_label.setText("未选择曲线")

    def reset_view(self) -> None:
        if self.parsed_log is None or self.active_panel is None:
            return
        self.active_panel.reset_view()

    def set_zoom_mode(self, mode: str) -> None:
        self.zoom_x_button.setChecked(mode == "x")
        self.zoom_y_button.setChecked(mode == "y")
        self.zoom_xy_button.setChecked(mode == "xy")
        if self.active_panel is not None:
            self.active_panel.set_zoom_mode(mode)

    def _current_zoom_mode(self) -> str:
        if self.zoom_x_button.isChecked():
            return "x"
        if self.zoom_y_button.isChecked():
            return "y"
        return "xy"

    def apply_filter(self, text: str) -> None:
        query = text.strip().lower()

        def filter_item(item: QTreeWidgetItem) -> bool:
            own_text = f"{item.text(0)} {item.toolTip(0)}".lower()
            child_visible = False

            for child_index in range(item.childCount()):
                child_visible = filter_item(item.child(child_index)) or child_visible

            visible = not query or query in own_text or child_visible
            item.setHidden(not visible)

            if query and child_visible:
                item.setExpanded(True)

            return visible

        for index in range(self.tree.topLevelItemCount()):
            filter_item(self.tree.topLevelItem(index))

    def _update_panel_status(self, panel: BasePlotPanel, text: str) -> None:
        if panel is self.active_panel:
            self.cursor_label.setText(text)

    def _sync_plot_cursors(self, source_panel: Plot2DPanel, index: int | None) -> None:
        for panel in self.panels:
            if panel is source_panel:
                continue
            if isinstance(panel, Plot2DPanel):
                panel.sync_cursor_to_index(index)


def launch_app(initial_path: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_path=initial_path)
    window.show()
    return app.exec()
