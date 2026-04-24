from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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


class MainWindow(QMainWindow):
    COLORS = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#17becf",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("机器人状态曲线查看器")
        self.resize(1400, 860)

        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.plot_items: dict[str, pg.PlotDataItem] = {}
        self.legend: pg.LegendItem | None = None
        self._suppress_item_changed = False
        self._initial_path = initial_path

        self._build_ui()

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
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索字段，例如 J1 / Tcp / ErrCode")

        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.search_input, 1)

        splitter = QSplitter(Qt.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("信号字段")
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_tree_item_changed)

        self.plot_widget = pg.PlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setLabel("bottom", "相对时间", units="s")
        self.plot_widget.setLabel("left", "数值")
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.getPlotItem().setDownsampling(mode="peak")
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot_widget.setMouseEnabled(x=True, y=True)

        self.legend = self.plot_widget.addLegend(offset=(10, 10))
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#555555", width=1))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#999999", width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.v_line.hide()
        self.h_line.hide()
        self.mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        splitter.addWidget(self.tree)
        splitter.addWidget(self.plot_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])

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
        self.search_input.textChanged.connect(self.apply_filter)

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
        QApplication.setOverrideCursor(Qt.WaitCursor)
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
        self._refresh_plot()

        self.info_label.setText(
            "文件: "
            f"{parsed.meta.path.name} | 采样点: {parsed.meta.sample_count} | 字段: {parsed.meta.field_count} | "
            f"起始: {parsed.meta.start_time_raw} | 结束: {parsed.meta.end_time_raw} | "
            f"跳过坏行: {parsed.meta.skipped_rows}"
        )
        self.cursor_label.setText("已加载文件，勾选左侧信号开始绘图")
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
                    item.setData(0, Qt.UserRole, signal.signal_id)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                    item.setCheckState(0, Qt.Unchecked)
                    item.setToolTip(0, signal.full_path)
                    if not signal.available:
                        item.setDisabled(True)
                        item.setText(0, f"{part} (无数据列)")

        for index in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(index).setExpanded(True)

        self._suppress_item_changed = False
        self.apply_filter(self.search_input.text())

    def _checked_signal_ids(self) -> list[str]:
        checked_ids: list[str] = []

        def visit(item: QTreeWidgetItem) -> None:
            signal_id = item.data(0, Qt.UserRole)
            if signal_id and item.checkState(0) == Qt.Checked and item.isDisabled() is False:
                checked_ids.append(signal_id)
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for index in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(index))

        return checked_ids

    def _refresh_plot(self) -> None:
        for item in self.plot_items.values():
            self.plot_widget.removeItem(item)
        self.plot_items.clear()

        if self.legend is not None:
            self.legend.clear()

        if self.parsed_log is None:
            self.v_line.hide()
            self.h_line.hide()
            return

        for order, signal_id in enumerate(self._checked_signal_ids()):
            signal = self.signal_lookup.get(signal_id)
            if signal is None:
                continue

            curve = self.plot_widget.plot(
                self.parsed_log.time_seconds,
                self.parsed_log.get_series(signal_id),
                pen=pg.mkPen(self.COLORS[order % len(self.COLORS)], width=2),
                name=signal.full_path,
            )
            self.plot_items[signal_id] = curve

        has_curves = bool(self.plot_items)
        self.v_line.setVisible(has_curves)
        self.h_line.setVisible(has_curves)

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0 or self._suppress_item_changed:
            return
        if not item.data(0, Qt.UserRole):
            return
        self._refresh_plot()
        self.reset_view()

    def clear_selection(self) -> None:
        self._suppress_item_changed = True

        def visit(item: QTreeWidgetItem) -> None:
            if item.data(0, Qt.UserRole) and item.isDisabled() is False:
                item.setCheckState(0, Qt.Unchecked)
            for child_index in range(item.childCount()):
                visit(item.child(child_index))

        for index in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(index))

        self._suppress_item_changed = False
        self._refresh_plot()
        self.cursor_label.setText("未选择曲线")

    def reset_view(self) -> None:
        if self.parsed_log is None:
            return
        self.plot_widget.enableAutoRange()
        self.plot_widget.autoRange()

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

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self.parsed_log is None or not self.plot_items:
            self.cursor_label.setText("未选择曲线")
            return

        scene_pos = event[0]
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)
        x_value = float(mouse_point.x())
        times = self.parsed_log.time_seconds
        index = int(np.searchsorted(times, x_value))
        if index >= len(times):
            index = len(times) - 1
        elif index > 0 and abs(times[index - 1] - x_value) <= abs(times[index] - x_value):
            index -= 1

        nearest_x = float(times[index])
        self.v_line.setPos(nearest_x)

        y_value = None
        preview_parts = [f"t={nearest_x:.4f}s"]
        for signal_id in list(self.plot_items)[:3]:
            signal = self.signal_lookup.get(signal_id)
            if signal is None:
                continue
            value = float(self.parsed_log.get_series(signal_id)[index])
            if y_value is None:
                y_value = value
            preview_parts.append(f"{signal.name}={value:.6f}")

        if y_value is not None:
            self.h_line.setPos(y_value)

        self.cursor_label.setText(" | ".join(preview_parts))


def launch_app(initial_path: Path | None = None) -> int:
    pg.setConfigOptions(antialias=True)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(initial_path=initial_path)
    window.show()
    return app.exec()
