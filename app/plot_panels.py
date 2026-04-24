from __future__ import annotations

from collections.abc import Callable
import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .models import ParsedLog, SignalNode

try:
    import pyqtgraph.opengl as gl

    GL_AVAILABLE = True
    GL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on optional local dependency
    gl = None
    GL_AVAILABLE = False
    GL_IMPORT_ERROR = str(exc)


class AxisZoomPlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._zoom_mode = "xy"

    def set_zoom_mode(self, mode: str) -> None:
        self._zoom_mode = mode

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._zoom_mode not in {"x", "y", "xy"}:
            return super().wheelEvent(event)

        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)

        scene_pos = self.mapToScene(event.position().toPoint())
        mouse_point = self.getPlotItem().vb.mapSceneToView(scene_pos)
        factor = 0.85 if delta > 0 else 1.0 / 0.85
        x_factor = factor if self._zoom_mode in {"x", "xy"} else 1.0
        y_factor = factor if self._zoom_mode in {"y", "xy"} else 1.0
        self.getPlotItem().vb.scaleBy(x=x_factor, y=y_factor, center=mouse_point)
        event.accept()


class BasePlotPanel(QFrame):
    def __init__(
        self,
        mode: str,
        activate_callback: Callable[["BasePlotPanel"], None],
        remove_callback: Callable[["BasePlotPanel"], None],
    ) -> None:
        super().__init__()
        self.mode = mode
        self.selected_signal_ids: list[str] = []
        self._activate_callback = activate_callback
        self._remove_callback = remove_callback
        self._status_callback: Callable[[str], None] | None = None

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("plotPanel")
        self.setStyleSheet(
            """
            QFrame#plotPanel {
                border: 2px solid #d9d9d9;
                border-radius: 8px;
                background: #ffffff;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.detail_label = QLabel()
        self.detail_label.setStyleSheet("color: #666666;")
        self.active_label = QLabel("当前图")
        self.active_label.setStyleSheet("color: #1f6feb; font-weight: 700;")
        self.active_label.hide()
        self.activate_button = QPushButton("设为当前图")
        self.remove_button = QPushButton("关闭此图")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.detail_label, 1)
        header_layout.addWidget(self.active_label)
        header_layout.addWidget(self.activate_button)
        header_layout.addWidget(self.remove_button)
        layout.addLayout(header_layout)

        self.message_label = QLabel("No data loaded")
        self.message_label.setStyleSheet("color: #666666;")
        layout.addWidget(self.message_label)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        layout.addWidget(self.content_widget, 1)

        self.activate_button.clicked.connect(lambda: self._activate_callback(self))
        self.remove_button.clicked.connect(lambda: self._remove_callback(self))

    @property
    def selection_limit(self) -> int | None:
        return None

    def set_status_callback(self, callback: Callable[[str], None]) -> None:
        self._status_callback = callback

    def emit_status(self, text: str) -> None:
        if self._status_callback is not None:
            self._status_callback(text)

    def set_panel_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_active(self, active: bool) -> None:
        if active:
            self.setStyleSheet(
                """
                QFrame#plotPanel {
                    border: 2px solid #1f6feb;
                    border-radius: 8px;
                    background: #ffffff;
                }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#plotPanel {
                    border: 2px solid #d9d9d9;
                    border-radius: 8px;
                    background: #ffffff;
                }
                """
            )
        self.active_label.setVisible(active)
        self.activate_button.setVisible(not active)
        self.activate_button.setEnabled(not active)

    def set_zoom_mode(self, mode: str) -> None:
        del mode

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        del parsed_log, signal_lookup

    def reset_view(self) -> None:
        return


class Plot2DPanel(BasePlotPanel):
    def __init__(
        self,
        mode: str,
        activate_callback: Callable[[BasePlotPanel], None],
        remove_callback: Callable[[BasePlotPanel], None],
    ) -> None:
        super().__init__(mode=mode, activate_callback=activate_callback, remove_callback=remove_callback)
        self.curves: list[pg.PlotDataItem] = []
        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self._default_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._cursor_sync_callback: Callable[[int | None], None] | None = None

        self.plot_widget = AxisZoomPlotWidget(background="w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getPlotItem().setClipToView(True)
        self.plot_widget.getPlotItem().setDownsampling(mode="peak")
        self.plot_widget.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plot_widget.setMouseEnabled(x=True, y=True)

        self.legend = None
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#555555", width=1))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#999999", width=1, style=Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)
        self.v_line.setZValue(20)
        self.h_line.setZValue(19)
        self._hide_cursor()

        self.content_layout.addWidget(self.plot_widget, 1)

        self.value_overlay = QLabel(self.plot_widget)
        self.value_overlay.setStyleSheet(
            """
            QLabel {
                background: rgba(255, 255, 255, 210);
                border: 1px solid rgba(31, 111, 235, 120);
                border-radius: 6px;
                padding: 6px 8px;
                color: #1f2937;
                font-size: 12px;
            }
            """
        )
        self.value_overlay.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.value_overlay.setWordWrap(True)
        self.value_overlay.setMinimumWidth(220)
        self.value_overlay.setMaximumWidth(320)
        self._set_overlay_text("Values will appear here")
        self.value_overlay.raise_()
        self.value_overlay.show()

        self.plot_widget.installEventFilter(self)
        self.plot_widget.viewport().installEventFilter(self)

        self.mouse_proxy = pg.SignalProxy(
            self.plot_widget.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )

        self.set_zoom_mode("xy")

    @property
    def selection_limit(self) -> int | None:
        if self.mode == "xy":
            return 2
        return None

    def set_zoom_mode(self, mode: str) -> None:
        self.plot_widget.set_zoom_mode(mode)
        self.detail_label.setText(f"Zoom wheel: {mode.upper()}")

    def set_cursor_sync_callback(self, callback: Callable[[int | None], None]) -> None:
        self._cursor_sync_callback = callback

    @staticmethod
    def _display_signal_label(signal: SignalNode) -> str:
        if len(signal.path_parts) > 1:
            return " / ".join(signal.path_parts[1:])
        return signal.full_path

    @staticmethod
    def _series_color(index: int) -> str:
        return MainPalette.colors[index % len(MainPalette.colors)]

    def _format_overlay_rows(self, rows: list[tuple[str, float, str]]) -> str:
        html_rows = []
        for label, value, color in rows:
            safe_label = html.escape(label)
            html_rows.append(
                f"<span style='color:{color}; font-weight:600;'>{safe_label}</span>: "
                f"<span>{value:.6f}</span>"
            )
        return "<br>".join(html_rows) if html_rows else "No values"

    def _clear_curves(self) -> None:
        for curve in self.curves:
            self.plot_widget.removeItem(curve)
        self.curves.clear()
        if self.legend is not None:
            self.legend.clear()
        self._default_ranges = None
        self._hide_cursor()

    def _set_overlay_text(self, text: str) -> None:
        self.value_overlay.setText(text)
        self.value_overlay.adjustSize()
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        margin = 12
        x_pos = max(self.plot_widget.width() - self.value_overlay.width() - margin, margin)
        self.value_overlay.move(x_pos, margin)

    def _hide_cursor(self, *, broadcast: bool = False) -> None:
        self.v_line.hide()
        self.h_line.hide()
        if broadcast and self._cursor_sync_callback is not None:
            self._cursor_sync_callback(None)
        if hasattr(self, "plot_widget"):
            self.plot_widget.viewport().update()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.plot_widget and event.type() in {QEvent.Type.Resize, QEvent.Type.Show}:
            self._reposition_overlay()
        elif watched is self.plot_widget.viewport() and event.type() in {QEvent.Type.Leave, QEvent.Type.Hide}:
            self._hide_cursor(broadcast=True)
        return super().eventFilter(watched, event)

    @staticmethod
    def _compute_padded_range(min_value: float, max_value: float) -> tuple[float, float]:
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            return -1.0, 1.0
        if min_value == max_value:
            padding = max(abs(min_value) * 0.05, 1.0)
            return min_value - padding, max_value + padding
        padding = (max_value - min_value) * 0.05
        return min_value - padding, max_value + padding

    def _set_default_ranges(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        x_min, x_max = self._compute_padded_range(float(np.min(x_values)), float(np.max(x_values)))
        y_min, y_max = self._compute_padded_range(float(np.min(y_values)), float(np.max(y_values)))
        self._default_ranges = ((x_min, x_max), (y_min, y_max))

    def _apply_default_ranges(self) -> None:
        if self._default_ranges is None:
            return
        (x_min, x_max), (y_min, y_max) = self._default_ranges
        self.plot_widget.setXRange(x_min, x_max, padding=0.0)
        self.plot_widget.setYRange(y_min, y_max, padding=0.0)

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        self.parsed_log = parsed_log
        self.signal_lookup = signal_lookup
        self._clear_curves()

        if parsed_log is None:
            self.message_label.setText("No data loaded")
            self._set_overlay_text("No data loaded")
            return

        if self.mode == "xt":
            self._update_xt_plot(parsed_log, signal_lookup)
        else:
            self._update_xy_plot(parsed_log, signal_lookup)

    def _update_xt_plot(self, parsed_log: ParsedLog, signal_lookup: dict[str, SignalNode]) -> None:
        self.plot_widget.setLabel("bottom", "Relative Time", units="s")
        self.plot_widget.setLabel("left", "Value")

        if not self.selected_signal_ids:
            self.message_label.setText("Select one or more signals to draw an X-T plot.")
            self._set_overlay_text("X-T plot is waiting for signal selection")
            return

        all_series: list[np.ndarray] = []
        for order, signal_id in enumerate(self.selected_signal_ids):
            if signal_id not in parsed_log.signals_by_id:
                continue
            signal = signal_lookup[signal_id]
            series = parsed_log.get_series(signal_id)
            curve = self.plot_widget.plot(
                parsed_log.time_seconds,
                series,
                pen=pg.mkPen(self._series_color(order), width=2),
                name=self._display_signal_label(signal),
            )
            self.curves.append(curve)
            all_series.append(series)

        self.message_label.setText(f"X-T plot with {len(self.curves)} signal(s)")
        if all_series:
            combined = np.concatenate(all_series)
            self._set_default_ranges(parsed_log.time_seconds, combined)
            first_signal = signal_lookup[self.selected_signal_ids[0]]
            first_value = float(parsed_log.get_series(self.selected_signal_ids[0])[0])
            self._set_overlay_text(
                self._format_overlay_rows(
                    [(self._display_signal_label(first_signal), first_value, self._series_color(0))]
                )
            )

    def _update_xy_plot(self, parsed_log: ParsedLog, signal_lookup: dict[str, SignalNode]) -> None:
        if len(self.selected_signal_ids) != 2:
            self.plot_widget.setLabel("bottom", "X")
            self.plot_widget.setLabel("left", "Y")
            self.message_label.setText("Select exactly two signals for the current X-Y plot.")
            self._set_overlay_text("X-Y plot is waiting for 2 selected signals")
            return

        x_signal_id, y_signal_id = self.selected_signal_ids
        x_signal = signal_lookup[x_signal_id]
        y_signal = signal_lookup[y_signal_id]
        x_series = parsed_log.get_series(x_signal_id)
        y_series = parsed_log.get_series(y_signal_id)

        x_label = self._display_signal_label(x_signal)
        y_label = self._display_signal_label(y_signal)
        self.plot_widget.setLabel("bottom", x_label)
        self.plot_widget.setLabel("left", y_label)
        curve = self.plot_widget.plot(
            x_series,
            y_series,
            pen=pg.mkPen(MainPalette.colors[0], width=2),
            name=f"{y_label} vs {x_label}",
        )
        self.curves.append(curve)
        self.message_label.setText(f"X-Y plot: X={x_label} | Y={y_label}")
        self._set_default_ranges(x_series, y_series)
        self._set_overlay_text(
            self._format_overlay_rows(
                [
                    (f"X - {x_label}", float(x_series[0]), self._series_color(0)),
                    (f"Y - {y_label}", float(y_series[0]), self._series_color(1)),
                ]
            )
        )

    def reset_view(self) -> None:
        self._apply_default_ranges()

    def sync_cursor_to_index(self, index: int | None) -> None:
        if index is None or self.parsed_log is None or not self.curves:
            self._hide_cursor()
            return

        sample_count = self.parsed_log.time_seconds.shape[0]
        if sample_count == 0:
            self._hide_cursor()
            return

        clamped_index = max(0, min(index, sample_count - 1))
        if self.mode == "xt":
            self._show_xt_cursor_at_index(clamped_index)
        else:
            self._show_xy_cursor_at_index(clamped_index)

    def _on_mouse_moved(self, event: tuple[object]) -> None:
        if self.parsed_log is None or not self.curves:
            self.emit_status("No visible data in the current plot")
            self._hide_cursor(broadcast=True)
            return

        scene_pos = event[0]
        view_rect = self.plot_widget.getPlotItem().vb.sceneBoundingRect()
        if not view_rect.contains(scene_pos):
            self._hide_cursor(broadcast=True)
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)

        if self.mode == "xt":
            self._update_xt_cursor(mouse_point.x())
        else:
            self._update_xy_cursor(mouse_point.x(), mouse_point.y())

    def _update_xt_cursor(self, x_value: float) -> None:
        assert self.parsed_log is not None
        times = self.parsed_log.time_seconds
        index = int(np.searchsorted(times, x_value))
        if index >= len(times):
            index = len(times) - 1
        elif index > 0 and abs(times[index - 1] - x_value) <= abs(times[index] - x_value):
            index -= 1
        self._show_xt_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)

    def _show_xt_cursor_at_index(self, index: int) -> None:
        assert self.parsed_log is not None
        nearest_x = float(self.parsed_log.time_seconds[index])
        self.v_line.setPos(nearest_x)

        preview_parts = [f"t={nearest_x:.4f}s"]
        y_value = None
        for signal_id in self.selected_signal_ids[:3]:
            value = float(self.parsed_log.get_series(signal_id)[index])
            signal = self.signal_lookup[signal_id]
            preview_parts.append(f"{signal.name}={value:.6f}")
            if y_value is None:
                y_value = value

        if y_value is not None:
            self.h_line.setPos(y_value)

        overlay_rows: list[tuple[str, float, str]] = []
        for order, signal_id in enumerate(self.selected_signal_ids[:6]):
            value = float(self.parsed_log.get_series(signal_id)[index])
            signal = self.signal_lookup[signal_id]
            color = self._series_color(order)
            overlay_rows.append((self._display_signal_label(signal), value, color))

        self._set_overlay_text(self._format_overlay_rows(overlay_rows))
        self.v_line.show()
        self.h_line.show()
        self.plot_widget.viewport().update()
        self.emit_status(" | ".join(preview_parts))

    def _update_xy_cursor(self, x_value: float, y_value: float) -> None:
        if len(self.selected_signal_ids) != 2:
            self._hide_cursor()
            return

        assert self.parsed_log is not None
        x_series = self.parsed_log.get_series(self.selected_signal_ids[0])
        y_series = self.parsed_log.get_series(self.selected_signal_ids[1])
        x_span = max(float(np.max(x_series) - np.min(x_series)), 1e-9)
        y_span = max(float(np.max(y_series) - np.min(y_series)), 1e-9)
        distances = ((x_series - x_value) / x_span) ** 2 + ((y_series - y_value) / y_span) ** 2
        index = int(np.argmin(distances))
        self._show_xy_cursor_at_index(index)
        if self._cursor_sync_callback is not None:
            self._cursor_sync_callback(index)

    def _show_xy_cursor_at_index(self, index: int) -> None:
        assert self.parsed_log is not None
        x_series = self.parsed_log.get_series(self.selected_signal_ids[0])
        y_series = self.parsed_log.get_series(self.selected_signal_ids[1])
        nearest_x = float(x_series[index])
        nearest_y = float(y_series[index])

        self.v_line.setPos(nearest_x)
        self.h_line.setPos(nearest_y)

        x_signal = self.signal_lookup[self.selected_signal_ids[0]]
        y_signal = self.signal_lookup[self.selected_signal_ids[1]]
        self._set_overlay_text(
            self._format_overlay_rows(
                [
                    (f"X - {self._display_signal_label(x_signal)}", nearest_x, self._series_color(0)),
                    (f"Y - {self._display_signal_label(y_signal)}", nearest_y, self._series_color(1)),
                ]
            )
        )
        self.v_line.show()
        self.h_line.show()
        self.plot_widget.viewport().update()
        self.emit_status(f"Point {index} | {x_signal.name}={nearest_x:.6f} | {y_signal.name}={nearest_y:.6f}")


class Plot3DPanel(BasePlotPanel):
    def __init__(
        self,
        activate_callback: Callable[[BasePlotPanel], None],
        remove_callback: Callable[[BasePlotPanel], None],
    ) -> None:
        super().__init__(mode="xyz", activate_callback=activate_callback, remove_callback=remove_callback)
        self.parsed_log: ParsedLog | None = None
        self.signal_lookup: dict[str, SignalNode] = {}
        self.line_item = None
        self._default_distance = 40.0

        if GL_AVAILABLE:
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setMinimumHeight(280)
            self.grid_item = gl.GLGridItem()
            self.gl_widget.addItem(self.grid_item)
            try:
                self.axis_item = gl.GLAxisItem()
                self.gl_widget.addItem(self.axis_item)
            except Exception:  # pragma: no cover - depends on backend support
                self.axis_item = None
            self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
            self.gl_widget.setCameraPosition(distance=self._default_distance)
            self.content_layout.addWidget(self.gl_widget, 1)
        else:
            self.gl_widget = None
            self.axis_item = None
            missing_label = QLabel(
                "3D dependencies are missing, so the XYZ plot is unavailable.\n"
                f"Install PyOpenGL and restart the app.\nError: {GL_IMPORT_ERROR}"
            )
            missing_label.setWordWrap(True)
            missing_label.setStyleSheet("color: #b54708;")
            self.content_layout.addWidget(missing_label)

    @property
    def selection_limit(self) -> int | None:
        return 3

    def update_plot(self, parsed_log: ParsedLog | None, signal_lookup: dict[str, SignalNode]) -> None:
        self.parsed_log = parsed_log
        self.signal_lookup = signal_lookup
        self.detail_label.setText("3D trajectory")

        if not GL_AVAILABLE or self.gl_widget is None:
            self.message_label.setText("XYZ plot unavailable. Install PyOpenGL first.")
            return

        if self.line_item is not None:
            self.gl_widget.removeItem(self.line_item)
            self.line_item = None

        if parsed_log is None:
            self.message_label.setText("No data loaded")
            return

        if len(self.selected_signal_ids) != 3:
            self.message_label.setText("Select exactly 3 signals for the XYZ plot in X, Y, Z order.")
            return

        x_signal_id, y_signal_id, z_signal_id = self.selected_signal_ids
        x_signal = signal_lookup[x_signal_id]
        y_signal = signal_lookup[y_signal_id]
        z_signal = signal_lookup[z_signal_id]
        points = np.column_stack(
            [
                parsed_log.get_series(x_signal_id),
                parsed_log.get_series(y_signal_id),
                parsed_log.get_series(z_signal_id),
            ]
        ).astype(np.float32)
        center = points.mean(axis=0)
        centered_points = points - center

        self.line_item = gl.GLLinePlotItem(
            pos=centered_points,
            color=(0.12, 0.47, 0.71, 1.0),
            width=2,
            antialias=True,
            mode="line_strip",
        )
        self.gl_widget.addItem(self.line_item)

        span = np.ptp(centered_points, axis=0)
        max_span = max(float(np.max(span)), 1.0)
        distance = max(max_span * 2.5, 20.0)
        grid_scale = max(max_span / 10.0, 1.0)
        self.grid_item.resetTransform()
        self.grid_item.scale(grid_scale, grid_scale, 1.0)
        if self.axis_item is not None:
            try:
                self.axis_item.setSize(max_span, max_span, max_span)
            except Exception:  # pragma: no cover - backend dependent
                pass
        self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
        self.gl_widget.setCameraPosition(distance=distance)
        self._default_distance = distance
        self.message_label.setText(f"XYZ plot: X={x_signal.name} | Y={y_signal.name} | Z={z_signal.name}")
        self.emit_status(
            f"XYZ trajectory: X={x_signal.full_path} | Y={y_signal.full_path} | Z={z_signal.full_path}"
        )

    def reset_view(self) -> None:
        if self.gl_widget is not None:
            self.gl_widget.opts["center"] = QVector3D(0.0, 0.0, 0.0)
            self.gl_widget.setCameraPosition(distance=self._default_distance)


class MainPalette:
    colors = [
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
