"""Moderne, skalierbare Netzwerkansicht auf Basis von QGraphicsView."""

from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6 import QtCore, QtGui, QtWidgets

from activations import ACTIVATION_COLORS
from model import ModularMLP
from services.network_projection_service import InputProjection


@dataclass(frozen=True)
class VisualOptions:
    min_scene_width: float = 920.0
    min_scene_height: float = 540.0
    horizontal_margin: float = 92.0
    vertical_margin: float = 80.0
    header_band: float = 52.0
    min_layer_gap: float = 136.0
    max_layer_gap: float = 220.0
    min_node_gap: float = 18.0
    max_node_gap: float = 58.0
    min_node_radius: float = 7.0
    max_node_radius: float = 18.0
    fit_margin: float = 26.0
    zoom_factor: float = 1.15


class _NeuronItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, rect: QtCore.QRectF, layer_index: int, neuron_index: int, callback) -> None:
        super().__init__(rect)
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self._callback = callback
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:  # pragma: no cover - GUI event
        self._callback(self.layer_index, self.neuron_index)
        super().mousePressEvent(event)


class NetworkViewWidget(QtWidgets.QGraphicsView):
    """Visualisiert ein MLP als klickbare Netzwerkansicht."""

    hiddenNeuronSelected = QtCore.Signal(int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setMinimumHeight(240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._model: ModularMLP | None = None
        self._options = VisualOptions()
        self._selected_hidden: tuple[int, int] | None = None
        self._highlighted_hidden: frozenset[tuple[int, int]] = frozenset()
        self._highlight_status: str | None = None
        self._input_projection: InputProjection | None = None
        self._node_positions: dict[tuple[str, int, int], QtCore.QPointF] = {}
        self._scene_rect = QtCore.QRectF(0.0, 0.0, self._options.min_scene_width, self._options.min_scene_height)
        self._auto_fit_enabled = True
        self.setBackgroundBrush(QtGui.QColor("#edf3f8"))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.BoundingRectViewportUpdate)
        self.viewport().setAttribute(QtCore.Qt.WA_AcceptTouchEvents, True)
        self.setToolTip("Wheel: zoom, drag: pan, double-click: fit to window")

    def set_model(self, model: ModularMLP | None) -> None:
        self._model = model
        self._redraw()

    def set_selected_hidden(self, selection: tuple[int, int] | None) -> None:
        self._selected_hidden = selection
        self._redraw()

    def set_highlighted_hidden(
        self,
        positions: frozenset[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...],
        status: str | None,
    ) -> None:
        """Markiert geänderte Hidden-Neuronen als akzeptiert oder verworfen."""

        if status not in {None, "accepted", "rejected"}:
            raise ValueError("status muss accepted, rejected oder None sein.")
        self._highlighted_hidden = frozenset(positions)
        self._highlight_status = status
        self._redraw()

    def set_input_projection(self, projection: InputProjection | None) -> None:
        self._input_projection = projection
        self._redraw()

    def is_auto_fit_enabled(self) -> bool:
        return self._auto_fit_enabled

    def reset_view_to_fit(self) -> None:
        self._auto_fit_enabled = True
        self._fit_scene()

    def zoom_in(self) -> None:
        self._apply_zoom(self._options.zoom_factor)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / self._options.zoom_factor)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # pragma: no cover - GUI event
        super().resizeEvent(event)
        self._fit_scene()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # pragma: no cover - GUI event
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._fit_scene)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # pragma: no cover - GUI event
        if self._model is None:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() == 0:
            event.ignore()
            return
        factor = self._options.zoom_factor if event.angleDelta().y() > 0 else 1.0 / self._options.zoom_factor
        self._apply_zoom(factor)
        event.accept()

    def viewportEvent(self, event: QtCore.QEvent) -> bool:  # pragma: no cover - GUI event
        if event.type() == QtCore.QEvent.Type.NativeGesture and isinstance(event, QtGui.QNativeGestureEvent):
            if event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
                zoom_factor = max(0.25, 1.0 + float(event.value()))
                self._apply_zoom(zoom_factor)
                event.accept()
                return True
            if event.gestureType() == QtCore.Qt.NativeGestureType.SmartZoomNativeGesture:
                self.reset_view_to_fit()
                event.accept()
                return True
        return super().viewportEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover - GUI event
        self.reset_view_to_fit()
        super().mouseDoubleClickEvent(event)

    def _redraw(self) -> None:
        scene = self.scene()
        assert scene is not None
        scene.clear()
        self._node_positions.clear()
        if self._model is None:
            message = scene.addText("No model loaded yet.")
            message.setDefaultTextColor(QtGui.QColor("#475569"))
            self._scene_rect = QtCore.QRectF(0.0, 0.0, self._options.min_scene_width, self._options.min_scene_height)
            scene.setSceneRect(self._scene_rect)
            self._fit_scene()
            return

        projection = self._effective_projection(self._model.input_size)
        input_indices = list(projection.displayed_indices)
        layer_sizes = [len(input_indices), *self._model.hidden_sizes, self._model.output_size]
        scene_width, scene_height, x_positions, top_y, node_gap, radius = self._layout_metrics(layer_sizes)
        self._scene_rect = QtCore.QRectF(0.0, 0.0, scene_width, scene_height)
        self._draw_column_headers(x_positions, projection)

        for layer_index, displayed_index in enumerate(input_indices):
            point = self._position_for_node(x_positions[0], top_y, len(input_indices), layer_index, node_gap)
            self._node_positions[("input", 0, displayed_index)] = point
            self._draw_node(
                point,
                "#cbd5e1",
                projection.displayed_labels[layer_index],
                len(input_indices),
                layer_index,
                radius,
            )

        for hidden_layer_index, layer in enumerate(self._model.layout.layers):
            x = x_positions[hidden_layer_index + 1]
            for neuron_index, activation_name in enumerate(layer):
                point = self._position_for_node(x, top_y, len(layer), neuron_index, node_gap)
                self._node_positions[("hidden", hidden_layer_index, neuron_index)] = point
                selected = self._selected_hidden == (hidden_layer_index, neuron_index)
                self._draw_hidden_node(
                    point,
                    activation_name,
                    hidden_layer_index,
                    neuron_index,
                    len(layer),
                    selected,
                    radius,
                )

        output_x = x_positions[-1]
        for output_index in range(self._model.output_size):
            point = self._position_for_node(output_x, top_y, self._model.output_size, output_index, node_gap)
            self._node_positions[("output", 0, output_index)] = point
            self._draw_node(point, "#94a3b8", f"y{output_index}", self._model.output_size, output_index, radius)

        self._draw_edges(input_indices)
        scene.setSceneRect(self._scene_rect)
        self._fit_scene()

    def _effective_projection(self, input_size: int) -> InputProjection:
        if self._input_projection is None:
            indices = tuple(range(input_size))
            return InputProjection(
                displayed_indices=indices,
                displayed_labels=tuple(f"x{index}" for index in indices),
                total_input_size=input_size,
            )
        return self._input_projection

    def _layout_metrics(
        self,
        layer_sizes: list[int],
    ) -> tuple[float, float, list[float], float, float, float]:
        total_columns = len(layer_sizes)
        max_layer_size = max(layer_sizes)
        density = max(0.0, max_layer_size - 8)
        node_gap = max(
            self._options.min_node_gap,
            min(self._options.max_node_gap, self._options.max_node_gap - density * 0.62),
        )
        layer_gap = max(
            self._options.min_layer_gap,
            min(self._options.max_layer_gap, self._options.max_layer_gap - max(0, total_columns - 4) * 18.0),
        )
        content_height = max(1, max_layer_size - 1) * node_gap
        content_width = max(1, total_columns - 1) * layer_gap
        scene_height = max(
            self._options.min_scene_height,
            self._options.header_band + self._options.vertical_margin * 2 + content_height,
        )
        scene_width = max(
            self._options.min_scene_width,
            self._options.horizontal_margin * 2 + content_width,
        )
        left_x = (scene_width - content_width) / 2
        top_y = self._options.header_band + (scene_height - self._options.header_band - content_height) / 2
        x_positions = [left_x + index * layer_gap for index in range(total_columns)]
        radius = max(
            self._options.min_node_radius,
            min(self._options.max_node_radius, node_gap * 0.32),
        )
        return scene_width, scene_height, x_positions, top_y, node_gap, radius

    def _position_for_node(
        self,
        x: float,
        top_y: float,
        layer_size: int,
        node_index: int,
        node_gap: float,
    ) -> QtCore.QPointF:
        y = top_y + node_index * node_gap
        return QtCore.QPointF(x, y)

    def _draw_column_headers(self, x_positions: list[float], projection: InputProjection) -> None:
        scene = self.scene()
        assert scene is not None
        labels = [
            f"Input ({len(projection.displayed_indices)}/{projection.total_input_size})",
            *(f"L{index + 1}" for index in range(len(x_positions) - 2)),
            "Output",
        ]
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSizeF(10.0)
        for label, x in zip(labels, x_positions, strict=False):
            item = QtWidgets.QGraphicsSimpleTextItem(label)
            item.setFont(font)
            item.setBrush(QtGui.QBrush(QtGui.QColor("#4b5563")))
            item.setPos(x - item.boundingRect().width() / 2, 16.0)
            scene.addItem(item)

    def _draw_node(
        self,
        point: QtCore.QPointF,
        color: str,
        label: str,
        layer_size: int,
        node_index: int,
        radius: float,
    ) -> None:
        scene = self.scene()
        assert scene is not None
        item = scene.addEllipse(
            point.x() - radius,
            point.y() - radius,
            radius * 2,
            radius * 2,
            QtGui.QPen(QtGui.QColor("#cbd5e1"), 1.2),
            QtGui.QBrush(QtGui.QColor(color)),
        )
        item.setToolTip(label)
        self._draw_label(point, label, layer_size, node_index, radius, QtGui.QColor("#111827"))

    def _draw_hidden_node(
        self,
        point: QtCore.QPointF,
        activation_name: str,
        layer_index: int,
        neuron_index: int,
        layer_size: int,
        selected: bool,
        radius: float,
    ) -> None:
        scene = self.scene()
        assert scene is not None
        highlighted = (layer_index, neuron_index) in self._highlighted_hidden
        if highlighted and self._highlight_status == "accepted":
            pen = QtGui.QPen(QtGui.QColor("#16a34a"), 3.6)
        elif highlighted and self._highlight_status == "rejected":
            pen = QtGui.QPen(QtGui.QColor("#dc2626"), 3.6)
        else:
            pen = QtGui.QPen(QtGui.QColor("#1d4ed8" if selected else "#cbd5e1"), 2.4 if selected else 1.2)
        brush = QtGui.QBrush(QtGui.QColor(ACTIVATION_COLORS[activation_name]))
        item = _NeuronItem(
            QtCore.QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2),
            layer_index,
            neuron_index,
            self._on_hidden_clicked,
        )
        item.setPen(pen)
        item.setBrush(brush)
        item.setToolTip(f"L{layer_index + 1}:n{neuron_index} ({activation_name})")
        scene.addItem(item)
        self._draw_label(point, str(neuron_index), layer_size, neuron_index, radius, QtGui.QColor("#0f172a"))

    def _draw_label(
        self,
        point: QtCore.QPointF,
        label: str,
        layer_size: int,
        node_index: int,
        radius: float,
        color: QtGui.QColor,
    ) -> None:
        if not self._should_label_node(layer_size, node_index):
            return
        scene = self.scene()
        assert scene is not None
        text_item = QtWidgets.QGraphicsSimpleTextItem(label)
        font = QtGui.QFont()
        font.setPointSizeF(max(7.5, min(10.5, radius * 0.78)))
        text_item.setFont(font)
        text_item.setBrush(QtGui.QBrush(color))
        bounds = text_item.boundingRect()
        text_item.setPos(point.x() - bounds.width() / 2, point.y() - bounds.height() / 2)
        scene.addItem(text_item)

    def _should_label_node(self, layer_size: int, node_index: int) -> bool:
        if layer_size <= 12:
            return True
        if layer_size <= 24:
            step = 2
        elif layer_size <= 48:
            step = 4
        elif layer_size <= 96:
            step = 8
        else:
            step = max(8, math.ceil(layer_size / 12))
        return node_index in {0, layer_size - 1} or node_index % step == 0

    def _draw_edges(self, input_indices: list[int]) -> None:
        scene = self.scene()
        assert scene is not None
        assert self._model is not None
        pen = QtGui.QPen(QtGui.QColor(148, 163, 184, 86), 0.9)
        selected_pen = QtGui.QPen(QtGui.QColor("#2563eb"), 1.8)

        for displayed_input_index in input_indices:
            start = self._node_positions[("input", 0, displayed_input_index)]
            for hidden_index in range(self._model.hidden_sizes[0]):
                end = self._node_positions[("hidden", 0, hidden_index)]
                line_pen = selected_pen if self._selected_hidden == (0, hidden_index) else pen
                scene.addLine(start.x(), start.y(), end.x(), end.y(), line_pen)

        for layer_index in range(self._model.num_hidden_layers - 1):
            for source_index in range(self._model.hidden_sizes[layer_index]):
                start = self._node_positions[("hidden", layer_index, source_index)]
                for target_index in range(self._model.hidden_sizes[layer_index + 1]):
                    end = self._node_positions[("hidden", layer_index + 1, target_index)]
                    selected = self._selected_hidden in {
                        (layer_index, source_index),
                        (layer_index + 1, target_index),
                    }
                    scene.addLine(start.x(), start.y(), end.x(), end.y(), selected_pen if selected else pen)

        last_hidden_index = self._model.num_hidden_layers - 1
        for source_index in range(self._model.hidden_sizes[last_hidden_index]):
            start = self._node_positions[("hidden", last_hidden_index, source_index)]
            for target_index in range(self._model.output_size):
                end = self._node_positions[("output", 0, target_index)]
                line_pen = selected_pen if self._selected_hidden == (last_hidden_index, source_index) else pen
                scene.addLine(start.x(), start.y(), end.x(), end.y(), line_pen)

    def _on_hidden_clicked(self, layer_index: int, neuron_index: int) -> None:
        self._selected_hidden = (layer_index, neuron_index)
        self.hiddenNeuronSelected.emit(layer_index, neuron_index)
        self._redraw()

    def _apply_zoom(self, factor: float) -> None:
        self._auto_fit_enabled = False
        self.scale(factor, factor)

    def _fit_scene(self) -> None:
        if not self._auto_fit_enabled:
            return
        scene = self.scene()
        if scene is None or scene.sceneRect().isEmpty():
            return
        target = scene.sceneRect().adjusted(
            -self._options.fit_margin,
            -self._options.fit_margin,
            self._options.fit_margin,
            self._options.fit_margin,
        )
        self.fitInView(target, QtCore.Qt.KeepAspectRatio)
        self.centerOn(scene.sceneRect().center())
