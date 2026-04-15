"""Gemeinsamer Layout-Editor mit Layer- und Neuron-Steuerung."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from activations import ActivationLayout, parse_layout_spec
from configs import SUPPORTED_ACTIVATIONS


class LayoutEditorWidget(QtWidgets.QWidget):
    """Bearbeitet Layout-Strings und neuronweise Aktivierungszuweisungen."""

    layoutSpecChanged = QtCore.Signal(str)

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        layout_spec: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._hidden_sizes = tuple(hidden_sizes)
        self._layout = parse_layout_spec(layout_spec, self._hidden_sizes)
        self._neuron_combos: list[list[QtWidgets.QComboBox]] = []
        self._layer_fill_combos: list[QtWidgets.QComboBox] = []
        self._synchronizing = False

        root = QtWidgets.QVBoxLayout(self)
        self.layout_line_edit = QtWidgets.QLineEdit(self._layout.to_compact_spec())
        self.layout_line_edit.setPlaceholderText("relu|tanh")
        self.layout_line_edit.editingFinished.connect(self._apply_line_edit)
        root.addWidget(self.layout_line_edit)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)
        self._rebuild_tabs()

    def hidden_sizes(self) -> tuple[int, ...]:
        return self._hidden_sizes

    def layout_spec(self) -> str:
        return self._layout.to_compact_spec()

    def current_layout(self) -> ActivationLayout:
        return self._layout

    def set_hidden_sizes(self, hidden_sizes: tuple[int, ...]) -> None:
        if tuple(hidden_sizes) == self._hidden_sizes:
            return
        self._hidden_sizes = tuple(hidden_sizes)
        self._layout = self._layout.with_hidden_sizes(hidden_sizes)
        self.layout_line_edit.setText(self._layout.to_compact_spec())
        self._rebuild_tabs()
        self.layoutSpecChanged.emit(self.layout_spec())

    def set_layout_spec(self, layout_spec: str) -> None:
        self._layout = parse_layout_spec(layout_spec, self._hidden_sizes)
        self.layout_line_edit.setText(self._layout.to_compact_spec())
        self.layout_line_edit.setStyleSheet("")
        self._rebuild_tabs()
        self.layoutSpecChanged.emit(self.layout_spec())

    def _rebuild_tabs(self) -> None:
        self._synchronizing = True
        self.tabs.clear()
        self._neuron_combos.clear()
        self._layer_fill_combos.clear()
        for layer_index, layer in enumerate(self._layout.layers):
            page = QtWidgets.QWidget()
            page_layout = QtWidgets.QVBoxLayout(page)
            fill_row = QtWidgets.QHBoxLayout()
            fill_row.addWidget(QtWidgets.QLabel(f"Layer {layer_index + 1}"))
            fill_combo = QtWidgets.QComboBox()
            fill_combo.addItems(SUPPORTED_ACTIVATIONS)
            fill_combo.currentTextChanged.connect(
                lambda activation_name, idx=layer_index: self._fill_layer(idx, activation_name)
            )
            fill_row.addWidget(fill_combo)
            fill_row.addStretch(1)
            page_layout.addLayout(fill_row)

            grid = QtWidgets.QGridLayout()
            neuron_row: list[QtWidgets.QComboBox] = []
            for neuron_index, activation_name in enumerate(layer):
                grid.addWidget(QtWidgets.QLabel(f"n{neuron_index}"), neuron_index // 4 * 2, neuron_index % 4)
                combo = QtWidgets.QComboBox()
                combo.addItems(SUPPORTED_ACTIVATIONS)
                combo.setCurrentText(activation_name)
                combo.currentTextChanged.connect(
                    lambda value, l=layer_index, n=neuron_index: self._set_neuron(l, n, value)
                )
                grid.addWidget(combo, neuron_index // 4 * 2 + 1, neuron_index % 4)
                neuron_row.append(combo)
            page_layout.addLayout(grid)
            page_layout.addStretch(1)
            self.tabs.addTab(page, f"L{layer_index + 1}")
            self._neuron_combos.append(neuron_row)
            self._layer_fill_combos.append(fill_combo)
            if len(set(layer)) == 1:
                fill_combo.setCurrentText(layer[0])
        self._synchronizing = False

    def _fill_layer(self, layer_index: int, activation_name: str) -> None:
        if self._synchronizing:
            return
        self._layout = self._layout.replace_layer(layer_index, activation_name)
        self._sync_after_widget_change()

    def _set_neuron(self, layer_index: int, neuron_index: int, activation_name: str) -> None:
        if self._synchronizing:
            return
        self._layout = self._layout.replace_neuron(layer_index, neuron_index, activation_name)
        self._sync_after_widget_change()

    def _sync_after_widget_change(self) -> None:
        self.layout_line_edit.setText(self._layout.to_compact_spec())
        self.layout_line_edit.setStyleSheet("")
        self._rebuild_tabs()
        self.layoutSpecChanged.emit(self.layout_spec())

    def _apply_line_edit(self) -> None:
        if self._synchronizing:
            return
        try:
            self._layout = parse_layout_spec(self.layout_line_edit.text(), self._hidden_sizes)
        except ValueError:
            self.layout_line_edit.setStyleSheet("border: 1px solid #dc2626;")
            return
        self.layout_line_edit.setStyleSheet("")
        self.layout_line_edit.setText(self._layout.to_compact_spec())
        self._rebuild_tabs()
        self.layoutSpecChanged.emit(self.layout_spec())
