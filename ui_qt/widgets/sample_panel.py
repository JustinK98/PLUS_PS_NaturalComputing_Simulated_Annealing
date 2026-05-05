"""Benchmark-spezifische Sample-Panels fuer die Qt-GUI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore, QtWidgets

from benchmarks import DatasetBundle
from services.analysis_sample_service import (
    AnalysisSample,
    build_feature_rows,
    build_test_activation_rows,
    format_probability_lines,
)


@dataclass(frozen=True)
class SampleDisplayPayload:
    dataset: DatasetBundle
    analysis_sample: AnalysisSample
    prediction_name: str
    probabilities: np.ndarray


class FeatureTableWidget(QtWidgets.QTableWidget):
    """Tabellenansicht fuer top Features eines Samples."""

    def __init__(self, headers: list[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().hide()
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def set_rows(self, rows: list[tuple[object, ...]]) -> None:
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem(str(value))
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.setItem(row_index, column_index, item)


class SamplePanelWidget(QtWidgets.QWidget):
    """Zeigt benchmark-spezifische Sample-Ansichten inklusive Zusammenfassung."""

    def __init__(self, language: str = "en", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.language = language

        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setProperty("role", "muted")
        self.probabilities_label = QtWidgets.QLabel()
        self.probabilities_label.setWordWrap(True)
        self.probabilities_label.setProperty("role", "muted")
        layout.addWidget(self.summary_label)
        layout.addWidget(self.hint_label)
        layout.addWidget(self.probabilities_label)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.generic_table = FeatureTableWidget(["#", "Feature", "Raw", "Scaled"])
        self.test_activation_table = FeatureTableWidget(["Feature", "Raw", "Scaled"])
        self.stack.addWidget(self.generic_table)
        self.stack.addWidget(self.test_activation_table)

    def set_language(self, language: str) -> None:
        self.language = language

    def set_sample(self, payload: SampleDisplayPayload) -> None:
        dataset = payload.dataset
        analysis_sample = payload.analysis_sample

        if self.language == "de":
            summary = (
                f"<b>Quelle:</b> {analysis_sample.source_label}<br>"
                f"<b>Split:</b> {analysis_sample.split_name} | "
                f"<b>Sample-Index:</b> {analysis_sample.sample_index}<br>"
                f"<b>Echtes Ziel:</b> {analysis_sample.actual_target_name} | "
                f"<b>Analyse-Ziel:</b> {analysis_sample.effective_target_name}<br>"
                f"<b>Vorhersage:</b> {payload.prediction_name}"
            )
        else:
            summary = (
                f"<b>Source:</b> {analysis_sample.source_label}<br>"
                f"<b>Split:</b> {analysis_sample.split_name} | "
                f"<b>Sample index:</b> {analysis_sample.sample_index}<br>"
                f"<b>True target:</b> {analysis_sample.actual_target_name} | "
                f"<b>Analysis target:</b> {analysis_sample.effective_target_name}<br>"
                f"<b>Prediction:</b> {payload.prediction_name}"
            )
        self.summary_label.setText(summary)
        self.probabilities_label.setText(format_probability_lines(dataset, payload.probabilities))

        if dataset.name == "test_activation":
            self.stack.setCurrentWidget(self.test_activation_table)
            rows = [
                (feature_name, f"{raw_value:+.3f}", f"{scaled_value:+.3f}")
                for feature_name, raw_value, scaled_value in build_test_activation_rows(dataset, analysis_sample)
            ]
            self.test_activation_table.set_rows(rows)
            self.hint_label.setText(
                "Kompaktes Rechenlabor mit Roh- und skalierten Eingaben."
                if self.language == "de"
                else "Compact lab view with raw and scaled inputs."
            )
            return

        self.stack.setCurrentWidget(self.generic_table)
        rows = [
            (rank, feature_name, f"{raw_value:+.3f}", f"{scaled_value:+.3f}")
            for rank, feature_name, raw_value, scaled_value in build_feature_rows(dataset, analysis_sample)
        ]
        self.generic_table.set_rows(rows)
        self.hint_label.setText(
            "Die Tabelle zeigt die staerksten Eingabefeatures des aktuellen Samples."
            if self.language == "de"
            else "The table shows the strongest input features of the current sample."
        )
