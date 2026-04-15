"""Qt-Panel fuer Baseline-Vergleiche im Demo-Workspace."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from services.compare_service import ComparePayload


class ComparePanel(QtWidgets.QWidget):
    storeBaselineRequested = QtCore.Signal()

    def __init__(self, language: str = "de", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._language = language

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QtWidgets.QHBoxLayout()
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-weight: 600; color: #0f172a;")
        header.addWidget(self.summary_label, 1)
        self.store_button = QtWidgets.QPushButton()
        self.store_button.clicked.connect(self.storeBaselineRequested.emit)
        header.addWidget(self.store_button)
        layout.addLayout(header)

        self.detail_text = QtWidgets.QTextBrowser()
        layout.addWidget(self.detail_text, 1)
        self.set_language(language)
        self.set_placeholder(language)

    def set_language(self, language: str) -> None:
        self._language = language
        self.store_button.setText(
            "Store as baseline" if language == "en" else "Als Baseline speichern"
        )

    def set_placeholder(self, language: str | None = None) -> None:
        if language is not None:
            self.set_language(language)
        self.summary_label.setText(
            "No baseline stored yet." if self._language == "en" else "Noch keine Baseline gespeichert."
        )
        self.detail_text.setPlainText(
            "Store one state as a baseline and compare it later."
            if self._language == "en"
            else "Speichere einen Zustand als Baseline und vergleiche ihn spaeter."
        )

    def set_payload(self, payload: ComparePayload, language: str | None = None) -> None:
        if language is not None:
            self.set_language(language)
        self.summary_label.setText(payload.summary_text)

        lines = [
            payload.summary_text,
            "=" * len(payload.summary_text),
            "",
            f"{'Current benchmark' if self._language == 'en' else 'Aktueller Benchmark'}: {payload.current_benchmark}",
            f"{'Baseline benchmark' if self._language == 'en' else 'Benchmark der Baseline'}: {payload.baseline_benchmark}",
            f"{'Current hidden sizes' if self._language == 'en' else 'Hidden-Sizes aktuell'}: {' / '.join(str(v) for v in payload.current_hidden_sizes)}",
            f"{'Baseline hidden sizes' if self._language == 'en' else 'Hidden-Sizes Baseline'}: {' / '.join(str(v) for v in payload.baseline_hidden_sizes)}",
            f"{'Current layout' if self._language == 'en' else 'Layout aktuell'}: {payload.current_layout_spec}",
            f"{'Baseline layout' if self._language == 'en' else 'Layout Baseline'}: {payload.baseline_layout_spec}",
            "",
        ]
        if payload.compatibility_issue is not None:
            lines.extend(
                [
                    "Comparison currently unavailable"
                    if self._language == "en"
                    else "Vergleich aktuell nicht moeglich",
                    "-" * 32,
                    payload.compatibility_issue,
                ]
            )
            self.detail_text.setPlainText("\n".join(lines))
            return

        if payload.sample_source is not None:
            lines.extend(
                [
                    f"{'Sample source' if self._language == 'en' else 'Sample-Quelle'}: {payload.sample_source}",
                    f"{'Baseline prediction' if self._language == 'en' else 'Baseline-Vorhersage'}: {payload.baseline_prediction}",
                    f"{'Current prediction' if self._language == 'en' else 'Aktuelle Vorhersage'}: {payload.current_prediction}",
                    "",
                    "Baseline probabilities:" if self._language == "en" else "Wahrscheinlichkeiten Baseline:",
                    ", ".join(f"{label}={value:.3f}" for label, value in payload.baseline_probabilities),
                    "",
                    "Current probabilities:" if self._language == "en" else "Wahrscheinlichkeiten aktuell:",
                    ", ".join(f"{label}={value:.3f}" for label, value in payload.current_probabilities),
                    "",
                ]
            )

        if payload.baseline_metrics is not None:
            lines.extend(
                [
                    "Baseline Metrics" if self._language == "en" else "Baseline-Metriken",
                    "-" * 16,
                    f"{'Epochs' if self._language == 'en' else 'Epochen'}: {payload.baseline_metrics.epochs}",
                    f"Test-Acc: {payload.baseline_metrics.test_accuracy:.4f}",
                    f"Test-Loss: {payload.baseline_metrics.test_loss:.4f}",
                    "",
                ]
            )
        if payload.current_metrics is not None:
            lines.extend(
                [
                    "Current Metrics" if self._language == "en" else "Aktuelle Metriken",
                    "-" * 15,
                    f"{'Epochs' if self._language == 'en' else 'Epochen'}: {payload.current_metrics.epochs}",
                    f"Test-Acc: {payload.current_metrics.test_accuracy:.4f}",
                    f"Test-Loss: {payload.current_metrics.test_loss:.4f}",
                    "",
                ]
            )

        lines.extend(
            [
                "Layout Diff" if self._language == "en" else "Layout-Diff",
                "-" * 11,
                *(
                    payload.layout_diff_lines
                    if payload.layout_diff_lines
                    else (("No differences." if self._language == "en" else "Keine Unterschiede."),)
                ),
                "",
                "Natural Computing View" if self._language == "en" else "Natural-Computing-Sicht",
                "-" * 22,
                (
                    f"Number of direct single-step neighbors: {payload.natural_neighbor_count}"
                    if self._language == "en"
                    else f"Anzahl direkter Single-Step-Nachbarn: {payload.natural_neighbor_count}"
                ),
                "",
            ]
        )

        if payload.confusion_matrix:
            lines.append(
                "Confusion Matrix on Test Split"
                if self._language == "en"
                else "Confusion-Matrix auf dem Testsplit"
            )
            lines.append("-" * 30)
            header = "true\\pred".ljust(12) + " ".join(label[:8].rjust(8) for label in payload.confusion_labels)
            lines.append(header)
            for label, row in zip(payload.confusion_labels, payload.confusion_matrix, strict=True):
                lines.append(label[:10].ljust(12) + " ".join(f"{value:8d}" for value in row))
            lines.append("")

        if payload.uncertain_samples:
            lines.append(
                "Most Uncertain Test Samples"
                if self._language == "en"
                else "Unsicherste Test-Samples"
            )
            lines.append("-" * 27)
            for entry in payload.uncertain_samples:
                lines.append(
                    (
                        f"#{entry.sample_index}: true={entry.true_label}, pred={entry.predicted_label}, conf={entry.confidence:.3f}"
                        if self._language == "en"
                        else f"#{entry.sample_index}: wahr={entry.true_label}, pred={entry.predicted_label}, conf={entry.confidence:.3f}"
                    )
                )

        self.detail_text.setPlainText("\n".join(lines))
