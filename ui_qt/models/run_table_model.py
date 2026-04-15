"""Tabellenmodell fuer Builder-Run-Ergebnisse."""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore


class RunTableModel(QtCore.QAbstractTableModel):
    """Zeigt gespeicherte Run-Payloads in einer stabilen Tabellenansicht."""

    COLUMNS = (
        ("run_id", "Run ID"),
        ("config_id", "Config"),
        ("seed", "Seed"),
        ("run_mode", "Mode"),
        ("val_accuracy", "Val Acc"),
        ("val_loss", "Val Loss"),
        ("test_accuracy", "Test Acc"),
        ("best_objective", "Best Obj"),
    )

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._runs: list[dict[str, Any]] = []

    def set_runs(self, runs: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._runs = list(runs)
        self.endResetModel()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._runs)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._runs)):
            return None
        run_payload = self._runs[index.row()]
        run_definition = run_payload["run_definition"]
        metrics = run_payload["metrics"]
        key = self.COLUMNS[index.column()][0]
        if role == QtCore.Qt.DisplayRole:
            if key in run_definition:
                return str(run_definition[key])
            if key in metrics:
                return f"{float(metrics[key]):.4f}"
            return ""
        if role == QtCore.Qt.UserRole:
            return run_payload
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ) -> Any:
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return self.COLUMNS[section][1]
        return str(section + 1)

    def run_payload(self, row: int) -> dict[str, Any] | None:
        if 0 <= row < len(self._runs):
            return self._runs[row]
        return None
