"""Zustandsobjekte fuer die Qt-GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkspacePreferences:
    """Globale UI-Praeferenzen fuer alle Qt-Workspaces."""

    language: str = "de"
    detail_mode: str = "beginner"


@dataclass
class AppState:
    """Aktueller Shell-Zustand der Qt-App."""

    workspace: str
    preferences: WorkspacePreferences
