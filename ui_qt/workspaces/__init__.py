"""Qt-Workspaces fuer Presentation, Demo, Playground und Experiment Builder."""

from .demo_workspace import DemoWorkspace
from .playground_workspace import PlaygroundWorkspace
from .experiment_builder_workspace import ExperimentBuilderWorkspace
from .presentation_workspace import PresentationWorkspace

__all__ = [
    "PresentationWorkspace",
    "DemoWorkspace",
    "PlaygroundWorkspace",
    "ExperimentBuilderWorkspace",
]
