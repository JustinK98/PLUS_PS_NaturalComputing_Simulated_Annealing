"""Kompakte bilinguale Texte fuer die Qt-Oberflaeche."""

from __future__ import annotations


TEXTS = {
    "de": {
        "workspace.workflow": "Activation Workflow",
        "workspace.builder": "Experiment Builder",
        "detail.beginner": "Beginner",
        "detail.expert": "Expert",
        "shell.language": "Sprache",
        "shell.detail": "Detailstufe",
        "shell.workspace": "Arbeitsmodus",
        "shell.guide": "Anleitung",
        "shell.status.ready": "Bereit",
        "common.run": "Starten",
        "common.load": "Laden",
        "common.reset": "Zurücksetzen",
        "common.export": "Exportieren",
        "common.benchmark": "Benchmark",
        "common.hidden_sizes": "Hidden Sizes",
        "common.layout": "Layout",
        "common.training": "Training",
        "common.help": "Hilfe",
        "common.recipes": "Rezepte",
        "common.summary": "Übersicht",
        "common.analysis": "Analyse",
        "common.runs": "Runs",
        "common.network": "Netzwerk",
        "common.neuron_detail": "Neuron Detail",
        "common.sample": "Sample",
        "common.split": "Split",
        "common.seed": "Seed",
        "common.epochs": "Epochen",
        "common.learning_rate": "Lernrate",
        "common.batch_size": "Batch-Größe",
        "common.weight_scale": "Weight Scale",
        "common.mode.manual": "Manuelles Training",
        "common.mode.sa": "Simulated Annealing",
        "builder.setup": "Experiment Setup",
        "builder.seeds": "Seeds",
        "builder.method": "Methode",
        "builder.search": "Search Space",
        "builder.storage": "Storage",
        "builder.run": "Run Control",
    },
    "en": {
        "workspace.workflow": "Activation Workflow",
        "workspace.builder": "Experiment Builder",
        "detail.beginner": "Beginner",
        "detail.expert": "Expert",
        "shell.language": "Language",
        "shell.detail": "Detail Level",
        "shell.workspace": "Workspace",
        "shell.guide": "Guide",
        "shell.status.ready": "Ready",
        "common.run": "Run",
        "common.load": "Load",
        "common.reset": "Reset",
        "common.export": "Export",
        "common.benchmark": "Benchmark",
        "common.hidden_sizes": "Hidden Sizes",
        "common.layout": "Layout",
        "common.training": "Training",
        "common.help": "Help",
        "common.recipes": "Recipes",
        "common.summary": "Summary",
        "common.analysis": "Analysis",
        "common.runs": "Runs",
        "common.network": "Network",
        "common.neuron_detail": "Neuron Detail",
        "common.sample": "Sample",
        "common.split": "Split",
        "common.seed": "Seed",
        "common.epochs": "Epochs",
        "common.learning_rate": "Learning Rate",
        "common.batch_size": "Batch Size",
        "common.weight_scale": "Weight Scale",
        "common.mode.manual": "Manual Training",
        "common.mode.sa": "Simulated Annealing",
        "builder.setup": "Experiment Setup",
        "builder.seeds": "Seeds",
        "builder.method": "Method",
        "builder.search": "Search Space",
        "builder.storage": "Storage",
        "builder.run": "Run Control",
    },
}


RECIPES = {
    "relu_vs_tanh_two_moons": {
        "title": {"de": "ReLU vs. Tanh auf two_moons", "en": "ReLU vs Tanh on two_moons"},
        "body": {
            "de": "Vergleiche zwei kleine Layouts auf two_moons mit identischen Seeds. Beobachte Val-Accuracy und ob tanh stabiler, aber langsamer konvergiert.",
            "en": "Compare two small layouts on two_moons with identical seeds. Watch validation accuracy and whether tanh converges more smoothly but more slowly.",
        },
    },
    "circles_layout_compare": {
        "title": {"de": "Layout-Vergleich auf concentric_circles", "en": "Layout comparison on concentric_circles"},
        "body": {
            "de": "Nutze concentric_circles mit 8 Hidden-Neuronen und vergleiche homogene vs. gemischte Aktivierungen.",
            "en": "Use concentric_circles with 8 hidden neurons and compare homogeneous versus mixed activations.",
        },
    },
    "sa_homogeneous_relu": {
        "title": {"de": "SA von homogener ReLU-Basis", "en": "SA from homogeneous ReLU baseline"},
        "body": {
            "de": "Starte SA mit relu und beobachte, ob einzelne Neuronen in andere Aktivierungen kippen.",
            "en": "Start SA from relu and observe whether single neurons move to other activations.",
        },
    },
    "multi_seed_compare": {
        "title": {"de": "Multi-Seed-Vergleich", "en": "Multi-seed comparison"},
        "body": {
            "de": "Führe dieselbe Konfiguration mit mehreren Seeds aus und interpretiere Mittelwert und Streuung statt Einzelruns.",
            "en": "Run the same configuration over multiple seeds and interpret mean and variance instead of a single run.",
        },
    },
    "test_activation_lab": {
        "title": {"de": "test_activation als Rechenlabor", "en": "test_activation as activation lab"},
        "body": {
            "de": "Nutze das kleine Lernlabor, um Forward-Pfade, Aktivierungen und SA-Schritte auf sehr kleinem Zustand zu verstehen.",
            "en": "Use the tiny learning lab to understand forward paths, activations, and SA steps on a very small state space.",
        },
    },
}


def text(language: str, key: str) -> str:
    """Liefert einen Textschluessel mit englischem Fallback."""

    return TEXTS.get(language, TEXTS["en"]).get(key, TEXTS["en"].get(key, key))
