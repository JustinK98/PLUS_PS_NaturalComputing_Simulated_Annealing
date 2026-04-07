# Activation Playground

Basisprojekt fuer das Proseminar **Natural Computing** an der **Paris Lodron Universitaet Salzburg**.

Die Codebasis dient als kompakter Playground fuer kleine neuronale Netze mit frei belegbaren Aktivierungsfunktionen. Sie bildet die Grundlage fuer spaetere Erweiterungen wie Suche ueber Layouts und Simulated Annealing.

## Schnellstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
python main.py
```

Das startet den interaktiven Assistenten im Terminal.

Direkt in die GUI:

```bash
python main.py --gui
```

Hilfe zur CLI:

```bash
python main.py --help
```

Hilfe zu Layout- und Neighbor-Syntax:

```bash
python main.py --show-layout-help
```

## Was mit der aktuellen Codebasis moeglich ist

- kleine MLPs mit **1 bis 4 Hidden-Layern**
- Aktivierungen pro Layer oder pro Neuron frei setzen
- unterstuetzte Aktivierungen: `relu`, `tanh`, `sigmoid`, `leaky_relu`
- Benchmarks: `breast_cancer`, `wine`, `digits`, `test_activation`
- Training mit **Loss**, **Accuracy**, **Validation** und **Test**
- Layouts als String definieren und veraendern
- Neighbor-Operationen auf Layouts ausfuehren
- Layout-Diffs sichtbar machen
- ASCII-Ausgaben im Terminal
- Matplotlib-Plots fuer Lernkurven und Layouts
- GUI fuer interaktive Analyse von Netz, Datenfluss und Neuronen

## Startmodi

### 1. Interaktiver Start

```bash
python main.py
```

Der Assistent fuehrt durch Benchmark, Hidden-Layer, Layout, Neighbor-Operationen, Training und optionale GUI-Nutzung.

### 2. GUI

```bash
python main.py --gui
```

Expertenmodus:

```bash
python main.py --gui --gui-mode expert
```

Die GUI zeigt:

- das Netzwerk als Knoten- und Verbindungsansicht
- Eingabedaten und Zielwerte
- Aktivierungsfunktionen in Farben
- frei aenderbare Hidden-Layer im Expertenmodus
- Trainingsplots
- Neuron-Inspektion per Klick
- Forward/Backward-Stepper fuer ein einzelnes Sample
- Vergleich von Baseline und aktuellem Experiment

### 3. Reproduzierbare CLI-Runs

```bash
python main.py --benchmark wine --hidden-sizes 16 8 --layout "relu|tanh" --epochs 120
```

Mit Plot-Speicherung:

```bash
python main.py --benchmark digits --layout "relu|relu" --epochs 40 --save-prefix demo/digits_run
```

## Benchmarks

### `breast_cancer`

- 30 numerische Eingaben
- 2 Klassen
- gut fuer kompakte binaere Klassifikation

### `wine`

- 13 numerische Eingaben
- 3 Klassen
- gut fuer erste Layout-Vergleiche

### `digits`

- 64 Eingaben
- 10 Klassen
- 8x8 Grauwertbilder von Ziffern

### `test_activation`

- kuenstlicher Mini-Datensatz mit 3 Inputs
- fuer rohe Vorwaertsrechnungen und Aktivierungsvergleiche

## Layout-Syntax

Ein Layout beschreibt die Aktivierungsfunktionen aller Hidden-Layer.

Beispiele:

```text
relu|relu
relu|tanh|sigmoid
relu*16|tanh*8
relu*5,tanh*5|sigmoid*4
```

Bedeutung:

- `|` trennt Hidden-Layer
- `,` trennt Aktivierungen innerhalb eines Layers
- `relu*16` steht fuer 16 Neuronen mit `relu`

## Neighbor-Operationen

Neighbor-Operationen veraendern Layouts lokal. Das ist wichtig fuer spaetere Suchverfahren.

Beispiele:

```text
set:L1:0:sigmoid
fill:L2:tanh
cycle:L1:3
swap:L2:1:4
```

## Projektstruktur

- `main.py`: CLI, interaktiver Assistent, Programmstart
- `benchmarks.py`: Datensaetze und Splits
- `activations.py`: Aktivierungen, Layouts, Neighbor-Logik
- `model.py`: MLP, Forward-Pass, Backpropagation, Inspektion
- `trainer.py`: Trainingsschleife und Metriken
- `plotting.py`: Matplotlib-Plots
- `terminal_viz.py`: ASCII-Darstellungen im Terminal
- `gui.py`: interaktive GUI
- `configs.py`: zentrale Konfigurationen und Defaults

## Aktueller Stand

Die aktuelle Version ist das erste Basisprojekt. Sie deckt Modell, Layouts, Neighbor-Logik, Training, Visualisierung und GUI bereits ab und ist so aufgebaut, dass spaeter Optimierungsverfahren auf Aktivierungs-Layouts direkt darauf aufsetzen koennen.
