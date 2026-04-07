# Activation Playground fuer Natural Computing

## Schnellstart

Dieses Projekt ist ein didaktischer Playground fuer ein Proseminar im Kontext
von Natural Computing. Es soll Studierenden helfen, kleine neuronale Netze,
Aktivierungsfunktionen, Layouts, Training und spaeter auch Suchverfahren wie
Simulated Annealing nachvollziehbar zu verstehen.

Wenn du das Programm einfach ausprobieren willst, genuegen diese Schritte:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
python main.py
```

Das startet den interaktiven Assistenten im Terminal.

Wenn du direkt die GUI starten willst:

```bash
python main.py --gui
```

Wenn du nur die Hilfe sehen willst:

```bash
python main.py --help
```

Wenn du nur die Layout- und Neighbor-Syntax erklaert bekommen willst:

```bash
python main.py --show-layout-help
```

## Worum geht es in diesem Projekt?

Dieses Projekt wurde als leicht verstaendliche, modulare Codebasis fuer ein
Proseminar entwickelt. Es ist kein produktionsreifes Deep-Learning-Framework,
sondern ein Lernwerkzeug.

Der Fokus liegt auf diesen Fragen:

- Was macht eine Aktivierungsfunktion in einem neuronalen Netz?
- Wie veraendert sich das Verhalten, wenn verschiedene Aktivierungen gemischt
  werden?
- Was passiert, wenn Aktivierungen nicht nur pro Layer, sondern pro Neuron
  unterschiedlich gesetzt werden?
- Wie sieht der Zusammenhang zwischen Layout, Training, Validation und Test aus?
- Wie kann man solche Layouts spaeter als Suchraum fuer Natural-Computing-
  Verfahren wie Simulated Annealing interpretieren?

Das Projekt ist bewusst einfach gehalten:

- kleines MLP mit variabler Hidden-Layer-Anzahl
- keine externen Deep-Learning-Frameworks wie PyTorch oder TensorFlow
- eigene Implementierung von Forward-Pass, Backpropagation und Training
- klare Trennung zwischen Daten, Modell, Training, Visualisierung und GUI

Gerade fuer ein Proseminar ist das wichtig: Studierende sollen die Mechanik des
Modells lesen und verstehen koennen, statt nur Bibliotheksaufrufe zu benutzen.

## Was das Programm kann

- kleine MLPs mit 1 bis 4 Hidden-Layern trainieren
- Aktivierungen pro Layer oder pro einzelnem Neuron frei setzen
- unterstuetzte Aktivierungen:
  - `relu`
  - `tanh`
  - `sigmoid`
  - `leaky_relu`
- Benchmarks laden:
  - `breast_cancer`
  - `wine`
  - `digits`
  - `test_activation` als kuenstliches Lernlabor
- Trainings-, Validierungs- und Testmetriken berechnen
- Accuracy und Loss verfolgen
- Layouts im Terminal als ASCII darstellen
- Neighbor-Layouts und Diffs anzeigen
- Matplotlib-Plots fuer Lernkurven und Layouts erzeugen
- didaktische GUI starten
- in der GUI Samples, Ziele, Aktivierungen und Netzstruktur sichtbar machen
- in der GUI einzelne Neuronen anklicken und lokale Berechnungen inspizieren

## Warum ist das Natural Computing?

Im Natural Computing geht es haeufig um Such-, Optimierungs- und
Adaptionsprozesse, die sich an natuerlichen Systemen orientieren. Dieses
Projekt bereitet genau so einen Suchraum vor.

Hier ist ein Aktivierungs-Layout nicht nur eine feste Konfiguration, sondern ein
moeglicher Zustand im Suchraum:

- Ein Zustand ist ein konkretes Layout der Aktivierungsfunktionen.
- Eine lokale Veraenderung ist eine Neighbor-Operation.
- Eine Bewertungsfunktion kann spaeter zum Beispiel die Validation-Accuracy oder
  der Validation-Loss sein.
- Ein Suchverfahren wie Simulated Annealing kann dann ueber diesen Layoutraum
  laufen.

Die aktuelle Codebasis stellt dafuer bereits die Grundlagen bereit:

- Layout-Syntax
- per-Neuron-Belegung
- Neighbor-Operationen
- Single-Step-Neighbor-Generierung
- Train/Validation/Test-Auswertung

## Startmodi

### 1. Interaktiver Assistent im Terminal

Standardstart:

```bash
python main.py
```

Oder explizit:

```bash
python main.py --interactive
```

Der Assistent fuehrt Schritt fuer Schritt durch:

- Benchmark-Auswahl
- Hidden-Layer-Groessen
- Layout-Modus
- Layout-Definition
- Neighbor-Operationen
- Trainingsparameter
- optionale GUI-Nutzung
- Plot-Speicherung

Dieser Modus ist fuer erste Lern- und Seminarversuche am einfachsten.

### 2. GUI-Modus

```bash
python main.py --gui
```

Oder direkt im Expertenmodus:

```bash
python main.py --gui --gui-mode expert
```

Die GUI ist als didaktische Lernoberflaeche gedacht. Sie zeigt:

- Einsteiger-Modus und Experten-Modus
- das Netzwerk als Kreise und Verbindungen
- variable Hidden-Layer-Anzahl mit Hinzufuegen und Entfernen von Layern
- die gesetzten Aktivierungen in Farben
- Input-Daten und Zielwerte
- bei `digits` eine 8x8-Darstellung der Eingabe
- bei `wine` und `breast_cancer` wichtige Input-Features
- Trainingsplots
- einen echten Forward/Backward-Stepper fuer das aktuell sichtbare Sample
- eine Aktivierungskurve mit markiertem aktuellem `z`-Wert
- neuronale Detailansichten
- einen Baseline-vs-Experiment-Vergleich
- lokale Rechnungen eines selektierten Hidden-Neurons

Die GUI eignet sich besonders fuer:

- Lehrveranstaltungen
- Demonstrationen
- explorative Experimente
- Diskussionen im Proseminar

### 3. Klassische CLI fuer reproduzierbare Experimente

Beispiel:

```bash
python main.py --benchmark wine --layout "relu*16|tanh*8" --epochs 120 --save-prefix demo/wine_run
```

Das ist sinnvoll, wenn du:

- mehrere Runs vergleichen willst
- bestimmte Layouts reproduzierbar testen willst
- spaeter Skripte oder Suchverfahren darauf aufbauen willst

## Benchmarks

### `breast_cancer`

- 30 numerische Eingabefeatures
- 2 Klassen
- relativ kompakter Datensatz
- gut fuer erste Trainingslaeufe

Didaktische Einordnung:

- Dieser Benchmark ist ein guter Einstieg, weil die Klassifikation nur zwei
  Klassen besitzt und sich Lernkurven oft schnell stabilisieren.
- Er eignet sich besonders, um den Unterschied zwischen Training,
  Validation und Test zu besprechen, ohne dass die Datenstruktur selbst zu
  komplex wird.
- In diesem Setting kann man gut beobachten, wie sich verschiedene
  Aktivierungsfunktionen auf einen vergleichsweise einfachen tabellarischen
  Datensatz auswirken.

Worauf man achten sollte:

- Lernt das Netz sehr schnell oder eher langsam?
- Bleiben Validation und Test nahe an der Trainingsleistung?
- Aendert ein Aktivierungswechsel die Stabilitaet der Lernkurven?

### `wine`

- 13 numerische Eingabefeatures
- 3 Klassen
- ueberschaubar und oft gut lernbar
- sehr gut fuer erste Layout-Vergleiche

Didaktische Einordnung:

- `wine` ist haeufig der angenehmste Benchmark fuer erste echte Vergleiche von
  Aktivierungs-Layouts.
- Er ist klein genug, um schnell trainiert zu werden, aber mehrklassig genug,
  damit Unterschiede zwischen Layouts und Wahrscheinlichkeitsverteilungen
  sichtbar werden.
- Wer verstehen will, wie sich `relu|relu`, `relu|tanh` oder gemischte Layouts
  unterschiedlich verhalten, bekommt hier oft schnell aussagekraeftige Resultate.

Worauf man achten sollte:

- Wie veraendern sich die Klassenwahrscheinlichkeiten bei anderen Layouts?
- Sind gemischte Aktivierungen im Hidden-Bereich sichtbar anders als uniforme?
- Verbessert sich die Validation-Accuracy oder nur die Trainings-Accuracy?

### `digits`

- 64 Eingabefeatures
- 10 Klassen
- jedes Sample entspricht einem 8x8 Grauwertbild
- in der GUI besonders anschaulich

Didaktische Einordnung:

- `digits` ist der visuell interessanteste Benchmark im Projekt.
- Hier laesst sich sehr gut zeigen, dass die Eingabe nicht nur eine abstrakte
  Zahlenliste ist, sondern ein Bild, das durch das Netz verarbeitet wird.
- Dadurch eignet sich dieser Benchmark hervorragend, um den Input-Datenstrom,
  Vorhersagewahrscheinlichkeiten und Fehlklassifikationen anschaulich zu
  diskutieren.

Worauf man achten sollte:

- Welche Pixelveraenderungen fuehren zu einer anderen Vorhersage?
- Wie sicher oder unsicher ist das Netz bei aehnlichen Ziffern?
- Welche Hidden-Aktivierungen reagieren stark auf bestimmte Bildmuster?

### `test_activation`

- kuenstlicher Mini-Datensatz mit 3 Inputs
- 2 Klassen
- ideal fuer rohe Aktivierungs- und Rechenexperimente
- besonders gut geeignet, um Forward-Pass und Aktivierungsunterschiede zu
  verstehen

Didaktische Einordnung:

- `test_activation` ist absichtlich kein realistischer Benchmark.
- Der Zweck dieses Modus ist nicht hoechstmoegliche Accuracy, sondern das
  Verstehen der Rechenschritte in einem moeglichst kleinen Netz.
- Gerade im Proseminar ist dieser Modus hilfreich, um Fragen wie diese konkret
  durchzusprechen:
  - Wie entsteht `z = sum(x_i * w_i) + b`?
  - Wie unterscheiden sich `relu`, `tanh`, `sigmoid` und `leaky_relu` direkt?
  - Wie veraendert eine kleine Input-Aenderung die Ausgabe eines einzelnen
    Neurons?

Die interne Lernregel ist bewusst einfach:

- Klasse 1, wenn mindestens zwei der drei Inputs deutlich aktiv sind
- sonst Klasse 0

Worauf man achten sollte:

- Wie wirkt dieselbe Eingabe unter verschiedenen Aktivierungsfunktionen?
- Wann saettigt `sigmoid`, wann bleibt `leaky_relu` noch reaktionsfaehig?
- Welche Summanden treiben ein einzelnes Hidden-Neuron besonders stark?

## Parameter und Stellschrauben

Dieses Projekt lebt davon, dass nicht nur trainiert, sondern bewusst variiert
und beobachtet wird. Die wichtigsten veraenderbaren Parameter sind:

### Benchmark

Bestimmt:

- welcher Datensatz verwendet wird
- wie viele Input-Features es gibt
- wie viele Zielklassen vorhergesagt werden
- wie anschaulich der Input sichtbar ist

Typische Nutzung:

- `breast_cancer` fuer erste stabile Laeufe
- `wine` fuer Layout-Vergleiche
- `digits` fuer visuelle Experimente
- `test_activation` fuer rohe Rechenschritte

### Hidden-Layer

Diese Werte bestimmen die Anzahl der Neuronen in den versteckten Schichten.

Wirkung:

- mehr Neuronen = mehr Modellkapazitaet
- mehr Neuronen = mehr Gewichte und Biases
- mehr Neuronen = oft unuebersichtlichere Visualisierung
- mehr Layer = tiefere, aber auch schwerer lesbare Rechenkette

Didaktischer Hinweis:

- Fuer den Einstieg lieber klein bleiben
- Fuer `test_activation` ist ein kleines Netz fast immer sinnvoller als ein grosses
- Im Expertenmodus der GUI koennen Layer hinzugefuegt und wieder entfernt werden

### Layout

Das Layout bestimmt die Aktivierungsfunktionen in allen Hidden-Layern.

Beispiele:

- `relu|relu`
- `relu|tanh`
- `relu|tanh|sigmoid`
- `relu*16|sigmoid*8`
- `relu*8,tanh*8|sigmoid*4,leaky_relu*4`

Wirkung:

- legt fest, wie aus Praeaktivierungen `z` die Aktivierungen `a` entstehen
- beeinflusst damit Lernverhalten, Sattigung, Sparsity und Gradientenfluss

### Neighbor-Operationen

Neighbor-Operationen veraendern ein Layout lokal.

Beispiele:

- `set:L1:0:tanh`
- `fill:L2:sigmoid`
- `cycle:L1:3`
- `swap:L2:0:4`

Didaktischer Sinn:

- man sieht, wie kleinste Layout-Aenderungen aussehen
- man bekommt ein Gefuehl fuer den Suchraum
- genau darauf kann spaeter Simulated Annealing aufbauen

### Datensplit

Der Datensplit in der GUI bestimmt, aus welchem Bereich das aktuell gezeigte
Analyse-Sample stammt:

- `train`
- `val`
- `test`

Wichtig:

- Das Training findet trotzdem immer auf dem Trainingssplit statt.
- Der Datensplit in der GUI ist vor allem eine Beobachtungs- und Analysewahl.

### Sample-Index

Waehlt ein einzelnes Beispiel aus dem aktuell gewaehlten Split.

Didaktischer Sinn:

- man kann ein konkretes Beispiel durch das Netz verfolgen
- man sieht, wie dasselbe Sample vor und nach dem Training reagiert

### Eigenes Sample verwenden

Nur in bestimmten Modi sinnvoll:

- `digits`: eigenes 8x8 Bild zusammensetzen
- `test_activation`: eigene rohe Inputwerte setzen

Didaktischer Sinn:

- gezielte, kontrollierte Experimente statt nur Datensatzbeispiele
- besonders wertvoll fuer Demonstrationen im Seminar

### Analyse-Ziel

Dieses Ziel wird verwendet, um den aktuell angezeigten Loss im Analysefenster
zu berechnen.

Wichtig:

- Das echte Label bleibt davon unberuehrt.
- Man kann hier absichtlich ein anderes Ziel setzen, um zu sehen, wie sich der
  Loss fuer alternative Klassen verhaelt.

### Epochen

Legt fest, wie oft das Training ueber den gesamten Trainingssplit laeuft.

Faustregel:

- mehr Epochen = laengeres Training
- mehr Epochen = oft bessere Anpassung
- zu viele Epochen koennen Overfitting beguenstigen

### Lernrate

Die Lernrate steuert die Schrittweite der Gewichtsupdates.

Faustregel:

- zu klein = langsames Lernen
- zu gross = instabiles oder springendes Lernen

Didaktisch besonders interessant:

- dieselbe Architektur kann mit schlechter Lernrate deutlich schlechter wirken

### Batch-Groesse

Bestimmt, wie viele Trainingsbeispiele pro Update gemeinsam verarbeitet werden.

Wirkung:

- kleine Batches = noisigere Updates, oft didaktisch interessant
- groessere Batches = ruhigere Updates, manchmal stabiler

### Weight-Scale

Steuert die Groessenordnung der zufaelligen Startgewichte.

Warum wichtig:

- zu kleine Startgewichte koennen Aktivierungen sehr schwach machen
- zu grosse Startgewichte koennen bestimmte Aktivierungen schnell in problematische
  Bereiche treiben

### Seed

Sorgt fuer Reproduzierbarkeit.

Damit beeinflusst der Seed:

- die Daten-Splits
- die Startgewichte
- indirekt den Trainingsverlauf

Gerade fuer ein Proseminar ist das wichtig, damit Ergebnisse spaeter erneut
erklaert und verglichen werden koennen.

## Aktivierungs-Layouts

Das Projekt unterstuetzt aktuell 1 bis 4 Hidden-Layer. Fuer jeden Layer kann
festgelegt werden, welche Aktivierungsfunktion pro Neuron verwendet wird.

### Einfache Layer-Syntax

```bash
relu|tanh
```

Bedeutung:

- Hidden-Layer 1 nutzt nur `relu`
- Hidden-Layer 2 nutzt nur `tanh`

### Drei Layer

```bash
relu|tanh|sigmoid
```

Bedeutung:

- Hidden-Layer 1 nutzt `relu`
- Hidden-Layer 2 nutzt `tanh`
- Hidden-Layer 3 nutzt `sigmoid`

### Layer-Syntax mit expliziter Neuronanzahl

```bash
relu*16|tanh*8
```

Bedeutung:

- Hidden-Layer 1 hat 16 Neuronen mit `relu`
- Hidden-Layer 2 hat 8 Neuronen mit `tanh`

### Gemischte Aktivierungen innerhalb eines Layers

```bash
relu*8,tanh*8|sigmoid*4,leaky_relu*4
```

### Voll explizite Belegung einzelner Neuronen

```bash
relu,relu,tanh,sigmoid|leaky_relu*4
```

Diese Layout-Sprache ist didaktisch wichtig, weil sie:

- einfache Basiskonfigurationen erlaubt
- gemischte Layouts erlaubt
- pro-Neuron-Layouts erlaubt
- direkt als Suchraum fuer spaetere Optimierung dient

## Neighbor-Operationen

Neighbor-Operationen veraendern ein Layout lokal. Genau das ist spaeter auch
fuer Simulated Annealing relevant.

### Aktivierung eines einzelnen Neurons setzen

```bash
set:L1:0:tanh
```

### Kompletten Layer fuellen

```bash
fill:L2:sigmoid
```

### Aktivierung eines Neurons zyklisch weiterschalten

```bash
cycle:L1:3
```

### Aktivierungen zweier Neuronen tauschen

```bash
swap:L2:0:4
```

Du kannst mehrere Neighbor-Operationen hintereinander angeben:

```bash
python main.py --benchmark wine --layout "relu*16|tanh*8" \
  --neighbor-op "set:L1:0:sigmoid" \
  --neighbor-op "cycle:L2:3"
```

## Beispiele zum Starten

### Einfacher Einstieg

```bash
python main.py
```

### GUI direkt mit `digits`

```bash
python main.py --gui --benchmark digits
```

### GUI mit dem Rechenlabor

```bash
python main.py --gui --benchmark test_activation
```

### Terminal-Experiment mit gespeichertem Plot

```bash
python main.py --benchmark wine --layout "relu*16|tanh*8" --epochs 120 --save-prefix demo/wine_run
```

### Layout-Hilfe anzeigen

```bash
python main.py --show-layout-help
```

## Was in der GUI sichtbar ist

Die GUI ist als Lernprogramm fuer Studierende gedacht, nicht nur als
Konfigurationsfenster.

Sie bietet unter anderem:

- einen gefuehrten Konfigurationsbereich
- Einsteiger-Modus und Experten-Modus
- Presets fuer schnelle Einstiege
- eine Netzwerkvisualisierung mit Neuronen und Verbindungen
- farbliche Darstellung der Aktivierungsfunktionen
- Auswahl einzelner Neuronen per Klick
- Anzeige der lokalen Rechnung in einem selektierten Hidden-Neuron
- Aktivierungskurve des selektierten Neurons mit markiertem aktuellem `z`
- Input- und Zielansicht
- eingebettete Trainingsplots
- einen echten Forward/Backward-Stepper fuer genau das sichtbare Sample
- einen Baseline-vs-Experiment-Vergleich
- Lernhilfe und Erklaertexte

Spezielle GUI-Faehigkeiten:

- `digits` kann als 8x8-Eingabe betrachtet werden
- `digits` kann in der GUI auch mit eigenen Pixelwerten manipuliert werden
- `test_activation` erlaubt einfache manuelle Inputs
- Vorhersage, echtes Ziel und Analyse-Ziel koennen sichtbar verglichen werden
- Im Expertenmodus koennen Hidden-Layer hinzugefuegt und entfernt werden

## Was im Terminal sichtbar ist

Der Terminalmodus ist bewusst nicht nur eine nackte Zahlenliste.

Er zeigt:

- Datensatz-Zusammenfassung
- ASCII-Layout des Basis-Layouts
- optional das Nachbar-Layout
- Diff zwischen Basis und Nachbar
- Vorschau auf Single-Step-Neighbors
- Trainingsergebnisse mit Loss und Accuracy

Das ist besonders hilfreich fuer:

- schnelle Vergleiche
- reproduzierbare Runs
- spaetere Automatisierung
- Seminarprotokolle und Screenshots

## Plotting

Das Projekt kann mit Matplotlib folgende Diagramme erzeugen:

- Lernkurven fuer Train-Loss und Validation-Loss
- Lernkurven fuer Train-Accuracy und Validation-Accuracy
- Layout-Heatmaps
- Vergleich von Basis- und Neighbor-Layout

Plots koennen:

- in der GUI eingebettet sichtbar sein
- im Terminalmodus als PNG gespeichert werden

Gespeicherte Dateien landen standardmaessig unter `outputs/`.

Beispiel:

```bash
python main.py --benchmark wine --save-prefix demo/wine_run
```

Dann entstehen zum Beispiel:

- `outputs/demo/wine_run_history.png`
- `outputs/demo/wine_run_layouts.png`

## Train, Validation und Test

Das Projekt trennt die Daten sauber in drei Bereiche:

- Training: zum Anpassen der Gewichte
- Validation: zur Bewertung waehrend des Trainings
- Test: fuer die abschliessende Bewertung

Warum ist das wichtig?

- Nur Training allein sagt nicht, ob das Modell generalisiert.
- Validation zeigt, ob ein Layout waehrend des Trainings gut funktioniert.
- Test ist die finale, moeglichst faire Bewertung.

Im Kontext Natural Computing ist das besonders relevant, weil spaetere
Suchverfahren typischerweise auf der Validation-Metrik optimieren sollten und
den Test-Split fuer die finale Beurteilung zurueckhalten sollten.

## Projektstruktur

Die wichtigsten Dateien dieses Projekts sind:

| Datei | Aufgabe |
| --- | --- |
| `main.py` | Programmeinstieg, Parser, interaktiver Assistent, GUI-Start, Terminal-Workflow |
| `configs.py` | zentrale Defaultwerte, erlaubte Benchmarks, Dataclasses fuer Konfiguration |
| `benchmarks.py` | Laden, Splitten und Standardisieren der Datensaetze |
| `activations.py` | Aktivierungsfunktionen, Layout-Logik, Parsing, Neighbor-Operationen |
| `model.py` | MLP, Forward-Pass, Backpropagation, Evaluation, Neuron-Inspektion |
| `trainer.py` | Epochen-Loop, Mini-Batches, Train/Val/Test-Auswertung |
| `terminal_viz.py` | ASCII-Darstellung fuer Layouts, Diffs und Trainingszusammenfassung |
| `plotting.py` | Matplotlib-Plots fuer Lernkurven und Layouts |
| `gui.py` | didaktische GUI mit Netzwerkvisualisierung und Interaktion |

## Warum kein PyTorch oder TensorFlow?

Fuer ein Proseminar ist Transparenz wichtiger als maximale Performance. Das
Projekt verzichtet deshalb bewusst auf grosse Deep-Learning-Frameworks.

Vorteile davon:

- Forward-Pass ist direkt lesbar
- Backpropagation ist direkt lesbar
- Aktivierungswechsel pro Neuron bleiben transparent
- die Verbindung zu Natural-Computing-Suchverfahren bleibt klar

## Typische Seminarfragen, die mit dem Projekt untersucht werden koennen

- Lernt ein kleines Netz mit `relu` schneller als mit `sigmoid`?
- Was passiert, wenn innerhalb eines Layers Aktivierungen gemischt werden?
- Welche Unterschiede sieht man auf `wine`, `digits` und `breast_cancer`?
- Wie veraendert ein einzelner Neighbor-Schritt das Layout?
- Welche Validation-Metrik ergibt sich fuer verschiedene Layouts?
- Wie koennte ein Simulated-Annealing-Lauf ueber diesen Layoutraum definiert
  werden?

## Erweiterungen fuer spaeter

Die aktuelle Struktur ist bewusst so gebaut, dass spaetere Proseminar- oder
Projektarbeiten darauf aufsetzen koennen.

Naheliegende Erweiterungen:

- Simulated Annealing auf Aktivierungs-Layouts
- weitere Neighbor-Typen
- Layout-Vergleich mehrerer Modelle
- Visualisierung von Backpropagation und Gradienten
- editierbare Gewichte und Biases im Lernmodus
- weitere Benchmarks
- Export von Versuchsergebnissen als CSV oder JSON

## Installation und Voraussetzungen

### Python

Empfohlen ist eine aktuelle Python-3-Version. Getestet wurde das Projekt lokal
mit:

- Python `3.13.12`

### Python-Pakete

Die benoetigten Python-Pakete stehen in `requirements.txt`.

Installation:

```bash
pip install -r requirements.txt
```

### Tkinter fuer die GUI

Die GUI benoetigt `tkinter`. Das ist in Python oft schon enthalten, auf manchen
Systemen aber nur ueber ein zusaetzliches Systempaket verfuegbar.

Typische Beispiele:

- macOS mit Homebrew-Python: `brew install python-tk@3.13`
- Ubuntu/Debian: `sudo apt install python3-tk`

Wenn du nur den Terminalmodus verwenden willst, ist `tkinter` nicht zwingend
noetig.

## Hinweise zu Matplotlib

In manchen Umgebungen ist das Standard-Cache-Verzeichnis von Matplotlib nicht
schreibbar. Dann empfiehlt sich:

```bash
mkdir -p .mplconfig
export MPLCONFIGDIR=.mplconfig
```

Die GUI setzt dieses Verzeichnis bereits selbst sinnvoll, aber fuer
Terminal-Experimente ist die obige Umgebungsvariable oft hilfreich.

## Typischer Arbeitsablauf fuer das Proseminar

Ein sinnvoller didaktischer Ablauf kann so aussehen:

1. Das Projekt mit `python main.py` starten.
2. Einen kleinen Benchmark wie `wine` waehlen.
3. Ein einfaches Layout wie `relu|relu` testen.
4. Ein gemischtes Layout ausprobieren.
5. Neighbor-Operationen anwenden und das Diff ansehen.
6. Validation- und Testwerte vergleichen.
7. Dasselbe in der GUI betrachten.
8. Im `test_activation`-Modus gezielt kleine Rechenbeispiele untersuchen.
9. Spaeter dieselben Layouts als Zustaende eines Suchraums interpretieren.

## Kurzfazit

Dieses Projekt ist ein bewusst ueberschaubarer, didaktisch orientierter
Playground fuer ein Proseminar im Bereich Natural Computing. Es kombiniert:

- erklaerbare kleine MLPs
- flexible Aktivierungs-Layouts
- Terminal- und GUI-Visualisierung
- Trainingsmetriken
- Neighbor-Strukturen als Vorbereitung fuer Suchverfahren

Damit eignet es sich sowohl fuer erste Lernschritte als auch als Grundlage fuer
eine spaetere Erweiterung in Richtung Simulated Annealing.
