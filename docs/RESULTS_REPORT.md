# Activation Playground: aktueller Ergebnisstand

Stand: 2026-05-04

## Forschungsfrage

Das Projekt untersucht, wie sich die Verteilung von Aktivierungsfunktionen in kleinen Feedforward-Netzen mit Simulated Annealing optimieren lässt. Wichtig ist die Trennung der Rollen:

- Das normale Training optimiert die Gewichte eines festen Netzes.
- Simulated Annealing sucht über diskrete Aktivierungs-Layouts.
- Ein Layout wird bewertet, indem das zugehörige Netz kurz trainiert und auf Validation-Daten gemessen wird.
- Die finale Aussage entsteht erst, wenn ausgewählte Layouts unter gleichen Trainingsbedingungen erneut trainiert und verglichen werden.

Der Gesamtworkflow lautet:

`CSV Benchmark laden -> Layout wählen -> optional SA suchen -> finale Layouts trainieren -> Layouts vergleichen -> Ergebnisse berichten`

## Benchmark-Basis

Die aktuelle Codebasis nutzt die offiziellen CSV-Benchmarks der Basics-Gruppe:

| Benchmark | Schwierigkeit | Topologie | Training |
|---|---:|---:|---:|
| `concentric_circles` | easy | `2-8-1` | 100 epochs |
| `iris` | medium | `4-8-3` | 150 epochs |
| `crossing_spirals` | hard | `6-16-16-1` | 250 epochs |

Der Suchraum verwendet den offiziellen Aktivierungsfunktionssatz:

`relu`, `gelu`, `sigmoid`, `tanh`, `swish`, `identity`

## Aktuelle Report-Artefakte

Die konsolidierten Tabellen und Plots werden mit folgendem Befehl erzeugt:

```bash
python main.py experiment report --output-dir outputs/report_assets
```

Erzeugte Artefakte:

- `outputs/report_assets/benchmark_overview.csv`
- `outputs/report_assets/layout_grid_summary.csv`
- `outputs/report_assets/sa_summary.csv`
- `outputs/report_assets/neighborhood_summary.csv`
- `outputs/report_assets/layout_grid_best_accuracy.png`
- `outputs/report_assets/sa_vs_grid_accuracy.png`
- `outputs/report_assets/neighborhood_comparison.png`
- `outputs/report_assets/training_curves_best_layouts.png`

## Layout-Grid-Ergebnisse

Das Layout-Grid dient als starke, reproduzierbare Baseline. Es trainiert mehrere Layout-Kandidaten unter gleichen Bedingungen und rankt primär nach `mean_val_loss`.

| Benchmark | Bestes Grid-Layout | Mean Val Loss | Mean Val Acc | Mean Test Acc |
|---|---|---:|---:|---:|
| `concentric_circles` | `all_relu` (`relu*8`) | 0.4302 | 0.9104 | 0.8967 |
| `iris` | `all_identity` (`identity*8`) | 0.2369 | 0.9028 | 0.9333 |
| `crossing_spirals` | `all_swish` (`swish*16\|swish*16`) | 0.6925 | 0.5188 | 0.5233 |

Interpretation:

- `concentric_circles` ist als Demo-Benchmark gut geeignet, weil das Netz sichtbar lernt und klare Accuracy-Zuwächse zeigt.
- `iris` bestätigt die Mehrklassenfähigkeit des Setups, wirkt aber empfindlich gegenüber Layout und Seed.
- `crossing_spirals` ist im aktuellen Setup noch nicht gelöst; die Ergebnisse liegen nahe Zufallsniveau.

## Simulated-Annealing-Ergebnisse

Die SA-Runs verwenden aktuell ein kleines Demo-Budget. Die Suchphase bewertet Kandidaten mit kurzen Trainingsläufen. Danach wird das gefundene `best_layout_from_sa` mit den offiziellen Epochen final trainiert.

| Benchmark | Annealing Score | Search Val Acc | Final Val Acc | Final Test Acc |
|---|---:|---:|---:|---:|
| `concentric_circles` | 0.6919 | 0.5729 | 0.9104 | 0.8967 |
| `iris` | 1.0674 | 0.7222 | 0.8750 | 0.9111 |
| `crossing_spirals` | 0.6931 | 0.4958 | 0.5104 | 0.5100 |

Interpretation:

- Auf `concentric_circles` findet SA in diesem Zwischenstand ein starkes Layout, das nach finalem Training mit der Grid-Baseline gleichzieht.
- Auf `iris` findet SA ein brauchbares Layout, bleibt aber hinter der besten Grid-Baseline zurück.
- Auf `crossing_spirals` reicht der aktuelle Trainings- und Suchaufbau noch nicht aus. Hier muss zuerst das Basistraining verbessert werden, bevor SA sinnvoll bewertet werden kann.

Wichtig für Präsentation und Bericht: Die Search-Metriken und finalen Metriken sind nicht dasselbe. Der `annealing_score` bewertet Kandidaten während der Suche. Die finale Aussage entsteht über das erneute Training des gefundenen Layouts.

## Neighborhood-Vergleich

Die Neighborhood-Experimente vergleichen, welche Änderungsoperatoren bei gleichem Budget bessere Layouts finden.

| Benchmark | Bestes Neighborhood-Set | Mean Val Loss | Mean Val Acc | Mean Test Acc |
|---|---|---:|---:|---:|
| `concentric_circles` | `set_neuron_swap_neurons` | 0.6910 | 0.6088 | 0.6190 |
| `iris` | `fill_layer` | 1.0388 | 0.7583 | 0.6933 |
| `crossing_spirals` | `fill_layer` | 0.6930 | 0.5225 | 0.5010 |

Diese Ergebnisse sind als Suchphasenvergleich zu lesen, nicht als finaler Layout-Test. Sie zeigen aktuell vor allem:

- lokale Änderungen einzelner Neuronen sind erklärbar, aber mit kleinem Budget nicht automatisch stark;
- `fill_layer` ist für kleine Netze teilweise konkurrenzfähig, weil es größere strukturelle Sprünge erlaubt;
- mehr Operatoren sind nicht automatisch besser, wenn das Bewertungsbudget klein ist.

## Demo-Story

Für eine stabile Demonstration eignet sich folgende Reihenfolge:

1. `concentric_circles`: zeigen, dass ein kleines Netz mit geeignetem Layout klar lernt.
2. Manuelles Layouttraining: erklären, dass Gewichte gelernt werden, während das Layout fix bleibt.
3. Simulated Annealing: erklären, dass SA Layouts sucht und Kandidaten über Validation Loss bewertet.
4. Finaler Vergleich: Startlayout, SA-Best und Grid-Baseline unter gleichen Bedingungen vergleichen.
5. `iris`: zeigen, dass der Workflow auch für Mehrklassenprobleme funktioniert.
6. `crossing_spirals`: als ehrlichen Hard-Case zeigen, bei dem weitere Tuning-Arbeit nötig ist.

## Kritische Einordnung

Der aktuelle Stand ist für Bericht und Demo belastbar als Zwischenstand, aber noch keine finale Optimierungsaussage.

Stärken:

- offizielle CSV-Benchmarks sind integriert;
- Aktivierungsfunktionen entsprechen dem offiziellen Set;
- Layout-Grid, SA und Neighborhood-Vergleich sind reproduzierbar ausführbar;
- Ergebnisartefakte werden konsolidiert reportfähig ausgegeben.

Offene Punkte:

- SA-Parameter sind noch nicht systematisch getunt;
- `crossing_spirals` braucht zuerst besseres Basistraining;
- Neighborhood-Vergleiche sollten zusätzlich mit final trainierten Bestlayouts berichtet werden;
- alle Aussagen sollten über mehrere Seeds berichtet werden, nicht über Einzelruns.

## Nächste Tuning-Runde

Die nächste sinnvolle Experimentrunde sollte nicht breit alles erhöhen, sondern gezielt vorgehen:

- zuerst `concentric_circles` und `iris`;
- dann erst `crossing_spirals`;
- Suchparameter:
  - `candidate_epochs`: 10, 20, 40
  - `max_steps`: 20, 50, 100
  - `start_temperature`: 0.5, 1.0, 1.5, 2.0
  - `cooling_parameter`: 0.85, 0.9, 0.95
  - Neighborhoods: `set_neuron`, `fill_layer`, `set_neuron+swap_neurons`, `all_operations`

Testdaten bleiben für die finale Bewertung reserviert. Ranking und Tuning erfolgen über Validation-Metriken.
