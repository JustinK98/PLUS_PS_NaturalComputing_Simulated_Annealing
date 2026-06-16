# Ergebnisbericht: Mayer-korrigiertes Online-Delta-SA

## Status

Aktives Evaluationsprofil: `mayer_corrected_20260604`

Quellartefakt:

```text
outputs/hyperparameter_tuning/20260602_122626_mayer-corrected
```

Der fruehere Lauf `tuned_20260530` bleibt nur als
`legacy_methodology_v1` archiviert und darf nicht fuer neue Aussagen verwendet
werden.

Die unten dokumentierten Kennzahlen stammen noch aus dem historischen
Confirmation-Artefakt vor der Trennung aller Zufallsstroeme. Die fixierten
Hyperparameter bleiben aktiv; belastbare neue Ergebniswerte muessen mit dem
neuen Design `10 Layouts x 3 Replikate` erneut bestaetigt werden.

## Methode

Alle drei Basics-Benchmarks nutzen die fixe offizielle Topologie, Binary Cross
Entropy, SGD, Xavier/Glorot-Initialisierung und zufaellige gemischte
Startlayouts. Online-Delta-SA verwendet nur `set_neuron`: pro Kandidat wird
genau ein Hidden-Neuron gleichverteilt ausgewaehlt und auf eine andere der
sechs Basics-Aktivierungen gesetzt.

Die SA-Entscheidung basiert auf dem Batch-Loss-Delta mit denselben Gewichten
und demselben Mini-Batch:

```text
delta = candidate_batch_loss - current_batch_loss
```

Bei `delta <= 0` wird akzeptiert. Bei `delta > 0` wird mit
`exp(-delta / temperature)` akzeptiert. Erst nach Akzeptanz wird das behaltene
Layout auf diesem Mini-Batch trainiert. Rejections trainieren nicht.

Die finale Bewertung vergleicht gepaart:

- `same_random_start_retrained`: dasselbe zufaellige Startlayout, neu trainiert
- `end_layout_from_sa_retrained`: finales SA-Layout, neu trainiert
- `best_online_delta_value`: bestes geerbtes Online-Delta-Modell nur diagnostisch

Neue Confirmation-Runs verwenden `10` zufaellige Layouts mit jeweils `3`
unabhaengigen Replikaten. Die Zufallsstroeme fuer Layout, Gewichte,
Mini-Batches, SA-Proposals und SA-Akzeptanz sind getrennt. Random-Start und
finales SA-Layout werden mit identischem Split, identischen
Retraining-Startgewichten und identischer Batch-Reihenfolge verglichen.

Primaer zaehlt Validation-Loss. Testmetriken werden nur final berichtet.

## Hauptergebnisse

| Benchmark | Random Start Val Loss | End SA Layout Val Loss | Gepaarte Verbesserung | Win Rate | End SA Test Acc | Urteil |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `two_moons` | 0.0907 | 0.0885 | +0.0022 | 0.50 | 0.9895 | schwach positiv |
| `concentric_circles` | 0.0296 | 0.0282 | +0.0014 | 0.47 | 0.9868 | praktisch neutral |
| `crossing_spirals` | 0.6623 | 0.6890 | -0.0268 | 0.47 | 0.7392 | negativ |

Interpretation:

- `two_moons`: Online-Delta verbessert den gepaarten Random-Start im Mittel
  leicht. Der Effekt ist klein.
- `concentric_circles`: Der Benchmark ist nahezu gesaettigt. Online-Delta
  liefert eine kleine mittlere Loss-Verbesserung, aber keine starke praktische
  Ueberlegenheit.
- `crossing_spirals`: Die SGD-Trainingskonfiguration ist lernfaehig, aber
  Online-Delta verschlechtert das gepaarte Random-Startlayout im Mittel. Fuer
  diesen Benchmark gibt es mit diesem Setup keine positive SA-Aussage.

## Fixierte Parameter

| Benchmark | Training LR | Epochs | Batch | Weight Scale | SA T0 | Cooling | Iter/T | Max Steps | Online LR | Online Batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `two_moons` | 0.2 | 400 | 64 | 2.0 | 0.02424 | 0.9990 | 10 | 12500 | 0.2 | 128 |
| `concentric_circles` | 0.02 | 400 | 16 | 0.5 | 0.00390 | 0.9975 | 25 | 50000 | 0.04 | 32 |
| `crossing_spirals` | 0.1 | 1500 | 8 | 0.75 | 0.00246 | 0.9900 | 5 | 2500 | 0.2 | 64 |

## SA-Gesundheit

| Benchmark | Acceptance Rate | Effective Online Epochs | Bewertung |
| --- | ---: | ---: | --- |
| `two_moons` | 0.663 | 1657.6 | viele akzeptierte Vorschlaege, lange Online-Trainingsphase |
| `concentric_circles` | 0.482 | 1108.2 | gesunde Akzeptanz, aber Benchmark bereits sehr leicht |
| `crossing_spirals` | 0.586 | 134.4 | Akzeptanz gesund, aber Layoutsuche hilft nicht stabil |

Eine gesunde Acceptance Rate allein reicht nicht. Entscheidend ist, ob das
finale SA-Layout nach sauberem Retraining besser ist als dasselbe
Random-Startlayout.

## Plots

### two_moons

![two_moons validation progress](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/two_moons/aggregate/online_delta_validation_progress_mean.png)

![two_moons fitness](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/two_moons/aggregate/online_delta_fitness_mean.png)

![two_moons activation counts](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/two_moons/aggregate/activation_counts_end.png)

two_moons zeigt einen sinkenden mittleren Validation-Verlauf und eine kleine
mittlere Verbesserung des finalen SA-Layouts gegenueber dem gepaarten
Random-Start. Der Effekt ist sichtbar, aber klein.

### concentric_circles

![concentric_circles validation progress](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/concentric_circles/aggregate/online_delta_validation_progress_mean.png)

![concentric_circles fitness](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/concentric_circles/aggregate/online_delta_fitness_mean.png)

![concentric_circles activation counts](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/concentric_circles/aggregate/activation_counts_end.png)

concentric_circles ist bereits mit normalen Layouts sehr gut loesbar. Daher ist
die Online-Delta-Verbesserung klein und schwer von Run-Varianz zu trennen.

### crossing_spirals

![crossing_spirals validation progress](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/crossing_spirals/aggregate/online_delta_validation_progress_mean.png)

![crossing_spirals fitness](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/crossing_spirals/aggregate/online_delta_fitness_mean.png)

![crossing_spirals activation counts](../outputs/hyperparameter_tuning/20260602_122626_mayer-corrected/confirmation/crossing_spirals/aggregate/activation_counts_end.png)

crossing_spirals bleibt der kritische Benchmark. Die Validation- und
Retraining-Ergebnisse sprechen dagegen, dass Online-Delta in der aktuellen
fixen Topologie und mit SGD robust bessere Aktivierungsverteilungen findet.

## Metriken

| Metrik | Bedeutung |
| --- | --- |
| `batch_loss_before` | BCE des aktuellen Layouts auf dem aktuellen Mini-Batch |
| `candidate_loss_after` | BCE des Kandidaten mit denselben Gewichten und demselben Mini-Batch |
| `delta` | Kandidaten-Loss minus aktueller Loss; entscheidet die SA-Akzeptanz |
| `post_training_batch_loss` | BCE nach Training eines akzeptierten Kandidaten auf einem Mini-Batch |
| `trained_mini_batch_updates` | Anzahl akzeptierter Schritte, die ein SGD-Update ausgeloest haben |
| `effective_online_epochs` | verbrauchte Trainingsbeispiele geteilt durch Trainingsset-Groesse |
| `val_loss` | BCE auf Validation-Split; primaere Auswahlmetrik |
| `val_accuracy` | Accuracy auf Validation-Split; sekundaere Diagnose |
| `test_loss` | BCE auf Test-Split; nur finale Berichtsmetrik |
| `test_accuracy` | Accuracy auf Test-Split; nur finale Berichtsmetrik |
| `acceptance_rate` | akzeptierte Vorschlaege geteilt durch alle Vorschlaege |

## Schlussfolgerung

Das Projekt bleibt methodisch auf dem einfachen Online-Delta-SA-Ansatz:
zufaelliges gemischtes Startlayout, `set_neuron` only, viele Runs, Progress
Graph und AF-Statistik.

Die Ergebnisse sind wissenschaftlich brauchbar, aber nicht euphorisch:
Online-Delta zeigt auf zwei einfachen Benchmarks kleine positive Tendenzen und
auf `crossing_spirals` ein negatives Ergebnis. Der richtige Bericht ist daher:
Der Ansatz ist implementiert und messbar, aber unter der fixen Basics-Topologie
und mit SGD nicht durchgehend besser als gepaarte Random-Startlayouts.
