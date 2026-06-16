# GUI Multi-Seed-Demomodus

## Bedienung

1. Vor dem Benchmark das **methodische Design** wählen:
   - `Blocked Layout Comparison`
   - `Robustheitsanalyse`
2. In der GUI unter **Run-Modus** `Multi Seeds` auswählen.
3. Zwischen `2` und `10` Random-Mixed-Layouts wählen. Standard sind `10`.
4. Die Sammelstufen sequenziell ausführen:
   - `Alle Baselines trainieren`
   - `SA alle bis Ende`
   - `Alle final vergleichen`
4. Oberhalb der Ergebnisansichten zwischen `Run 1 ... Run N` und `Aggregiert` wechseln.

Einzelaktionen wie `Start bewerten`, `1 Schritt`, `10 Schritte` und `Reset` wirken
nur auf den aktiven Run. Ein manuell geändertes Layout invalidiert ebenfalls nur
diesen Run.

## Aggregierte Ansichten

Kurven zeigen den Median und ein transparentes IQR-Band vom 25. bis zum
75. Perzentil. Training, SA-Verlauf, Layoutbelegung und finaler gepaarter
Vergleich besitzen echte Aggregatansichten. Datenpunkt, Neuron-Details,
Rechenschritte und SA-Timeline bleiben bewusst Run-spezifisch.

## Seeds und Interpretation

Der sichtbare `Master-Seed` erzeugt deterministisch alle benötigten Seed-Streams.
Er ist kein einzelner Run- oder Weight-Seed.

### Blocked Layout Comparison

Nur das Random-Mixed-Startlayout variiert. Alle Layouts teilen:

- Datensplit
- Online-Startgewichte
- Online-Batch-Reihenfolge
- SA-Proposal- und Acceptance-Streams
- finale Retraining-Gewichte und Retraining-Batch-Reihenfolge

Damit werden Layoutunterschiede möglichst isoliert. Die Runs sind bewusst
korreliert und dürfen statistisch nicht als vollständig unabhängige
Beobachtungen interpretiert werden.

### Robustheitsanalyse

Jeder GUI-Multi-Run besitzt getrennte Seeds für Layout, Datensplit,
Online-Gewichte, Online-Batches, SA-Proposals, SA-Acceptance sowie finales
Retraining. Damit wird die Stabilität der vollständigen Methode unter
wechselnden Zufallsbedingungen untersucht.

In beiden Designs verwenden Random-Start- und finales SA-Layout innerhalb eines
Runs dieselben Retraining-Gewichte und dieselbe Batch-Reihenfolge.

Für einen belastbaren wissenschaftlichen Layoutvergleich sollten mehrere
Blocked-Layout-Blöcke mit unterschiedlichen Master-Seeds ausgeführt werden.

## Autosave

Multi-Sessions werden nach Layouterzeugung und nach jedem bearbeiteten Run unter
`outputs/gui_multi_runs/<timestamp>_<benchmark>_<profile>/session.json`
gespeichert. Eine unterbrochene Session wird an der letzten sicheren Run-Grenze
fortgesetzt; ein teilweise ausgeführter Run startet erneut.
