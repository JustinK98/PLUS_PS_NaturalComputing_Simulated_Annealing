# Experiment Notes

Stand: 2026-05-12

## Gemerkte Iris-Multi-Seed-Laeufe

Gemeinsames Setup:

- Benchmark: `iris`
- Topology: `4-8-3`
- Seeds: `42, 43, 44, 45, 46, 47, 48, 49`
- SA mode: `online_delta`
- Online train policy: `accepted`
- Max steps: `200`
- Start temperature: `0.5`
- Cooling: `geometric`, parameter `0.92`
- Iterations per temperature: `5`
- Final training: `150` epochs, learning rate `0.03`, batch size `32`, weight scale `0.05`
- Primary metric: `validation_loss`

### Startlayout `relu*8`

- Search mean val loss: `0.5875`
- Search mean val acc: `0.7448`
- Search mean test acc: `0.6958`
- Final `best_layout_from_sa` mean val loss: `0.2533`
- Final `best_layout_from_sa` mean test acc: `0.9333`
- `all_identity` mean val loss: `0.2363`
- `all_identity` mean test acc: `0.9333`

Interpretation:

SA verbessert klar gegenueber `relu*8`, schlaegt aber `all_identity` nicht nach Validation Loss. Test Accuracy ist gleich stark.

### Startlayout `tanh*8`

- Search mean val loss: `0.5765`
- Search mean val acc: `0.7448`
- Search mean test acc: `0.6917`
- Final `best_layout_from_sa` mean val loss: `0.2469`
- Final `best_layout_from_sa` mean test acc: `0.9333`
- `all_identity` mean val loss: `0.2363`
- `all_identity` mean test acc: `0.9333`

Interpretation:

SA verbessert klar gegenueber `tanh*8` und kommt naeher an `all_identity` heran, bleibt aber leicht schlechter nach Validation Loss.

### Startlayout `sigmoid*8`

- Search mean val loss: `0.5891`
- Search mean val acc: `0.7500`
- Search mean test acc: `0.7000`
- Final `best_layout_from_sa` mean val loss: `0.2468`
- Final `best_layout_from_sa` mean test acc: `0.9333`
- `start_layout / all_sigmoid` mean val loss: `0.8168`
- `start_layout / all_sigmoid` mean test acc: `0.7042`
- `all_identity` mean val loss: `0.2363`
- `all_identity` mean test acc: `0.9333`

Interpretation:

Dieser Lauf ist didaktisch am staerksten: SA startet von einer schwachen `sigmoid*8`-Baseline und findet nach Retraining ein Layout, das fast `all_identity` erreicht und deutlich besser ist als Startlayout, Random und mehrere homogene Baselines.

## Gemeinsame Kernaussage

Online-Delta-SA findet auf `iris` konkurrenzfaehige gemischte Aktivierungs-Layouts und verbessert schwache manuelle Startlayouts deutlich. In diesen Runs ist `all_identity` nach Validation Loss weiterhin die staerkste einfache Baseline. Die inherited SA-Modelle bleiben deutlich schwaecher als die retrainierten SA-Layouts; der Nutzen liegt aktuell also vor allem im gefundenen Layout, nicht im waehrend der Suche geerbten Modellzustand.
