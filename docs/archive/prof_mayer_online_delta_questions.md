# Rückfrage zu Online-Delta-SA: Neighborhood Operations und Startlayouts

Sehr geehrter Herr Professor Mayer,

vielen Dank nochmals für Ihre Hinweise zur letzten Präsentation. Ich würde gerne kurz sicherstellen, dass ich Ihre Empfehlung zu den Neighborhood Operations richtig verstanden habe, bevor wir den finalen Experimentaufbau festlegen.

Bei unserem Online-Delta-SA wählen wir pro Schritt ein Kandidatenlayout aus der Nachbarschaft und bewerten es mit gleichem Mini-Batch und gleichen Gewichten über das Loss-Delta.

Aktuell unterscheiden wir drei mögliche Neighborhood Operations:

- `set_neuron`: eine Aktivierungsfunktion eines einzelnen Hidden-Neurons ändern
- `swap_neurons`: zwei Aktivierungsfunktionen innerhalb eines Layers tauschen
- `fill_layer`: einen ganzen Layer auf eine Aktivierungsfunktion setzen

Wenn ich Ihre Mail richtig verstehe, würden Sie `fill_layer` für unsere kleinen Netze eher nicht verwenden, weil dadurch bei kleinen Architekturen sehr viele Aktivierungsfunktionen auf einmal geändert werden. Außerdem ist `swap_neurons` möglicherweise nicht besonders hilfreich, weil ein entsprechender Effekt auch über zwei einzelne `set_neuron`-Schritte erreichbar wäre.

Meine konkrete Rückfrage dazu wäre:

**Sollen wir für den Hauptversuch daher nur `set_neuron` als Neighborhood Operation verwenden und `swap_neurons` bzw. `fill_layer` höchstens als optionale Ablation betrachten?**

Das würde den Experimentaufbau deutlich einfacher machen. Gleichzeitig würde sich auch die Proposal-Frage vereinfachen: Wenn wir nur `set_neuron` verwenden, müssen wir keine Wahrscheinlichkeiten zwischen `set_neuron`, `swap_neurons` und `fill_layer` festlegen. Dann wäre die Auswahl nur noch zufällig über die möglichen Einzel-Neuron-Änderungen.

Falls wir doch mehrere Neighborhood Operations verwenden, müssten wir aus meiner Sicht explizit entscheiden, ob die Proposal-Verteilung

1. uniform über alle konkreten Nachbarn läuft, oder
2. erst eine Operation auswählt und dann innerhalb dieser Operation einen konkreten Nachbarn.

Der Unterschied ist, dass bei uniformer Auswahl über alle konkreten Nachbarn `set_neuron` automatisch dominiert, weil es deutlich mehr konkrete Nachbarn erzeugt als `fill_layer` oder `swap_neurons`. Das ist zwar zufällig, aber nicht operationsneutral.

Als zweiten Punkt wollte ich noch die Startlayouts klären. Bisher betrachten wir homogene Startlayouts wie `all_relu`, `all_tanh`, `all_sigmoid` oder `all_identity`. Das ist kontrollierbar und leicht interpretierbar. Gleichzeitig wäre es, wie Sie angedeutet haben, eventuell sinnvoller, direkt mit zufälligen gemischten Aktivierungs-Layouts zu starten.

Wenn wir Random-Startlayouts verwenden, würden wir `layout_seed` und `training_seed` getrennt behandeln:

- `layout_seed`: bestimmt das zufällige Startlayout
- `training_seed`: bestimmt Split, Initialisierung und Mini-Batch-Reihenfolge

Außerdem bräuchten wir für jedes Random-Startlayout eine faire Baseline: dasselbe Startlayout einmal ohne SA normal trainieren und einmal mit Online-Delta-SA suchen lassen.

Wäre aus Ihrer Sicht ein Design mit Random-Startlayouts methodisch sinnvoller als homogene Starts? Oder sollten wir beides verwenden:

- homogene Startlayouts als kontrollierte Referenz
- Random-Startlayouts als Robustheitstest

Ihre Hinweise zur Numerik bei SA würden wir zusätzlich explizit berücksichtigen. Insbesondere würden wir dokumentieren:

- Anzahl tatsächlicher SA-Schritte
- Temperaturverlauf
- Akzeptanzrate
- Loss-Delta-Verteilung
- Validation-Loss über SA-Schritte

Damit sollten wir vermeiden, dass SA effektiv nur sehr wenige sinnvolle Iterationen macht oder dass Temperatur und Delta-Skala nicht zusammenpassen.

Meine aktuelle Tendenz wäre daher:

- Hauptversuch mit `set_neuron` only
- `swap_neurons` und `fill_layer` nur als Ablation, falls überhaupt
- Random-Startlayouts zusätzlich zu homogenen Starts oder eventuell als Hauptvariante
- numerische Diagnoseplots für Temperatur, Delta und Akzeptanzverhalten

Wäre das aus Ihrer Sicht ein sinnvoller und sauberer Experimentaufbau?

Vielen Dank im Voraus. Ich hoffe, dass wir damit den endgültigen Experimentansatz für nächste Woche festlegen können.

Mit freundlichen Grüßen  
Justin Klein
