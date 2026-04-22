# Presentation Script: Activation Playground

Ziel: 10-15 Minuten Low-Level-Demo ohne Code-Architektur. Die App ist die Praesentation.

Start:

```bash
python main.py gui --mode presentation --detail-level beginner --language de
```

Notfallroute:

- Wenn ein Live-Klick nicht klappt: kurz sagen, dass die Zustandslogik deterministisch ist, Track resetten und weiter.
- Wenn SA langsam wirkt: nicht warten, sondern `Bis zum Ende laufen` klicken.
- Builder nicht live zeigen. Nur am Ende erwaehnen: systematische Experimente laufen ueber Builder/CLI.

## Track A: Neural Network Preview

### Folie 1: Neuronale Netze als Lernmodell

Klick: keinen.

Sagen:

> Wir starten bewusst mit einem kleinen Netz. Das Ziel ist nicht medizinische Bestleistung, sondern Verstaendlichkeit: Wie wird aus numerischen Messwerten eine Vorhersage? Im ersten Teil ist das Aktivierungs-Layout fest. Trainiert werden spaeter nur Gewichte und Biases.

Kernaussage:

- Demo = ein festes Netzwerk verstehen.
- SA kommt erst danach als Suche ueber Aktivierungs-Layouts.

### Folie 2: Vom Sample zur Vorhersage

Klick: keinen.

Sagen:

> Hier sehen wir ein konkretes Sample aus dem Breast-Cancer-Datensatz. Das echte Ziel kommt aus dem Datensatz. Die Vorhersage ist die aktuelle Entscheidung des Modells. Das Analyse-Ziel verwenden wir fuer lokale Erklaerungen, etwa Loss und Gradienten.

Kernaussage:

- Echtes Ziel: Dataset-Label.
- Analyse-Ziel: Referenz fuer Erklaerungen.
- Vorhersage: aktuelle Modellentscheidung.

### Folie 3: Netzwerkstruktur und Belegung

Klick: optional ein Hidden-Neuron anklicken.

Sagen:

> Das Netzwerk transformiert Inputs ueber Hidden Layers zu Output-Wahrscheinlichkeiten. Jedes Hidden-Neuron hat eine Aktivierungsfunktion. Das Layout beschreibt genau diese Belegung, zum Beispiel `relu|relu`.

Kernaussage:

- Aktivierungsfunktionen sitzen in Hidden-Neuronen.
- Die Kurve gehoert zum markierten Neuron.

### Folie 4: Wie formt eine Aktivierung das Signal?

Klick: Layer/Neuron waehlen, Aktivierung wechseln.

Sagen:

> Ein Neuron bildet zuerst eine gewichtete Summe: z gleich Summe aus Gewicht mal Eingabe plus Bias. Danach formt die Aktivierungsfunktion daraus a. Wenn wir ReLU, tanh oder sigmoid wechseln, sehen wir, dass dieselbe lineare Idee anders weitergegeben wird.

Kernaussage:

- `z = sum(w_i * x_i) + b`
- `a = f(z)`
- Layout ist eine Strukturentscheidung, nicht nur Kosmetik.

### Folie 5: Konkrete Rechnung im Neuron

Klick: optional anderes Hidden-Neuron anklicken.

Sagen:

> Hier schauen wir lokal in ein Neuron. Nicht jede Zahl muss gelesen werden. Wichtig ist die Kette: staerkste Eingangssummanden plus Bias ergeben z, die Aktivierung macht daraus a, und die Ableitung zeigt lokal, wie sensitiv das Signal ist.

Kernaussage:

- Die Rechnung ist sample-spezifisch.
- Der Tracker zeigt Innenansicht eines festen Netzwerks.

### Folie 6: Was wird beim Training gelernt?

Klick: `Train 10`.

Sagen:

> Jetzt trainieren wir das Modell. Dabei aendern sich Gewichte und Biases. Das Aktivierungs-Layout bleibt gleich. Loss sollte sinken, Accuracy sollte steigen. An den Karten sieht man, ob sich die Vorhersage fuer dieses Sample veraendert.

Kernaussage:

- Gradient Descent optimiert Gewichte.
- Nicht das Aktivierungs-Layout.

### Folie 7: Uebergang zu SA

Klick: keinen.

Sagen:

> Damit bleibt die eigentliche Seminarfrage offen: Wenn Training nur Gewichte anpasst, wer entscheidet dann, welche Aktivierungsfunktion in welchem Neuron sitzt? Daraus machen wir jetzt ein diskretes Suchproblem.

Kernaussage:

- Demo beantwortet: Was passiert in einem festen Netz?
- SA beantwortet: Wie suchen wir nach einem besseren Aktivierungs-Layout?

## Track B: Simulated Annealing Preview

### Folie 1: Unsere Seminar-Frage

Klick: keinen.

Sagen:

> Simulated Annealing ist hier kein Gewichtstrainer. Wir verwenden SA, um ueber Aktivierungs-Layouts zu suchen. Ein Layout ist eine Belegung der Hidden-Neuronen mit ReLU, tanh, sigmoid oder leaky ReLU.

Kernaussage:

- SA-State = Aktivierungs-Layout.
- Bewertung entsteht durch kurzes Training dieses Layouts.

### Folie 2: State als Aktivierungs-Layout

Klick: keinen.

Sagen:

> Links sehen wir ein Startlayout, rechts ein beispielhaft anderes Layout. Die Netzwerkstruktur bleibt gleich, aber die Aktivierungsbelegung aendert sich. Genau dieser diskrete Raum wird durchsucht.

Kernaussage:

- Zustand ist nicht ein Gewichtszustand.
- Zustand ist Layout.

### Folie 3: Neighbor als lokale Layout-Aenderung

Klick: keinen.

Sagen:

> Ein Candidate entsteht durch eine kleine Operation auf dem aktuellen Layout. Hier ist es `set_neuron`: ein einzelnes Neuron bekommt eine andere Aktivierungsfunktion. Andere Operationen koennen Layer fuellen oder Neuronen tauschen.

Kernaussage:

- Neighbor = kleiner Schritt im Layout-Suchraum.
- Candidate = vorgeschlagenes Nachbarlayout.

### Folie 4: Startbewertung

Klick: `Start bewerten`.

Sagen:

> Um ein Layout zu bewerten, bauen wir ein frisches Modell mit diesem Layout, trainieren es wenige Candidate Epochs und messen auf Validation-Daten. Der Score ist hier `validation_loss`, also kleiner ist besser. Accuracy wird zusaetzlich angezeigt, ist aber hier nicht der Optimierungsscore.

Kernaussage:

- Score kommt aus kurzer Trainings- und Validation-Auswertung.
- `validation_loss` kleiner ist besser.
- Bei `validation_accuracy` wuerde man intern z. B. `1 - accuracy` minimieren.

### Folie 5: Akzeptanzentscheidung

Klick: `Ein SA-Schritt`, danach optional `10 SA-Schritte`.

Sagen:

> Jetzt sehen wir Current, Candidate und Best. Bessere Kandidaten werden akzeptiert. Schlechtere Kandidaten koennen bei hoher Temperatur trotzdem akzeptiert werden. Das ist der Unterschied zu einer rein gierigen Suche.

Kernaussage:

- Current = aktuell akzeptiertes Layout.
- Candidate = neuer Vorschlag.
- Best = bestes bisher gefundenes Layout.
- Temperatur erlaubt Exploration.

### Folie 6: Den Verlauf lesen

Klick: optional `10 SA-Schritte`.

Sagen:

> Best sollte nur besser werden oder gleich bleiben. Current kann schwanken, weil SA Exploration erlaubt. Die Temperatur sinkt im Verlauf, dadurch wird die Suche konservativer.

Kernaussage:

- Best zeigt Fortschritt.
- Current zeigt Suchpfad.
- Temperature zeigt Explorationsbereitschaft.

### Folie 7: Bis zum Ende laufen

Klick: `Bis zum Ende laufen`.

Sagen:

> Am Ende vergleichen wir Start mit Best beziehungsweise Current. Das ist kein Beweis fuer globale Optimalitaet. Es ist ein nachvollziehbarer, reproduzierbarer Suchpfad durch den Layout-Raum.

Kernaussage:

- SA findet ein gutes gefundenes Layout, nicht garantiert das globale Optimum.
- Fuer echte Aussagen braucht man mehrere Seeds und systematische Experimente.

### Folie 8: Abschluss

Klick: keinen.

Sagen:

> Der erste Teil hat gezeigt, wie ein festes neuronales Netz rechnet und trainiert. Der zweite Teil hat gezeigt, wie Simulated Annealing ueber Aktivierungs-Layouts sucht. Damit verbinden wir neuronale Netze, Aktivierungsfunktionen und diskrete Optimierung in einer kontrollierten Experimentierplattform.

Kernaussage:

- Demo: einzelnes Netz verstehen.
- Playground: Layout-Suche verstehen.
- Builder/CLI: spaeter systematisch auswerten.
