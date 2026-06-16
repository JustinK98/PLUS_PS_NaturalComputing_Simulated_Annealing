# Online-Delta Simulated Annealing: vollständige methodische und mathematische Spezifikation

## 1. Untersuchungsziel

Das Projekt untersucht, ob die Aktivierungsfunktionen einzelner Hidden-Neuronen
automatisch durch Simulated Annealing (SA) ausgewählt werden können.

Die Netzwerktopologie bleibt fest. Verändert wird ausschließlich das
**Aktivierungs-Layout**

\[
L = \left(L^{(1)}, L^{(2)}, \ldots, L^{(H)}\right),
\]

wobei

\[
L^{(h)} =
\left(\phi^{(h)}_1,\phi^{(h)}_2,\ldots,\phi^{(h)}_{n_h}\right)
\]

die Aktivierungsfunktionen der \(n_h\) Neuronen im Hidden-Layer \(h\)
beschreibt.

Jede Aktivierung stammt aus der diskreten Menge

\[
\mathcal A =
\{\operatorname{ReLU},\operatorname{GELU},\operatorname{Sigmoid},
\tanh,\operatorname{Swish},\operatorname{Identity}\}.
\]

Die zentrale Forschungsfrage lautet:

> Erzeugt Online-Delta-SA ein finales Aktivierungs-Layout, das nach einem
> fairen Retraining besser generalisiert als sein zufälliges Startlayout?

Die primäre Effektgröße ist:

\[
\Delta_{\text{paired}}
=
\mathcal L_{\text{val}}(L_{\text{start}},\theta^{*}_{\text{start}})
-
\mathcal L_{\text{val}}(L_{\text{end}},\theta^{*}_{\text{end}}).
\]

Positive Werte sprechen für das finale SA-Layout.

---

## 2. Was ist ein Benchmark?

Ein Benchmark ist eine vollständig definierte, reproduzierbare
Experimentaufgabe. Er legt fest:

- Eingabefeatures
- Zielklassen
- Trainings- und Testdaten
- Netzwerktopologie
- Verlustfunktion
- Evaluationsmetriken

Die offiziellen Benchmarks sind:

| Benchmark | Eingaben | Hidden-Topologie | Output | Aufgabe |
| --- | ---: | --- | ---: | --- |
| `two_moons` | 2 | `2-8-1` | 1 Sigmoid-Neuron | binäre Klassifikation |
| `concentric_circles` | 2 | `2-8-8-1` | 1 Sigmoid-Neuron | binäre Klassifikation |
| `crossing_spirals` | 6 | `6-16-16-1` | 1 Sigmoid-Neuron | binäre Klassifikation |

Ein Datenpunkt ist ein Paar

\[
(\mathbf x_i,y_i),
\qquad
\mathbf x_i\in\mathbb R^d,\quad y_i\in\{0,1\}.
\]

Der Benchmark liefert eine feste Trainings-CSV und eine separat versionierte,
unangetastete Test-CSV.

---

## 3. Datensplit und Standardisierung

### 3.1 Aufteilung

Die bereitgestellte Trainingsmenge wird stratifiziert in Training und
Validierung aufgeteilt:

\[
\mathcal D_{\text{train-full}}
\longrightarrow
\mathcal D_{\text{train}}\cup\mathcal D_{\text{val}}.
\]

Der Testdatensatz bleibt separat:

\[
\mathcal D_{\text{test}}.
\]

Stratifiziert bedeutet, dass die Klassenanteile in Training und Validierung
möglichst erhalten bleiben.

Der `data_split_seed` bestimmt reproduzierbar, welche Beispiele in Training und
Validierung landen.

### 3.2 Rollen der Splits

- **Training:** Gewichtsupdates und Online-Delta-Batches
- **Validierung:** Konfigurationsauswahl und diagnostische Verlaufsmessung
- **Test:** ausschließlich finale Berichterstattung nach Fixierung der Methode

Testdaten dürfen weder SA-Entscheidungen noch Hyperparameter-Tuning beeinflussen.

### 3.3 Standardisierung

Für jedes Feature \(j\) werden ausschließlich auf dem Trainingssplit Mittelwert
und Standardabweichung bestimmt:

\[
\mu_j
=
\frac{1}{|\mathcal D_{\text{train}}|}
\sum_{i\in\mathcal D_{\text{train}}}x_{ij},
\]

\[
\sigma_j
=
\sqrt{
\frac{1}{|\mathcal D_{\text{train}}|}
\sum_{i\in\mathcal D_{\text{train}}}(x_{ij}-\mu_j)^2
}.
\]

Danach wird transformiert:

\[
\tilde x_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
\]

Die auf Training geschätzten \(\mu_j,\sigma_j\) werden unverändert auf
Validierung und Test angewendet. Dadurch entsteht kein Data Leakage.

---

## 4. Das neuronale Netz

### 4.1 Parameter und Layout

Das Netz besitzt Gewichte und Biases

\[
\theta =
\left\{
W^{(1)},b^{(1)},\ldots,W^{(H+1)},b^{(H+1)}
\right\}.
\]

Das Aktivierungs-Layout \(L\) ist kein kontinuierlicher Parameter, sondern ein
diskreter Suchzustand.

### 4.2 Forward Pass

Mit

\[
\mathbf a^{(0)}=\mathbf x
\]

berechnet Hidden-Layer \(h\):

\[
\mathbf z^{(h)}
=
\mathbf a^{(h-1)}W^{(h)}+\mathbf b^{(h)}.
\]

Die Aktivierung wird neuronweise angewendet:

\[
a^{(h)}_j
=
\phi^{(h)}_j\left(z^{(h)}_j\right).
\]

Anders als in einem klassischen MLP können in demselben Layer unterschiedliche
Aktivierungen verwendet werden.

Für binäre Klassifikation gilt am Output:

\[
z_{\text{out}}
=
\mathbf a^{(H)}W^{(H+1)}+b^{(H+1)},
\]

\[
\hat p
=
\sigma(z_{\text{out}})
=
\frac{1}{1+\exp(-z_{\text{out}})}.
\]

Die vorhergesagte Klasse ist:

\[
\hat y=
\begin{cases}
1,&\hat p\ge 0.5\\
0,&\hat p<0.5.
\end{cases}
\]

### 4.3 Aktivierungsfunktionen

#### ReLU

\[
\operatorname{ReLU}(z)=\max(0,z),
\qquad
\operatorname{ReLU}'(z)=\mathbb 1[z>0].
\]

#### Tanh

\[
\tanh'(z)=1-\tanh^2(z).
\]

#### Sigmoid

\[
\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)).
\]

#### Swish

\[
\operatorname{Swish}(z)=z\sigma(z),
\]

\[
\operatorname{Swish}'(z)
=
\sigma(z)+z\sigma(z)(1-\sigma(z)).
\]

#### Identity

\[
\operatorname{Identity}(z)=z,
\qquad
\operatorname{Identity}'(z)=1.
\]

#### GELU-Approximation

\[
\operatorname{GELU}(z)
\approx
\frac{z}{2}
\left[
1+\tanh
\left(
\sqrt{\frac{2}{\pi}}
(z+0.044715z^3)
\right)
\right].
\]

In der Implementierung wird die zugehörige analytische Ableitung der
tanh-basierten Approximation verwendet.

### 4.4 Verlustfunktion

Für einen Batch \(B\) mit \(m=|B|\) Beispielen wird Binary Cross-Entropy
minimiert:

\[
\mathcal L_B(\theta,L)
=
-\frac{1}{m}
\sum_{i\in B}
\left[
y_i\log(\hat p_i)
+
(1-y_i)\log(1-\hat p_i)
\right].
\]

### 4.5 Gewichtsinitialisierung

Gewichte werden Xavier-uniform initialisiert:

\[
W_{ij}^{(h)}
\sim
\mathcal U(-a_h,a_h),
\]

\[
a_h
=
s\sqrt{\frac{6}{\operatorname{fan\_in}_h+\operatorname{fan\_out}_h}},
\]

wobei \(s\) die konfigurierbare Xavier-Skalierung ist.

Alle Biases starten bei:

\[
b_j^{(h)}=0.
\]

### 4.6 Klassisches SGD-Update

Für Lernrate \(\eta\):

\[
\theta_{t+1}
=
\theta_t-\eta\nabla_{\theta}\mathcal L_B(\theta_t,L).
\]

Das Aktivierungs-Layout bleibt während eines klassischen Trainingslaufs fest.

---

## 5. Der Layout-Suchraum

Sind insgesamt

\[
N=\sum_{h=1}^{H}n_h
\]

Hidden-Neuronen vorhanden und stehen \(|\mathcal A|=6\) Aktivierungen zur
Verfügung, besitzt der vollständige Layout-Suchraum:

\[
|\mathcal S|=6^N
\]

mögliche Zustände.

Beispiele:

- `two_moons`, \(N=8\):

  \[
  6^8=1\,679\,616
  \]

- `concentric_circles`, \(N=16\):

  \[
  6^{16}\approx2.82\cdot10^{12}
  \]

- `crossing_spirals`, \(N=32\):

  \[
  6^{32}\approx7.96\cdot10^{24}
  \]

Eine vollständige Enumeration ist daher für größere Netze unpraktikabel.

---

## 6. Random-Mixed-Startlayout

Für jedes Hidden-Neuron wird unabhängig eine Aktivierung gezogen:

\[
\phi^{(h)}_j
\sim
\operatorname{Uniform}(\mathcal A).
\]

Der `layout_seed` macht diese Ziehung reproduzierbar.

Im GUI-Multi-Modus werden doppelte Startlayouts verworfen und neu gezogen,
bis die angeforderte Anzahl eindeutiger Layouts vorliegt.

---

## 7. Seed-System und methodische Designs

### 7.1 Master-Seed

Der sichtbare GUI-Regler ist ein **Master-Seed** \(s_{\text{master}}\).
Er wird nicht direkt für alle Zufallsoperationen wiederverwendet.

Stattdessen werden mittels stabiler SHA-256-basierter Ableitung getrennte Seeds
erzeugt:

\[
s_r^{(q)}
=
H(s_{\text{master}},\text{benchmark},q,\text{run-index}),
\]

wobei \(q\) den Zufallsstrom bezeichnet:

- Layout
- Datensplit
- Online-Gewichte
- Online-Batches
- SA-Proposals
- SA-Acceptance
- Retraining-Gewichte
- Retraining-Batches

Dadurch sind die Ergebnisse deterministisch reproduzierbar, ohne denselben
Pseudozufallsstrom versehentlich für verschiedene Rollen zu verwenden.

### 7.2 Blocked Layout Comparison

Ziel: den Einfluss unterschiedlicher Startlayouts möglichst isolieren.

Für Layouts \(L_1,\ldots,L_K\) innerhalb eines Blocks gilt:

\[
s^{\text{split}}_1=\cdots=s^{\text{split}}_K,
\]

\[
s^{\text{online-weight}}_1=\cdots=s^{\text{online-weight}}_K,
\]

\[
s^{\text{online-batch}}_1=\cdots=s^{\text{online-batch}}_K,
\]

\[
s^{\text{proposal}}_1=\cdots=s^{\text{proposal}}_K,
\]

\[
s^{\text{acceptance}}_1=\cdots=s^{\text{acceptance}}_K,
\]

\[
s^{\text{retrain-weight}}_1=\cdots=s^{\text{retrain-weight}}_K,
\]

\[
s^{\text{retrain-batch}}_1=\cdots=s^{\text{retrain-batch}}_K.
\]

Nur:

\[
s^{\text{layout}}_i\ne s^{\text{layout}}_j
\quad\text{für }i\ne j.
\]

Dies ist ein Common-Random-Numbers-Design. Die Layoutläufe sind absichtlich
korreliert. Ihre Unterschiede besitzen geringere Störvarianz, dürfen aber nicht
als vollständig unabhängige Beobachtungen behandelt werden.

Für belastbare Inferenz sind mehrere Blöcke mit unterschiedlichen Master-Seeds
notwendig.

### 7.3 Robustheitsanalyse

Ziel: Stabilität der vollständigen Methode unter wechselnden Zufallsbedingungen.

Für unterschiedliche Runs variieren:

- Startlayout
- Datensplit
- Online-Startgewichte
- Online-Batch-Reihenfolge
- Proposal- und Acceptance-Zufall
- Retraining-Gewichte und Retraining-Batches

Die beobachtete Varianz enthält damit Layout-, Split-, Initialisierungs-,
Trainings- und Suchvarianz.

### 7.4 Gepaarter finaler Vergleich

Unabhängig vom Design teilen Start- und Endlayout innerhalb desselben Runs:

\[
s_{\text{start}}^{\text{retrain-weight}}
=
s_{\text{end}}^{\text{retrain-weight}},
\]

\[
s_{\text{start}}^{\text{retrain-batch}}
=
s_{\text{end}}^{\text{retrain-batch}},
\]

\[
s_{\text{start}}^{\text{split}}
=
s_{\text{end}}^{\text{split}}.
\]

Dadurch wird beim finalen Vergleich möglichst nur der Layouteffekt gemessen.

---

## 8. Online-Delta-SA: Zustand

Der vollständige Online-Zustand zum Schritt \(t\) ist:

\[
S_t=
\left(
L_t,
\theta_t,
B_t,
T_t,
c_t
\right),
\]

mit:

- \(L_t\): aktuell behaltenes Aktivierungs-Layout
- \(\theta_t\): aktuelle Gewichte und Biases
- \(B_t\): aktueller Trainings-Minibatch
- \(T_t\): aktuelle Temperatur
- \(c_t\): Position des Batch-Cursors

Wichtig:

> Online-Delta-SA optimiert nicht nur ein Layout bei eingefrorenen Gewichten.
> Nach akzeptierten Layoutänderungen werden die Gewichte online weitertrainiert.

Damit verändert sich die Bewertungslandschaft während der Suche.

---

## 9. Initialisierung einer Online-SA-Session

Für einen Run:

1. Lade Benchmark mit `data_split_seed`.
2. Erzeuge Startlayout \(L_0\) mit `layout_seed`.
3. Initialisiere \(\theta_0\) mit `online_weight_seed`.
4. Erzeuge deterministische Batch-Reihenfolge mit `online_batch_seed`.
5. Initialisiere Proposal-RNG und Acceptance-RNG getrennt.
6. Setze:

   \[
   T_0>0,\quad t=0.
   \]

7. Bewerte das Startmodell auf dem ersten aktuellen Batch:

   \[
   f_0=\mathcal L_{B_0}(\theta_0,L_0).
   \]

Die vollständige Trainings- und Validierungsbewertung des Starts ist
diagnostisch. Sie entscheidet nicht über einzelne SA-Proposals.

---

## 10. Nachbarschaftsoperation

Der Hauptpfad verwendet ausschließlich `set_neuron`.

Ausgehend von \(L_t\):

1. Wähle eine Hidden-Neuron-Position

   \[
   (h,j)\sim\operatorname{Uniform}(\text{alle Hidden-Neuronen}).
   \]

2. Ziehe eine andere Aktivierung:

   \[
   \phi'\sim
   \operatorname{Uniform}
   \left(
   \mathcal A\setminus\{\phi^{(h)}_j\}
   \right).
   \]

3. Erzeuge Kandidat \(L'_t\), der sich an genau einer Position unterscheidet.

Die Netzwerktopologie und alle Gewichte bleiben dabei unverändert.

---

## 11. Der Online-Delta-Vergleich

Aktuelles Layout und Kandidat werden auf **demselben aktuellen Batch** und mit
**identischen Gewichten** verglichen.

Aktueller Loss:

\[
\ell_t
=
\mathcal L_{B_t}(\theta_t,L_t).
\]

Kandidaten-Loss:

\[
\ell'_t
=
\mathcal L_{B_t}(\theta_t,L'_t).
\]

Online-Delta:

\[
\delta_t=\ell'_t-\ell_t.
\]

Interpretation:

- \(\delta_t<0\): Kandidat ist auf demselben Batch besser.
- \(\delta_t=0\): Kandidat ist gleich gut.
- \(\delta_t>0\): Kandidat ist schlechter.

Der Vergleich ist fair bezüglich Gewichten und Datenbatch:

\[
\theta_t^{\text{current}}=\theta_t^{\text{candidate}},
\qquad
B_t^{\text{current}}=B_t^{\text{candidate}}.
\]

Zwischen aktueller und Kandidatenbewertung findet kein Training statt.

---

## 12. SA-Akzeptanzregel

Für \(\delta_t\le0\) wird der Kandidat immer akzeptiert:

\[
P(\text{accept}\mid\delta_t\le0)=1.
\]

Für \(\delta_t>0\):

\[
P(\text{accept}\mid\delta_t,T_t)
=
\exp\left(-\frac{\delta_t}{T_t}\right).
\]

Ziehe:

\[
u_t\sim\mathcal U(0,1).
\]

Akzeptiere den schlechteren Kandidaten genau dann, wenn:

\[
u_t\le
\exp\left(-\frac{\delta_t}{T_t}\right).
\]

Hohe Temperaturen erlauben häufiger schlechtere Schritte und fördern
Exploration. Niedrige Temperaturen machen die Suche zunehmend greedy.

---

## 13. Was geschieht nach Akzeptanz?

Bei Akzeptanz:

1. Das Kandidatenlayout wird behalten:

   \[
   L_{t+1}=L'_t.
   \]

2. Genau ein SGD-Update wird auf demselben Batch ausgeführt:

   \[
   \theta_{t+1}
   =
   \theta_t
   -
   \eta_{\text{online}}
   \nabla_{\theta}
   \mathcal L_{B_t}(\theta_t,L_{t+1}).
   \]

3. Der Post-Training-Batch-Loss wird gemessen.
4. Der Batch-Cursor wird weitergeschoben:

   \[
   B_t\longrightarrow B_{t+1}.
   \]

Bei Akzeptanz verändert sich daher sowohl das Layout als auch der
Gewichtszustand.

---

## 14. Was geschieht nach Ablehnung?

Bei Ablehnung:

\[
L_{t+1}=L_t,
\]

\[
\theta_{t+1}=\theta_t,
\]

\[
B_{t+1}=B_t.
\]

Es gibt:

- kein Gewichtsupdate
- keinen Batch-Fortschritt
- keine Änderung des aktuellen Layouts

Dadurch wird derselbe Batch so lange erneut verwendet, bis ein Proposal
akzeptiert wird.

---

## 15. Batch-Cursor und effektive Online-Epochen

Der Batch-Cursor verwaltet eine deterministische Permutation des
Trainingssplits. Nur akzeptierte Proposals konsumieren einen Batch.

Nach vollständigem Durchlaufen des Trainingssplits wird eine neue, durch
denselben RNG deterministisch erzeugte Permutation verwendet.

Die effektiven Online-Epochen sind:

\[
E_{\text{online}}
=
\frac{\text{Anzahl konsumierter Trainingsbeispiele}}
{|\mathcal D_{\text{train}}|}.
\]

Diese Größe misst den tatsächlichen Trainingsaufwand besser als die reine
Anzahl der SA-Proposals, weil abgelehnte Proposals kein Training verursachen.

---

## 16. Cooling

Die Temperatur bleibt jeweils für
`iterations_per_temperature` Proposal-Schritte konstant.

Für Proposal-Index \(t\) ist der Cooling-Index:

\[
k(t)
=
\left\lfloor
\frac{t}{n_{\text{iter-per-temp}}}
\right\rfloor.
\]

### Geometrisch

\[
T_k=T_0\alpha^k,
\qquad 0<\alpha<1.
\]

### Linear

\[
T_k=\max(0,T_0-dk),
\qquad d>0.
\]

### Logarithmisch

\[
T_k
=
\frac{T_0}{1+c\log(1+k)},
\qquad c>0.
\]

Die Temperatur wird nach jedem Proposal fortgeschrieben, unabhängig davon, ob
es akzeptiert oder abgelehnt wurde.

---

## 17. Diagnostische Validierung während SA

Die SA-Entscheidung verwendet ausschließlich:

\[
\delta_t
=
\mathcal L_{B_t}(\theta_t,L'_t)
-
\mathcal L_{B_t}(\theta_t,L_t).
\]

Validierungs-Loss und Validierungs-Accuracy sind nur Diagnostik.

Standardmäßig werden vollständige Train-/Validierungsdiagnosen aktualisiert,
wenn durch akzeptierte Batches eine Online-Epoche abgeschlossen wurde.

Das diagnostisch beste geerbte Modell ist:

\[
(\hat L_{\text{diag}},\hat\theta_{\text{diag}})
=
\arg\min_{(L_t,\theta_t)\text{ an Diagnosepunkten}}
\mathcal L_{\text{val}}(\theta_t,L_t).
\]

Es ist **nicht** das offizielle SA-Endergebnis, weil seine adaptive Auswahl nach
Validierungs-Loss eine andere Methode definieren würde.

---

## 18. Stopkriterien

Eine Session endet, sobald mindestens ein Stopkriterium erfüllt ist:

### Maximale Proposal-Schritte

\[
t\ge t_{\max}.
\]

### Mindesttemperatur

\[
T_t\le T_{\min}.
\]

### Optionales Online-Epochen-Ziel

Nur wenn explizit als Stopregel aktiviert:

\[
E_{\text{online}}\ge E_{\text{target}}.
\]

### Keine Nachbarn

Falls kein gültiger Nachbar mehr erzeugt werden kann.

Das offizielle finale SA-Layout ist das beim Stopzeitpunkt behaltene Layout:

\[
L_{\text{end}}=L_t.
\]

---

## 19. Finaler fairer Layoutvergleich

Nach der Suche werden mindestens zwei Layouts von Grund auf neu trainiert:

1. Random-Startlayout \(L_{\text{start}}\)
2. finales SA-Layout \(L_{\text{end}}\)

Beide erhalten:

- denselben Datensplit
- dieselben frisch initialisierten Gewichte
- dieselbe Batch-Reihenfolge
- dieselben Trainingshyperparameter

Formal:

\[
\theta_{0,\text{start}}
=
\theta_{0,\text{end}},
\]

aber die Vorwärts- und Rückwärtsrechnung unterscheidet sich aufgrund der
Layouts. Nach dem ersten Update können sich die Gewichte daher auseinander
entwickeln.

Nach vollständigem Retraining entstehen:

\[
\theta^{*}_{\text{start}}
=
\operatorname{Train}
(L_{\text{start}},s_{\text{retrain-weight}},s_{\text{retrain-batch}}),
\]

\[
\theta^{*}_{\text{end}}
=
\operatorname{Train}
(L_{\text{end}},s_{\text{retrain-weight}},s_{\text{retrain-batch}}).
\]

Die primäre gepaarte Verbesserung ist:

\[
\Delta_{\text{paired}}
=
\mathcal L_{\text{val}}
(L_{\text{start}},\theta^{*}_{\text{start}})
-
\mathcal L_{\text{val}}
(L_{\text{end}},\theta^{*}_{\text{end}}).
\]

- \(\Delta_{\text{paired}}>0\): finales SA-Layout besser
- \(\Delta_{\text{paired}}=0\): kein Unterschied
- \(\Delta_{\text{paired}}<0\): finales SA-Layout schlechter

---

## 20. Metriken

### Lokale Suchmetriken

| Metrik | Definition |
| --- | --- |
| `batch_loss_before` | \(\mathcal L_{B_t}(\theta_t,L_t)\) |
| `candidate_loss_after` | \(\mathcal L_{B_t}(\theta_t,L'_t)\) |
| `delta` | Kandidaten-Loss minus aktueller Loss |
| `post_training_batch_loss` | Loss nach akzeptiertem SGD-Update |
| `temperature` | \(T_t\) |
| `accepted` | SA-Akzeptanzentscheidung |

### Laufdiagnostik

Akzeptanzrate:

\[
r_{\text{accept}}
=
\frac{\#\text{akzeptierte Proposals}}
{\#\text{alle Proposals}}.
\]

Effektive Online-Epochen:

\[
E_{\text{online}}
=
\frac{\text{konsumierte Trainingsbeispiele}}
{|\mathcal D_{\text{train}}|}.
\]

### Finale Effektmetriken

Run-Win-Rate:

\[
\operatorname{WinRate}_{\text{run}}
=
\frac{1}{R}
\sum_{r=1}^{R}
\mathbb 1[\Delta_{\text{paired},r}>0].
\]

Median der gepaarten Verbesserungen:

\[
\operatorname{Median}
\left(
\Delta_{\text{paired},1},\ldots,\Delta_{\text{paired},R}
\right).
\]

---

## 21. Aggregierte GUI-Darstellungen

Für eine Kurve \(y_r(t)\) über Runs zeigt die GUI:

Median:

\[
\tilde y(t)
=
\operatorname{Median}_r(y_r(t)).
\]

Interquartilsabstand:

\[
\operatorname{IQR}(t)
=
\left[
Q_{0.25}(y_r(t)),
Q_{0.75}(y_r(t))
\right].
\]

Die Medianlinie zeigt die typische Entwicklung. Das transparente IQR-Band
zeigt die mittleren 50 Prozent der Run-Verteilung.

Unterschiedlich lange SA-Kurven werden nach Proposal-Schritt mit fehlenden
Tail-Werten behandelt. Validierungsverläufe werden auf effektive
Online-Epochen ausgerichtet.

---

## 22. Statistische Interpretation der Designs

### 22.1 Blocked Layout Comparison

Innerhalb eines Blocks sind Layoutläufe abhängig, weil sie gemeinsame
Zufallsbedingungen verwenden.

Sei \(\Delta_{b,i}\) der Effekt des Layouts \(i\) im Block \(b\). Eine
wissenschaftlich sinnvolle Zusammenfassung verwendet zunächst den Block:

\[
\bar\Delta_b
=
\frac{1}{K}
\sum_{i=1}^{K}\Delta_{b,i}.
\]

Erst die Blockmittel

\[
\bar\Delta_1,\ldots,\bar\Delta_B
\]

sind die primären unabhängigen Einheiten für Inferenz über mehrere
Master-Seeds.

Ein einzelner GUI-Block mit zehn Layouts ist eine starke kontrollierte
Demonstration, aber noch keine belastbare populationsweite Inferenz.

### 22.2 Robustheitsanalyse

Bei unabhängigen Runs ist die beobachtete Verteilung breiter:

\[
\operatorname{Var}(\Delta)
=
\operatorname{Var}_{\text{Layout}}
+
\operatorname{Var}_{\text{Split}}
+
\operatorname{Var}_{\text{Initialisierung}}
+
\operatorname{Var}_{\text{Batch}}
+
\operatorname{Var}_{\text{SA}}
+
\text{Interaktionen}.
\]

Sie beantwortet die Frage:

> Funktioniert die Methode unter zufällig wechselnden Gesamtbedingungen?

### 22.3 Empfohlenes Validierungsdesign

Für den stärksten wissenschaftlichen Claim:

```text
B unabhängige Blöcke
× K Random-Mixed-Startlayouts pro Block
× gepaarter Start-vs-End-Retraining-Vergleich pro Layout
```

Innerhalb jedes Blocks teilen die \(K\) Layouts alle Störzufallsströme.
Zwischen Blöcken werden Master-Seed und damit alle Störbedingungen gewechselt.

Zusätzlich sollte eine unabhängige Robustheitsanalyse berichtet werden.

---

## 23. Vollständiger Algorithmus als Pseudocode

```text
Eingabe:
    Benchmark D
    Startlayout L0
    Online-Lernrate eta_online
    Starttemperatur T0
    Cooling-Schedule
    Maximalzahl Schritte
    getrennte Seed-Streams

Lade und splitte Benchmark
Standardisiere mit Trainingsstatistik
Initialisiere Gewichte theta0
Initialisiere Batch-Cursor, Proposal-RNG, Acceptance-RNG

L <- L0
theta <- theta0
t <- 0

solange kein Stopkriterium erfüllt:
    B <- aktueller Trainingsbatch

    loss_current <- Loss(B, theta, L)

    L_candidate <- set_neuron_neighbor(L)
    loss_candidate <- Loss(B, theta, L_candidate)

    delta <- loss_candidate - loss_current
    T <- temperature(t)

    falls delta <= 0:
        accepted <- wahr
    sonst:
        u <- Uniform(0, 1)
        accepted <- u <= exp(-delta / T)

    falls accepted:
        L <- L_candidate
        theta <- theta - eta_online * Gradient_theta Loss(B, theta, L)
        Batch-Cursor weiterschieben
    sonst:
        L und theta unverändert lassen
        Batch-Cursor nicht weiterschieben

    Diagnostik gegebenenfalls aktualisieren
    t <- t + 1

L_end <- L

Trainiere L0 und L_end frisch mit gepaarten Retraining-Seeds
Berechne paired_val_loss_improvement
```

---

## 24. Konkretes Ein-Schritt-Beispiel

Angenommen:

\[
\ell_t=0.420,
\qquad
\ell'_t=0.435.
\]

Dann:

\[
\delta_t=0.435-0.420=0.015.
\]

Bei:

\[
T_t=0.030
\]

ist die Akzeptanzwahrscheinlichkeit:

\[
p_t
=
\exp\left(-\frac{0.015}{0.030}\right)
=
\exp(-0.5)
\approx0.6065.
\]

Falls:

\[
u_t=0.41,
\]

wird der schlechtere Kandidat akzeptiert, weil:

\[
0.41\le0.6065.
\]

Danach wird der Kandidat auf demselben Batch trainiert und der Batch-Cursor
weitergeschoben.

Falls stattdessen:

\[
u_t=0.80,
\]

wird der Kandidat abgelehnt. Layout, Gewichte und Batch bleiben unverändert.

---

## 25. Was Online-Delta-SA beweisen kann und was nicht

Ein positiver gepaarter Effekt zeigt:

> Das finale, durch SA gefundene Layout trainiert unter denselben
> Retraining-Bedingungen besser als sein zufälliges Startlayout.

Ein hoher Acceptance-Rate-Wert allein zeigt das nicht.

Ein sinkender geerbter Online-Loss allein zeigt das ebenfalls nicht, weil dabei
Layoutänderung und kontinuierliches Gewichtstraining vermischt sind.

Das diagnostisch beste geerbte Modell kann nützlich sein, darf aber nicht mit
dem Effekt des finalen retrainierten Layouts gleichgesetzt werden.

Korrelationen zwischen Aktivierungsanteilen und Verbesserung sind
assoziativ, nicht kausal. Unterschiedliche Aktivierungsfunktionen interagieren
mit Position, Gewichten, Datensatz und anderen Neuronen.

---

## 26. Methodische Hauptaussage des Projekts

Die sauberste Trennung lautet:

1. **Online-SA-Suche:** Findet unter kontinuierlichem Training ein finales
   diskretes Aktivierungs-Layout.
2. **Gepaarter Retraining-Vergleich:** Prüft, ob dieses Layout unabhängig vom
   geerbten Online-Gewichtszustand besser trainierbar ist.
3. **Blocked Layout Comparison:** Isoliert Layoutunterschiede unter gemeinsamen
   Störbedingungen.
4. **Robustheitsanalyse:** Prüft die Stabilität unter wechselnden
   Gesamtbedingungen.
5. **Testauswertung:** Erfolgt erst nach Fixierung von Methode und Parametern.

Nur die Kombination dieser Ebenen erlaubt eine wissenschaftlich belastbare
Interpretation des Online-Delta-SA-Ansatzes.
