"""Strukturierte didaktische Hilfetexte fuer Qt-Workspaces und Handbuch."""

from __future__ import annotations

from html import escape


def _language(language: str) -> str:
    return "de" if language == "de" else "en"


def _wrap_html(title: str, sections: list[tuple[str, list[str]]]) -> str:
    parts = [
        "<html><head><style>"
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.45; }"
        "h2 { margin-bottom: 10px; }"
        "h3 { margin-top: 18px; margin-bottom: 6px; }"
        "ul { margin-top: 4px; }"
        "li { margin-bottom: 4px; }"
        "p { margin-top: 4px; margin-bottom: 8px; }"
        "</style></head><body>"
    ]
    parts.append(f"<h2>{escape(title)}</h2>")
    for heading, lines in sections:
        parts.append(f"<h3>{escape(heading)}</h3>")
        parts.append("<ul>")
        for line in lines:
            parts.append(f"<li>{escape(line)}</li>")
        parts.append("</ul>")
    parts.append("</body></html>")
    return "".join(parts)


TOPICS: dict[str, dict[str, object]] = {
    "benchmark": {
        "title": {
            "de": "Benchmark",
            "en": "Benchmark",
        },
        "sections": {
            "de": [
                (
                    "Was du hier festlegst",
                    [
                        "Der Benchmark bestimmt Datensatz, Eingabedarstellung, Klassenzahl und Schwierigkeit.",
                        "Offiziell genutzt werden concentric_circles, iris und crossing_spirals als CSV-Benchmarks des Basics-Teams.",
                    ],
                ),
                (
                    "Wie du das lesen solltest",
                    [
                        "Ein Benchmark ist keine einzelne Aufgabe im GUI, sondern die gesamte Spielwiese, auf der Training und Simulated Annealing bewertet werden.",
                        "Wenn du den Benchmark wechselst, aendern sich auch sinnvolle Hidden Sizes, Samples und typische Hyperparameter.",
                    ],
                ),
            ],
            "en": [
                (
                    "What this controls",
                    [
                        "The benchmark determines the dataset, input representation, number of classes, and overall difficulty.",
                        "The official suite uses concentric_circles, iris, and crossing_spirals from the Basics team's CSV files.",
                    ],
                ),
                (
                    "How to read it",
                    [
                        "A benchmark is not a single GUI task. It defines the full environment used for training and simulated annealing evaluation.",
                        "When you switch benchmarks, sensible hidden sizes, samples, and hyperparameters also change.",
                    ],
                ),
            ],
        },
    },
    "hidden_sizes": {
        "title": {"de": "Hidden Sizes", "en": "Hidden Sizes"},
        "sections": {
            "de": [
                (
                    "Bedeutung",
                    [
                        "Hidden Sizes legen fest, wie viele Neuronen in jeder versteckten Schicht existieren.",
                        "Mehr Neuronen oder Layer erhoehen Kapazitaet, machen das Netz aber schwerer lesbar.",
                    ],
                ),
                (
                    "Didaktischer Tipp",
                    [
                        "Beginne fuer Verstehen klein und erweitere erst spaeter. test_activation funktioniert besonders gut mit sehr kleinen Netzen.",
                        "Wenn du Rechnungen pro Neuron nachvollziehen willst, sind 4/3 oder 8/4 oft besser als 64/32.",
                    ],
                ),
            ],
            "en": [
                (
                    "Meaning",
                    [
                        "Hidden sizes define how many neurons exist in each hidden layer.",
                        "More neurons or layers increase capacity but also make the network harder to read.",
                    ],
                ),
                (
                    "Didactic tip",
                    [
                        "Start small for understanding and only scale up later. test_activation works best with very small networks.",
                        "If you want to follow neuron-level computations, 4/3 or 8/4 are usually more useful than 64/32.",
                    ],
                ),
            ],
        },
    },
    "layout": {
        "title": {"de": "Aktivierungs-Layout", "en": "Activation Layout"},
        "sections": {
            "de": [
                (
                    "Was das Layout beschreibt",
                    [
                        "Jedes Hidden-Neuron bekommt eine Aktivierungsfunktion wie relu, gelu, sigmoid, tanh, swish oder identity.",
                        "Das Layout bestimmt also nicht die Gewichte, sondern die Form der nichtlinearen Transformation im Netz.",
                    ],
                ),
                (
                    "Warum das fuer SA wichtig ist",
                    [
                        "Im Activation Workflow und Builder ist ein Zustand genau so ein Aktivierungs-Layout.",
                        "Simulated Annealing sucht also nicht direkt nach besseren Gewichten, sondern nach besseren Aktivierungsbelegungen.",
                    ],
                ),
            ],
            "en": [
                (
                    "What the layout describes",
                    [
                        "Each hidden neuron receives an activation function such as relu, gelu, sigmoid, tanh, swish, or identity.",
                        "The layout therefore does not define weights. It defines the nonlinear transform used inside the network.",
                    ],
                ),
                (
                    "Why this matters for SA",
                    [
                        "In Activation Workflow and Builder, one state is exactly such an activation layout.",
                        "Simulated annealing therefore searches for better activation assignments, not directly for better weights.",
                    ],
                ),
            ],
        },
    },
    "sample_selection": {
        "title": {"de": "Split und Sample", "en": "Split and Sample"},
        "sections": {
            "de": [
                (
                    "Was sichtbar ist",
                    [
                        "Der Split waehlt aus, ob das sichtbare Analyse-Sample aus train, val oder test stammt.",
                        "Der Sample-Index bestimmt genau ein Beispiel, das du durch Netz, Stepper und Neuron-Tracker verfolgst.",
                    ],
                ),
                (
                    "Wichtige Einordnung",
                    [
                        "Das sichtbare Sample ist ein Analysefenster. Training und SA bewerten nicht nur dieses eine Sample.",
                        "Die Plots und Metriken beziehen sich auf ganze Splits, waehrend rechts ein einzelnes Beispiel erklaert wird.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is visible",
                    [
                        "The split chooses whether the visible analysis sample comes from train, val, or test.",
                        "The sample index selects one exact example that you follow through the network, stepper, and neuron tracker.",
                    ],
                ),
                (
                    "Important context",
                    [
                        "The visible sample is an analysis window. Training and SA do not judge only that one sample.",
                        "Plots and metrics refer to full splits, while the right side explains one example in detail.",
                    ],
                ),
            ],
        },
    },
    "training_hyperparameters": {
        "title": {"de": "Trainingsparameter", "en": "Training Hyperparameters"},
        "sections": {
            "de": [
                (
                    "Kernparameter",
                    [
                        "Epochen geben an, wie oft der Trainingssplit komplett durchlaufen wird.",
                        "Lernrate steuert die Schrittweite der Gewichtsupdates.",
                        "Batch-Groesse legt fest, wie viele Beispiele pro Update gemeinsam verarbeitet werden.",
                        "Weight Scale beeinflusst die Groesse der Startgewichte und damit die Staerke frueher Aktivierungen.",
                    ],
                ),
                (
                    "Wie Studenten das lesen sollten",
                    [
                        "Zu grosse Lernraten fuehren oft zu unruhigen oder divergierenden Kurven.",
                        "Zu kleine Lernraten machen das Lernen sehr langsam.",
                        "Groessere Batch-Groessen glaetten oft das Training, kleinere zeigen mehr Rauschen.",
                    ],
                ),
            ],
            "en": [
                (
                    "Core parameters",
                    [
                        "Epochs define how many full passes over the training split are executed.",
                        "Learning rate controls the step size of weight updates.",
                        "Batch size sets how many examples are processed together per update.",
                        "Weight scale affects the size of the initial weights and therefore the strength of early activations.",
                    ],
                ),
                (
                    "How students should read them",
                    [
                        "Too large a learning rate often creates unstable or diverging curves.",
                        "Too small a learning rate makes learning slow.",
                        "Larger batch sizes often smooth training, while smaller ones show more noise.",
                    ],
                ),
            ],
        },
    },
    "training_plot": {
        "title": {"de": "Trainingsplot lesen", "en": "Reading the Training Plot"},
        "sections": {
            "de": [
                (
                    "Achsen und Linien",
                    [
                        "Die x-Achse zeigt Epochen.",
                        "Loss-Linien sollten langfristig fallen, Accuracy-Linien langfristig steigen.",
                        "train_* beschreibt Verhalten auf Trainingsdaten, val_* auf Validierungsdaten.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "Wenn train besser wird, val aber stagniert oder kippt, deutet das auf Overfitting hin.",
                        "Wenn beides kaum besser wird, sind Layout, Hyperparameter oder Modellkapazitaet oft unpassend.",
                    ],
                ),
            ],
            "en": [
                (
                    "Axes and lines",
                    [
                        "The x-axis shows epochs.",
                        "Loss curves should trend downward over time, accuracy curves upward.",
                        "train_* describes behavior on training data, val_* on validation data.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "If train improves but val stagnates or drops, that often indicates overfitting.",
                        "If neither improves much, the layout, hyperparameters, or model capacity may be poorly chosen.",
                    ],
                ),
            ],
        },
    },
    "network_view": {
        "title": {"de": "Netzansicht", "en": "Network View"},
        "sections": {
            "de": [
                (
                    "Was du hier siehst",
                    [
                        "Die Netzansicht ist didaktisch reduziert. Nicht jede Eingabe muss als einzelner Knoten sichtbar sein.",
                        "Bei groesseren CSV-Benchmarks zeigt das Netz nur eine sinnvolle Projektion relevanter Inputs.",
                    ],
                ),
                (
                    "Interaktion",
                    [
                        "Klicke auf Hidden-Neuronen, um lokale Rechnungen und die Aktivierungskurve zu sehen.",
                        "Fit und Zoom dienen nur zur Navigation. Das Standardbild sollte bereits lesbar sein.",
                    ],
                ),
            ],
            "en": [
                (
                    "What you see here",
                    [
                        "The network view is didactically reduced. Not every input must be shown as an explicit node.",
                        "For larger CSV benchmarks, the network shows a readable projection of relevant inputs.",
                    ],
                ),
                (
                    "Interaction",
                    [
                        "Click hidden neurons to inspect local computations and the activation curve.",
                        "Fit and zoom are navigation tools only. The default view should already be readable.",
                    ],
                ),
            ],
        },
    },
    "neuron_tracker": {
        "title": {"de": "Neuron-Tracker", "en": "Neuron Tracker"},
        "sections": {
            "de": [
                (
                    "Was erklaert wird",
                    [
                        "Der Tracker zeigt die lokale Rechnung fuer genau ein ausgewaehltes Hidden-Neuron.",
                        "Wichtig sind Bias, gewichtete Summe z, Aktivierung a, Ableitung und die staerksten Summanden.",
                    ],
                ),
                (
                    "Didaktischer Fokus",
                    [
                        "Die Rechnung ist sample-spezifisch. Dieselben Gewichte koennen fuer ein anderes Sample zu einem ganz anderen z fuehren.",
                        "Der Tracker verbindet Gewichte, Aktivierungsfunktion und sichtbares Sample zu einer konkreten mathematischen Geschichte.",
                    ],
                ),
            ],
            "en": [
                (
                    "What it explains",
                    [
                        "The tracker shows the local computation for one selected hidden neuron.",
                        "The key quantities are bias, weighted sum z, activation a, derivative, and the strongest terms in the sum.",
                    ],
                ),
                (
                    "Didactic focus",
                    [
                        "The computation is sample-specific. The same weights can produce a very different z for another sample.",
                        "The tracker connects weights, activation function, and visible sample into one concrete mathematical story.",
                    ],
                ),
            ],
        },
    },
    "activation_curve": {
        "title": {"de": "Aktivierungskurve", "en": "Activation Curve"},
        "sections": {
            "de": [
                (
                    "Was dargestellt wird",
                    [
                        "Die Kurve zeigt die Form der aktuell gewaehlten Aktivierungsfunktion.",
                        "Die markierte Stelle zeigt den aktuellen z-Wert des selektierten Neurons und den dazugehoerigen Aktivierungswert a.",
                    ],
                ),
                (
                    "Wie du das liest",
                    [
                        "ReLU ist fuer negative z null und fuer positive z linear.",
                        "tanh und sigmoid koennen saettigen. Dann aendern sich Ausgaben trotz grosser z-Aenderungen nur noch wenig.",
                        "Leaky ReLU behaelt auch fuer negative z einen kleinen Durchlass.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is shown",
                    [
                        "The curve displays the shape of the currently selected activation function.",
                        "The marker shows the current z-value of the selected neuron and the resulting activation value a.",
                    ],
                ),
                (
                    "How to read it",
                    [
                        "ReLU is zero for negative z and linear for positive z.",
                        "tanh and sigmoid can saturate. Then even large changes in z only move the output slightly.",
                        "Leaky ReLU keeps a small slope for negative z.",
                    ],
                ),
            ],
        },
    },
    "stepper": {
        "title": {"de": "Forward-/Backward-Stepper", "en": "Forward/Backward Stepper"},
        "sections": {
            "de": [
                (
                    "Zweck",
                    [
                        "Der Stepper zerlegt einen einzelnen Forward- und Backward-Pfad in kleine Schritte.",
                        "Du siehst Eingaben, Aktivierungen, Outputs, Loss und Rueckwaerts-Signale in didaktischer Reihenfolge.",
                    ],
                ),
                (
                    "Nutzen",
                    [
                        "So wird klar, dass Training nicht magisch ist: erst entsteht ein Output, dann ein Fehler, dann Rueckmeldungen fuer Gewichtsupdates.",
                        "Der Stepper ist besonders stark bei kleinen Benchmarks wie test_activation.",
                    ],
                ),
            ],
            "en": [
                (
                    "Purpose",
                    [
                        "The stepper breaks a single forward and backward path into small stages.",
                        "You see inputs, activations, outputs, loss, and backward signals in a didactic order.",
                    ],
                ),
                (
                    "Benefit",
                    [
                        "This makes it clear that training is not magic: first an output is produced, then an error, then backward signals for weight updates.",
                        "The stepper is especially useful on small benchmarks such as test_activation.",
                    ],
                ),
            ],
        },
    },
    "compare": {
        "title": {"de": "Compare / Baseline", "en": "Compare / Baseline"},
        "sections": {
            "de": [
                (
                    "Was verglichen wird",
                    [
                        "Du speicherst einen Referenzzustand als Baseline und vergleichst spaeter Layout, Prediction, Wahrscheinlichkeiten und Metriken.",
                        "So sieht man nicht nur, dass ein Modell anders ist, sondern worin genau der Unterschied besteht.",
                    ],
                ),
                (
                    "Wichtige Regel",
                    [
                        "Verglichen werden nur kompatible Modelle mit demselben Benchmark sowie passender Input- und Output-Dimension.",
                        "Die Baseline ist didaktisch stark, weil sie gute und absichtlich verschlechterte Zustaende direkt gegeneinander stellt.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is compared",
                    [
                        "You store a reference state as a baseline and later compare layout, prediction, probabilities, and metrics against it.",
                        "This shows not only that a model changed, but exactly how it changed.",
                    ],
                ),
                (
                    "Important rule",
                    [
                        "Only compatible models with the same benchmark and matching input/output dimensions can be compared.",
                        "The baseline is didactically useful because it lets you contrast good and intentionally degraded states directly.",
                    ],
                ),
            ],
        },
    },
    "objective": {
        "title": {"de": "Objective", "en": "Objective"},
        "sections": {
            "de": [
                (
                    "Was optimiert wird",
                    [
                        "Online-Delta-SA nutzt Loss als Suchsignal: gleicher Mini-Batch, gleiche Gewichte, Layoutaenderung, direkter Loss-Vergleich.",
                        "Accuracy bleibt eine Reporting-Metrik, ist aber fuer kleine Layoutaenderungen zu grob als Akzeptanzsignal.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "Loss reagiert feiner auf kleine Verbesserungen, selbst wenn die vorhergesagte Klasse noch gleich bleibt.",
                        "Nach akzeptierten Online-Delta-Schritten wird dasselbe Netz weitertrainiert; die Suche ist dadurch ein kontinuierlicher Trainingsprozess.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is optimized",
                    [
                        "Online-delta SA uses loss as the search signal: same mini-batch, same weights, layout change, direct loss comparison.",
                        "Accuracy remains a reporting metric, but it is too coarse as an acceptance signal for small layout changes.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "Loss reacts to small improvements even when the predicted class has not changed yet.",
                        "After accepted online-delta steps, the same network keeps training, so the search is a continuous training process.",
                    ],
                ),
            ],
        },
    },
    "annealing_config": {
        "title": {"de": "Annealing-Parameter", "en": "Annealing Parameters"},
        "sections": {
            "de": [
                (
                    "Kernparameter",
                    [
                        "Start Temperature steuert, wie offen der Start gegen schlechtere Kandidaten ist.",
                        "Cooling Schedule und Cooling Parameter regeln, wie schnell diese Offenheit sinkt.",
                        "Iterations per Temperature legt fest, wie viele Schritte vor jeder Abkuehlung passieren.",
                        "Max Steps und Min Temperature sind Stopregeln.",
                    ],
                ),
                (
                    "Didaktische Lesart",
                    [
                        "Hohe Temperatur bedeutet mehr Exploration, niedrige Temperatur mehr Ausnutzung.",
                        "Wenn fast nie schlechtere Kandidaten angenommen werden, ist die Suche oft zu kalt oder kuehlt zu schnell ab.",
                    ],
                ),
            ],
            "en": [
                (
                    "Core parameters",
                    [
                        "Start temperature controls how open the search is to worse candidates at the beginning.",
                        "Cooling schedule and cooling parameter define how quickly that openness decreases.",
                        "Iterations per temperature sets how many steps happen before the next cooling event.",
                        "Max steps and min temperature are stop rules.",
                    ],
                ),
                (
                    "Didactic reading",
                    [
                        "High temperature means more exploration, low temperature means more exploitation.",
                        "If worse candidates are almost never accepted, the search is often too cold or cools too aggressively.",
                    ],
                ),
            ],
        },
    },
    "neighborhood": {
        "title": {"de": "Neighborhood", "en": "Neighborhood"},
        "sections": {
            "de": [
                (
                    "Was das ist",
                    [
                        "Neighborhood Operations bestimmen, welche kleinen Layout-Aenderungen Simulated Annealing vorschlagen darf.",
                        "set_neuron aendert ein einzelnes Neuron, fill_layer setzt einen ganzen Layer, swap_neurons vertauscht Belegungen.",
                    ],
                ),
                (
                    "Warum das wichtig ist",
                    [
                        "Neighborhoods bestimmen die Suchdynamik. Zu kleine Moves erkunden langsam, zu grobe Moves machen die Suche unruhig.",
                        "Didaktisch sieht man hier sehr gut, was ein 'Nachbar' im SA-Kontext konkret bedeutet.",
                    ],
                ),
            ],
            "en": [
                (
                    "What this is",
                    [
                        "Neighborhood operations define which small layout changes simulated annealing is allowed to propose.",
                        "set_neuron changes one neuron, fill_layer rewrites a whole layer, and swap_neurons swaps assignments.",
                    ],
                ),
                (
                    "Why it matters",
                    [
                        "Neighborhoods shape the search dynamics. Too-small moves explore slowly, while overly large moves make the search noisy.",
                        "Didactically, this is where students can see what a 'neighbor' means in SA.",
                    ],
                ),
            ],
        },
    },
    "annealing_snapshots": {
        "title": {"de": "SA-Snapshshots", "en": "SA Snapshots"},
        "sections": {
            "de": [
                (
                    "Zustaende",
                    [
                        "start = Layout vor Beginn der Suche",
                        "candidate = aktuell vorgeschlagener Nachbar",
                        "current = aktuell akzeptierter Zustand",
                        "best = bestes bisher gefundenes Layout",
                        "end = Endzustand nach Abschluss",
                    ],
                ),
                (
                    "Didaktischer Sinn",
                    [
                        "Die Snapshots zeigen, dass SA nicht nur einen Endpunkt hat. Der Weg durch den Zustandsraum ist Teil der Erklaerung.",
                    ],
                ),
            ],
            "en": [
                (
                    "States",
                    [
                        "start = layout before search begins",
                        "candidate = newly proposed neighbor",
                        "current = currently accepted state",
                        "best = best layout found so far",
                        "end = final state after completion",
                    ],
                ),
                (
                    "Didactic role",
                    [
                        "The snapshots show that SA is not only about one final result. The path through the state space is part of the explanation.",
                    ],
                ),
            ],
        },
    },
    "annealing_live": {
        "title": {"de": "SA Live-Ansicht", "en": "SA Live View"},
        "sections": {
            "de": [
                (
                    "Was sichtbar ist",
                    [
                        "Die Live-Ansicht verdichtet Schrittzahl, Temperatur, Candidate-Score, Current-Score und Best-Score.",
                        "Sie ist die schnellste Uebersicht ueber den aktuellen Stand der Suche.",
                    ],
                ),
                (
                    "Wie du sie liest",
                    [
                        "Best sollte ueber die Zeit besser oder gleich gut werden.",
                        "Candidate darf stark schwanken. Gerade das macht SA interessant.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is visible",
                    [
                        "The live view summarizes step count, temperature, candidate score, current score, and best score.",
                        "It is the fastest overview of the current search state.",
                    ],
                ),
                (
                    "How to read it",
                    [
                        "Best should improve or stay stable over time.",
                        "Candidate can fluctuate strongly. That fluctuation is part of what makes SA interesting.",
                    ],
                ),
            ],
        },
    },
    "annealing_decision": {
        "title": {"de": "SA-Entscheidung", "en": "SA Decision"},
        "sections": {
            "de": [
                (
                    "Was erklaert wird",
                    [
                        "Hier wird erklaert, ob ein Kandidat besser war, als schlechterer Zustand trotzdem akzeptiert wurde oder verworfen wurde.",
                        "Das Panel verbindet Objective, Temperatur und Annahmeentscheidung direkt.",
                    ],
                ),
                (
                    "Warum das wichtig ist",
                    [
                        "Viele Missverstaendnisse bei SA kommen daher, dass schlechtere Kandidaten manchmal akzeptiert werden. Genau das soll hier sichtbar und plausibel werden.",
                    ],
                ),
            ],
            "en": [
                (
                    "What it explains",
                    [
                        "This panel explains whether a candidate was better, accepted despite being worse, or rejected.",
                        "It directly links objective, temperature, and acceptance decision.",
                    ],
                ),
                (
                    "Why it matters",
                    [
                        "Many misunderstandings about SA come from the fact that worse candidates may still be accepted. This panel makes that behavior visible and interpretable.",
                    ],
                ),
            ],
        },
    },
    "annealing_history": {
        "title": {"de": "SA-History und Plots", "en": "SA History and Plots"},
        "sections": {
            "de": [
                (
                    "Wie du die Verlaeufe liest",
                    [
                        "Score-Plots zeigen, wie Candidate, Current und Best sich ueber die Schritte entwickeln.",
                        "Temperatur sinkt typischerweise ueber die Zeit.",
                        "Akzeptanzrate zeigt, wie haeufig Schritte insgesamt akzeptiert wurden.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "Frueh sind mehr Akzeptanzen normal. Spaeter wird die Suche konservativer.",
                        "Wenn Best lange flach bleibt, kann die Suche stecken oder das Neighborhood/Training-Budget unpassend sein.",
                    ],
                ),
            ],
            "en": [
                (
                    "How to read the traces",
                    [
                        "Score plots show how candidate, current, and best evolve across steps.",
                        "Temperature usually decreases over time.",
                        "Acceptance rate shows how frequently steps were accepted overall.",
                    ],
                ),
                (
                    "Interpretation",
                    [
                        "More acceptance early on is normal. Later the search becomes more conservative.",
                        "If best stays flat for a long time, the search may be stuck or the neighborhood/training budget may be poorly chosen.",
                    ],
                ),
            ],
        },
    },
    "experiment_setup": {
        "title": {"de": "Experiment Setup", "en": "Experiment Setup"},
        "sections": {
            "de": [
                (
                    "Was hier definiert wird",
                    [
                        "Experiment-ID, Benchmark, Hidden Sizes, Run Mode und primaere Metrik bilden die Identitaet eines Builder-Experiments.",
                        "Hier legst du fest, ob ein einzelnes fixes Layout trainiert wird oder ob SA zuerst das Layout optimiert.",
                    ],
                ),
                (
                    "Didaktischer Sinn",
                    [
                        "Der Builder ist kein Einzelsample-Tool, sondern ein reproduzierbarer Vergleichsmodus fuer viele Seeds und Konfigurationen.",
                    ],
                ),
            ],
            "en": [
                (
                    "What is defined here",
                    [
                        "Experiment ID, benchmark, hidden sizes, run mode, and primary metric define the identity of a builder experiment.",
                        "This is where you decide whether one fixed layout is trained or whether SA first optimizes the layout.",
                    ],
                ),
                (
                    "Didactic role",
                    [
                        "Builder is not a single-sample tool. It is a reproducible comparison mode for many seeds and configurations.",
                    ],
                ),
            ],
        },
    },
    "seeds": {
        "title": {"de": "Seeds", "en": "Seeds"},
        "sections": {
            "de": [
                (
                    "Warum mehrere Seeds?",
                    [
                        "Ein einzelner Seed kann ein verzerrtes Bild geben. Mehrere Seeds zeigen, ob ein Ergebnis stabil oder zufallsgetrieben ist.",
                        "Initialisierung, Reihenfolge und Splits koennen Leistung messbar beeinflussen.",
                    ],
                ),
                (
                    "Wie du das interpretierst",
                    [
                        "Interessant ist nicht nur der beste Seed, sondern Mittelwert und Streuung.",
                        "Ein leicht schlechterer Mittelwert mit kleiner Streuung kann didaktisch wertvoller sein als ein gluecklicher Spitzenwert.",
                    ],
                ),
            ],
            "en": [
                (
                    "Why multiple seeds?",
                    [
                        "A single seed can give a distorted picture. Multiple seeds show whether a result is stable or mostly random.",
                        "Initialization, ordering, and splits can materially affect performance.",
                    ],
                ),
                (
                    "How to interpret it",
                    [
                        "The best seed is not the only interesting quantity. Mean and variance matter as well.",
                        "A slightly weaker mean with lower variance can be more didactically useful than one lucky peak result.",
                    ],
                ),
            ],
        },
    },
    "search_space": {
        "title": {"de": "Search Space", "en": "Search Space"},
        "sections": {
            "de": [
                (
                    "Was hier passiert",
                    [
                        "Grid Search prueft alle diskreten Kombinationen des Suchraums.",
                        "Random Search zieht reproduzierbar nur einen Teil derselben diskreten Kandidatenmenge.",
                    ],
                ),
                (
                    "Didaktischer Nutzen",
                    [
                        "So wird sichtbar, dass gute Konfigurationen nicht nur aus dem Bauch kommen, sondern systematisch verglichen werden koennen.",
                        "Der Builder bleibt dennoch CLI-first fuer groessere Suchraeume.",
                    ],
                ),
            ],
            "en": [
                (
                    "What happens here",
                    [
                        "Grid search evaluates all discrete combinations from the search space.",
                        "Random search reproducibly samples only a subset of the same discrete candidate pool.",
                    ],
                ),
                (
                    "Didactic value",
                    [
                        "This shows that good configurations are not just guessed. They can be compared systematically.",
                        "Builder still remains CLI-first for larger searches.",
                    ],
                ),
            ],
        },
    },
    "builder_storage": {
        "title": {"de": "Storage und Ergebnisdateien", "en": "Storage and Result Files"},
        "sections": {
            "de": [
                (
                    "Dateien",
                    [
                        "Der Builder speichert Manifest, Summary und einzelne Run-JSONs.",
                        "Dadurch lassen sich Ergebnisse spaeter wieder laden, analysieren und vergleichen, ohne alles neu zu rechnen.",
                    ],
                ),
                (
                    "Praxis",
                    [
                        "Fuer schwere Multi-Seed- oder Search-Runs ist der CLI-Pfad sauberer. Die GUI bleibt fuer Konfiguration und Analyse stark.",
                    ],
                ),
            ],
            "en": [
                (
                    "Files",
                    [
                        "Builder stores a manifest, a summary, and individual run JSON files.",
                        "This allows later loading, analysis, and comparison without recomputing everything.",
                    ],
                ),
                (
                    "Practice",
                    [
                        "For heavy multi-seed or search runs, the CLI path is cleaner. The GUI remains strong for configuration and analysis.",
                    ],
                ),
            ],
        },
    },
    "builder_results": {
        "title": {"de": "Builder-Analyse", "en": "Builder Analysis"},
        "sections": {
            "de": [
                (
                    "Was du auswertest",
                    [
                        "Summary, Run-Tabelle, Ranking und Per-Run-Details zeigen nicht nur den besten Run, sondern die Struktur des gesamten Experiments.",
                        "Wichtig sind Ranking, Konfigurationsanzahl, Seed-Streuung und die Unterschiede zwischen Start-, Best- und Endlayout.",
                    ],
                ),
                (
                    "Wie Studenten das lesen sollten",
                    [
                        "Eine Konfiguration ist nur dann wirklich gut, wenn sie nicht nur einmal, sondern ueber mehrere Seeds stabil wirkt.",
                    ],
                ),
            ],
            "en": [
                (
                    "What you evaluate",
                    [
                        "Summary, run table, ranking, and per-run detail show not only the best run but the structure of the full experiment.",
                        "Important signals are ranking, number of configurations, seed variance, and the differences between start, best, and end layout.",
                    ],
                ),
                (
                    "How students should read it",
                    [
                        "A configuration is only truly strong if it looks stable across multiple seeds, not just in one run.",
                    ],
                ),
            ],
        },
    },
    "recipes": {
        "title": {"de": "Rezepte", "en": "Recipes"},
        "sections": {
            "de": [
                (
                    "Wozu sie dienen",
                    [
                        "Rezepte sind kuratierte Startideen fuer erste Experimente.",
                        "Sie zeigen nicht 'die Wahrheit', sondern sinnvolle Pfade fuer Vergleich, Beobachtung und Diskussion.",
                    ],
                ),
            ],
            "en": [
                (
                    "What they are for",
                    [
                        "Recipes are curated starting ideas for first experiments.",
                        "They do not represent 'the truth'; they provide useful paths for comparison, observation, and discussion.",
                    ],
                ),
            ],
        },
    },
    "workspace_help": {
        "title": {"de": "Workspace-Hilfe", "en": "Workspace Help"},
        "sections": {
            "de": [
                (
                    "Inhalt",
                    [
                        "Dieser Reiter fasst Workflow, Plot-Lesen, Benchmark-Hinweise und typische Beobachtungsfragen fuer den aktuellen Workspace zusammen.",
                        "Nutze die kleinen Info-Buttons fuer schnelle Kontext-Hilfe und diesen Reiter fuer den groesseren Zusammenhang.",
                    ],
                ),
            ],
            "en": [
                (
                    "Content",
                    [
                        "This tab summarizes workflow, plot reading, benchmark notes, and observation questions for the current workspace.",
                        "Use the small info buttons for quick context help and this tab for the bigger picture.",
                    ],
                ),
            ],
        },
    },
}


def topic_title(topic_key: str, language: str) -> str:
    lang = _language(language)
    topic = TOPICS.get(topic_key) or TOPICS["workspace_help"]
    return topic["title"][lang]  # type: ignore[index]


def topic_html(topic_key: str, language: str) -> str:
    lang = _language(language)
    topic = TOPICS.get(topic_key) or TOPICS["workspace_help"]
    return _wrap_html(topic["title"][lang], topic["sections"][lang])  # type: ignore[index]


def _benchmark_section(benchmark: str, language: str) -> tuple[str, list[str]]:
    lang = _language(language)
    section_title = "Aktueller Benchmark" if lang == "de" else "Current Benchmark"
    benchmark_lines = {
        "de": {
            "concentric_circles": [
                "2 numerische Eingaben, 2 Klassen, offizieller Easy-Benchmark.",
                "Gut fuer erste SA-Vergleiche, weil die Topologie klein und die Entscheidung geometrisch anschaulich ist.",
            ],
            "iris": [
                "4 numerische Eingaben, 3 Klassen, offizieller Medium-Benchmark.",
                "Ein kompakter Mehrklassen-Benchmark fuer erste Vergleiche zwischen homogenen und gemischten Layouts.",
            ],
            "crossing_spirals": [
                "6 numerische Eingaben, 2 Klassen, offizieller Hard-Benchmark.",
                "Gut fuer robuste SA-Vergleiche, weil die Topologie zwei Hidden-Layer nutzt.",
            ],
            "test_activation": [
                "Sehr kleiner kuenstlicher Datensatz mit Fokus auf Rechenwegen statt Benchmark-Staerke.",
                "Ideal fuer lokale z-, a- und Ableitungsdiskussionen.",
            ],
        },
        "en": {
            "concentric_circles": [
                "2 numeric inputs, 2 classes, official easy benchmark.",
                "Useful for first SA comparisons because the topology is small and the decision boundary is geometrically intuitive.",
            ],
            "iris": [
                "4 numeric inputs, 3 classes, official medium benchmark.",
                "A compact multi-class benchmark for early comparisons between homogeneous and mixed layouts.",
            ],
            "crossing_spirals": [
                "6 numeric inputs, 2 classes, official hard benchmark.",
                "Useful for robust SA comparisons because the topology uses two hidden layers.",
            ],
            "test_activation": [
                "Very small artificial dataset focused on computation paths rather than benchmark strength.",
                "Ideal for local discussions of z, a, and derivatives.",
            ],
        },
    }
    lines = benchmark_lines[lang].get(benchmark, benchmark_lines[lang]["concentric_circles"])
    return section_title, lines


def workspace_help_html(workspace_id: str, benchmark: str, language: str) -> str:
    lang = _language(language)
    if workspace_id == "activation_workflow":
        title = "Activation Workflow"
        sections = [
            (
                "Recommended workflow" if lang == "en" else "Empfohlener Ablauf",
                [
                    "Choose a benchmark, hidden sizes, and a readable activation layout."
                    if lang == "en"
                    else "Waehle Benchmark, Hidden Sizes und ein lesbares Aktivierungs-Layout.",
                    "Inspect one sample before training so that later changes are meaningful."
                    if lang == "en"
                    else "Untersuche zuerst ein einzelnes Sample, damit spaetere Veraenderungen Sinn ergeben.",
                    "Train in small chunks and read the sample, plot, stepper, and neuron tracker together."
                    if lang == "en"
                    else "Trainiere in kleinen Schritten und lies Sample, Plot, Stepper und Neuron-Tracker gemeinsam.",
                    "Run Online Delta SA and then compare start, best, end, random, and homogeneous baselines under identical training."
                    if lang == "en"
                    else "Fuehre Online-Delta-SA aus und vergleiche danach Start, Best, End, Random und homogene Baselines unter gleichen Trainingsbedingungen.",
                ],
            ),
            _benchmark_section(benchmark, language),
            (
                "How to read the right side" if lang == "en" else "Wie du die rechte Seite liest",
                [
                    "Sample tab explains one concrete example."
                    if lang == "en"
                    else "Der Sample-Tab erklaert ein konkretes Beispiel.",
                    "Training plot explains the learning dynamics over full splits."
                    if lang == "en"
                    else "Der Trainingsplot erklaert die Lerndynamik ueber ganze Splits.",
                    "Neuron tracker and activation curve explain one local hidden computation."
                    if lang == "en"
                    else "Neuron-Tracker und Aktivierungskurve erklaeren eine lokale Hidden-Rechnung.",
                    "Final comparison contrasts SA-found layouts against start, random, and homogeneous baselines."
                    if lang == "en"
                    else "Der finale Vergleich stellt SA-Layouts gegen Start, Random und homogene Baselines.",
                ],
            ),
            (
                "Useful questions" if lang == "en" else "Nutzbare Beobachtungsfragen",
                [
                    "Which activation makes this sample easier or harder to classify?"
                    if lang == "en"
                    else "Welche Aktivierung macht dieses Sample leichter oder schwerer klassifizierbar?",
                    "Do changes in the plot also show up in the prediction of the visible sample?"
                    if lang == "en"
                    else "Spiegeln sich Plot-Veraenderungen auch in der Prediction des sichtbaren Samples?",
                    "Which inputs dominate the selected hidden neuron?"
                    if lang == "en"
                    else "Welche Inputs dominieren das selektierte Hidden-Neuron?",
                ],
            ),
        ]
        return _wrap_html(title, sections)
    title = "Experiment Builder" if lang == "en" else "Experiment Builder"
    sections = [
        (
            "Recommended workflow" if lang == "en" else "Empfohlener Ablauf",
            [
                "Configure one clean experiment definition before using search."
                if lang == "en"
                else "Baue erst eine saubere Experimentdefinition, bevor du Suche aktivierst.",
                "Use several seeds when you want robust conclusions."
                if lang == "en"
                else "Nutze mehrere Seeds, wenn du robuste Aussagen willst.",
                "Load and compare stored results instead of relying on one run."
                if lang == "en"
                else "Lade und vergleiche gespeicherte Resultate statt dich auf einen einzelnen Run zu verlassen.",
            ],
        ),
        _benchmark_section(benchmark, language),
        (
            "How to read builder results" if lang == "en" else "Wie du Builder-Resultate liest",
            [
                "The summary describes the whole experiment."
                if lang == "en"
                else "Die Summary beschreibt das gesamte Experiment.",
                "The run table contains concrete executions for one configuration and one seed."
                if lang == "en"
                else "Die Run-Tabelle enthaelt konkrete Ausfuehrungen fuer jeweils eine Konfiguration und einen Seed.",
                "Ranking, mean, and variance matter more than one lucky best run."
                if lang == "en"
                else "Ranking, Mittelwert und Streuung sind wichtiger als ein einzelner Gluecksrun.",
            ],
        ),
        (
            "Practical note" if lang == "en" else "Praxis-Hinweis",
            [
                "Use the GUI for configuration and analysis, but prefer the CLI for heavy batch workloads."
                if lang == "en"
                else "Nutze die GUI fuer Konfiguration und Analyse, aber fuer schwere Batch-Laeufe bevorzugt die CLI.",
            ],
        ),
    ]
    return _wrap_html(title, sections)


def program_handbook_html(language: str) -> str:
    lang = _language(language)
    title = "Activation Playground Guide" if lang == "en" else "Anleitung zum Activation Playground"
    sections = [
        (
            "What this program is" if lang == "en" else "Was dieses Programm ist",
            [
                "A didactic environment for small neural networks with editable activation layouts."
                if lang == "en"
                else "Eine didaktische Experimentierplattform fuer kleine neuronale Netze mit editierbaren Aktivierungs-Layouts.",
                "It connects benchmarks, activation functions, training, and simulated annealing in one coherent interface."
                if lang == "en"
                else "Sie verbindet Benchmarks, Aktivierungsfunktionen, Training und Simulated Annealing in einer zusammenhaengenden Oberflaeche.",
            ],
        ),
        (
            "Main workspaces" if lang == "en" else "Zentrale Arbeitsmodi",
            [
                "Activation Workflow connects layout choice, manual training, simulated annealing, and final layout comparison."
                if lang == "en"
                else "Activation Workflow verbindet Layoutwahl, manuelles Training, Simulated Annealing und finalen Layout-Vergleich.",
                "Experiment Builder is for reproducible multi-seed experiments and later analysis."
                if lang == "en"
                else "Experiment Builder ist fuer reproduzierbare Multi-Seed-Experimente und spaetere Analyse gedacht.",
            ],
        ),
        (
            "Main idea" if lang == "en" else "Kernidee",
            [
                "Weights learn during training, but the activation layout defines how hidden neurons transform signals."
                if lang == "en"
                else "Gewichte werden im Training gelernt, aber das Aktivierungs-Layout bestimmt, wie Hidden-Neuronen Signale transformieren.",
                "Simulated annealing searches over these layouts. The default Online Delta mode measures an immediate mini-batch loss change before training accepted states further."
                if lang == "en"
                else "Simulated Annealing sucht ueber diese Layouts. Der Standardmodus Online Delta misst zuerst die direkte Mini-Batch-Loss-Aenderung und trainiert akzeptierte Zustaende danach weiter.",
            ],
        ),
        (
            "How to work well with the app" if lang == "en" else "Wie du mit dem Programm gut arbeitest",
            [
                "Start with a benchmark and a small readable layout."
                if lang == "en"
                else "Starte mit einem Benchmark und einem kleinen, lesbaren Layout.",
                "Inspect one sample before changing anything."
                if lang == "en"
                else "Untersuche erst ein einzelnes Sample, bevor du etwas veraenderst.",
                "Change one thing at a time and compare against a baseline."
                if lang == "en"
                else "Veraendere immer nur eine Sache und vergleiche gegen eine Baseline.",
                "Use Builder and CLI when you want evidence across seeds."
                if lang == "en"
                else "Nutze Builder und CLI, wenn du Aussagen ueber mehrere Seeds absichern willst.",
            ],
        ),
        (
            "How to read the interface" if lang == "en" else "Wie du die Oberflaeche liest",
            [
                "Left side = control strip for benchmark, layout, sample, training, or annealing settings."
                if lang == "en"
                else "Linke Seite = Steuerleiste fuer Benchmark, Layout, Sample, Training oder Annealing-Parameter.",
                "Right side = live network, plots, sample view, neuron tracker, stepper, comparison, and builder analysis."
                if lang == "en"
                else "Rechte Seite = Live-Netz, Plots, Sample-Ansicht, Neuron-Tracker, Stepper, Vergleich und Builder-Analyse.",
                "Use the small ? buttons for local explanations and the help tabs for the bigger picture."
                if lang == "en"
                else "Nutze die kleinen ?-Buttons fuer lokale Erklaerungen und die Hilfe-Reiter fuer den groesseren Zusammenhang.",
            ],
        ),
        (
            "First experiments" if lang == "en" else "Erste sinnvolle Experimente",
            [
                "Compare relu vs tanh on iris."
                if lang == "en"
                else "Vergleiche relu vs tanh auf iris.",
                "Use concentric_circles for the first complete SA workflow."
                if lang == "en"
                else "Nutze concentric_circles fuer den ersten vollstaendigen SA-Workflow.",
                "Use test_activation to read local neuron computations without distraction."
                if lang == "en"
                else "Nutze test_activation, um lokale Neuron-Rechnungen ohne Ablenkung zu lesen.",
                "Use the SA Delta and SA History tabs to study why SA accepts some worse candidates early in the run."
                if lang == "en"
                else "Nutze SA Delta und SA History, um zu verstehen, warum SA frueh im Lauf auch schlechtere Kandidaten akzeptiert.",
            ],
        ),
    ]
    return _wrap_html(title, sections)
