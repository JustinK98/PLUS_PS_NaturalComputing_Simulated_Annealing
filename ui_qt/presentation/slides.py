"""Feste Story-Struktur fuer den gefuehrten Praesentationsmodus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresentationSlide:
    track_id: str
    visual: str
    action: str | None
    right_panel: str
    title_de: str
    title_en: str
    question_de: str
    question_en: str
    bullets_de: tuple[str, ...]
    bullets_en: tuple[str, ...]
    formula: str = ""

    def title(self, language: str) -> str:
        return self.title_en if language == "en" else self.title_de

    def question(self, language: str) -> str:
        return self.question_en if language == "en" else self.question_de

    def bullets(self, language: str) -> tuple[str, ...]:
        return self.bullets_en if language == "en" else self.bullets_de


TRACK_TITLES = {
    "neural_network": {
        "de": "Neural Network Preview",
        "en": "Neural Network Preview",
    },
    "simulated_annealing": {
        "de": "Simulated Annealing Preview",
        "en": "Simulated Annealing Preview",
    },
}


TRACKS: dict[str, tuple[PresentationSlide, ...]] = {
    "neural_network": (
        PresentationSlide(
            track_id="neural_network",
            visual="intro_problem",
            action=None,
            right_panel="none",
            title_de="Neuronale Netze als Lernmodell",
            title_en="Neural Networks as Learning Models",
            question_de="Welche Aufgabe betrachten wir zuerst?",
            question_en="Which task do we look at first?",
            bullets_de=(
                "Ziel dieses ersten Blocks: verstehen, wie ein kleines neuronales Netz aus Messwerten eine Entscheidung berechnet.",
                "Breast Cancer ist dafuer ein uebersichtlicher binaerer Klassifikationsdatensatz.",
                "Das Netz bekommt numerische Zell-Messwerte und gibt Wahrscheinlichkeiten fuer zwei Klassen aus.",
                "Wichtig: Hier trainieren wir Gewichte. Die Wahl der Aktivierungsfunktionen wird erst im SA-Teil zur Suchfrage.",
            ),
            bullets_en=(
                "Goal of this first block: understand how a small neural network turns measurements into a decision.",
                "Breast Cancer is a compact binary classification dataset for that purpose.",
                "The network receives numeric cell measurements and outputs probabilities for two classes.",
                "Important: here we train weights. The activation choice becomes the search question in the SA part.",
            ),
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="sample_with_definitions",
            action=None,
            right_panel="definitions",
            title_de="Vom Sample zur Vorhersage",
            title_en="From Sample to Prediction",
            question_de="Was bedeuten Ziel, Analyse-Ziel und Vorhersage?",
            question_en="What do target, analysis target, and prediction mean?",
            bullets_de=(
                "Echtes Ziel: das Label aus dem Datensatz.",
                "Analyse-Ziel: die Klasse, gegen die wir lokale Loss-/Gradienten-Erklaerungen aufbauen.",
                "Vorhersage: die aktuelle Entscheidung des noch kleinen Modells.",
                "Die Sample-Ansicht ist ein Blickfenster, nicht der gesamte Trainingsprozess.",
            ),
            bullets_en=(
                "True target: the label from the dataset.",
                "Analysis target: the class used for local loss/gradient explanations.",
                "Prediction: the current decision of the small model.",
                "The sample view is a window into one example, not the whole training process.",
            ),
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="network_with_curve",
            action="select_neuron",
            right_panel="network",
            title_de="Netzwerkstruktur und Belegung",
            title_en="Network Structure and Layout",
            question_de="Wo sitzen die Aktivierungsfunktionen?",
            question_en="Where do activation functions sit?",
            bullets_de=(
                "Eingabefeatures werden durch Hidden Layers zur Output-Entscheidung transformiert.",
                "Jedes Hidden-Neuron besitzt eine Aktivierungsfunktion.",
                "Das Layout beschreibt diese Belegung, zum Beispiel relu|relu.",
                "Beim markierten Neuron sehen wir direkt die passende Aktivierungskurve.",
            ),
            bullets_en=(
                "Input features are transformed through hidden layers into an output decision.",
                "Each hidden neuron has an activation function.",
                "The layout describes this assignment, for example relu|relu.",
                "For the highlighted neuron we immediately see the matching activation curve.",
            ),
            formula="Input → Hidden Layers → Output",
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="activation_lab",
            action="cycle_activation",
            right_panel="activation",
            title_de="Wie formt eine Aktivierung das Signal?",
            title_en="How Does an Activation Shape the Signal?",
            question_de="Was passiert zwischen z und a?",
            question_en="What happens between z and a?",
            bullets_de=(
                "z ist die gewichtete Summe plus Bias.",
                "Die Aktivierungsfunktion formt daraus a.",
                "Aendere die Beispielaktivierung und beobachte die Kurve.",
                "Damit wird sichtbar, warum das Layout eine Strukturentscheidung ist.",
            ),
            bullets_en=(
                "z is the weighted sum plus bias.",
                "The activation function turns it into a.",
                "Change the example activation and observe the curve.",
                "This shows why the layout is a structural decision.",
            ),
            formula="z = Σᵢ wᵢ · xᵢ + b\n\na = f(z)",
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="neuron_calculation",
            action="select_neuron",
            right_panel="calculation",
            title_de="Konkrete Rechnung im Neuron",
            title_en="Concrete Neuron Calculation",
            question_de="Welche Zahlen erzeugen das Signal dieses Neurons?",
            question_en="Which numbers create this neuron's signal?",
            bullets_de=(
                "Jetzt betrachten wir ein echtes Hidden-Neuron fuer dieses Sample.",
                "Die staerksten Summanden zeigen, welche Eingaben z besonders beeinflussen.",
                "Bias, z, a und Ableitung erklaeren die lokale Verarbeitung.",
                "Das ist die Innenansicht eines festen Netzwerks.",
            ),
            bullets_en=(
                "Now we inspect one real hidden neuron for this sample.",
                "The strongest terms show which inputs influence z the most.",
                "Bias, z, a, and derivative explain the local processing.",
                "This is the internal view of one fixed network.",
            ),
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="training_with_prediction",
            action="train_10",
            right_panel="training",
            title_de="Was wird beim Training gelernt?",
            title_en="What Is Learned During Training?",
            question_de="Was veraendert Gradient Descent?",
            question_en="What does gradient descent change?",
            bullets_de=(
                "Trainiert werden Gewichte und Biases.",
                "Das Aktivierungs-Layout bleibt in diesem Modus fest.",
                "Loss und Accuracy zeigen, ob das feste Layout trainierbar ist.",
                "Diese Beobachtung fuehrt direkt zur Frage nach besseren Layouts.",
            ),
            bullets_en=(
                "Weights and biases are trained.",
                "The activation layout remains fixed in this mode.",
                "Loss and accuracy show whether this fixed layout is trainable.",
                "This observation leads directly to the question of better layouts.",
            ),
            formula="weights ← weights − η · gradient",
        ),
        PresentationSlide(
            track_id="neural_network",
            visual="transition_to_sa",
            action=None,
            right_panel="none",
            title_de="Uebergang: Wer waehlt das Layout?",
            title_en="Transition: Who Chooses the Layout?",
            question_de="Wenn Training nur Gewichte aendert, wer entscheidet ueber ReLU, tanh oder sigmoid?",
            question_en="If training only changes weights, who chooses ReLU, tanh, or sigmoid?",
            bullets_de=(
                "Bisher war das Aktivierungs-Layout vorgegeben.",
                "Aber verschiedene Belegungen koennen unterschiedlich gut trainierbar sein.",
                "Daraus machen wir jetzt ein diskretes Suchproblem.",
                "Simulated Annealing sucht ueber Aktivierungs-Layouts.",
            ),
            bullets_en=(
                "So far the activation layout was given.",
                "But different assignments may train differently well.",
                "We now turn this into a discrete search problem.",
                "Simulated annealing searches over activation layouts.",
            ),
        ),
    ),
    "simulated_annealing": (
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_intro",
            action=None,
            right_panel="none",
            title_de="Unsere Seminar-Frage",
            title_en="Our Seminar Question",
            question_de="Wie nutzen wir Simulated Annealing in diesem Projekt?",
            question_en="How do we use simulated annealing in this project?",
            bullets_de=(
                "Wir betrachten Aktivierungsfunktionen als verteilbare Struktur im Hidden Layer.",
                "Ein Layout ist eine diskrete Belegung von Neuronen mit Aktivierungen.",
                "SA ist hier kein Gewichtstrainer, sondern ein Suchverfahren ueber Layouts.",
                "Jeder Kandidat wird kurz trainiert, um seine Validation-Performance zu messen.",
            ),
            bullets_en=(
                "We treat activation functions as a distributable hidden-layer structure.",
                "A layout is a discrete assignment of activations to neurons.",
                "SA is not the weight trainer here; it searches over layouts.",
                "Each candidate is trained briefly to measure validation performance.",
            ),
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="layout_comparison",
            action=None,
            right_panel="layout_compare",
            title_de="State: ein Aktivierungs-Layout",
            title_en="State: An Activation Layout",
            question_de="Was ist ein Zustand im Suchraum?",
            question_en="What is a state in the search space?",
            bullets_de=(
                "Ein Zustand ist nicht ein Gewichtszustand.",
                "Ein Zustand ist die Aktivierungsbelegung des Netzwerks.",
                "Startlayout und bestes gefundenes Layout koennen verschieden sein.",
                "Das trainierte Modell wird pro Layout frisch bewertet.",
            ),
            bullets_en=(
                "A state is not a weight state.",
                "A state is the network's activation assignment.",
                "Start layout and best found layout may differ.",
                "The trained model is freshly evaluated per layout.",
            ),
            formula="state = activation layout",
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="neighbor_comparison",
            action=None,
            right_panel="neighbor",
            title_de="Neighbor: lokale Layout-Aenderung",
            title_en="Neighbor: Local Layout Change",
            question_de="Wie entsteht ein Kandidat?",
            question_en="How is a candidate created?",
            bullets_de=(
                "Ein Kandidat entsteht aus dem aktuellen Layout.",
                "set_neuron: ein Neuron bekommt eine neue Aktivierung.",
                "fill_layer: eine ganze Schicht wird neu belegt.",
                "swap_neurons: zwei Neuronen tauschen ihre Aktivierungen.",
            ),
            bullets_en=(
                "A candidate is created from the current layout.",
                "set_neuron: one neuron receives a new activation.",
                "fill_layer: one whole layer is reassigned.",
                "swap_neurons: two neurons exchange activations.",
            ),
            formula="current layout → neighbor layout",
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_evaluation",
            action="evaluate_start",
            right_panel="none",
            title_de="Startbewertung",
            title_en="Start Evaluation",
            question_de="Wie bekommt ein Layout einen Score?",
            question_en="How does a layout receive a score?",
            bullets_de=(
                "Aus dem Layout wird ein frisches kleines Netz gebaut.",
                "Dieses Netz wird fuer wenige Candidate Epochs trainiert.",
                "Validation Loss bewertet die Trainierbarkeit des Layouts.",
                "Validation Accuracy zeigt zusaetzlich die Klassifikationsleistung.",
            ),
            bullets_en=(
                "A fresh small network is built from the layout.",
                "This network is trained for a few candidate epochs.",
                "Validation loss evaluates how trainable the layout is.",
                "Validation accuracy additionally shows classification performance.",
            ),
            formula="score = validation_loss",
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_decision_cards",
            action="step_once",
            right_panel="decision",
            title_de="Akzeptanzentscheidung",
            title_en="Acceptance Decision",
            question_de="Wann wird ein Kandidat uebernommen?",
            question_en="When is a candidate accepted?",
            bullets_de=(
                "Bessere Kandidaten werden akzeptiert.",
                "Schlechtere Kandidaten koennen bei hoher Temperatur trotzdem akzeptiert werden.",
                "Das verhindert, dass die Suche zu frueh stecken bleibt.",
                "Current, Candidate und Best muessen deshalb getrennt gelesen werden.",
            ),
            bullets_en=(
                "Better candidates are accepted.",
                "Worse candidates can still be accepted at high temperature.",
                "This prevents the search from getting stuck too early.",
                "Current, candidate, and best must therefore be read separately.",
            ),
            formula="Δ = score(candidate) − score(current)\n\np = exp(−Δ / T)",
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_history_explained",
            action="step_once",
            right_panel="none",
            title_de="Den Verlauf lesen",
            title_en="Reading the Run",
            question_de="Was sagen Plot und Karten aus?",
            question_en="What do the plot and cards tell us?",
            bullets_de=(
                "Candidate darf schwanken, weil neue Layouts ausprobiert werden.",
                "Best zeigt den besten bisher gefundenen Score.",
                "Current ist der aktuell akzeptierte Zustand.",
                "Temperature sinkt und macht die Suche konservativer.",
            ),
            bullets_en=(
                "Candidate may fluctuate because new layouts are tested.",
                "Best shows the best score found so far.",
                "Current is the currently accepted state.",
                "Temperature decreases and makes the search more conservative.",
            ),
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_run_summary",
            action="run_to_completion",
            right_panel="summary",
            title_de="Bis zum Ende laufen",
            title_en="Run to Completion",
            question_de="Was ist das Ergebnis dieses kurzen Laufs?",
            question_en="What is the result of this short run?",
            bullets_de=(
                "Startlayout: Ausgangspunkt der Suche.",
                "Endlayout: letzter akzeptierter Zustand.",
                "Bestlayout: bestes gefundenes Layout.",
                "Das ist nicht global garantiert optimal, aber ein nachvollziehbarer Suchpfad.",
            ),
            bullets_en=(
                "Start layout: starting point of the search.",
                "End layout: last accepted state.",
                "Best layout: best layout found.",
                "This is not guaranteed globally optimal, but it is an explainable search path.",
            ),
        ),
        PresentationSlide(
            track_id="simulated_annealing",
            visual="sa_final_takeaway",
            action=None,
            right_panel="none",
            title_de="Was zeigt Simulated Annealing hier?",
            title_en="What Does Simulated Annealing Show Here?",
            question_de="Welche Idee nehmen wir mit?",
            question_en="What is the main idea?",
            bullets_de=(
                "Aktivierungsverteilungen bilden einen diskreten Suchraum.",
                "Nachbarschaften machen diesen Raum schrittweise begehbar.",
                "Temperatur erlaubt Exploration statt reiner Greedy-Suche.",
                "Validation-Performance verbindet Suchprozess und Lernproblem.",
            ),
            bullets_en=(
                "Activation distributions form a discrete search space.",
                "Neighborhoods make this space traversable step by step.",
                "Temperature enables exploration instead of pure greedy search.",
                "Validation performance connects the search process to the learning problem.",
            ),
        ),
    ),
}


def track_title(track_id: str, language: str) -> str:
    return TRACK_TITLES[track_id]["en" if language == "en" else "de"]
