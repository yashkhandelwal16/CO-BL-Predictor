import threading
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import mainthread
from kivy.uix.spinner import Spinner

from predictor import predict, CURRICULUM

# Desktop Window Size
Window.size = (540, 780)
Window.clearcolor = (0.95, 0.97, 0.99, 1)

KV = """
BoxLayout:
    orientation: "vertical"
    padding: ["24dp", "20dp", "24dp", "20dp"]
    spacing: "12dp"

    # --- Header Section ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: "64dp"
        spacing: "4dp"

        Label:
            text: "CO–BL Predictor"
            color: 0.07, 0.23, 0.33, 1
            font_size: "26sp"
            bold: True
            size_hint_y: None
            height: "36dp"
            halign: "center"

        Label:
            text: "AI-Powered Academic Question Classification"
            color: 0.35, 0.48, 0.56, 1
            font_size: "13sp"
            size_hint_y: None
            height: "20dp"
            halign: "center"

    # --- Subject Selector Dropdown ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: "64dp"
        spacing: "4dp"

        Label:
            text: "Select Subject / Course:"
            color: 0.15, 0.25, 0.32, 1
            font_size: "13sp"
            bold: True
            size_hint_y: None
            height: "20dp"
            halign: "left"
            text_size: self.width, None

        Spinner:
            id: subject_spinner
            text: "Microwave Engineering and Antennas"
            values: app.get_subject_list()
            size_hint_y: None
            height: "40dp"
            background_normal: ""
            background_color: 0.15, 0.40, 0.50, 1
            color: 1, 1, 1, 1
            bold: True

    # --- Question Input Box ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: 0.38
        spacing: "4dp"

        Label:
            text: "Enter Examination Question:"
            color: 0.15, 0.25, 0.32, 1
            font_size: "13sp"
            bold: True
            size_hint_y: None
            height: "20dp"
            halign: "left"
            text_size: self.width, None

        TextInput:
            id: question_input
            text: "Explain the radiation mechanism of a half-wave dipole antenna."
            hint_text: "Type or paste an exam question here..."
            multiline: True
            padding: "12dp"
            background_color: 1, 1, 1, 1
            foreground_color: 0.10, 0.18, 0.22, 1
            font_size: "14sp"

    # --- Buttons Row ---
    BoxLayout:
        orientation: "horizontal"
        size_hint_y: None
        height: "46dp"
        spacing: "10dp"

        Button:
            id: predict_btn
            text: "Predict CO & BL"
            bold: True
            font_size: "15sp"
            background_normal: ""
            background_color: 0.08, 0.65, 0.72, 1
            on_release: app.on_predict_click()

        Button:
            text: "Clear"
            size_hint_x: 0.3
            font_size: "14sp"
            background_normal: ""
            background_color: 0.70, 0.75, 0.80, 1
            color: 0.15, 0.25, 0.30, 1
            on_release: app.clear_fields()

    # --- Status / Loading Label ---
    Label:
        id: status_label
        text: "Ready to predict"
        color: 0.40, 0.55, 0.62, 1
        font_size: "12sp"
        size_hint_y: None
        height: "18dp"
        halign: "center"

    # --- Result Cards (Dual Grid) ---
    BoxLayout:
        orientation: "horizontal"
        size_hint_y: 0.28
        spacing: "12dp"

        # Card 1: Course Outcome
        BoxLayout:
            orientation: "vertical"
            padding: "12dp"
            spacing: "4dp"
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

            Label:
                text: "PREDICTED CO"
                color: 0.40, 0.50, 0.58, 1
                font_size: "11sp"
                bold: True
                size_hint_y: None
                height: "16dp"

            Label:
                id: co_badge
                text: "CLO2"
                color: 0.05, 0.55, 0.65, 1
                font_size: "24sp"
                bold: True
                size_hint_y: None
                height: "36dp"

            Label:
                id: co_desc
                text: "Analyze radiation mechanisms and antenna parameters."
                color: 0.20, 0.30, 0.36, 1
                font_size: "11sp"
                halign: "center"
                valign: "top"
                text_size: self.width, None

        # Card 2: Bloom's Level
        BoxLayout:
            orientation: "vertical"
            padding: "12dp"
            spacing: "4dp"
            canvas.before:
                Color:
                    rgba: 1, 1, 1, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

            Label:
                text: "BLOOM'S LEVEL"
                color: 0.40, 0.50, 0.58, 1
                font_size: "11sp"
                bold: True
                size_hint_y: None
                height: "16dp"

            Label:
                id: bl_badge
                text: "L2"
                color: 0.88, 0.42, 0.15, 1
                font_size: "24sp"
                bold: True
                size_hint_y: None
                height: "36dp"

            Label:
                id: bl_name
                text: "Understand"
                color: 0.20, 0.30, 0.36, 1
                font_size: "14sp"
                bold: True
                size_hint_y: None
                height: "22dp"

    # --- Card 3: Model & Justification ---
    BoxLayout:
        orientation: "vertical"
        padding: "10dp"
        size_hint_y: 0.18
        canvas.before:
            Color:
                rgba: 0.90, 0.94, 0.97, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            id: source_label
            text: "Engine: Ready"
            color: 0.15, 0.35, 0.45, 1
            font_size: "11sp"
            bold: True
            size_hint_y: None
            height: "18dp"
            halign: "left"
            text_size: self.width, None

        Label:
            id: explanation_label
            text: "Predictions will automatically follow official syllabus definitions."
            color: 0.30, 0.42, 0.48, 1
            font_size: "11sp"
            halign: "left"
            valign: "middle"
            text_size: self.width, None
"""


class COBLApp(App):
    def build(self):
        self.title = "CO-BL Predictor"
        return Builder.load_string(KV)

    def get_subject_list(self):
        subjects = list(CURRICULUM.get("subjects", {}).keys())
        return subjects if subjects else ["Default Subject"]

    def on_predict_click(self):
        question = self.root.ids.question_input.text.strip()
        subject = self.root.ids.subject_spinner.text

        if not question:
            self.root.ids.status_label.text = "Error: Please enter an exam question first!"
            self.root.ids.status_label.color = (0.85, 0.2, 0.2, 1)
            return

        # Disable button & show loading state
        self.root.ids.predict_btn.disabled = True
        self.root.ids.status_label.text = "Analyzing question and mapping CO/BL..."
        self.root.ids.status_label.color = (0.08, 0.55, 0.65, 1)

        # Run in background thread to prevent UI freezing
        threading.Thread(target=self._run_async_prediction, args=(question, subject), daemon=True).start()

    def _run_async_prediction(self, question, subject):
        result = predict(question, subject)
        self.update_results(result)

    @mainthread
    def update_results(self, res):
        # Update UI with prediction output safely on the main UI thread
        self.root.ids.co_badge.text = res.get("co", "CO1")
        self.root.ids.co_desc.text = res.get("co_desc", "")
        self.root.ids.bl_badge.text = res.get("bl", "L2")
        self.root.ids.bl_name.text = res.get("bl_name", "Understand")
        self.root.ids.source_label.text = f"Engine: {res.get('source', 'Unknown')}"
        self.root.ids.explanation_label.text = res.get("explanation", "")

        self.root.ids.status_label.text = "Prediction Complete!"
        self.root.ids.status_label.color = (0.1, 0.6, 0.3, 1)
        self.root.ids.predict_btn.disabled = False

    def clear_fields(self):
        self.root.ids.question_input.text = ""
        self.root.ids.status_label.text = "Cleared"
        self.root.ids.status_label.color = (0.4, 0.5, 0.6, 1)


if __name__ == "__main__":
    COBLApp().run()
