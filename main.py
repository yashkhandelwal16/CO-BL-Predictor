import threading
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import mainthread
from kivy.uix.spinner import Spinner

from predictor import predict, CURRICULUM

# Desktop Window Size
Window.size = (540, 800)

# =====================================================================
# 🎨 COLOR THEME PALETTE: Sleek Yellow & Black (Cyber Dark)
# You can easily adjust any color here!
# Format: [Red, Green, Blue, Alpha] (values from 0.0 to 1.0)
# =====================================================================
THEME = {
    "bg_window": [0.07, 0.07, 0.09, 1],       # Deep Obsidian Black
    "card_bg": [0.13, 0.13, 0.16, 1],         # Dark Matte Charcoal Cards
    "card_subtle": [0.10, 0.10, 0.12, 1],     # Slightly darker background
    "accent_yellow": [1.0, 0.80, 0.02, 1],    # Electric Cyber Yellow
    "accent_gold": [1.0, 0.65, 0.05, 1],      # Warm Amber Gold
    "text_primary": [0.96, 0.96, 0.97, 1],    # Crisp White
    "text_muted": [0.60, 0.63, 0.68, 1],      # Muted Gray
    "btn_text_dark": [0.06, 0.06, 0.07, 1],   # Pitch Black for button text
    "btn_clear": [0.22, 0.22, 0.26, 1],       # Slate Charcoal for Clear button
    "input_bg": [0.12, 0.12, 0.15, 1],        # Input box dark background
}

Window.clearcolor = THEME["bg_window"]

KV = """
BoxLayout:
    orientation: "vertical"
    padding: ["24dp", "20dp", "24dp", "20dp"]
    spacing: "12dp"
    canvas.before:
        Color:
            rgba: 0.07, 0.07, 0.09, 1
        Rectangle:
            pos: self.pos
            size: self.size

    # --- Header Section ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: "66dp"
        spacing: "4dp"

        Label:
            text: "CO–BL Predictor"
            color: 1.0, 0.82, 0.02, 1
            font_size: "26sp"
            bold: True
            size_hint_y: None
            height: "36dp"
            halign: "center"

        Label:
            text: "AI-Powered Academic Question Classification"
            color: 0.65, 0.68, 0.72, 1
            font_size: "13sp"
            size_hint_y: None
            height: "20dp"
            halign: "center"

    # --- Subject Selector Dropdown ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: "66dp"
        spacing: "4dp"

        Label:
            text: "Select Subject / Course:"
            color: 1.0, 0.80, 0.02, 1
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
            height: "42dp"
            background_normal: ""
            background_color: 0.14, 0.14, 0.18, 1
            color: 0.95, 0.95, 0.96, 1
            bold: True

    # --- Question Input Box ---
    BoxLayout:
        orientation: "vertical"
        size_hint_y: 0.36
        spacing: "4dp"

        Label:
            text: "Enter Examination Question:"
            color: 1.0, 0.80, 0.02, 1
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
            background_color: 0.12, 0.12, 0.15, 1
            foreground_color: 0.95, 0.95, 0.96, 1
            cursor_color: 1.0, 0.80, 0.02, 1
            font_size: "14sp"

    # --- Action Buttons Row ---
    BoxLayout:
        orientation: "horizontal"
        size_hint_y: None
        height: "48dp"
        spacing: "12dp"

        Button:
            id: predict_btn
            text: "⚡ PREDICT CO & BL"
            bold: True
            font_size: "15sp"
            background_normal: ""
            background_color: 1.0, 0.80, 0.02, 1
            color: 0.06, 0.06, 0.07, 1
            on_release: app.on_predict_click()

        Button:
            text: "Clear"
            size_hint_x: 0.28
            bold: True
            font_size: "14sp"
            background_normal: ""
            background_color: 0.20, 0.20, 0.25, 1
            color: 0.85, 0.85, 0.90, 1
            on_release: app.clear_fields()

    # --- Status / Loading Label ---
    Label:
        id: status_label
        text: "Ready to predict"
        color: 0.60, 0.63, 0.68, 1
        font_size: "12sp"
        size_hint_y: None
        height: "18dp"
        halign: "center"

    # --- Result Cards (Dual Grid) ---
    BoxLayout:
        orientation: "horizontal"
        size_hint_y: 0.28
        spacing: "12dp"

        # Card 1: Course Outcome (CO)
        BoxLayout:
            orientation: "vertical"
            padding: "12dp"
            spacing: "4dp"
            canvas.before:
                Color:
                    rgba: 0.13, 0.13, 0.17, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

            Label:
                text: "PREDICTED CO"
                color: 0.65, 0.68, 0.72, 1
                font_size: "11sp"
                bold: True
                size_hint_y: None
                height: "16dp"

            Label:
                id: co_badge
                text: "CLO2"
                color: 1.0, 0.82, 0.02, 1
                font_size: "26sp"
                bold: True
                size_hint_y: None
                height: "36dp"

            Label:
                id: co_desc
                text: "Analyze radiation mechanisms and antenna parameters."
                color: 0.88, 0.88, 0.92, 1
                font_size: "11sp"
                halign: "center"
                valign: "top"
                text_size: self.width, None

        # Card 2: Bloom's Level (BL)
        BoxLayout:
            orientation: "vertical"
            padding: "12dp"
            spacing: "4dp"
            canvas.before:
                Color:
                    rgba: 0.13, 0.13, 0.17, 1
                Rectangle:
                    pos: self.pos
                    size: self.size

            Label:
                text: "BLOOM'S LEVEL"
                color: 0.65, 0.68, 0.72, 1
                font_size: "11sp"
                bold: True
                size_hint_y: None
                height: "16dp"

            Label:
                id: bl_badge
                text: "L2"
                color: 1.0, 0.65, 0.05, 1
                font_size: "26sp"
                bold: True
                size_hint_y: None
                height: "36dp"

            Label:
                id: bl_name
                text: "Understand"
                color: 0.95, 0.95, 0.97, 1
                font_size: "14sp"
                bold: True
                size_hint_y: None
                height: "22dp"

    # --- Card 3: Engine & Justification ---
    BoxLayout:
        orientation: "vertical"
        padding: "10dp"
        size_hint_y: 0.18
        canvas.before:
            Color:
                rgba: 0.10, 0.10, 0.13, 1
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            id: source_label
            text: "Engine: Ready"
            color: 1.0, 0.80, 0.02, 1
            font_size: "11sp"
            bold: True
            size_hint_y: None
            height: "18dp"
            halign: "left"
            text_size: self.width, None

        Label:
            id: explanation_label
            text: "Predictions will automatically follow official syllabus definitions."
            color: 0.75, 0.78, 0.82, 1
            font_size: "11sp"
            halign: "left"
            valign: "middle"
            text_size: self.width, None
"""


class COBLApp(App):
    def build(self):
        self.title = "CO-BL Predictor (Cyber Dark)"
        return Builder.load_string(KV)

    def get_subject_list(self):
        subjects = list(CURRICULUM.get("subjects", {}).keys())
        return subjects if subjects else ["Default Subject"]

    def on_predict_click(self):
        question = self.root.ids.question_input.text.strip()
        subject = self.root.ids.subject_spinner.text

        if not question:
            self.root.ids.status_label.text = "Error: Please enter an exam question first!"
            self.root.ids.status_label.color = (1.0, 0.3, 0.3, 1)
            return

        # Disable button & show loading state
        self.root.ids.predict_btn.disabled = True
        self.root.ids.status_label.text = "Analyzing question and mapping CO/BL..."
        self.root.ids.status_label.color = (1.0, 0.80, 0.02, 1)

        # Run in background thread to prevent UI freezing
        threading.Thread(target=self._run_async_prediction, args=(question, subject), daemon=True).start()

    def _run_async_prediction(self, question, subject):
        result = predict(question, subject)
        self.update_results(result)

    @mainthread
    def update_results(self, res):
        self.root.ids.co_badge.text = res.get("co", "CO1")
        self.root.ids.co_desc.text = res.get("co_desc", "")
        self.root.ids.bl_badge.text = res.get("bl", "L2")
        self.root.ids.bl_name.text = res.get("bl_name", "Understand")
        self.root.ids.source_label.text = f"Engine: {res.get('source', 'Unknown')}"
        self.root.ids.explanation_label.text = res.get("explanation", "")

        self.root.ids.status_label.text = "Prediction Complete!"
        self.root.ids.status_label.color = (0.2, 0.9, 0.4, 1)
        self.root.ids.predict_btn.disabled = False

    def clear_fields(self):
        self.root.ids.question_input.text = ""
        self.root.ids.status_label.text = "Cleared"
        self.root.ids.status_label.color = (0.6, 0.63, 0.68, 1)


if __name__ == "__main__":
    COBLApp().run()
