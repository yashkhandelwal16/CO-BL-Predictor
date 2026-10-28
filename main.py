from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window

# Desktop window ki dimension set karte hain
Window.size = (480, 720)

KV = """
BoxLayout:
    orientation: "vertical"
    padding: "24dp"
    spacing: "16dp"
    canvas.before:
        Color:
            rgba: 0.96, 0.98, 0.99, 1
        Rectangle:
            pos: self.pos
            size: self.size

    # Header Title
    Label:
        text: "CO-BL Predictor"
        color: 0.08, 0.25, 0.35, 1
        font_size: "26sp"
        bold: True
        size_hint_y: None
        height: "50dp"

    Label:
        text: "AI-Powered Academic Question Classification"
        color: 0.30, 0.45, 0.55, 1
        font_size: "14sp"
        size_hint_y: None
        height: "25dp"

    # Input Box
    TextInput:
        id: question_input
        hint_text: "Enter your examination question here (e.g. Explain dipole antenna radiation)..."
        multiline: True
        size_hint_y: 0.45
        padding: "14dp"
        background_color: 1, 1, 1, 1
        foreground_color: 0.10, 0.20, 0.25, 1

    # Action Button
    Button:
        text: "Predict CO & BL"
        size_hint_y: None
        height: "50dp"
        bold: True
        background_normal: ""
        background_color: 0.09, 0.65, 0.72, 1
        on_release: app.predict_dummy()

    # Output Card
    BoxLayout:
        orientation: "vertical"
        padding: "16dp"
        size_hint_y: 0.35
        canvas.before:
            Color:
                rgba: 1, 1, 1, 1
            Rectangle:
                pos: self.pos
                size: self.size
        
        Label:
            id: result_co
            text: "Course Outcome (CO): Pending..."
            color: 0.10, 0.20, 0.30, 1
            font_size: "16sp"
            bold: True
            halign: "left"
            text_size: self.width, None

        Label:
            id: result_bl
            text: "Bloom's Level (BL): Pending..."
            color: 0.10, 0.55, 0.60, 1
            font_size: "16sp"
            bold: True
            halign: "left"
            text_size: self.width, None
"""


class COBLApp(App):
    def build(self):
        self.title = "CO-BL Predictor"
        return Builder.load_string(KV)

    def predict_dummy(self):
        question = self.root.ids.question_input.text.strip()
        if not question:
            self.root.ids.result_co.text = "Error: Please enter a question!"
            self.root.ids.result_bl.text = ""
            return

        # Dummy response (AI connect karne se pehle test ke liye)
        self.root.ids.result_co.text = "Predicted CO: CLO2 (Antenna Theory)"
        self.root.ids.result_bl.text = "Bloom's Level: L2 (Understand)"


if __name__ == "__main__":
    COBLApp().run()
