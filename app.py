# app.py
import gradio as gr
from fastai.vision.all import *
from pathlib import Path
import sys
import pathlib
# 🔧 Cross-platform patch for PosixPath in export.pkl
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# Load the trained model
learn_inf = load_learner(Path('export.pkl'))

# Prediction function
def classify_food(img):
    pred, idx, probs = learn_inf.predict(img)
    return {learn_inf.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Gradio UI
demo = gr.Interface(
    fn=classify_food,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🍱 Food Classifier",
    description="Upload a food image (Biryani, Pizza, or Sushi) and get a prediction!"
)

# Launch
demo.launch()
