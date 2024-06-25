
import gradio as gr
from fastai.vision.all import load_learner, PILImage
import torch

model_path = 'potato_classifier.pkl'  # Ensure this path is correct
learn = load_learner(model_path)

def classify_image(image):
    # Predict the class of the image
    pred, pred_idx, probs = learn.predict(image)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type='pil'),
    outputs=gr.outputs.Label(num_top_classes=3),
    title="Potato Plant Disease Classifier",
    description="Potato leaf: "
)

if __name__ == "__main__":
    interface.launch()
