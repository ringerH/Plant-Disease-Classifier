import gradio as gr
from fastai.vision.all import load_learner, PILImage
import torch
  
learn = load_learner('model.pkl')

def classify_image(image):
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
