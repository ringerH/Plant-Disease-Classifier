import gradio as gr
from fastai.vision.all import load_learner, PILImage

learn = load_learner('model.pkl')

def classify_image(image):
    pred, _, probs = learn.predict(image)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),  # Updated to new API
    outputs=gr.Label(num_top_classes=3),  # Updated to new API
    title="Potato Plant Disease Classifier",
    description="Potato leaf:"
)

if __name__ == "__main__":
    interface.launch(share=True)
