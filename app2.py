import gradio as gr
from transformers import pipeline
from PIL import Image

# Load the zero-shot image classification model
checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

# Candidate labels (you can customize or make this a Gradio input if you want)
default_labels = ["fox", "bear", "seagull", "owl"]

# Inference function
def classify_image(image, labels):
    labels_list = [label.strip() for label in labels.split(",")]
    image = image.convert("RGB")
    results = detector(image, candidate_labels=labels_list)
    return {res["label"]: round(res["score"], 4) for res in results}

# Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(label="Candidate Labels (comma-separated)", value=", ".join(default_labels))
    ],
    outputs=gr.Label(num_top_classes=4),
    title="Zero-Shot Image Classification with CLIP",
    description="Upload an image and enter comma-separated labels to classify it using OpenAI's CLIP model."
)

if __name__ == "__main__":
    interface.launch()
