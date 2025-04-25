import gradio as gr
from transformers import pipeline
from PIL import Image

# Load the zero-shot image classification model
checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

# Candidate labels (you can customize or make this a Gradio input if you want)
default_labels = ["fox", "bear", "seagull", "owl"]

# Inference function
def classify_image(image):
    labels = "cat, dog"
    print("labels_list")
    labels_list = [label.strip() for label in labels.split(",")]
    print("image")
    image = image.convert("RGB")
    print("results")
    results = detector(image, candidate_labels=labels_list)
    print("return")
    return {res["label"]: round(res["score"], 4) for res in results}

# Gradio interface
interface = gr.Interface(
    classify_image,
    # inputs=[
    #     gr.Image(type="pil"),
    #     gr.Textbox(label="Candidate Labels (comma-separated)")
    # ],
    inputs=gr.Image(label="Upload image", sources=['upload', 'webcam'], type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Zero-Shot Image Classification with CLIP",
    # description="Upload an image and enter comma-separated labels to classify it using OpenAI's CLIP model."
)

if __name__ == "__main__":
    interface.launch()