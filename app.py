# app.py

# 🛠️ Setup
# pip install -q gradio torch ftfy regex tqdm git+https://github.com/openai/CLIP.git matplotlib

# 📦 Imports
import gradio as gr
import torch
import clip
from PIL import Image
import numpy as np
from typing import List, Tuple, Union

# 🚀 Load CLIP Model
device: str = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



def predict(image: Image.Image, label_text: str) -> List[List[Union[str, float]]]:
    """
    Perform zero-shot classification using the CLIP model.

    Args:
        image (PIL.Image.Image): Input image.
        label_text (str): Comma-separated labels to classify against.

    Returns:
        List[List[Union[str, float]]]: A list of results with label, probability, and confidence bar HTML.
    """
    labels: List[str] = [label.strip() for label in label_text.split(",") if label.strip()]
    if not image or not labels:
        return []

    # Preprocess inputs
    image_input: torch.Tensor = preprocess(image).unsqueeze(0).to(device)
    text_inputs: torch.Tensor = clip.tokenize(labels).to(device)

    # Run model
    with torch.no_grad():
        image_features: torch.Tensor = model.encode_image(image_input)
        text_features: torch.Tensor = model.encode_text(text_inputs)
        logits_per_image, _ = model(image_input, text_inputs)
        probs: np.ndarray = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Create table with bar visualization
    results: List[List[Union[str, float]]] = []
    for label, prob in zip(labels, probs):
        bar_html: str = (
            f'<div style="background-color:#4caf50;width:{prob * 100:.1f}%;height:20px;"></div>'
        )
        results.append([label, f"{prob * 100:.2f}%", bar_html])

    return results


# 🎨 Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## CLIP Zero-Shot Classifier")

    with gr.Row():
        image = gr.Image(type="pil", label="Upload Image")
        label_text = gr.Textbox(
            lines=2,
            label="Enter comma-separated labels",
            placeholder="e.g., a cat, a dog, a diagram"
        )

    # Image Examples
    with gr.Row():
        gr.Examples(
            examples=[
                ["images/boy.jpg"],
                ["images/dog.jpg"],
                ["images/boy_dog.jpg"]
            ],
            inputs=[image],
            label="🖼️ Click to select example image"
        )

        # Label Text Examples
        gr.Examples(
            examples=[
                ["boy, girl, dog, cat"],
                ["a boy with a dog, a boy with a cat, a girl with a dog, a girl with a cat"],
                ["a cat, a dog, a diagram"]
            ],
            inputs=[label_text],
            label="📝 Click to autofill example labels"
        )

    submit = gr.Button("Classify")

    output = gr.Dataframe(
        headers=["Label", "Probability", "Confidence Bar"],
        datatype=["str", "str", "html"],
        row_count=5,
        interactive=False
    )

    submit.click(fn=predict, inputs=[image, label_text], outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
