
# 🖼️ CLIP Zero-Shot Classifier

This interactive web app demonstrates a **zero-shot image classification** system using **OpenAI's CLIP model** (`ViT-B/32`) and a custom Gradio interface.

## 🚀 What It Does

CLIP can understand images and text in the same embedding space. With this app, you can:
- Upload an image
- Enter any number of labels (comma-separated)
- Get predictions on how likely the image matches each label — **even without training!**

## 💡 How It Works

1. The input image is preprocessed and encoded using CLIP.
2. Your custom labels are tokenized and also encoded.
3. The cosine similarity between image and text embeddings is computed.
4. The results are displayed with a probability score and a visual bar indicator.

## 📦 Technologies Used

- [Gradio](https://www.gradio.app/) — for the interactive web interface
- [OpenAI CLIP](https://github.com/openai/CLIP) — the core model for zero-shot classification
- PyTorch — model backend
- Hugging Face Spaces — for easy and free deployment

## 📷 Example Use Cases

- Test if an image matches multiple tags
- Quickly validate custom labels
- Educational demos for multimodal ML

## 🛠️ How to Use

1. Upload an image.
2. Type in labels like: `a cat, a dog, a diagram, a spacecraft`
3. Click **Classify**.
4. See prediction probabilities and visual bars for each label.

## 📍 Notes

- You can enter *any text labels* — even abstract or creative ones!
- Works best on natural images (e.g., animals, objects, scenes)

## 📓 Notebook

You can explore the companion Jupyter notebook here:
[📘 Open notebook.ipynb](./notebook/clip_inspect.ipynb)

---

## 👤 About Me

I'm **Nikko**, a Machine Learning Engineer and AI enthusiast with a Master's degree in Artificial Intelligence from the University of the Philippines Diliman. With over a decade of experience in ICT consulting and telecommunications, I now specialize in **vision-language models**, **LLMs**, and **generative AI applications**.

I'm passionate about creating systems where AI and humans can collaborate seamlessly — working toward a future where **smart cities** and intelligent automation become reality.    

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/nikkoyabut/).

---

Made with ❤️ using CLIP + Gradio
