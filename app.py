import gradio as gr
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def process_passport(image):
    if image is None:
        return None

    img = Image.fromarray(image).convert("RGB")

    # Mild contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.18)

    # Mild sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.35)

    # Slight brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.03)

    # Light noise reduction
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return np.array(img)

demo = gr.Interface(
    fn=process_passport,
    inputs=gr.Image(type="numpy", label="Upload Passport Photo"),
    outputs=gr.Image(type="numpy", label="Cleaned Result", format="png"),
    title="Passport Cleaner",
    description="Upload a passport photo. Gentle cleaning for better clarity."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
