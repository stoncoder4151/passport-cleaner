import gradio as gr
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def process_passport(image):
    if image is None:
        return None

    # Convert to PIL Image
    img = Image.fromarray(image).convert("RGB")

    # 1. Basic enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.6)

    # 2. Slight brightness adjustment to help with glare
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)

    # 3. Light denoising
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return np.array(img)

# Create the web interface
demo = gr.Interface(
    fn=process_passport,
    inputs=gr.Image(type="numpy", label="Upload Passport Photo"),
    outputs=gr.Image(type="numpy", label="Cleaned Result"),
    title="Passport Cleaner",
    description="Upload a passport photo. The system will improve clarity, reduce glare, and enhance text. (Basic version)"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
