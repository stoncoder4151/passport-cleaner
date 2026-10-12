import gradio as gr
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def process_passport(image):
    if image is None:
        return None

    # Convert to PIL
    img = Image.fromarray(image).convert("RGB")

    # 1. Mild contrast improvement (helps readability)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.18)

    # 2. Mild sharpness (makes text clearer without destroying details)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.35)

    # 3. Very slight brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.03)

    # 4. Very light noise reduction (optional and gentle)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return np.array(img)

demo = gr.Interface(
    fn=process_passport,
    inputs=gr.Image(type="numpy", label="Upload Passport Photo"),
    outputs=gr.Image(type="numpy", label="Cleaned Result", format="png"),
    title="Passport Cleaner",
    description="Upload a passport photo. Gentle cleaning for better clarity without damaging the document."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)import gradio as gr
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2

def process_passport(image):
    if image is None:
        return None

    # Convert to OpenCV format
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 1. Reduce glare / bright reflections
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Denoise while keeping text sharp
    img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

    # 3. Convert back to PIL for final enhancements
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 4. Increase contrast and sharpness for clearer text
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.35)

    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(1.8)

    # 5. Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(1.05)

    return np.array(img_pil)

demo = gr.Interface(
    fn=process_passport,
    inputs=gr.Image(type="numpy", label="Upload Passport Photo"),
    outputs=gr.Image(type="numpy", label="Cleaned Result", format="png"),
    title="Passport Cleaner",
    description="Upload a passport photo. Improved version with better glare reduction and clearer text."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
