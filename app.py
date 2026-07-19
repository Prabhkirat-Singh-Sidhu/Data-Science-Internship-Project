import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
model = tf.keras.models.load_model(model_path)

# Image dimensions
img_height = 150
img_width = 150

# Class labels
class_labels = ['broken box', 'correct box']

def predict_box_quality(image):
    if image is None:
        return "Please upload an image or turn on your webcam."
    
    # Preprocess the image
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img = img.resize((img_height, img_width))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Perform prediction
    predictions = model.predict(img_array)[0]
    
    # Return predictions as dictionary for Gradio label component
    return {class_labels[i]: float(predictions[i]) for i in range(len(class_labels))}

# Custom CSS for Sleek Dark Glassmorphism Design
custom_css = """
body {
    background-color: #0f0f16;
    color: #f3f4f6;
    font-family: 'Outfit', 'Inter', sans-serif;
}
.gradio-container {
    background: radial-gradient(circle at top left, #1e1b4b, #0f0f16 60%);
    border-radius: 20px;
    padding: 30px;
}
h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-shadow: 0 0 10px rgba(168, 85, 247, 0.4);
}
.feedback-header {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
"""

# Build clean Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="indigo"), css=custom_css) as demo:
    with gr.Row():
        gr.Markdown(
            """
            # 📦 Product Quality Estimator
            ### Real-Time Box Defect Detection System
            Analyze boxes in real-time using your webcam or by uploading a photo. The system will automatically detect if a box is **correct** or **broken**.
            """
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab("📷 Webcam Live Stream"):
                webcam_input = gr.Image(sources=["webcam"], type="numpy", label="Webcam Feed")
                webcam_btn = gr.Button("Analyze Live Frame", variant="primary")
            
            with gr.Tab("📁 Upload Image File"):
                file_input = gr.Image(sources=["upload"], type="numpy", label="Upload Box Image")
                file_btn = gr.Button("Analyze Uploaded Image", variant="primary")
                
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Analysis Report")
            output_labels = gr.Label(num_top_classes=2, label="Classification Probability")

    # Wire up events
    webcam_btn.click(fn=predict_box_quality, inputs=webcam_input, outputs=output_labels)
    file_btn.click(fn=predict_box_quality, inputs=file_input, outputs=output_labels)

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
