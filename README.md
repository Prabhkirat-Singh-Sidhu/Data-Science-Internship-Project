# Broken Box Recognition Web App

This project is a real-time, in-browser recognition application. It uses a custom Convolutional Neural Network (CNN) model built with Keras and converted to TensorFlow.js to run inference directly in the browser, entirely client-side.

## Live Demo
The application is deployed via GitHub Pages and can be accessed here:
[https://prabhkirat-singh-sidhu.github.io/Data-Science-Internship-Project/](https://prabhkirat-singh-sidhu.github.io/Data-Science-Internship-Project/)

## Features
- **Real-time Inference**: Uses your device's webcam to perform continuous prediction.
- **Client-Side Processing**: All machine learning inference happens directly in your browser using TensorFlow.js. No images are sent to a server, ensuring absolute privacy.
- **Modern UI**: Clean, responsive, dark-mode design.
- **Zero Hosting Costs**: The entire application is static (HTML, CSS, JS, and model weights), allowing it to be hosted for free on GitHub Pages without the need for expensive backend servers.

## Tech Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Machine Learning Engine**: TensorFlow.js (`tfjs`)
- **Model Training (Python)**: TensorFlow / Keras (CNN architecture)
- **Deployment**: GitHub Pages

## Project Structure
- `index.html`: The main web interface containing the UI and webcam logic.
- `tfjs_model/`: Directory containing the converted TensorFlow.js model.
  - `model.json`: The model topology and weight manifestations (optimized for TF.js compatibility).
  - `*.bin`: The binary shard files containing the trained model weights.
- `convert_model.py`: The Python script used to convert the original `.h5` / `.keras` model to the TF.js format.

## How to Run Locally

Because the application needs to load external files (`model.json` and `.bin` weights) using the Fetch API, it cannot be run simply by double-clicking the `index.html` file (due to browser CORS security restrictions). You must run it through a local HTTP server.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/prabhkirat-singh-sidhu/Data-Science-Internship-Project.git
    cd Data-Science-Internship-Project
    ```

2.  **Start a local server**:
    If you have Python installed, you can easily start a server from your terminal:
    ```bash
    python -m http.server 8000
    ```

3.  **Open in Browser**:
    Navigate to `http://localhost:8000` in your web browser.

## Model Details
The underlying model is a custom Sequential Convolutional Neural Network consisting of multiple `Conv2D` and `MaxPooling2D` layers, followed by a `Flatten` layer, `Dropout` for regularization, and `Dense` layers for final classification.

**Note on Keras 3 Compatibility**: The `model.json` was manually patched to ensure compatibility with TensorFlow.js, resolving issues with Keras 3's `DTypePolicy` objects and weight naming conventions (`sequential/` prefixes).
