# Face Detection Web App

A simple web application for face detection using Flask and OpenCV.

🌐 **Live Demo**: [https://facedetection-webapp-go5s.onrender.com](https://facedetection-webapp-go5s.onrender.com)

## Features

- Upload images for face detection 
- Use webcam for real-time face detection
- Simple and clean user interface
- Error handling for various scenarios
- Local version supports emotion, age, and gender analysis using DeepFace
- Online version provides basic face detection capabilities

## Online vs Local Version

### Online Version (Render Deployment)
- Basic face detection using OpenCV's Haar Cascade
- Memory-optimized for Render's free tier
- Image size should be less than 500KB for optimal performance

### Local Version
- Enhanced features including emotion, age, and gender analysis via DeepFace
- Optional PyTorch integration for improved face detection accuracy
- No image size limitations (depends on your hardware)

## Requirements

### Core Requirements (For Basic Functionality)
- Python 3.7+ (3.11 recommended)
- Flask 2.3.3
- Werkzeug 2.3.7
- OpenCV (opencv-python-headless)
- NumPy 1.24.3
- Gunicorn 21.2.0 (for deployment)

### Optional Requirements (For Enhanced Features)
- DeepFace 0.0.79
- TensorFlow 2.13.0
- Keras 2.13.1
- PyTorch 2.0.1
- TorchVision 0.15.2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Param-10/FaceDetection_WebApp.git
cd FaceDetection_WebApp
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the core dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install enhanced features:
```bash
# For DeepFace analysis
pip install deepface==0.0.79 tensorflow==2.13.0 keras==2.13.1

# For PyTorch-based detection
pip install torch==2.0.1 torchvision==0.15.2
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to `http://127.0.0.1:5050`

3. Upload an image or use your webcam to detect faces

## How It Works

This application uses OpenCV's Haar Cascade classifier for face detection in the online version. The local version can additionally use a pre-trained Faster R-CNN model from PyTorch and DeepFace for advanced face analysis.

### Technical Details

- **Backend**: Flask web server handles HTTP requests and serves the web interface
- **Primary Face Detection**: OpenCV's Haar Cascade classifier for lightweight processing
- **Enhanced Face Detection** (local only): PyTorch's Faster R-CNN model
- **Face Analysis** (local only): DeepFace for emotion, age, and gender prediction
- **Image Processing**: OpenCV for image manipulation and drawing bounding boxes
- **Frontend**: HTML, CSS, and JavaScript for the user interface
- **Deployment**: Optimized for Render's free tier with memory constraints in mind

## Project Structure

```
FaceDetection_WebApp/
├── models/                    # Model directory
│   ├── __init__.py
│   └── face_detection_model.py  # Face detection model
├── static/                    # Static files
│   ├── css/
│   │   └── style.css          # CSS styles
│   └── js/
│       └── main.js            # JavaScript for frontend
├── templates/                 # HTML templates
│   └── index.html             # Main page
├── app.py                     # Flask application and entry point
├── requirements.txt           # Core dependencies
├── setup.py                   # Package setup with optional dependencies
├── render.yaml                # Render deployment configuration
├── Procfile                   # For deployment to Render
└── README.md                  # This file
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the versions in requirements.txt match your installed versions
3. For memory issues on deployment, ensure the enhanced dependencies are not installed
4. If you see errors related to NumPy or PyTorch compatibility, try the specific versions listed above

## License

This project is licensed under the MIT License - see the LICENSE file for details.
