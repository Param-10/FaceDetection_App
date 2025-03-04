# FaceDetection_WebApp
A simple web application for face detection using OpenCV and PyTorch.

## Features
- Upload images for face detection
- Use webcam for real-time face detection
- Simple and clean user interface

## Requirements
- Python 3.7+
- Flask
- OpenCV
- PyTorch
- TorchVision
- NumPy

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/FaceDetection_WebApp.git
cd FaceDetection_WebApp
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```
python app.py
```

2. Open your browser and go to `http://127.0.0.1:5000`

3. Upload an image or use your webcam to detect faces

## How It Works

This application uses a pre-trained Faster R-CNN model from PyTorch's torchvision library to detect faces in images. The model is loaded when the application starts and is used to process images uploaded by the user or captured from the webcam.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
