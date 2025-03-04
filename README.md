# Face Detection Web App

A simple web application for face detection using Flask, OpenCV, and PyTorch.

## Features

- Upload images for face detection
- Use webcam for real-time face detection
- Simple and clean user interface
- Error handling for various scenarios

## Requirements

- Python 3.7+
- Flask 2.0.1
- Werkzeug 2.0.1
- OpenCV (opencv-python-headless)
- PyTorch 2.2.0
- TorchVision 0.17.0
- NumPy 1.24.3

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FaceDetection_WebApp.git
cd FaceDetection_WebApp
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to `http://127.0.0.1:5000`

3. Upload an image or use your webcam to detect faces

## How It Works

This application uses a pre-trained Faster R-CNN model from PyTorch's torchvision library to detect faces in images. The model is loaded when the application starts and is used to process images uploaded by the user or captured from the webcam.

### Technical Details

- **Backend**: Flask web server handles HTTP requests and serves the web interface
- **Face Detection**: PyTorch's Faster R-CNN model pre-trained on COCO dataset
- **Image Processing**: OpenCV for image manipulation and drawing bounding boxes
- **Frontend**: HTML, CSS, and JavaScript for the user interface

## Project Structure

```
FaceDetection_WebApp/
├── FaceDetection_WebApp/       # Main package
│   ├── __init__.py             # Package initialization
│   ├── app.py                  # Flask application
│   ├── models/                 # Model directory
│   │   ├── __init__.py
│   │   └── face_detection_model.py  # Face detection model
│   ├── static/                 # Static files
│   │   ├── css/
│   │   │   └── style.css       # CSS styles
│   │   └── js/
│   │       └── main.js         # JavaScript for frontend
│   └── templates/              # HTML templates
│       └── index.html          # Main page
├── app.py                      # Entry point script
├── requirements.txt            # Dependencies
├── run.py                      # Alternative entry point
└── README.md                   # This file
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the versions in requirements.txt match your installed versions
3. If you see errors related to NumPy or PyTorch compatibility, try the specific versions in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.
