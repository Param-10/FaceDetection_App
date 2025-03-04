#!/usr/bin/env python
"""
Simple entry point for the Face Detection Web App.
This file is placed at the root directory for easier execution.
"""
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
import base64
import os
from models.face_detection_model import FaceDetectionModel

app = Flask(__name__)

# Initialize the face detection model
face_detector = FaceDetectionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def process_image():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'Empty file uploaded'}), 400
            
        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check if image was properly decoded
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
            
        # Detect faces with enhanced analysis
        result_img, face_data = face_detector.detect_faces(img)
        
        # Convert to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', result_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Return both the processed image and face data
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_str}',
            'faces': face_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Face Detection Web App...")
    # Get port from environment variable (Render sets this)
    # Default to 5050 instead of 5000 to avoid common port conflicts
    port = int(os.environ.get("PORT", 5050))
    print(f"Server will start on port {port}")
    # Use 0.0.0.0 to bind to all interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
