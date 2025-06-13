#!/usr/bin/env python
"""
Simple entry point for the Face Detection Web App.
This file is placed at the root directory for easier execution.
"""
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
import base64
import os
from face_detection_model import FaceDetectionModel
import traceback, sys

app = Flask(__name__)

# Initialize the face detection model
face_detector = FaceDetectionModel()

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Face Detection API is running'})

@app.route('/ready')
def ready_check():
    """Check if all models are loaded and ready for processing"""
    try:
        models_ready = True
        status_details = {}
        
        # Check if face detector is initialized
        if face_detector is None:
            models_ready = False
            status_details['face_detector'] = 'not_initialized'
        else:
            status_details['face_detector'] = 'ready'
        
        # Check DeepFace models if available
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            status_details['emotion_model'] = 'ready' if face_detector.emotion_model_loaded else 'loading'
            status_details['age_gender_model'] = 'ready' if face_detector.age_gender_model_loaded else 'loading'
            
            if not face_detector.emotion_model_loaded or not face_detector.age_gender_model_loaded:
                models_ready = False
        
        return jsonify({
            'ready': models_ready,
            'status': 'ready' if models_ready else 'loading',
            'models': status_details,
            'message': 'All models ready for processing' if models_ready else 'Models are still loading, please wait...'
        })
        
    except Exception as e:
        return jsonify({
            'ready': False,
            'status': 'error',
            'error': str(e),
            'message': 'Error checking model readiness'
        }), 500

@app.route('/dashboard')
def get_dashboard():
    """Get model performance dashboard data"""
    try:
        dashboard_data = face_detector.get_model_performance_dashboard()
        return jsonify(dashboard_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
            
        print(f"🔍 Processing image of size: {img.shape}")
        
        # Check if models are still loading
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            if not face_detector.emotion_model_loaded or not face_detector.age_gender_model_loaded:
                print("⏳ Models still loading - this may take a moment...")
                return jsonify({
                    'error': 'Models are still loading. Please wait a moment and try again.',
                    'loading': True,
                    'message': 'AI models are initializing for the first time. This usually takes 30-60 seconds.'
                }), 503  # Service Temporarily Unavailable
            
        # Detect faces with enhanced analysis and autonomous validation
        result_img, face_data, metadata = face_detector.detect_faces(img)
        
        # Convert to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', result_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Detection completed: {len(face_data)} faces found")
        
        # Return both the processed image, face data, and quality metadata
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_str}',
            'faces': face_data,
            'metadata': metadata
        })
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        traceback.print_exc()
        
        # Check if it's a model loading error
        if "model" in str(e).lower() or "deepface" in str(e).lower():
            return jsonify({
                'error': 'AI models are still initializing. Please wait a moment and try again.',
                'loading': True,
                'message': 'This is normal for the first few requests after server startup.'
            }), 503
        
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Face Detection Web App...")
    # Get port from environment variable (Render sets this)
    # Default to 5050 instead of 5000 to avoid common port conflicts
    port = int(os.environ.get("PORT", 5050))
    print(f"Server will start on port {port}")
    # Use 0.0.0.0 to bind to all interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
