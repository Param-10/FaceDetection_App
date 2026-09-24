#!/usr/bin/env python
"""
Simple entry point for the Face Detection Web App.
This file is placed at the root directory for easier execution.
"""
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
import base64
import os
from pathlib import Path
from face_detection_model import FaceDetectionModel
import traceback, sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_BYTES', 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get('MAX_IMAGE_PIXELS', 20_000_000))
FRONTEND_DIR = Path(__file__).resolve().parent / 'dist'

@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(error):
    return jsonify({
        'error': 'Uploaded image is too large',
        'max_bytes': app.config['MAX_CONTENT_LENGTH']
    }), 413

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
        
        # Check DeepFace models and distinguish missing/failed dependencies
        # from a genuinely in-progress load.
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            if face_detector.emotion_model_loaded and face_detector.age_gender_model_loaded:
                model_status = 'ready'
            elif not getattr(face_detector, 'deepface_available', False):
                model_status = 'unavailable'
            elif getattr(face_detector, 'model_preload_failed', False):
                model_status = 'error'
            else:
                model_status = 'loading'

            status_details['emotion_model'] = model_status
            status_details['age_gender_model'] = model_status

            if model_status != 'ready':
                models_ready = False
        
        if models_ready:
            readiness_message = 'All models ready for processing'
        elif status_details.get('emotion_model') == 'unavailable':
            readiness_message = 'DeepFace is not installed; install deepface and tensorflow, then restart the backend'
        elif status_details.get('emotion_model') == 'error':
            readiness_message = 'DeepFace models failed to load; check backend logs and restart the backend'
        else:
            readiness_message = 'Models are still loading, please wait...'

        return jsonify({
            'ready': models_ready,
            'status': 'ready' if models_ready else status_details.get('emotion_model', 'loading'),
            'models': status_details,
            'message': readiness_message
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

        if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
            return jsonify({
                'error': 'Image dimensions are too large',
                'max_pixels': MAX_IMAGE_PIXELS
            }), 400
            
        print(f"🔍 Processing image of size: {img.shape}")
        
        # The current API requires both DeepFace models before processing.
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            if not getattr(face_detector, 'deepface_available', False):
                return jsonify({
                    'error': 'DeepFace is not installed',
                    'loading': False,
                    'message': 'Install deepface and tensorflow, then restart the backend.'
                }), 503

            if getattr(face_detector, 'model_preload_failed', False):
                return jsonify({
                    'error': 'DeepFace models failed to load',
                    'loading': False,
                    'message': 'Check the backend logs, then restart the backend.'
                }), 503

            if not face_detector.emotion_model_loaded or not face_detector.age_gender_model_loaded:
                print("⏳ Models still loading - this may take a moment...")
                return jsonify({
                    'error': 'Models are still loading. Please wait a moment and try again.',
                    'loading': True,
                    'message': 'AI models are initializing. Check /ready for current status.'
                }), 503  # Service Temporarily Unavailable
            
        # Detect faces with DeepFace analysis and heuristic validation
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
        
    except RequestEntityTooLarge:
        return handle_request_entity_too_large(None)
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

@app.route('/')
@app.route('/<path:path>')
def frontend(path=''):
    """Serve the compiled React app for non-API routes."""
    if not FRONTEND_DIR.is_dir():
        return jsonify({
            'error': 'Frontend build not found',
            'message': 'Run npm run build before starting the production server.'
        }), 503

    if path:
        return send_from_directory(FRONTEND_DIR, path)

    return send_from_directory(FRONTEND_DIR, 'index.html')

if __name__ == '__main__':
    print("Starting Face Detection Web App...")
    # Get port from environment variable (Render sets this)
    # Default to 5050 instead of 5000 to avoid common port conflicts
    port = int(os.environ.get("PORT", 5050))
    print(f"Server will start on port {port}")
    # Use 0.0.0.0 to bind to all interfaces
    app.run(host='0.0.0.0', port=port, debug=False)
