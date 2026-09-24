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
        
        # Report whether analysis is ready while allowing the lightweight
        # OpenCV-only mode to serve detection requests.
        model_status = 'error'
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            if face_detector.emotion_model_loaded and face_detector.age_gender_model_loaded:
                model_status = 'ready'
            elif not getattr(face_detector, 'deepface_available', False):
                model_status = 'degraded'
            elif getattr(face_detector, 'model_preload_failed', False):
                model_status = 'degraded'
            else:
                model_status = 'loading'

            status_details['emotion_model'] = model_status
            status_details['age_gender_model'] = model_status
            status_details['analysis_available'] = model_status == 'ready'

            if model_status == 'loading':
                models_ready = False

        if models_ready and model_status == 'degraded':
            readiness_message = 'OpenCV detection is ready; DeepFace analysis is unavailable'
        elif models_ready:
            readiness_message = 'All models ready for processing'
        elif model_status == 'loading':
            readiness_message = 'Models are still loading, please wait...'
        else:
            readiness_message = 'Model initialization failed; check backend logs'

        return jsonify({
            'ready': models_ready,
            'status': model_status,
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
        
        # OpenCV-only mode is supported when DeepFace is unavailable or its
        # preload failed. A genuine in-progress load still returns 503.
        if hasattr(face_detector, 'emotion_model_loaded') and hasattr(face_detector, 'age_gender_model_loaded'):
            deepface_ready = face_detector.emotion_model_loaded and face_detector.age_gender_model_loaded
            deepface_unavailable = not getattr(face_detector, 'deepface_available', False)
            preload_failed = getattr(face_detector, 'model_preload_failed', False)

            if not deepface_ready and not deepface_unavailable and not preload_failed:
                print("⏳ Models still loading - this may take a moment...")
                return jsonify({
                    'error': 'Models are still loading. Please wait a moment and try again.',
                    'loading': True,
                    'message': 'AI models are initializing. Check /ready for current status.'
                }), 503  # Service Temporarily Unavailable

        # Detect faces with optional DeepFace analysis and heuristic validation
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
