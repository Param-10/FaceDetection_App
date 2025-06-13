#!/usr/bin/env python
"""
Enhanced Face Detection Model with Autonomous Improvement
Features:
- Self-validation of results
- Automatic data collection for retraining
- Confidence-based result filtering
- Quality assessment and feedback loops
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import hashlib
import sqlite3
from pathlib import Path

# No more PyTorch imports to save memory
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace imported successfully!")
except ImportError as e:
    print(f"❌ DeepFace import failed: {e}")
    DEEPFACE_AVAILABLE = False
except Exception as e:
    print(f"❌ DeepFace error: {e}")
    DEEPFACE_AVAILABLE = False

class ModelDataCollector:
    """Collects and manages training data for model improvement"""
    
    def __init__(self, data_dir="model_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "high_confidence").mkdir(exist_ok=True)
        (self.data_dir / "low_confidence").mkdir(exist_ok=True)
        (self.data_dir / "rejected").mkdir(exist_ok=True)
        (self.data_dir / "validated").mkdir(exist_ok=True)
        
        # Initialize database for tracking
        self.db_path = self.data_dir / "model_feedback.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for tracking model performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_hash TEXT,
                num_faces INTEGER,
                avg_confidence REAL,
                predictions TEXT,
                validation_score REAL,
                accepted BOOLEAN,
                feedback_source TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                total_predictions INTEGER,
                accepted_predictions INTEGER,
                avg_confidence REAL,
                accuracy_trend REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, image, predictions, validation_score, accepted, feedback_source="auto"):
        """Log a prediction for tracking and analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create image hash for deduplication
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        
        # Calculate average confidence
        avg_conf = np.mean([p['confidence'] for p in predictions]) if predictions else 0
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, image_hash, num_faces, avg_confidence, predictions, validation_score, accepted, feedback_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            image_hash,
            len(predictions),
            avg_conf,
            json.dumps(predictions),
            validation_score,
            accepted,
            feedback_source
        ))
        
        conn.commit()
        conn.close()
    
    def get_model_performance_stats(self, days=30):
        """Get model performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) as accepted,
                AVG(avg_confidence) as avg_conf,
                AVG(validation_score) as avg_validation
            FROM predictions 
            WHERE timestamp > datetime('now', '-{} days')
        '''.format(days))
        
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] > 0:
            total, accepted, avg_conf, avg_validation = result
            return {
                'total_predictions': total,
                'acceptance_rate': accepted / total if total > 0 else 0,
                'avg_confidence': avg_conf or 0,
                'avg_validation_score': avg_validation or 0
            }
        return None

class ResultValidator:
    """Validates model predictions for quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'min_confidence': 0.6,
            'max_faces_per_image': 10,
            'min_face_size_ratio': 0.02,  # Face should be at least 2% of image
            'age_bounds': (1, 100),
            'emotion_consistency_threshold': 0.3
        }
    
    def validate_predictions(self, image, predictions):
        """
        Comprehensive validation of model predictions
        Returns: (is_valid, validation_score, issues)
        """
        if not predictions:
            return False, 0.0, ["No faces detected"]
        
        issues = []
        validation_scores = []
        
        # Rule 1: Check number of faces (detect mass false positives)
        if len(predictions) > self.validation_rules['max_faces_per_image']:
            issues.append(f"Too many faces detected: {len(predictions)}")
            validation_scores.append(0.1)
        else:
            validation_scores.append(1.0)
        
        # Rule 2: Check individual face validations
        for i, face in enumerate(predictions):
            face_score = self._validate_single_face(image, face)
            validation_scores.append(face_score)
            
            if face_score < 0.5:
                issues.append(f"Face {i+1} failed validation (score: {face_score:.2f})")
        
        # Rule 3: Check for consistency across faces
        if len(predictions) > 1:
            consistency_score = self._check_consistency(predictions)
            validation_scores.append(consistency_score)
            
            if consistency_score < 0.7:
                issues.append("Inconsistent predictions across faces")
        
        # Rule 4: Statistical outlier detection
        outlier_score = self._detect_outliers(predictions)
        validation_scores.append(outlier_score)
        
        if outlier_score < 0.6:
            issues.append("Statistical outliers detected in predictions")
        
        # Calculate overall validation score
        overall_score = np.mean(validation_scores)
        is_valid = overall_score >= 0.7 and len(issues) == 0
        
        return is_valid, overall_score, issues
    
    def _validate_single_face(self, image, face):
        """Validate a single face prediction"""
        score = 1.0
        
        # Check confidence
        if face['confidence'] < self.validation_rules['min_confidence']:
            score *= 0.5
        
        # Check face size relative to image
        x1, y1, x2, y2 = face['box']
        face_area = (x2 - x1) * (y2 - y1)
        image_area = image.shape[0] * image.shape[1]
        size_ratio = face_area / image_area
        
        if size_ratio < self.validation_rules['min_face_size_ratio']:
            score *= 0.6
        
        # Check age bounds
        if face.get('age'):
            age = face['age']
            if not (self.validation_rules['age_bounds'][0] <= age <= self.validation_rules['age_bounds'][1]):
                score *= 0.3
        
        # Check for reasonable emotion confidence
        if face.get('emotion'):
            # If emotion is detected, confidence should be reasonable
            if face['confidence'] < 0.4:
                score *= 0.7
        
        return score
    
    def _check_consistency(self, predictions):
        """Check consistency across multiple face predictions"""
        if len(predictions) <= 1:
            return 1.0
        
        # Check age consistency (should be in reasonable ranges for group photos)
        ages = [p['age'] for p in predictions if p.get('age')]
        if len(ages) > 1:
            age_std = np.std(ages)
            # If age standard deviation is very high, might be inconsistent
            if age_std > 25:  # More than 25 years difference is suspicious
                return 0.5
        
        # Check if all detected genders are the same (might indicate bias)
        genders = [p['gender'] for p in predictions if p.get('gender')]
        if len(genders) > 2 and len(set(genders)) == 1:
            return 0.8  # Slightly suspicious but not necessarily wrong
        
        return 1.0
    
    def _detect_outliers(self, predictions):
        """Detect statistical outliers in predictions"""
        if len(predictions) <= 2:
            return 1.0
        
        confidences = [p['confidence'] for p in predictions]
        
        # Check for extreme confidence variations
        conf_std = np.std(confidences)
        conf_mean = np.mean(confidences)
        
        # If there's one very high confidence and others very low, suspicious
        outlier_count = sum(1 for c in confidences if abs(c - conf_mean) > 2 * conf_std)
        
        if outlier_count > len(predictions) * 0.3:  # More than 30% outliers
            return 0.4
        
        return 1.0

class AdaptiveLearningSystem:
    """Manages adaptive learning and model improvement"""
    
    def __init__(self, data_collector, validator):
        self.data_collector = data_collector
        self.validator = validator
        self.improvement_threshold = 0.8  # Trigger retraining when accuracy drops below this
        self.confidence_adjustment_factor = 0.95  # Gradually increase standards
    
    def should_trigger_retraining(self):
        """Determine if model retraining should be triggered"""
        stats = self.data_collector.get_model_performance_stats(days=7)
        
        if not stats:
            return False
        
        # Trigger retraining if:
        # 1. Acceptance rate is too low
        # 2. Average confidence is dropping
        # 3. We have enough data samples
        
        return (
            stats['acceptance_rate'] < self.improvement_threshold and
            stats['total_predictions'] > 50  # Minimum samples for meaningful retraining
        )
    
    def adjust_confidence_thresholds(self):
        """Dynamically adjust confidence thresholds based on performance"""
        stats = self.data_collector.get_model_performance_stats(days=14)
        
        if not stats:
            return
        
        # If model is performing well, we can be more strict
        if stats['acceptance_rate'] > 0.9:
            self.validator.validation_rules['min_confidence'] *= 1.02  # Increase threshold
        elif stats['acceptance_rate'] < 0.7:
            self.validator.validation_rules['min_confidence'] *= 0.98  # Decrease threshold
        
        # Keep thresholds in reasonable bounds
        self.validator.validation_rules['min_confidence'] = max(0.4, min(0.8, 
            self.validator.validation_rules['min_confidence']))

class FaceDetectionModel:
    def __init__(self):
        # Initialize the face cascade for OpenCV - primary detector now
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade classifier")
            print("Successfully loaded OpenCV face detector")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            self.face_cascade = None
            
        # Initialize eye cascade for face validation
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            if self.eye_cascade.empty():
                print("❌ Warning: Could not load eye cascade for validation")
                self.eye_cascade = None
            else:
                print("✅ Eye cascade loaded for face validation")
        except Exception as e:
            print(f"Warning: Eye cascade failed to load: {e}")
            self.eye_cascade = None
                
        # Initialize DeepFace models
        self.emotion_model_loaded = False
        self.age_gender_model_loaded = False
        
        # Don't try to load DeepFace models if the package is not available
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace is not available. Face analysis will be limited to detection only.")
        else:
            print("🧠 DeepFace is available! Enhanced analysis features enabled.")
        
        # Initialize autonomous improvement systems
        self.data_collector = ModelDataCollector()
        self.validator = ResultValidator()
        self.adaptive_system = AdaptiveLearningSystem(self.data_collector, self.validator)
        
        print("🤖 Autonomous model improvement system initialized!")
        
        # Preload DeepFace models during initialization to prevent first-request delays
        if DEEPFACE_AVAILABLE:
            print("🔄 Preloading DeepFace models for faster first-time inference...")
            self._preload_deepface_models()
        
    def _ensure_models_loaded(self):
        """Ensure DeepFace models are loaded when needed"""
        if not DEEPFACE_AVAILABLE:
            return
            
        if not self.emotion_model_loaded or not self.age_gender_model_loaded:
            print("🔄 Loading DeepFace models for the first time...")
            # This will trigger model downloads if needed
            try:
                # Pre-load models by running a simple analysis
                sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
                
                if not self.emotion_model_loaded:
                    try:
                        print("📊 Loading emotion analysis model...")
                        # Use more accurate emotion model
                        DeepFace.analyze(sample_img, actions=['emotion'], 
                                       detector_backend='opencv', 
                                       enforce_detection=False, silent=True)
                        self.emotion_model_loaded = True
                        print("✅ Emotion model loaded successfully!")
                    except Exception as e:
                        print(f"❌ Emotion model failed to load: {e}")
                
                if not self.age_gender_model_loaded:
                    try:
                        print("👤 Loading age/gender analysis model...")
                        # Use more accurate age/gender models
                        DeepFace.analyze(sample_img, actions=['age', 'gender'], 
                                       detector_backend='opencv',
                                       enforce_detection=False, silent=True)
                        self.age_gender_model_loaded = True
                        print("✅ Age/Gender model loaded successfully!")
                    except Exception as e:
                        print(f"❌ Age/Gender model failed to load: {e}")
                        
            except Exception as e:
                print(f"❌ Error loading DeepFace models: {e}")
                
        if self.emotion_model_loaded and self.age_gender_model_loaded:
            print("🎉 All AI models are ready for enhanced face analysis!")
    
    def _preload_deepface_models(self):
        """Preload DeepFace models during initialization to prevent first-request delays"""
        if not DEEPFACE_AVAILABLE:
            return
            
        try:
            print("📦 Preloading DeepFace models (this may take 30-60 seconds for first time)...")
            
            # Create a small sample image for model initialization
            sample_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            
            # Preload emotion model
            if not self.emotion_model_loaded:
                try:
                    print("   🧠 Loading emotion analysis model...")
                    DeepFace.analyze(sample_img, actions=['emotion'], 
                                   detector_backend='opencv', 
                                   enforce_detection=False, silent=True)
                    self.emotion_model_loaded = True
                    print("   ✅ Emotion model ready!")
                except Exception as e:
                    print(f"   ❌ Emotion model failed: {e}")
            
            # Preload age/gender model
            if not self.age_gender_model_loaded:
                try:
                    print("   👤 Loading age/gender analysis model...")
                    DeepFace.analyze(sample_img, actions=['age', 'gender'], 
                                   detector_backend='opencv',
                                   enforce_detection=False, silent=True)
                    self.age_gender_model_loaded = True
                    print("   ✅ Age/Gender model ready!")
                except Exception as e:
                    print(f"   ❌ Age/Gender model failed: {e}")
            
            if self.emotion_model_loaded and self.age_gender_model_loaded:
                print("🎉 All DeepFace models preloaded successfully!")
                print("🚀 Server is ready for immediate face detection requests!")
            else:
                print("⚠️  Some models failed to preload - first requests may be slower")
                
        except Exception as e:
            print(f"⚠️  Model preloading failed: {e}")
            print("🔄 Models will load on first request instead")
    
    def _detect_faces_opencv(self, image):
        """Primary method using OpenCV's built-in face detector with improved accuracy"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve detection
        gray = cv2.equalizeHist(gray)
        
        # Apply additional preprocessing for better face detection
        # Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Slightly increased for more stability  
            minNeighbors=8,    # Increased to reduce false positives significantly
            minSize=(80, 80),  # Larger minimum for better quality faces only
            maxSize=(350, 350), # Slightly reduced max size 
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        boxes = []
        scores = []
        
        # Filter overlapping detections
        filtered_faces = self._filter_overlapping_faces(faces)
        
        for (x, y, w, h) in filtered_faces:
            # Add padding around face for better analysis (10% on each side)
            padding = int(min(w, h) * 0.1)
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding) 
            w_padded = min(image.shape[1] - x_padded, w + 2*padding)
            h_padded = min(image.shape[0] - y_padded, h + 2*padding)
            
            boxes.append([x_padded, y_padded, x_padded + w_padded, y_padded + h_padded])
            scores.append(1.0)  # OpenCV doesn't provide confidence scores
            
        return boxes, scores
    
    def _filter_overlapping_faces(self, faces):
        """Filter out overlapping face detections to reduce duplicates"""
        if len(faces) == 0:
            return faces
            
        # Sort faces by area (larger faces first)
        faces_with_area = [(x, y, w, h, w*h) for x, y, w, h in faces]
        faces_with_area.sort(key=lambda x: x[4], reverse=True)
        
        filtered = []
        
        for i, (x1, y1, w1, h1, area1) in enumerate(faces_with_area):
            is_overlapping = False
            
            for x2, y2, w2, h2, area2 in filtered:
                # Calculate intersection over union (IoU)
                intersection_x = max(x1, x2)
                intersection_y = max(y1, y2)
                intersection_w = min(x1 + w1, x2 + w2) - intersection_x
                intersection_h = min(y1 + h1, y2 + h2) - intersection_y
                
                if intersection_w > 0 and intersection_h > 0:
                    intersection_area = intersection_w * intersection_h
                    union_area = area1 + area2 - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    
                    # If IoU is greater than 0.2, consider it overlapping (more aggressive filtering)
                    if iou > 0.2:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered.append((x1, y1, w1, h1, area1))
        
        # Return only the coordinates (remove area)
        return [(x, y, w, h) for x, y, w, h, area in filtered]

    def detect_faces(self, image):
        """
        Enhanced face detection with autonomous quality checking and improvement
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            result_img: Image with bounding boxes drawn around detected faces
            face_data: List of dictionaries containing face information
            metadata: Additional information about prediction quality and validation
        """
        # Detect faces using OpenCV
        if self.face_cascade is not None:
            boxes, scores = self._detect_faces_opencv(image)
        else:
            return image.copy(), [], {'error': 'No detection method available'}
        
        # Only attempt to load DeepFace models if the package is available
        if DEEPFACE_AVAILABLE:
            self._ensure_models_loaded()
        
        # Process detected faces
        face_data = []
        result_img = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Skip if face is too small
            if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue
                
            # Calculate face detection quality score
            face_quality_score = self._calculate_face_quality(face_img, x1, y1, x2, y2, image.shape)
            
            # Skip faces with low quality scores to reduce false positives
            if face_quality_score < 0.5:  # Threshold for minimum face quality
                continue
            
            # Additional validation: check for eyes if eye cascade is available
            if self.eye_cascade is not None and not self._validate_face_has_eyes(face_img):
                continue
            
            face_info = {
                'box': (x1, y1, x2, y2),
                'confidence': face_quality_score,  # Start with detection quality, enhance with analysis
                'emotion': None,
                'age': None,
                'gender': None
            }
            
            # Only attempt DeepFace analysis if available
            if DEEPFACE_AVAILABLE:
                # Analyze face for emotion, age, and gender
                try:
                    if self.emotion_model_loaded and self.age_gender_model_loaded:
                        # Run comprehensive analysis once
                        analysis_results = self._analyze_face_with_ensemble(face_img)
                        if analysis_results:
                            # Calculate overall confidence from individual confidences
                            confidences = []
                            if analysis_results.get('emotion'):
                                face_info['emotion'] = analysis_results['emotion']
                                confidences.append(analysis_results.get('emotion_confidence', 0))
                            if analysis_results.get('age'):
                                face_info['age'] = int(analysis_results['age'])  # Convert to int for JSON
                                confidences.append(analysis_results.get('age_confidence', 0))
                            if analysis_results.get('gender'):
                                face_info['gender'] = analysis_results['gender']
                                confidences.append(analysis_results.get('gender_confidence', 0))
                            
                            # Combine detection quality with analysis confidence
                            if confidences:
                                analysis_confidence = sum(confidences) / len(confidences)
                                # Weighted average: 30% detection quality, 70% analysis confidence
                                face_info['confidence'] = float(0.3 * face_quality_score + 0.7 * analysis_confidence)
                            else:
                                face_info['confidence'] = float(face_quality_score)  # Use detection quality only
                except Exception as e:
                    print(f"Error analyzing face: {e}")
            
            # Ensure all values are JSON serializable
            face_info['confidence'] = float(face_info['confidence'])
            if face_info['age'] is not None:
                face_info['age'] = int(face_info['age'])
            
            face_data.append(face_info)
        
        # Validate predictions using autonomous system
        is_valid, validation_score, issues = self.validator.validate_predictions(image, face_data)
        
        # Log prediction for learning
        self.data_collector.log_prediction(image, face_data, validation_score, is_valid)
        
        # Adjust thresholds based on recent performance
        self.adaptive_system.adjust_confidence_thresholds()
        
        # Check if retraining should be triggered
        should_retrain = self.adaptive_system.should_trigger_retraining()
        
        # Prepare metadata
        metadata = {
            'validation_score': validation_score,
            'is_valid': is_valid,
            'issues': issues,
            'should_retrain': should_retrain,
            'num_faces_detected': len(face_data),
            'detection_quality': 'high' if validation_score > 0.8 else 'medium' if validation_score > 0.6 else 'low'
        }
        
        # Only return results if they pass validation
        if not is_valid:
            print(f"⚠️ Prediction rejected (score: {validation_score:.2f}): {', '.join(issues)}")
            return image.copy(), [], metadata  # Return empty results for rejected predictions
        
        # Draw visualization for accepted results
        for face_info in face_data:
            x1, y1, x2, y2 = face_info['box']
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw labels with emotion, age, and gender
            label = ""
            if face_info['emotion']:
                label += f"Emotion: {face_info['emotion']} "
            if face_info['age']:
                label += f"Age: {face_info['age']} "
            if face_info['gender']:
                label += f"Gender: {face_info['gender']}"
                
            if label:
                # Add background for text
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(result_img, (x1, y1 - 20), (x1 + label_size[0], y1), (255, 0, 0), -1)
                cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        print(f"✅ Prediction accepted (score: {validation_score:.2f}, faces: {len(face_data)})")
        return result_img, face_data, metadata

    def _preprocess_face_for_analysis(self, face_img):
        """Enhanced face preprocessing for better analysis accuracy"""
        # Resize face to optimal size for analysis (224x224 is standard)
        target_size = (224, 224)
        face_resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply histogram equalization to improve contrast
        lab = cv2.cvtColor(face_resized, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        face_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight gaussian blur to reduce noise
        face_smooth = cv2.GaussianBlur(face_enhanced, (3, 3), 0)
        
        return face_smooth

    def _analyze_face_with_ensemble(self, face_img):
        """Use ensemble approach for more accurate predictions"""
        results = {
            'emotion': None,
            'age': None, 
            'gender': None,
            'emotion_confidence': 0,
            'age_confidence': 0,
            'gender_confidence': 0
        }
        
        try:
            # Preprocess face for better analysis
            processed_face = self._preprocess_face_for_analysis(face_img)
            
            # Try multiple detector backends for best results
            detector_backends = ['opencv', 'ssd', 'dlib']
            best_results = None
            best_confidence = 0
            
            for backend in detector_backends:
                try:
                    analysis = DeepFace.analyze(
                        processed_face, 
                        actions=['emotion', 'age', 'gender'],
                        detector_backend=backend,
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if analysis and len(analysis) > 0:
                        result = analysis[0]
                        
                        # Calculate overall confidence for this result
                        confidence_score = 0
                        confidence_count = 0
                        
                        if 'emotion' in result:
                            emotion_conf = max(result['emotion'].values()) / 100.0
                            confidence_score += emotion_conf
                            confidence_count += 1
                        
                        if 'gender' in result:
                            gender_conf = max(result['gender'].values()) / 100.0
                            confidence_score += gender_conf
                            confidence_count += 1
                        
                        if confidence_count > 0:
                            avg_confidence = confidence_score / confidence_count
                            if avg_confidence > best_confidence:
                                best_confidence = avg_confidence
                                best_results = result
                                break  # Use first good result for speed
                
                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
                    continue
             
            if best_results:
                result = best_results
                
                # Extract emotion with confidence
                if 'dominant_emotion' in result and 'emotion' in result:
                    emotion = result['dominant_emotion']
                    emotion_scores = result['emotion']
                    emotion_confidence = emotion_scores.get(emotion, 0) / 100.0
                    
                    # Only accept emotion predictions with >40% confidence
                    if emotion_confidence > 0.4:
                        results['emotion'] = emotion
                        results['emotion_confidence'] = emotion_confidence
                
                # Extract age with validation
                if 'age' in result:
                    predicted_age = result['age']
                    # Validate age range (reasonable bounds)
                    if 1 <= predicted_age <= 100:
                        results['age'] = predicted_age
                        # Calculate age confidence based on prediction certainty
                        # Ages in middle ranges (20-60) are typically more accurate
                        if 20 <= predicted_age <= 60:
                            age_confidence = 0.85
                        elif 15 <= predicted_age <= 70:
                            age_confidence = 0.75
                        else:
                            age_confidence = 0.65
                        results['age_confidence'] = age_confidence
                
                # Extract gender with confidence
                if 'dominant_gender' in result and 'gender' in result:
                    gender = result['dominant_gender']
                    gender_scores = result['gender']
                    
                    # DeepFace uses 'Man'/'Woman', normalize to standard format
                    if gender == 'Man':
                        gender = 'Male'
                        gender_confidence = gender_scores.get('Man', 0) / 100.0
                    elif gender == 'Woman':
                        gender = 'Female'  
                        gender_confidence = gender_scores.get('Woman', 0) / 100.0
                    else:
                        gender_confidence = 0
                    
                    # Only accept gender predictions with >55% confidence (slightly lower for better recall)
                    if gender_confidence > 0.55:
                        results['gender'] = gender
                        results['gender_confidence'] = gender_confidence
                
                print(f"🔍 Analysis results - Emotion: {results['emotion']} ({results['emotion_confidence']:.2f}), "
                      f"Age: {results['age']}, Gender: {results['gender']} ({results['gender_confidence']:.2f})")
                
        except Exception as e:
            print(f"❌ Enhanced face analysis failed: {e}")
            
        return results

    def _calculate_face_quality(self, face_img, x1, y1, x2, y2, image_shape):
        """Calculate face detection quality score based on multiple factors"""
        height, width = image_shape[:2]
        face_width = x2 - x1
        face_height = y2 - y1
        face_area = face_width * face_height
        
        quality_score = 1.0
        
        # 1. Size Quality (30% weight)
        # Optimal face size is 15-40% of image width
        face_size_ratio = face_width / width
        if 0.15 <= face_size_ratio <= 0.4:
            size_quality = 1.0
        elif 0.1 <= face_size_ratio <= 0.6:
            size_quality = 0.8
        elif 0.05 <= face_size_ratio <= 0.8:
            size_quality = 0.6
        else:
            size_quality = 0.4
        
        # 2. Position Quality (20% weight)
        # Face should be reasonably centered, not cut off at edges
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if face is too close to edges (potential cutoff)
        edge_margin = 0.05  # 5% margin from edges
        if (x1 < width * edge_margin or x2 > width * (1 - edge_margin) or
            y1 < height * edge_margin or y2 > height * (1 - edge_margin)):
            position_quality = 0.7
        else:
            position_quality = 1.0
        
        # 3. Aspect Ratio Quality (20% weight)
        # Face should have reasonable aspect ratio (not too stretched)
        aspect_ratio = face_width / face_height
        if 0.7 <= aspect_ratio <= 1.3:  # Good face aspect ratio
            aspect_quality = 1.0
        elif 0.5 <= aspect_ratio <= 1.8:
            aspect_quality = 0.8
        else:
            aspect_quality = 0.6
        
        # 4. Image Clarity (30% weight)
        # Check for blur using Laplacian variance
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Normalize blur score (higher variance = less blur = better quality)
        if laplacian_var > 500:
            clarity_quality = 1.0
        elif laplacian_var > 200:
            clarity_quality = 0.9
        elif laplacian_var > 100:
            clarity_quality = 0.7
        elif laplacian_var > 50:
            clarity_quality = 0.5
        else:
            clarity_quality = 0.3
        
        # Weighted combination
        quality_score = (0.3 * size_quality + 
                        0.2 * position_quality + 
                        0.2 * aspect_quality + 
                        0.3 * clarity_quality)
        
        # Ensure score is between 0.3 and 0.95 (never perfect, never too low)
        quality_score = max(0.3, min(0.95, quality_score))
        
        return quality_score
    
    def _validate_face_has_eyes(self, face_img):
        """Validate that a detected face actually contains eyes (reduces false positives)"""
        if self.eye_cascade is None:
            return True  # Skip validation if eye cascade not available
            
        try:
            # Convert face to grayscale for eye detection
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10),
                maxSize=(50, 50)
            )
            
            # A valid face should have at least 1 eye detected (sometimes profile faces show only 1 eye)
            return len(eyes) >= 1
            
        except Exception as e:
            print(f"Eye validation failed: {e}")
            return True  # If validation fails, assume face is valid
    
    def get_model_performance_dashboard(self):
        """Get comprehensive model performance data for monitoring"""
        stats_7d = self.data_collector.get_model_performance_stats(days=7)
        stats_30d = self.data_collector.get_model_performance_stats(days=30)
        
        dashboard = {
            'last_7_days': stats_7d,
            'last_30_days': stats_30d,
            'current_thresholds': {
                'min_confidence': self.validator.validation_rules['min_confidence'],
                'max_faces_per_image': self.validator.validation_rules['max_faces_per_image'],
                'min_face_size_ratio': self.validator.validation_rules['min_face_size_ratio']
            },
            'should_retrain': self.adaptive_system.should_trigger_retraining(),
            'recommendations': self._generate_recommendations()
        }
        
        return dashboard
    
    def _generate_recommendations(self):
        """Generate recommendations for model improvement"""
        recommendations = []
        
        stats = self.data_collector.get_model_performance_stats(days=7)
        if stats:
            if stats['acceptance_rate'] < 0.7:
                recommendations.append("Consider collecting more training data - low acceptance rate detected")
            
            if stats['avg_confidence'] < 0.6:
                recommendations.append("Model confidence is low - may need retraining with higher quality data")
            
            if stats['total_predictions'] > 100 and stats['acceptance_rate'] > 0.9:
                recommendations.append("Model performing well - consider increasing quality thresholds")
        
        return recommendations