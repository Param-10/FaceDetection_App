import cv2
import numpy as np

# No more PyTorch imports to save memory
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("DeepFace is not available or has compatibility issues. Face analysis features will be disabled.")
    DEEPFACE_AVAILABLE = False

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
                
        # Initialize DeepFace models
        self.emotion_model_loaded = False
        self.age_gender_model_loaded = False
        
        # Don't try to load DeepFace models if the package is not available
        if not DEEPFACE_AVAILABLE:
            print("DeepFace is not available. Face analysis will be limited to detection only.")
        
    def _ensure_models_loaded(self):
        """Ensure DeepFace models are loaded when needed"""
        if not DEEPFACE_AVAILABLE:
            return
            
        if not self.emotion_model_loaded or not self.age_gender_model_loaded:
            # This will trigger model downloads if needed
            try:
                # Pre-load models by running a simple analysis
                sample_img = np.zeros((100, 100, 3), dtype=np.uint8)
                try:
                    DeepFace.analyze(sample_img, actions=['emotion'], enforce_detection=False, silent=True)
                    self.emotion_model_loaded = True
                except:
                    print("Warning: Emotion model could not be loaded")
                
                try:
                    DeepFace.analyze(sample_img, actions=['age', 'gender'], enforce_detection=False, silent=True)
                    self.age_gender_model_loaded = True
                except:
                    print("Warning: Age/Gender model could not be loaded")
            except Exception as e:
                print(f"Error loading DeepFace models: {e}")
    
    def _detect_faces_opencv(self, image):
        """Primary method using OpenCV's built-in face detector"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        boxes = []
        scores = []
        
        for (x, y, w, h) in faces:
            boxes.append([x, y, x+w, y+h])
            scores.append(1.0)  # OpenCV doesn't provide confidence scores
            
        return boxes, scores
    
    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV's face detector.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            result_img: Image with bounding boxes drawn around detected faces
            face_data: List of dictionaries containing face information
        """
        # Draw bounding boxes and analyze faces
        result_img = image.copy()
        face_data = []
        
        # Detect faces using OpenCV
        if self.face_cascade is not None:
            boxes, scores = self._detect_faces_opencv(image)
        else:
            # No detection method available
            return result_img, []
        
        # Only attempt to load DeepFace models if the package is available
        if DEEPFACE_AVAILABLE:
            self._ensure_models_loaded()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Extract face region
            face_img = image[y1:y2, x1:x2]
            
            # Skip if face is too small
            if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                continue
                
            face_info = {
                'box': (x1, y1, x2, y2),
                'confidence': float(scores[i]) if isinstance(scores[i], (float, int)) else 1.0,
                'emotion': None,
                'age': None,
                'gender': None
            }
            
            # Only attempt DeepFace analysis if available
            if DEEPFACE_AVAILABLE:
                # Analyze face for emotion, age, and gender
                try:
                    if self.emotion_model_loaded:
                        emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                        if emotion_analysis and len(emotion_analysis) > 0:
                            face_info['emotion'] = emotion_analysis[0]['dominant_emotion']
                    
                    if self.age_gender_model_loaded:
                        age_gender_analysis = DeepFace.analyze(face_img, actions=['age', 'gender'], enforce_detection=False, silent=True)
                        if age_gender_analysis and len(age_gender_analysis) > 0:
                            face_info['age'] = age_gender_analysis[0]['age']
                            face_info['gender'] = age_gender_analysis[0]['dominant_gender']
                except Exception as e:
                    print(f"Error analyzing face: {e}")
            
            face_data.append(face_info)
            
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
        
        return result_img, face_data