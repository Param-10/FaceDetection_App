#!/usr/bin/env python
import cv2
import numpy as np
import os
from FaceDetection_WebApp.models.face_detection_model import FaceDetectionModel

def test_face_detection():
    """
    Test the face detection model with a sample image.
    If no sample image is available, create a simple test image.
    """
    print("Testing face detection model...")
    
    # Initialize the model
    model = FaceDetectionModel()
    
    # Create a simple test image if needed
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
    
    if not os.path.exists(test_image_path):
        # Create a blank image (640x480) with a simple rectangle to simulate a face
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(200)  # Light gray background
        
        # Draw a rectangle to simulate a face
        cv2.rectangle(img, (250, 150), (390, 330), (100, 100, 100), -1)
        
        # Save the test image
        cv2.imwrite(test_image_path, img)
        print(f"Created test image at {test_image_path}")
    
    # Load the test image
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return False
    
    # Process the image
    try:
        result_img = model.detect_faces(img)
        print("Face detection completed successfully!")
        
        # Save the result
        result_path = os.path.join(os.path.dirname(__file__), 'test_result.jpg')
        cv2.imwrite(result_path, result_img)
        print(f"Result saved to {result_path}")
        return True
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_face_detection()
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed!")
