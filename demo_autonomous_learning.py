#!/usr/bin/env python
"""
Demonstration of Autonomous Learning System
This script shows how the model automatically improves itself
"""

import cv2
import numpy as np
from face_detection_model import FaceDetectionModel
import time

def demonstrate_autonomous_learning():
    """Demonstrate the autonomous learning capabilities"""
    print("🤖 Autonomous Face Detection Learning System Demo")
    print("=" * 50)
    
    # Initialize the enhanced model
    detector = FaceDetectionModel()
    
    # Create some sample images for testing
    print("\n📊 Creating test scenarios...")
    
    # Scenario 1: Good quality face
    good_face = create_sample_face_image(200, 200, quality='high')
    
    # Scenario 2: Low quality/blurry face  
    poor_face = create_sample_face_image(50, 50, quality='low')
    
    # Scenario 3: Multiple faces (potential false positives)
    multi_face = create_multi_face_image()
    
    test_images = [
        ("High Quality Face", good_face),
        ("Low Quality Face", poor_face), 
        ("Multi-Face Scenario", multi_face)
    ]
    
    print("\n🔍 Testing Face Detection with Autonomous Validation...")
    
    for name, image in test_images:
        print(f"\n--- Testing: {name} ---")
        
        # Detect faces with autonomous validation
        result_img, face_data, metadata = detector.detect_faces(image)
        
        # Display results
        print(f"📋 Results:")
        print(f"   • Faces Detected: {metadata['num_faces_detected']}")
        print(f"   • Validation Score: {metadata['validation_score']:.3f}")
        print(f"   • Result Quality: {metadata['detection_quality']}")
        print(f"   • Prediction Valid: {'✅' if metadata['is_valid'] else '❌'}")
        
        if metadata['issues']:
            print(f"   • Issues Found: {', '.join(metadata['issues'])}")
        
        if metadata['should_retrain']:
            print(f"   • 🔄 Retraining Recommended")
    
    # Show performance dashboard
    print(f"\n📈 Model Performance Dashboard:")
    dashboard = detector.get_model_performance_dashboard()
    
    if dashboard['last_7_days']:
        stats = dashboard['last_7_days']
        print(f"   • Total Predictions (7 days): {stats['total_predictions']}")
        print(f"   • Acceptance Rate: {stats['acceptance_rate']:.1%}")
        print(f"   • Average Confidence: {stats['avg_confidence']:.3f}")
    else:
        print("   • No recent data available")
    
    print(f"\n⚙️  Current Thresholds:")
    thresholds = dashboard['current_thresholds']
    print(f"   • Min Confidence: {thresholds['min_confidence']:.3f}")
    print(f"   • Max Faces: {thresholds['max_faces_per_image']}")
    print(f"   • Min Face Size: {thresholds['min_face_size_ratio']:.3f}")
    
    if dashboard['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in dashboard['recommendations']:
            print(f"   • {rec}")
    
    print(f"\n🎯 Key Features Demonstrated:")
    print(f"   ✅ Automatic quality assessment")
    print(f"   ✅ Result validation and filtering") 
    print(f"   ✅ Performance tracking and analytics")
    print(f"   ✅ Adaptive threshold adjustment")
    print(f"   ✅ Retraining recommendations")
    print(f"   ✅ Data collection for improvement")

def create_sample_face_image(width, height, quality='high'):
    """Create a sample image for testing"""
    # Create a simple test image
    if quality == 'high':
        # Large, clear image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 200
        # Add some "face-like" features (just rectangles for demo)
        cv2.rectangle(image, (150, 100), (250, 200), (150, 150, 150), -1)  # Face
        cv2.rectangle(image, (170, 130), (180, 140), (0, 0, 0), -1)  # Eye
        cv2.rectangle(image, (220, 130), (230, 140), (0, 0, 0), -1)  # Eye
        cv2.rectangle(image, (190, 160), (210, 180), (100, 100, 100), -1)  # Nose/mouth
    else:
        # Small, blurry image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 180
        # Add noise to simulate poor quality
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        # Blur the image
        image = cv2.GaussianBlur(image, (15, 15), 0)
    
    return image

def create_multi_face_image():
    """Create an image that might trigger false positives"""
    # Create image with patterns that might confuse the detector
    image = np.ones((300, 400, 3), dtype=np.uint8) * 220
    
    # Add multiple rectangular patterns
    for i in range(5):
        x = 50 + i * 60
        y = 50 + (i % 2) * 100
        cv2.rectangle(image, (x, y), (x+40, y+60), (150, 150, 150), -1)
        # Add some dots that might look like eyes
        cv2.circle(image, (x+10, y+20), 3, (0, 0, 0), -1)
        cv2.circle(image, (x+30, y+20), 3, (0, 0, 0), -1)
    
    return image

if __name__ == "__main__":
    demonstrate_autonomous_learning() 