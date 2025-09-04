#!/usr/bin/env python3
"""
Debug script to test the face parsing and makeup functions
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.append('.')

try:
    from test import evaluate
    from makeup import hair, get_color_scheme
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_face_parsing():
    """Test face parsing with a sample image"""
    print("\n🔍 Testing face parsing...")
    
    # Check if model exists
    model_path = 'cp/79999_iter.pth'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Check if we have a test image
    test_image = 'temp_image.jpg'
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return None
    
    try:
        print(f"📸 Processing image: {test_image}")
        parsing = evaluate(test_image, model_path)
        
        if parsing is not None:
            print(f"✅ Parsing successful!")
            print(f"   Shape: {parsing.shape}")
            print(f"   Unique values: {np.unique(parsing)}")
            print(f"   Min/Max: {parsing.min()}/{parsing.max()}")
            return parsing
        else:
            print("❌ Parsing returned None")
            return None
            
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        return None

def test_makeup_function(parsing):
    """Test the makeup function with the parsing result"""
    print("\n💄 Testing makeup function...")
    
    if parsing is None:
        print("❌ No parsing result to test with")
        return False
    
    # Create a test image
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray image
    
    # Test different face parts
    test_parts = [2, 3, 4, 5, 9, 10, 12, 13, 17]
    test_color = [100, 50, 200]  # Purple color
    
    for part in test_parts:
        try:
            print(f"   Testing part {part}...")
            result = hair(test_image, parsing, part, test_color)
            print(f"   ✅ Part {part} processed successfully")
        except Exception as e:
            print(f"   ❌ Error with part {part}: {e}")
            return False
    
    return True

def main():
    print("🚀 Starting debug test...")
    
    # Test face parsing
    parsing = test_face_parsing()
    
    # Test makeup function
    if parsing is not None:
        success = test_makeup_function(parsing)
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
    else:
        print("\n❌ Cannot test makeup function without parsing result")

if __name__ == "__main__":
    main()