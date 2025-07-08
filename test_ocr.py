#!/usr/bin/env python3
"""
Test script for OCR detector
"""

import numpy as np
from PIL import Image
import os
import sys
from ocr_detector import detect_space_names

def test_ocr_with_sample_image():
    """Test OCR detector with a sample image"""
    
    # Look for sample images in the images directory
    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"Images directory '{images_dir}' not found")
        return
    
    # Find first image file
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No image files found in images directory")
        return
    
    # Use the first image
    image_path = os.path.join(images_dir, image_files[0])
    print(f"Testing OCR with image: {image_path}")
    
    try:
        # Load image
        image = Image.open(image_path)
        image_array = np.array(image)
        
        print(f"Image shape: {image_array.shape}")
        print("Starting OCR detection...")
        
        # Detect space names
        space_names = detect_space_names(image_array)
        
        print(f"\nOCR Results:")
        print(f"Total space names detected: {len(space_names)}")
        
        for i, space in enumerate(space_names):
            print(f"\nSpace {i+1}:")
            print(f"  Name: '{space['name']}'")
            print(f"  Confidence: {space['confidence']:.3f}")
            print(f"  Method: {space.get('method', 'unknown')}")
            print(f"  BBox: {space['bbox']}")
            print(f"  Center: ({space['center']['x']:.1f}, {space['center']['y']:.1f})")
        
        if len(space_names) == 0:
            print("\nNo space names detected. This could be due to:")
            print("1. No text in the image")
            print("2. Text is too small or unclear")
            print("3. OCR confidence is too low")
            print("4. Text is filtered out by our criteria")
            print("\nCheck the 'ocr_preprocessed_debug.png' file to see the preprocessed image.")
        
    except Exception as e:
        print(f"Error during OCR test: {str(e)}")
        import traceback
        traceback.print_exc()

def create_test_image():
    """Create a simple test image with text"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add some text
    draw.text((100, 100), "KITCHEN", fill='black', font=font)
    draw.text((400, 150), "LIVING ROOM", fill='black', font=font)
    draw.text((200, 300), "BEDROOM", fill='black', font=font)
    draw.text((500, 350), "BATHROOM", fill='black', font=font)
    draw.text((300, 500), "DINING", fill='black', font=font)
    
    # Save test image
    test_image_path = "test_floor_plan.png"
    img.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    
    return test_image_path

def test_ocr_with_created_image():
    """Test OCR with a created test image"""
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        # Load image
        image = Image.open(test_image_path)
        image_array = np.array(image)
        
        print(f"\nTesting OCR with created image: {test_image_path}")
        print(f"Image shape: {image_array.shape}")
        print("Starting OCR detection...")
        
        # Detect space names
        space_names = detect_space_names(image_array)
        
        print(f"\nOCR Results:")
        print(f"Total space names detected: {len(space_names)}")
        
        for i, space in enumerate(space_names):
            print(f"\nSpace {i+1}:")
            print(f"  Name: '{space['name']}'")
            print(f"  Confidence: {space['confidence']:.3f}")
            print(f"  Method: {space.get('method', 'unknown')}")
            print(f"  BBox: {space['bbox']}")
            print(f"  Center: ({space['center']['x']:.1f}, {space['center']['y']:.1f})")
        
        # Clean up test image
        os.remove(test_image_path)
        
    except Exception as e:
        print(f"Error during OCR test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("OCR Detector Test Script")
    print("=" * 50)
    
    # Test with created image first
    test_ocr_with_created_image()
    
    print("\n" + "=" * 50)
    
    # Test with sample image if available
    test_ocr_with_sample_image() 