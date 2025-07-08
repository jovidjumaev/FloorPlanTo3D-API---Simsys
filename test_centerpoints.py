#!/usr/bin/env python3
"""
Test script for space name centerpoints functionality
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import os

def create_test_floor_plan():
    """Create a test floor plan image with room names"""
    
    # Create a larger white image
    img = Image.new('RGB', (1200, 800), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a larger font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw some walls (simple rectangles)
    wall_color = (100, 100, 100)
    
    # Outer walls
    draw.rectangle([50, 50, 1150, 750], outline=wall_color, width=8)
    
    # Internal walls
    draw.line([400, 50, 400, 400], fill=wall_color, width=8)  # Vertical wall
    draw.line([50, 400, 800, 400], fill=wall_color, width=8)  # Horizontal wall
    draw.line([800, 50, 800, 750], fill=wall_color, width=8)  # Vertical wall
    
    # Add room names in different areas
    rooms = [
        ("KITCHEN", (150, 200), (0, 0, 0)),
        ("LIVING ROOM", (550, 200), (0, 0, 0)),
        ("BEDROOM", (950, 200), (0, 0, 0)),
        ("BATHROOM", (150, 550), (0, 0, 0)),
        ("DINING", (550, 550), (0, 0, 0)),
        ("OFFICE", (950, 550), (0, 0, 0))
    ]
    
    for room_name, position, color in rooms:
        draw.text(position, room_name, fill=color, font=font)
    
    # Add some descriptive text that should be filtered out
    draw.text((100, 20), "3 Bedrooms / 2 Bathrooms", fill=(50, 50, 50), font=small_font)
    draw.text((100, 770), "Floor Plan - Scale 1:100", fill=(50, 50, 50), font=small_font)
    
    # Save test image
    test_image_path = "test_centerpoints_floor_plan.png"
    img.save(test_image_path)
    print(f"Created test floor plan: {test_image_path}")
    
    return test_image_path

def test_centerpoints_api():
    """Test the visualize_walls endpoint with centerpoints"""
    
    # Create test image
    test_image_path = create_test_floor_plan()
    
    try:
        # Test the API endpoint
        url = "http://localhost:8080/visualize_walls"
        
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'scale_factor_mm_per_pixel': '2.0'}  # 1 pixel = 2mm
            
            print("Sending request to visualize_walls endpoint...")
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print("✓ API request successful!")
                
                # Check space names detection
                space_names = result.get('space_names', {})
                total_spaces = space_names.get('total_spaces_detected', 0)
                spaces = space_names.get('spaces', [])
                
                print(f"\nSpace Names Detection Results:")
                print(f"Total spaces detected: {total_spaces}")
                
                if total_spaces > 0:
                    print(f"\nDetected Spaces:")
                    for i, space in enumerate(spaces):
                        print(f"\nSpace {i+1}:")
                        print(f"  Name: '{space['name']}'")
                        print(f"  Confidence: {space['confidence']:.3f}")
                        print(f"  Method: {space.get('method', 'unknown')}")
                        
                        # Centerpoint in pixels
                        center_px = space.get('center', {})
                        print(f"  Center (pixels): ({center_px.get('x', 0):.1f}, {center_px.get('y', 0):.1f})")
                        
                        # Centerpoint in millimeters
                        center_mm = space.get('center_mm', {})
                        print(f"  Center (mm): ({center_mm.get('x', 0):.1f}, {center_mm.get('y', 0):.1f})")
                        
                        # Bounding box
                        bbox_px = space.get('bbox', [])
                        print(f"  BBox (pixels): {bbox_px}")
                        
                        bbox_mm = space.get('bbox_mm', {})
                        if bbox_mm:
                            print(f"  BBox (mm): x1={bbox_mm.get('x1', 0):.1f}, y1={bbox_mm.get('y1', 0):.1f}, x2={bbox_mm.get('x2', 0):.1f}, y2={bbox_mm.get('y2', 0):.1f}")
                    
                    # Check detection summary
                    detection_summary = space_names.get('detection_summary', {})
                    if detection_summary:
                        print(f"\nDetection Summary:")
                        print(f"  Average confidence: {detection_summary.get('average_confidence', 0):.3f}")
                        print(f"  Confidence range: {detection_summary.get('confidence_range', {})}")
                        print(f"  Detection methods: {detection_summary.get('detection_methods', [])}")
                        
                        centerpoints_mm = detection_summary.get('centerpoints_mm', [])
                        print(f"  All centerpoints (mm): {len(centerpoints_mm)} points")
                        for j, center in enumerate(centerpoints_mm):
                            print(f"    Point {j+1}: ({center.get('x', 0):.1f}, {center.get('y', 0):.1f})")
                
                # Check visualization file
                vis_file = result.get('visualization_file', '')
                if vis_file:
                    print(f"\n✓ Visualization saved as: {vis_file}")
                    print("Check the visualization image to see the pink centerpoint circles!")
                
                # Check analysis file
                analysis_file = result.get('analysis_file', '')
                if analysis_file:
                    print(f"✓ Analysis saved as: {analysis_file}")
                
                print(f"\n✓ Centerpoints test completed successfully!")
                print(f"Found {total_spaces} space names with centerpoints")
                
            else:
                print(f"✗ API request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("✗ Failed to connect to API. Make sure the server is running on localhost:8080")
        print("Run: python application.py")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Cleaned up test image: {test_image_path}")

if __name__ == "__main__":
    print("Space Name Centerpoints Test")
    print("=" * 50)
    
    test_centerpoints_api()
    
    print("\nTest completed!")
    print("If successful, check the generated visualization image for pink centerpoint circles.") 