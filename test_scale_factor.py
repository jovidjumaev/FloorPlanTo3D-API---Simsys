#!/usr/bin/env python3
"""
Test script to verify the scale factor conversion is working correctly.
"""

import requests
import json
import sys

def test_scale_factor_conversion():
    """Test the scale factor conversion with a known value."""
    
    # Your provided scale factor: 1 mm = 3.7795275591 pixels
    # So scale_factor_mm_per_pixel = 3.7795275591
    scale_factor = 3.7795275591
    print(f"Testing with scale factor: {scale_factor} mm/pixel")
    print(f"This means 1 pixel = {scale_factor} mm")
    
    # Test the conversion functions
    test_pixels = 100
    expected_mm = test_pixels * scale_factor
    print(f"Test: {test_pixels} pixels should equal {expected_mm} mm")
    
    # Test with a real API call if you have an image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTesting with image: {image_path}")
        
        url = "http://localhost:5000/analyze_walls"
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'scale_factor_mm_per_pixel': str(scale_factor)}
                
                print("Sending request to API...")
                response = requests.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check if scale factor is correctly set
                    metadata = result.get('metadata', {})
                    actual_scale = metadata.get('scale_factor_mm_per_pixel', 1.0)
                    print(f"API returned scale factor: {actual_scale}")
                    
                    # Check first wall measurements
                    walls = result.get('individual_walls', [])
                    if walls:
                        wall = walls[0]
                        length_px = wall.get('length_px', 0)
                        length_mm = wall.get('length_mm', 0)
                        print(f"First wall - Length: {length_px} px = {length_mm} mm")
                        print(f"Expected mm: {length_px * scale_factor}")
                        print(f"Conversion working: {abs(length_mm - length_px * scale_factor) < 0.001}")
                        
                        # Check thickness
                        thickness_px = wall.get('thickness_px', {}).get('average', 0)
                        thickness_mm = wall.get('thickness_mm', {}).get('average', 0)
                        print(f"First wall - Thickness: {thickness_px} px = {thickness_mm} mm")
                        print(f"Expected mm: {thickness_px * scale_factor}")
                        print(f"Conversion working: {abs(thickness_mm - thickness_px * scale_factor) < 0.001}")
                    
                    # Save results
                    output_file = f"test_scale_factor_{scale_factor}.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"Results saved to: {output_file}")
                    
                else:
                    print(f"Error: {response.status_code}")
                    print(f"Response: {response.text}")
                    
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("\nNo image provided. To test with an image:")
        print("python test_scale_factor.py your_image.jpg")

if __name__ == "__main__":
    test_scale_factor_conversion() 