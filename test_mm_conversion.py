#!/usr/bin/env python3
"""
Test script for the millimeter conversion feature in the Floor Plan to 3D API.

This script demonstrates how to use the new scale_factor_mm_per_pixel parameter
to convert pixel measurements to millimeters in the wall analysis.

Usage:
    python test_mm_conversion.py <image_path> [scale_factor]

Example:
    python test_mm_conversion.py floorplan.jpg 0.5
    # This means 1 pixel = 0.5 mm in the real world
"""

import requests
import sys
import json
import os

def test_mm_conversion(image_path, scale_factor=1.0):
    """
    Test the millimeter conversion feature with a floor plan image.
    
    Args:
        image_path (str): Path to the floor plan image
        scale_factor (float): Millimeters per pixel (default: 1.0)
    """
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # API endpoint
    url = "http://localhost:5000/analyze_walls"
    
    # Prepare the request
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'scale_factor_mm_per_pixel': str(scale_factor)}
        
        print(f"Analyzing floor plan with scale factor: {scale_factor} mm/pixel")
        print(f"Image: {image_path}")
        print("Sending request to API...")
        
        try:
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n✅ Analysis completed successfully!")
                print(f"Analysis file: {result.get('analysis_file', 'N/A')}")
                
                # Display summary
                summary = result.get('summary', {})
                print(f"\n📊 Summary:")
                print(f"  Total walls detected: {summary.get('total_walls_detected', 0)}")
                print(f"  Total junctions: {summary.get('total_junctions', 0)}")
                print(f"  Total wall length (pixels): {summary.get('total_wall_length_px', 0):.2f}")
                print(f"  Total wall length (mm): {summary.get('total_wall_length_mm', 0):.2f}")
                print(f"  Average wall thickness (pixels): {summary.get('average_wall_thickness_px', 0):.2f}")
                print(f"  Average wall thickness (mm): {summary.get('average_wall_thickness_mm', 0):.2f}")
                
                # Display first few walls with measurements
                walls = result.get('individual_walls', [])
                if walls:
                    print(f"\n🏗️  Sample wall measurements:")
                    for i, wall in enumerate(walls[:3]):  # Show first 3 walls
                        print(f"  Wall {wall.get('wall_id', 'Unknown')}:")
                        print(f"    Length: {wall.get('length_px', 0):.2f} px = {wall.get('length_mm', 0):.2f} mm")
                        thickness_px = wall.get('thickness_px', {})
                        thickness_mm = wall.get('thickness_mm', {})
                        print(f"    Thickness: {thickness_px.get('average', 0):.2f} px = {thickness_mm.get('average', 0):.2f} mm")
                
                # Save detailed results to file
                output_file = f"mm_conversion_test_results_{scale_factor}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Detailed results saved to: {output_file}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to the API server.")
            print("Make sure the Flask server is running on localhost:5000")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    image_path = sys.argv[1]
    scale_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    
    test_mm_conversion(image_path, scale_factor)

if __name__ == "__main__":
    main() 