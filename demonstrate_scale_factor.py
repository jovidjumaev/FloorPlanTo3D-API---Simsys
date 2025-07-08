#!/usr/bin/env python3
"""
Demonstrate the difference between old and new scale factors.
"""

def demonstrate_scale_factor():
    """Show the difference between scale factors."""
    
    # Example measurements from walls6.json
    length_px = 642.8284271247462  # Example wall length in pixels
    thickness_px = 15.814570540450802  # Example wall thickness in pixels
    
    print("=== Scale Factor Demonstration ===")
    print(f"Original measurements in pixels:")
    print(f"  Wall length: {length_px} pixels")
    print(f"  Wall thickness: {thickness_px} pixels")
    print()
    
    # Old scale factor (default 1.0)
    old_scale = 1.0
    length_mm_old = length_px * old_scale
    thickness_mm_old = thickness_px * old_scale
    
    print(f"With old scale factor ({old_scale} mm/pixel):")
    print(f"  Wall length: {length_mm_old} mm")
    print(f"  Wall thickness: {thickness_mm_old} mm")
    print()
    
    # New correct scale factor
    new_scale = 3.7795275591
    length_mm_new = length_px * new_scale
    thickness_mm_new = thickness_px * new_scale
    
    print(f"With new scale factor ({new_scale} mm/pixel):")
    print(f"  Wall length: {length_mm_new:.2f} mm")
    print(f"  Wall thickness: {thickness_mm_new:.2f} mm")
    print()
    
    print("=== Conversion Summary ===")
    print(f"1 pixel = {new_scale} mm")
    print(f"1 mm = {1/new_scale:.6f} pixels")
    print()
    
    print("To get correct millimeter measurements, use:")
    print("scale_factor_mm_per_pixel = 3.7795275591")
    print()
    
    print("Example API call:")
    print("""
import requests

with open('your_floorplan.jpg', 'rb') as f:
    files = {'image': f}
    data = {'scale_factor_mm_per_pixel': '3.7795275591'}
    response = requests.post('http://localhost:5000/analyze_walls', 
                           files=files, data=data)
    """)

if __name__ == "__main__":
    demonstrate_scale_factor() 