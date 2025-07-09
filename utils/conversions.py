import os
import json
from config.constants import ROOT_DIR, JSON_OUTPUT_DIR

def pixels_to_mm(pixels, scale_factor_mm_per_pixel):
    """Convert pixel measurements to millimeters"""
    # If scale_factor_mm_per_pixel represents mm per pixel, use it directly
    # If it represents pixels per mm, we need to invert it
    return pixels * scale_factor_mm_per_pixel

def mm_to_pixels(mm, scale_factor_mm_per_pixel):
    """Convert millimeter measurements to pixels"""
    return mm / scale_factor_mm_per_pixel

def convert_centerline_to_mm(centerline_pixels, scale_factor_mm_per_pixel):
    """Convert centerline coordinates from pixels to millimeters"""
    if not centerline_pixels:
        return []
    
    centerline_mm = []
    for point in centerline_pixels:
        if len(point) >= 2:
            x_mm = pixels_to_mm(point[0], scale_factor_mm_per_pixel)
            y_mm = pixels_to_mm(point[1], scale_factor_mm_per_pixel)
            centerline_mm.append([x_mm, y_mm])
    
    return centerline_mm

def convert_thickness_to_mm(thickness_data, scale_factor_mm_per_pixel):
    """Convert thickness measurements from pixels to millimeters"""
    if not thickness_data:
        return thickness_data
    
    thickness_mm = {
        "average": pixels_to_mm(thickness_data.get("average", 0), scale_factor_mm_per_pixel),
        "min": pixels_to_mm(thickness_data.get("min", 0), scale_factor_mm_per_pixel),
        "max": pixels_to_mm(thickness_data.get("max", 0), scale_factor_mm_per_pixel),
        "profile": [pixels_to_mm(t, scale_factor_mm_per_pixel) for t in thickness_data.get("profile", [])]
    }
    
    return thickness_mm

def convert_bbox_to_mm(bbox_pixels, scale_factor_mm_per_pixel):
    """Convert bounding box coordinates from pixels to millimeters"""
    if not bbox_pixels:
        return bbox_pixels
    
    bbox_mm = {
        "x1": pixels_to_mm(bbox_pixels.get("x1", 0), scale_factor_mm_per_pixel),
        "y1": pixels_to_mm(bbox_pixels.get("y1", 0), scale_factor_mm_per_pixel),
        "x2": pixels_to_mm(bbox_pixels.get("x2", 0), scale_factor_mm_per_pixel),
        "y2": pixels_to_mm(bbox_pixels.get("y2", 0), scale_factor_mm_per_pixel)
    }
    
    return bbox_mm

def convert_junction_position_to_mm(junction_data, scale_factor_mm_per_pixel):
    """Convert junction position from pixels to millimeters"""
    if not junction_data or "position" not in junction_data:
        return junction_data
    
    junction_mm = junction_data.copy()
    position_pixels = junction_data["position"]
    if len(position_pixels) >= 2:
        x_mm = pixels_to_mm(position_pixels[0], scale_factor_mm_per_pixel)
        y_mm = pixels_to_mm(position_pixels[1], scale_factor_mm_per_pixel)
        junction_mm["position"] = [x_mm, y_mm]
    
    return junction_mm

def save_wall_analysis(wall_data, filename):
	"""Save wall analysis to file in JSON output directory"""
	filepath = os.path.join(JSON_OUTPUT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(wall_data, f, indent=2)
		print(f"Wall analysis saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving wall analysis: {str(e)}")
		return None
