# Wall analysis module for floor plan processing
import numpy as np
from scipy.ndimage import distance_transform_edt
from utils.geometry import find_nearest_valid_point
from utils.conversions import pixels_to_mm, convert_thickness_to_mm, convert_centerline_to_mm, convert_bbox_to_mm
from analysis.junction_analysis import extract_centerline_coords, extract_centerline_coords_with_validation

def calculate_wall_thickness(wall_mask, centerline_coords):
	"""Calculate wall thickness along the centerline"""
	if not centerline_coords or len(centerline_coords) < 2:
		return {"average": 0, "min": 0, "max": 0, "profile": []}
	
	# Calculate distance transform
	distance_map = distance_transform_edt(wall_mask)
	
	thickness_values = []
	for coord in centerline_coords:
		x, y = coord
		if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
			# Thickness is approximately 2 * distance to edge
			thickness = distance_map[y, x] * 2
			thickness_values.append(float(thickness))
	
	if not thickness_values:
		return {"average": 0, "min": 0, "max": 0, "profile": []}
	
	return {
		"average": float(numpy.mean(thickness_values)),
		"min": float(numpy.min(thickness_values)),
		"max": float(numpy.max(thickness_values)),
		"profile": thickness_values
	}

def validate_centerline_in_walls(centerline_coords, wall_mask):
    """Additional validation to ensure centerline stays within wall boundaries"""
    if not centerline_coords or len(centerline_coords) < 2:
        return centerline_coords
    
    validated_coords = []
    for coord in centerline_coords:
        x, y = coord
        # Check if point is within image bounds and in wall area
        if (0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1] and 
    bool(wall_mask[y, x])): # FIXED: removed numpy.any()
            validated_coords.append(coord)
        else:
            # Try to find a nearby valid point
            nearest = find_nearest_valid_point(x, y, wall_mask, max_search_radius=3)
            if nearest is not None:
                validated_coords.append([nearest[1], nearest[0]])  # Convert back to [x, y]
    
    return validated_coords if len(validated_coords) >= 2 else centerline_coords

def calculate_wall_length(centerline_coords):
	"""Calculate total wall length from centerline coordinates"""
	if len(centerline_coords) < 2:
		return 0.0
	
	total_length = 0.0
	for i in range(1, len(centerline_coords)):
		x1, y1 = centerline_coords[i-1]
		x2, y2 = centerline_coords[i]
		segment_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
		total_length += segment_length
	
	return float(total_length)

def calculate_wall_orientation(centerline_coords):
	"""Calculate wall orientation angle in degrees"""
	if len(centerline_coords) < 2:
		return 0.0
	
	# Use first and last points for overall orientation
	start_point = centerline_coords[0]
	end_point = centerline_coords[-1]
	
	dx = end_point[0] - start_point[0]
	dy = end_point[1] - start_point[1]
	
	# Calculate angle in degrees
	angle = numpy.arctan2(dy, dx) * 180 / numpy.pi
	
	# Normalize to 0-180 degrees (walls don't have direction)
	if angle < 0:
		angle += 180
	if angle > 180:
		angle -= 180
	
	return float(angle)

def find_wall_connections(wall_segments, junctions, tolerance=10):
	"""Find which walls connect at which junctions"""
	connections = {}
	
	for i, segment in enumerate(wall_segments):
		wall_id = f"W{i+1}"
		connections[wall_id] = {"start_junction": None, "end_junction": None}
		
		if len(segment) < 2:
			continue
		
		# Check start and end points of wall
		start_point = [segment[0][1], segment[0][0]]  # Convert y,x to x,y
		end_point = [segment[-1][1], segment[-1][0]]
		
		# Find closest junctions to start and end points
		for j, junction in enumerate(junctions):
			junction_id = f"J{j+1}"
			
			# Distance to start point
			start_dist = ((start_point[0] - junction[0])**2 + (start_point[1] - junction[1])**2)**0.5
			if start_dist <= tolerance and not connections[wall_id]["start_junction"]:
				connections[wall_id]["start_junction"] = junction_id
			
			# Distance to end point  
			end_dist = ((end_point[0] - junction[0])**2 + (end_point[1] - junction[1])**2)**0.5
			if end_dist <= tolerance and not connections[wall_id]["end_junction"]:
				connections[wall_id]["end_junction"] = junction_id
	
	return connections

def analyze_junction_types(junctions, wall_connections):
	"""Analyze the type of each junction based on connected walls"""
	junction_analysis = []
	
	for i, junction in enumerate(junctions):
		junction_id = f"J{i+1}"
		
		# Count walls connected to this junction
		connected_walls = []
		for wall_id, connections in wall_connections.items():
			if (connections["start_junction"] == junction_id or 
				connections["end_junction"] == junction_id):
				connected_walls.append(wall_id)
		
		# Determine junction type
		junction_type = "unknown"
		if len(connected_walls) == 2:
			junction_type = "corner"
		elif len(connected_walls) == 3:
			junction_type = "T_junction"
		elif len(connected_walls) == 4:
			junction_type = "cross_junction"
		elif len(connected_walls) > 4:
			junction_type = "complex_junction"
		
		junction_analysis.append({
			"junction_id": junction_id,
			"position": [float(junction[0]), float(junction[1])],
			"connected_walls": connected_walls,
			"junction_type": junction_type,
			"wall_count": len(connected_walls)
		})
	
	return junction_analysis

def extract_wall_parameters(segments, wall_mask, junctions, scale_factor_mm_per_pixel=1.0):
    """Extract comprehensive parameters for each wall segment (output only mm fields)"""
    wall_parameters = []
    wall_connections = find_wall_connections(segments, junctions)
    for i, segment in enumerate(segments):
        wall_id = f"W{i+1}"
        centerline = extract_centerline_coords(segment)
        length_px = calculate_wall_length(centerline)
        thickness_px = calculate_wall_thickness(wall_mask, centerline)
        orientation = calculate_wall_orientation(centerline)
        if len(segment) > 0:
            min_y, min_x = numpy.min(segment, axis=0)
            max_y, max_x = numpy.max(segment, axis=0)
            bbox_px = {
                "x1": float(min_x), "y1": float(min_y),
                "x2": float(max_x), "y2": float(max_y)
            }
        else:
            bbox_px = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        length_mm = pixels_to_mm(length_px, scale_factor_mm_per_pixel)
        thickness_mm = convert_thickness_to_mm(thickness_px, scale_factor_mm_per_pixel)
        centerline_mm = convert_centerline_to_mm(centerline, scale_factor_mm_per_pixel)
        bbox_mm = convert_bbox_to_mm(bbox_px, scale_factor_mm_per_pixel)
        wall_params = {
            "wall_id": wall_id,
            "centerline": centerline_mm,
            "length": length_mm,
            "thickness": thickness_mm,
            "orientation_degrees": orientation,
            "bbox": bbox_mm,
            "connections": wall_connections.get(wall_id, {"start_junction": None, "end_junction": None}),
            "segment_area": float(len(segment))
        }
        wall_parameters.append(wall_params)
    return wall_parameters

def extract_wall_parameters_with_regions(all_wall_segments, wall_mask, junctions, scale_factor_mm_per_pixel=1.0):
    """Extract comprehensive parameters for each wall segment with region prefixes (output only mm fields)"""
    wall_parameters = []
    wall_connections = find_wall_connections([seg for seg, _ in all_wall_segments], junctions)
    for i, (segment, region_prefix) in enumerate(all_wall_segments):
        wall_id = f"{region_prefix}W{i+1}"
        centerline = extract_centerline_coords_with_validation(segment, wall_mask)
        length_px = calculate_wall_length(centerline)
        thickness_px = calculate_wall_thickness(wall_mask, centerline)
        orientation = calculate_wall_orientation(centerline)
        if len(segment) > 0:
            min_y, min_x = numpy.min(segment, axis=0)
            max_y, max_x = numpy.max(segment, axis=0)
            bbox_px = {
                "x1": float(min_x), "y1": float(min_y),
                "x2": float(max_x), "y2": float(max_y)
            }
        else:
            bbox_px = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        length_mm = pixels_to_mm(length_px, scale_factor_mm_per_pixel)
        thickness_mm = convert_thickness_to_mm(thickness_px, scale_factor_mm_per_pixel)
        centerline_mm = convert_centerline_to_mm(centerline, scale_factor_mm_per_pixel)
        bbox_mm = convert_bbox_to_mm(bbox_px, scale_factor_mm_per_pixel)
        wall_params = {
            "wall_id": wall_id,
            "centerline": centerline_mm,
            "length": length_mm,
            "thickness": thickness_mm,
            "orientation_degrees": orientation,
            "bbox": bbox_mm,
            "connections": wall_connections.get(wall_id, {"start_junction": None, "end_junction": None}),
            "segment_area": float(len(segment))
        }
        wall_parameters.append(wall_params)
    return wall_parameters



def generate_wall_insights(wall_parameters, junction_analysis):
	"""Generate architectural insights about wall layout"""
	insights = []
	
	if not wall_parameters:
		return ["No walls detected for analysis"]
	
	# Analyze wall thickness consistency
	thicknesses = [w["thickness"]["average"] for w in wall_parameters if w["thickness"]["average"] > 0]
	if thicknesses:
		thickness_std = numpy.std(thicknesses)
		if thickness_std < 2.0:
			insights.append("Consistent wall thickness throughout floor plan")
		else:
			insights.append("Variable wall thickness detected - may indicate different wall types")
	
	# Analyze wall length distribution
	lengths = [w["length_px"] for w in wall_parameters]
	if lengths:
		long_walls = len([l for l in lengths if l > numpy.mean(lengths) * 1.5])
		if long_walls > 0:
			insights.append(f"Found {long_walls} notably long walls - potential load-bearing structures")
	
	# Analyze junction complexity
	complex_junctions = [j for j in junction_analysis if j["wall_count"] > 3]
	if complex_junctions:
		insights.append(f"Found {len(complex_junctions)} complex junctions with 4+ walls")
	
	# Check for isolated walls
	isolated_walls = [w for w in wall_parameters if 
					 w["connections"]["start_junction"] is None and 
					 w["connections"]["end_junction"] is None]
	if isolated_walls:
		insights.append(f"Found {len(isolated_walls)} isolated wall segments")
	
	# Analyze wall orientation patterns
	orientations = [w["orientation_degrees"] for w in wall_parameters]
	horizontal_walls = len([o for o in orientations if o <= 30 or o >= 150])
	vertical_walls = len([o for o in orientations if 60 <= o <= 120])
	
	if horizontal_walls > vertical_walls * 1.5:
		insights.append("Predominantly horizontal wall layout")
	elif vertical_walls > horizontal_walls * 1.5:
		insights.append("Predominantly vertical wall layout")
	else:
		insights.append("Balanced horizontal and vertical wall layout")
	
	return insights


def identify_exterior_walls(wall_parameters, image_width, image_height, scale_factor_mm_per_pixel):
    """Identify walls that form the exterior boundary of the floor plan"""
    exterior_walls = []
    interior_walls = []

    # Convert image dimensions to mm for boundary analysis
    image_width_mm = pixels_to_mm(image_width, scale_factor_mm_per_pixel)
    image_height_mm = pixels_to_mm(image_height, scale_factor_mm_per_pixel)

    # Define boundary margins (walls within 3% of image edges are likely exterior)
    boundary_margin = 0.03  # 3% of image dimensions (more strict)
    x_margin = image_width_mm * boundary_margin
    y_margin = image_height_mm * boundary_margin

    for wall in wall_parameters:
        bbox = wall.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
        
        # Check if wall is near any image boundary
        is_near_left = x1 <= x_margin
        is_near_right = x2 >= (image_width_mm - x_margin)
        is_near_top = y1 <= y_margin
        is_near_bottom = y2 >= (image_height_mm - y_margin)
        
        # Additional criteria: walls with fewer connections are more likely to be exterior
        connections = wall.get("connections", {})
        start_connected = connections.get("start_junction") is not None
        end_connected = connections.get("end_junction") is not None
        connection_count = sum([start_connected, end_connected])
        
        # Determine if this is likely an exterior wall
        is_exterior = False
        exterior_reasons = []
        
        # Criterion 1: Near image boundaries (more strict)
        if is_near_left or is_near_right or is_near_top or is_near_bottom:
            is_exterior = True
            if is_near_left:
                exterior_reasons.append("left_boundary")
            if is_near_right:
                exterior_reasons.append("right_boundary")
            if is_near_top:
                exterior_reasons.append("top_boundary")
            if is_near_bottom:
                exterior_reasons.append("bottom_boundary")
        
        # Criterion 2: Poorly connected walls (likely exterior) - more strict
        if connection_count == 0 and not is_exterior:  # Only unconnected walls
            is_exterior = True
            exterior_reasons.append("unconnected")
        
        # Criterion 3: Long walls near boundaries (likely perimeter walls)
        wall_length = wall.get("length", 0)
        if wall_length > 150 and (is_near_left or is_near_right or is_near_top or is_near_bottom):  # Increased threshold
            is_exterior = True
            exterior_reasons.append("long_boundary_wall")
        
        # Criterion 4: Walls that span a significant portion of the image edge
        if is_near_left or is_near_right:
            wall_span = abs(y2 - y1)
            if wall_span > image_height_mm * 0.3:  # Spans 30% of image height
                is_exterior = True
                exterior_reasons.append("spans_vertical_edge")
        
        if is_near_top or is_near_bottom:
            wall_span = abs(x2 - x1)
            if wall_span > image_width_mm * 0.3:  # Spans 30% of image width
                is_exterior = True
                exterior_reasons.append("spans_horizontal_edge")
        
        if is_exterior:
            exterior_wall_data = wall.copy()
            exterior_wall_data["exterior_reasons"] = exterior_reasons
            exterior_walls.append(exterior_wall_data)
        else:
            interior_walls.append(wall)

    return exterior_walls, interior_walls


def calculate_perimeter_dimensions(exterior_walls):
	"""Calculate perimeter dimensions from exterior walls"""
	if not exterior_walls:
		return {
			"total_perimeter_length": 0,
			"exterior_wall_count": 0,
			"perimeter_area": 0,
			"boundary_coverage": {
				"left": 0, "right": 0, "top": 0, "bottom": 0
			}
		}
	
	total_perimeter_length = sum(wall.get("length", 0) for wall in exterior_walls)
	exterior_wall_count = len(exterior_walls)
	
	# Calculate approximate perimeter area (assuming rectangular shape)
	# This is a rough estimate based on the longest walls in each direction
	wall_lengths = [wall.get("length", 0) for wall in exterior_walls]
	if len(wall_lengths) >= 4:
		# Sort by length and assume the 4 longest walls form the rectangle
		sorted_lengths = sorted(wall_lengths, reverse=True)
		width_estimate = sorted_lengths[0]  # Longest wall
		height_estimate = sorted_lengths[1]  # Second longest wall
		perimeter_area = width_estimate * height_estimate
	else:
		perimeter_area = 0
	
	# Analyze boundary coverage
	boundary_coverage = {"left": 0, "right": 0, "top": 0, "bottom": 0}
	for wall in exterior_walls:
		reasons = wall.get("exterior_reasons", [])
		for reason in reasons:
			if reason in boundary_coverage:
				boundary_coverage[reason] += 1
	
	return {
		"total_perimeter_length": total_perimeter_length,
		"exterior_wall_count": exterior_wall_count,
		"perimeter_area": perimeter_area,
		"boundary_coverage": boundary_coverage,
		"average_exterior_wall_length": total_perimeter_length / exterior_wall_count if exterior_wall_count > 0 else 0
	}

def calculate_centered_straight_centerline(wall_mask, bbox=None):
    """
    Calculate a straight, centered centerline using distance transform and geometric analysis.
    
    Args:
        wall_mask: Binary mask of the wall
        bbox: Optional bounding box to guide centerline direction
    
    Returns:
        List of [x, y] coordinates for the centerline
    """
    if numpy.sum(wall_mask) < 10:
        return []
    
    # Calculate distance transform to find medial axis
    distance_map = distance_transform_edt(wall_mask)
    
    # Find the ridge (medial axis) points with high distance values
    threshold = numpy.percentile(distance_map[distance_map > 0], 70)  # Top 30% of distances
    ridge_points = numpy.where(distance_map >= threshold)
    
    if len(ridge_points[0]) < 2:
        return []
    
    # Convert to (x, y) coordinates
    ridge_coords = list(zip(ridge_points[1], ridge_points[0]))  # (x, y)
    
    # Use bounding box to determine primary direction
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        is_horizontal = width > height
    else:
        # Calculate orientation from ridge points
        ridge_array = numpy.array(ridge_coords)
        x_range = numpy.max(ridge_array[:, 0]) - numpy.min(ridge_array[:, 0])
        y_range = numpy.max(ridge_array[:, 1]) - numpy.min(ridge_array[:, 1])
        is_horizontal = x_range > y_range
    
    # Sort ridge points along the primary direction
    if is_horizontal:
        ridge_coords.sort(key=lambda p: p[0])  # Sort by x
    else:
        ridge_coords.sort(key=lambda p: p[1])  # Sort by y
    
    # Create a straight centerline by fitting a line through the ridge points
    if len(ridge_coords) >= 2:
        # Use robust line fitting
        points = numpy.array(ridge_coords)
        
        # For horizontal walls, fit y = mx + b
        # For vertical walls, fit x = my + b
        if is_horizontal:
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Find the line that best fits through the center
            if len(x_coords) > 1:
                # Use weighted average of y-coordinates for each x region
                x_min, x_max = numpy.min(x_coords), numpy.max(x_coords)
                
                # Create evenly spaced points along the line
                num_points = min(max(3, int((x_max - x_min) / 10)), 20)
                straight_x = numpy.linspace(x_min, x_max, num_points)
                
                # For each x position, find the average y position of nearby ridge points
                straight_y = []
                for x in straight_x:
                    # Find ridge points within a small window around this x
                    window = 15  # pixels
                    nearby_y = y_coords[numpy.abs(x_coords - x) <= window]
                    if len(nearby_y) > 0:
                        # Use weighted average, giving more weight to points with higher distance values
                        nearby_x = x_coords[numpy.abs(x_coords - x) <= window]
                        nearby_points = [(int(nx), int(ny)) for nx, ny in zip(nearby_x, nearby_y)]
                        weights = [distance_map[py, px] for px, py in nearby_points if 0 <= py < distance_map.shape[0] and 0 <= px < distance_map.shape[1]]
                        
                        if weights:
                            weighted_y = numpy.average(nearby_y, weights=weights)
                        else:
                            weighted_y = numpy.mean(nearby_y)
                        straight_y.append(weighted_y)
                    else:
                        # Interpolate from previous points
                        if len(straight_y) > 0:
                            straight_y.append(straight_y[-1])
                        else:
                            straight_y.append(numpy.mean(y_coords))
                
                centerline = [[x, y] for x, y in zip(straight_x, straight_y)]
        else:
            # Vertical wall - similar logic but swapped axes
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            if len(y_coords) > 1:
                y_min, y_max = numpy.min(y_coords), numpy.max(y_coords)
                
                # Create evenly spaced points along the line
                num_points = min(max(3, int((y_max - y_min) / 10)), 20)
                straight_y = numpy.linspace(y_min, y_max, num_points)
                
                # For each y position, find the average x position of nearby ridge points
                straight_x = []
                for y in straight_y:
                    window = 15  # pixels
                    nearby_x = x_coords[numpy.abs(y_coords - y) <= window]
                    if len(nearby_x) > 0:
                        # Use weighted average
                        nearby_y = y_coords[numpy.abs(y_coords - y) <= window]
                        nearby_points = [(int(nx), int(ny)) for nx, ny in zip(nearby_x, nearby_y)]
                        weights = [distance_map[py, px] for px, py in nearby_points if 0 <= py < distance_map.shape[0] and 0 <= px < distance_map.shape[1]]
                        
                        if weights:
                            weighted_x = numpy.average(nearby_x, weights=weights)
                        else:
                            weighted_x = numpy.mean(nearby_x)
                        straight_x.append(weighted_x)
                    else:
                        # Interpolate from previous points
                        if len(straight_x) > 0:
                            straight_x.append(straight_x[-1])
                        else:
                            straight_x.append(numpy.mean(x_coords))
                
                centerline = [[x, y] for x, y in zip(straight_x, straight_y)]
    
    # Final validation - ensure all points are within the wall mask
    if 'centerline' in locals():
        validated_centerline = []
        for point in centerline:
            x, y = int(point[0]), int(point[1])
            if (0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1] and wall_mask[y, x]):
                validated_centerline.append(point)
            else:
                # Find nearest valid point
                nearest = find_nearest_valid_point(x, y, wall_mask, max_search_radius=10)
                if nearest is not None:
                    validated_centerline.append([nearest[1], nearest[0]])  # Convert to [x, y]
        
        return validated_centerline if len(validated_centerline) >= 2 else []
    
    return []