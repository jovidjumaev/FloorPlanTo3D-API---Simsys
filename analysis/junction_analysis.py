import numpy as np
import numpy  # Add this for backward compatibility
from skimage.measure import label
from scipy.ndimage import binary_erosion, binary_dilation
from utils.geometry import find_nearest_valid_point

def find_junction_points(skeleton):
    """Find junction points where multiple walls meet - only real intersections"""
    junctions = []
    h, w = skeleton.shape
    
    # Check each point in skeleton with a more restrictive approach
    for y in range(2, h-2):  # Stay further from edges
        for x in range(2, w-2):
            if skeleton[y, x]:  # FIXED: removed numpy.any()
                # Count connected components in 3x3 neighborhood
                # A true junction should connect 3 or more separate line segments
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = False  # Remove center pixel
                
                # Label connected components in the neighborhood
                labeled_neighbors = label(neighbors, connectivity=2)
                num_components = np.max(labeled_neighbors)
                
                # Only consider it a junction if it connects 3+ separate components
                if num_components >= 3:
                    junctions.append((x, y))
    
    # More aggressive duplicate removal (within 15 pixels)
    filtered_junctions = []
    for junction in junctions:
        is_duplicate = False
        for existing in filtered_junctions:
            dist = ((junction[0] - existing[0])**2 + (junction[1] - existing[1])**2)**0.5
            if dist < 15:  # Larger radius to remove more duplicates
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_junctions.append(junction)
    
    return filtered_junctions

def find_junction_points_simple(skeleton):
    """Ultra-conservative junction detection - only real intersections"""
    # Clean the skeleton first to remove noise
    # Erode then dilate to remove small artifacts
    cleaned_skeleton = binary_erosion(skeleton.astype(bool), iterations=1)
    cleaned_skeleton = binary_dilation(cleaned_skeleton, iterations=1)
    
    junctions = []
    h, w = cleaned_skeleton.shape
    
    # Only check every 5th pixel to avoid noise
    for y in range(5, h-5, 5):
        for x in range(5, w-5, 5):
            if cleaned_skeleton[y, x].item():   # FIXED: removed numpy.any()
                # Check in a larger 5x5 neighborhood to be more stable
                region = cleaned_skeleton[y-2:y+3, x-2:x+3]
                
                # Count distinct line directions
                # Look for pixels in cardinal and diagonal directions
                directions = [
    							bool(region[0, 2].item()),  # North
    							bool(region[4, 2].item()),  # South  
    							bool(region[2, 0].item()),  # West
    							bool(region[2, 4].item()),  # East
    							bool(region[0, 0].item()),  # NW
    							bool(region[0, 4].item()),  # NE
    							bool(region[4, 0].item()),  # SW
    							bool(region[4, 4].item())   # SE
								]
                
                # Count how many directions have connections
                direction_count = sum(directions)
                
                # Only consider it a junction if it connects 3+ directions
                # AND is far from image boundaries
                if direction_count >= 3 and x > 20 and x < w-20 and y > 20 and y < h-20:
                    junctions.append((x, y))
    
    # Very aggressive duplicate removal (within 30 pixels)
    filtered_junctions = []
    for junction in junctions:
        is_duplicate = False
        for existing in filtered_junctions:
            dist = np.sqrt(((junction[0] - existing[0])**2 + (junction[1] - existing[1])**2))
            if dist < 30:  # Much larger radius
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_junctions.append(junction)
    
    # Final filter: only keep junctions that are actually meaningful
    # Limit to maximum 20 junctions for the entire floor plan
    if len(filtered_junctions) > 20:
        # Keep only the strongest junctions (those with most connections)
        junction_scores = []
        for jx, jy in filtered_junctions:
            if 0 <= jy < h and 0 <= jx < w:
                y_start = max(0, jy-3)
                y_end = min(h, jy+4)
                x_start = max(0, jx-3)
                x_end = min(w, jx+4)
                region = cleaned_skeleton[y_start:y_end, x_start:x_end]
                score = np.sum(region)
                junction_scores.append((score, (jx, jy)))
        
        # Sort by score and keep top 20
        junction_scores.sort(reverse=True)
        filtered_junctions = [junction for score, junction in junction_scores[:20]]
    
    return filtered_junctions

def find_junctions_from_bboxes(wall_bboxes):
	"""Alternative: Find junctions from wall bounding box intersections"""
	junctions = []
	
	# Check each pair of wall bounding boxes for intersections
	for i, bbox1 in enumerate(wall_bboxes):
		for j, bbox2 in enumerate(wall_bboxes[i+1:], i+1):
			y1_1, x1_1, y2_1, x2_1 = bbox1
			y1_2, x1_2, y2_2, x2_2 = bbox2
			
			# Check if bounding boxes intersect
			if not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1):
				# Calculate intersection point
				intersection_x = (max(x1_1, x1_2) + min(x2_1, x2_2)) / 2
				intersection_y = (max(y1_1, y1_2) + min(y2_1, y2_2)) / 2
				junctions.append((intersection_x, intersection_y))
	
	# Remove duplicates
	filtered_junctions = []
	for junction in junctions:
		is_duplicate = False
		for existing in filtered_junctions:
			dist = ((junction[0] - existing[0])**2 + (junction[1] - existing[1])**2)**0.5
			if dist < 25:
				is_duplicate = True
				break
		if not is_duplicate:
			filtered_junctions.append(junction)
	
	return filtered_junctions



def extract_centerline_coords(segment_coords):
	"""Legacy extractor that now leverages connectivity ordering."""
	if len(segment_coords) < 2:
		return segment_coords.tolist()
	ordered = order_centerline_points_connectivity(segment_coords)
	return ordered if ordered else segment_coords.tolist()



def extract_centerline_coords_with_validation(segment_coords, wall_mask, min_length=2):
    """Extract centered and straight centerline coordinates using distance transform."""
    if len(segment_coords) < 2:
        return []
    
    # Calculate bounding box from segment coordinates
    if len(segment_coords) > 0:
        coords_array = np.array(segment_coords)
        min_y, min_x = np.min(coords_array, axis=0)
        max_y, max_x = np.max(coords_array, axis=0)
        bbox = (min_x, min_y, max_x, max_y)
    else:
        bbox = None
    
    # Try the new centered straight centerline approach first
    # Import locally to avoid circular import
    from analysis.wall_analysis import calculate_centered_straight_centerline
    centerline = calculate_centered_straight_centerline(wall_mask, bbox)
    
    if len(centerline) >= min_length:
        return centerline
    
    # Fallback to original approach if new method fails
    ordered_centerline = order_centerline_points_connectivity(segment_coords)
    if len(ordered_centerline) < min_length:
        return []
    
    # For angled walls, we need to preserve the skeleton curve better
    # Instead of drawing lines between points, we validate each skeleton point individually
    validated_coords = []
    
    for point in ordered_centerline:
        x, y = point
        # Check if point is within image bounds and in wall area
        if (0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1] and 
            bool(wall_mask[y, x])):
            validated_coords.append(point)
        else:
            # Try to find a nearby valid point
            nearest = find_nearest_valid_point(x, y, wall_mask, max_search_radius=3)
            if nearest is not None:
                validated_coords.append([nearest[1], nearest[0]])  # Convert back to [x, y]
    
    # Apply smoothing to reduce noise while preserving the curve
    if len(validated_coords) > 4:
        validated_coords = smooth_centerline_curve(validated_coords)
    
    return validated_coords if len(validated_coords) >= min_length else []

def smooth_centerline_curve(centerline_coords, window_size=3):
    """Apply smoothing to centerline coordinates to reduce noise while preserving curve shape."""
    if len(centerline_coords) < window_size:
        return centerline_coords
    
    smoothed_coords = []
    half_window = window_size // 2
    
    for i in range(len(centerline_coords)):
        # For points near the edges, use smaller windows
        start_idx = max(0, i - half_window)
        end_idx = min(len(centerline_coords), i + half_window + 1)
        
        # Calculate average position within the window
        x_sum = sum(point[0] for point in centerline_coords[start_idx:end_idx])
        y_sum = sum(point[1] for point in centerline_coords[start_idx:end_idx])
        count = end_idx - start_idx
        
        smoothed_x = x_sum / count
        smoothed_y = y_sum / count
        
        smoothed_coords.append([smoothed_x, smoothed_y])
    
    return smoothed_coords

def order_centerline_points_connectivity(segment_coords):
    """Improved ordering that better handles angled walls and curved segments."""
    # Convert to set of (x, y) tuples for fast lookup
    coords_set = set((int(c[1]), int(c[0])) for c in segment_coords)
    if not coords_set:
        return []
    
    # 8-neighbour offsets
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    
    # Compute degree of each pixel
    degrees = {p: sum(((p[0]+dx, p[1]+dy) in coords_set) for dx, dy in nbrs) for p in coords_set}
    
    # Find endpoints (degree 1) - these are the natural start/end points
    endpoints = [p for p, d in degrees.items() if d == 1]
    
    # If we have endpoints, start from one of them
    if endpoints:
        start = endpoints[0]
    else:
        # If no clear endpoints (closed loop), find a point with minimum degree
        min_degree = min(degrees.values())
        start = next(p for p, d in degrees.items() if d == min_degree)
    
    # Traverse the skeleton following connectivity
    ordered = [start]
    visited = {start}
    current = start
    
    while True:
        # Find next unvisited neighbor
        next_pixel = None
        min_degree_neighbor = float('inf')
        
        # Prefer neighbors with lower degree to follow the main path
        for dx, dy in nbrs:
            cand = (current[0]+dx, current[1]+dy)
            if cand in coords_set and cand not in visited:
                cand_degree = degrees[cand]
                if cand_degree < min_degree_neighbor:
                    min_degree_neighbor = cand_degree
                    next_pixel = cand
        
        if next_pixel is None:
            break
            
        ordered.append(next_pixel)
        visited.add(next_pixel)
        current = next_pixel
    
    # Convert back to [x, y] format
    return [[p[0], p[1]] for p in ordered]


def order_centerline_points(points):
    """Simple nearest-neighbor ordering kept for backward compatibility."""
    if len(points) < 3:
        return points
    ordered = [points[0]]
    remaining = points[1:]
    while remaining:
        current = ordered[-1]
        # Choose the nearest remaining point
        distances = [((p[0]-current[0])**2 + (p[1]-current[1])**2) for p in remaining]
        idx = int(np.argmin(distances))
        ordered.append(remaining.pop(idx))
    return ordered