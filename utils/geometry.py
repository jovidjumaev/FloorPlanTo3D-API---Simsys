# Geometric utility functions for floor plan analysis
import numpy

def line_intersects_rectangle(x1, y1, x2, y2, rect_x1, rect_y1, rect_x2, rect_y2):
    """
    Check if a line segment intersects with a rectangle.
    
    Args:
        x1, y1, x2, y2: Line segment endpoints
        rect_x1, rect_y1, rect_x2, rect_y2: Rectangle bounds
    
    Returns:
        True if the line intersects the rectangle, False otherwise
    """
    # First check if either endpoint is inside the rectangle
    if ((rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2) or
        (rect_x1 <= x2 <= rect_x2 and rect_y1 <= y2 <= rect_y2)):
        return True
    
    # Check if line intersects any of the four rectangle edges
    # Top edge
    if line_segments_intersect(x1, y1, x2, y2, rect_x1, rect_y1, rect_x2, rect_y1):
        return True
    # Bottom edge  
    if line_segments_intersect(x1, y1, x2, y2, rect_x1, rect_y2, rect_x2, rect_y2):
        return True
    # Left edge
    if line_segments_intersect(x1, y1, x2, y2, rect_x1, rect_y1, rect_x1, rect_y2):
        return True
    # Right edge
    if line_segments_intersect(x1, y1, x2, y2, rect_x2, rect_y1, rect_x2, rect_y2):
        return True
    
    return False

def line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Check if two line segments intersect.
    
    Args:
        x1, y1, x2, y2: First line segment endpoints
        x3, y3, x4, y4: Second line segment endpoints
    
    Returns:
        True if the line segments intersect, False otherwise
    """
    # Calculate direction vectors
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return False
    
    # Calculate intersection parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection point is within both line segments
    return 0 <= t <= 1 and 0 <= u <= 1

def split_line_around_windows(x1, y1, x2, y2, bboxes, class_ids):
    """
    Split a line segment around windows, returning only the parts that don't cross windows.
    
    Args:
        x1, y1, x2, y2: Line segment endpoints
        bboxes: All bounding boxes from detection
        class_ids: Class IDs corresponding to bboxes
    
    Returns:
        List of line segments [(start_point, end_point), ...] that don't cross windows
    """
    # Start with the original line segment
    segments_to_process = [((x1, y1), (x2, y2))]
    final_segments = []
    
    # Get all window bounding boxes with margin
    window_rects = []
    for idx, cid in enumerate(class_ids):
        if cid == 2:  # window
            wy1, wx1, wy2, wx2 = bboxes[idx]
            # Add slightly larger margin around window to ensure better exclusion
            margin = 6  # Slightly increased for better window avoidance
            wx1 -= margin
            wy1 -= margin
            wx2 += margin
            wy2 += margin
            window_rects.append((wx1, wy1, wx2, wy2))
    
    # Process each line segment against all windows
    for seg_start, seg_end in segments_to_process:
        sx, sy = seg_start
        ex, ey = seg_end
        
        # Check if this segment intersects any window
        intersects_window = False
        
        for wx1, wy1, wx2, wy2 in window_rects:
            # Check if line crosses window or has significant overlap
            line_crosses = line_intersects_rectangle(sx, sy, ex, ey, wx1, wy1, wx2, wy2)
            
            # Check endpoint containment for short segments or if endpoints are very close to window
            segment_length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            if segment_length < 15:  # Slightly increased from 10 to 15
                start_inside = (wx1 <= sx <= wx2 and wy1 <= sy <= wy2)
                end_inside = (wx1 <= ex <= wx2 and wy1 <= ey <= wy2)
                if start_inside or end_inside:
                    line_crosses = True
            
            # Additional check: if either endpoint is very close to window edge (within 2 pixels)
            else:
                close_margin = 2
                start_very_close = (wx1 - close_margin <= sx <= wx2 + close_margin and 
                                   wy1 - close_margin <= sy <= wy2 + close_margin)
                end_very_close = (wx1 - close_margin <= ex <= wx2 + close_margin and 
                                 wy1 - close_margin <= ey <= wy2 + close_margin)
                if start_very_close or end_very_close:
                    line_crosses = True
            
            if line_crosses:
                intersects_window = True
                # Split the line at the window boundaries
                split_segments = split_line_at_rectangle(sx, sy, ex, ey, wx1, wy1, wx2, wy2)
                final_segments.extend(split_segments)
                break
        
        # If no intersection with any window, keep the original segment
        if not intersects_window:
            final_segments.append((seg_start, seg_end))
    
    return final_segments

def split_line_at_rectangle(x1, y1, x2, y2, rect_x1, rect_y1, rect_x2, rect_y2):
    """
    Split a line segment at the boundaries of a rectangle, returning the parts outside the rectangle.
    
    Args:
        x1, y1, x2, y2: Line segment endpoints
        rect_x1, rect_y1, rect_x2, rect_y2: Rectangle bounds
    
    Returns:
        List of line segments that are outside the rectangle
    """
    segments = []
    
    # Check if line actually crosses the rectangle
    line_crosses = line_intersects_rectangle(x1, y1, x2, y2, rect_x1, rect_y1, rect_x2, rect_y2)
    
    # Check endpoint containment
    start_inside = (rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2)
    end_inside = (rect_x1 <= x2 <= rect_x2 and rect_y1 <= y2 <= rect_y2)
    
    # Additional check: if endpoints are very close to window edge
    close_margin = 2
    start_very_close = (rect_x1 - close_margin <= x1 <= rect_x2 + close_margin and 
                       rect_y1 - close_margin <= y1 <= rect_y2 + close_margin)
    end_very_close = (rect_x1 - close_margin <= x2 <= rect_x2 + close_margin and 
                     rect_y1 - close_margin <= y2 <= rect_y2 + close_margin)
    
    # If no intersection and endpoints aren't inside or very close, return original line
    if not line_crosses and not (start_inside or end_inside or start_very_close or end_very_close):
        return [((x1, y1), (x2, y2))]
    
    # If both endpoints are inside rectangle or very close to it, don't draw anything
    if (start_inside and end_inside) or (start_very_close and end_very_close):
        return []
    
    # Find intersection points with rectangle edges
    intersections = []
    
    # Check intersection with each edge and collect intersection points
    edges = [
        (rect_x1, rect_y1, rect_x2, rect_y1),  # top edge
        (rect_x1, rect_y2, rect_x2, rect_y2),  # bottom edge  
        (rect_x1, rect_y1, rect_x1, rect_y2),  # left edge
        (rect_x2, rect_y1, rect_x2, rect_y2)   # right edge
    ]
    
    for edge_x1, edge_y1, edge_x2, edge_y2 in edges:
        intersection = line_intersection_point(x1, y1, x2, y2, edge_x1, edge_y1, edge_x2, edge_y2)
        if intersection:
            intersections.append(intersection)
    
    # Remove duplicate intersections (within small tolerance)
    unique_intersections = []
    for ix, iy in intersections:
        is_duplicate = False
        for ux, uy in unique_intersections:
            if abs(ix - ux) < 1 and abs(iy - uy) < 1:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_intersections.append((ix, iy))
    
    if not unique_intersections:
        # Line is entirely inside rectangle, don't draw it
        return []
    
    if len(unique_intersections) == 1:
        # Line starts or ends inside rectangle
        ix, iy = unique_intersections[0]
        if rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2:
            # Start point is inside, draw from intersection to end
            segments.append(((ix, iy), (x2, y2)))
        else:
            # End point is inside, draw from start to intersection
            segments.append(((x1, y1), (ix, iy)))
    
    elif len(unique_intersections) >= 2:
        # Line crosses through rectangle, sort intersections by distance from start
        intersections_with_dist = []
        for ix, iy in unique_intersections:
            dist = ((ix - x1) ** 2 + (iy - y1) ** 2) ** 0.5
            intersections_with_dist.append((dist, ix, iy))
        
        intersections_with_dist.sort()
        
        # Take the first two intersections (entry and exit points)
        if len(intersections_with_dist) >= 2:
            _, ix1, iy1 = intersections_with_dist[0]
            _, ix2, iy2 = intersections_with_dist[1]
            
            # Draw segment from start to first intersection
            if not (rect_x1 <= x1 <= rect_x2 and rect_y1 <= y1 <= rect_y2):
                segments.append(((x1, y1), (ix1, iy1)))
            
            # Draw segment from second intersection to end
            if not (rect_x1 <= x2 <= rect_x2 and rect_y1 <= y2 <= rect_y2):
                segments.append(((ix2, iy2), (x2, y2)))
    
    return segments

def line_intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Find the intersection point of two line segments.
    
    Returns:
        (x, y) intersection point if segments intersect, None otherwise
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection point is within both line segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    
    return None

def calculateOverlap(bbox1, bbox2):
	"""Calculate the overlap ratio between two bounding boxes"""
	y1_1, x1_1, y2_1, x2_1 = bbox1
	y1_2, x1_2, y2_2, x2_2 = bbox2
	
	# Calculate intersection
	x_left = max(x1_1, x1_2)
	y_top = max(y1_1, y1_2)
	x_right = min(x2_1, x2_2)
	y_bottom = min(y2_1, y2_2)
	
	if x_right < x_left or y_bottom < y_top:
		return 0.0
	
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	
	# Calculate union
	bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
	bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
	union_area = bbox1_area + bbox2_area - intersection_area
	
	return intersection_area / union_area if union_area > 0 else 0.0

def safe_logical_and(a, b):
    """Safe wrapper for numpy.logical_and with proper boolean conversion"""
    return numpy.logical_and(numpy.asarray(a, dtype=bool), numpy.asarray(b, dtype=bool))

def safe_logical_or(a, b):
    """Safe wrapper for numpy.logical_or with proper boolean conversion"""
    return numpy.logical_or(numpy.asarray(a, dtype=bool), numpy.asarray(b, dtype=bool))

def safe_logical_not(a):
    """Safe wrapper for numpy.logical_not with proper boolean conversion"""
    return numpy.logical_not(numpy.asarray(a, dtype=bool))

def find_nearest_valid_point(x, y, wall_mask, max_search_radius=5):
    """Find the nearest valid wall point within a search radius"""
    height, width = wall_mask.shape
    
    for radius in range(1, max_search_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                # Check if this is a point on the radius boundary
                if abs(dx) == radius or abs(dy) == radius:
                    new_y, new_x = y + dy, x + dx
                    if (0 <= new_y < height and 0 <= new_x < width and 
     			        bool(wall_mask[new_y, new_x].item())): # FIXED: removed numpy.any()
                        return (new_y, new_x)
    
    return None

