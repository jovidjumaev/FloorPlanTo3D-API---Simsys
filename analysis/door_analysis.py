def analyzeDoorOrientation(door_mask, door_bbox, image_width, image_height):
    """Determine door swing direction using door mask geometry (leaf + arc).
    Returns dict with door_type (horizontal/vertical), estimated_swing, hinge_side, confidence.
    """
    orientation_analysis = {"door_type": "unknown", "estimated_swing": "unknown", "hinge_side": "unknown", "confidence": 0.3, "analysis_method": "geometric_arc"}
    try:
        if door_mask is None:
            return orientation_analysis
        # Ensure binary uint8 mask
        dm = (door_mask.astype("uint8") * 255)
        dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, numpy.ones((3,3), numpy.uint8))
        contours,_ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("no contour")
        cnt = max(contours, key=cv2.contourArea)
        (cx,cy),(w_box,h_box),theta = cv2.minAreaRect(cnt)
        vertical_leaf = h_box > w_box
        orientation_analysis["door_type"] = "horizontal" if vertical_leaf else "vertical"
        # Straight-edge mask
        edge_mask = numpy.zeros_like(dm)
        box_pts = cv2.boxPoints(((cx,cy),(w_box,h_box),theta)).astype(int)
        cv2.drawContours(edge_mask, [box_pts], -1, 255, 1)
        dist = cv2.distanceTransform(255-edge_mask, cv2.DIST_L2, 3)
        arc_pixels = safe_logical_and((dm > 0).astype(bool), (dist > 2).astype(bool))
        # FIXED: Check for arc pixels properly
        if numpy.count_nonzero(arc_pixels) == 0:
            raise ValueError("no arc")
        ys,xs = numpy.nonzero(arc_pixels)
        xc_arc, yc_arc = float(xs.mean()), float(ys.mean())
        vec_x, vec_y = xc_arc - cx, yc_arc - cy
        if vertical_leaf:
            # Door opens up/down (horizontal door)
            if vec_y > 0:
                orientation_analysis.update({"estimated_swing":"opens_downward","hinge_side":"top_edge"})
            else:
                orientation_analysis.update({"estimated_swing":"opens_upward","hinge_side":"bottom_edge"})
        else:
            # Door opens left/right (vertical door)
            if vec_x > 0:
                orientation_analysis.update({"estimated_swing":"opens_rightward","hinge_side":"left_edge"})
            else:
                orientation_analysis.update({"estimated_swing":"opens_leftward","hinge_side":"right_edge"})
        orientation_analysis["confidence"] = 0.9
        return orientation_analysis
    except Exception:
        # Fallback to simple image-center heuristic
        (y1,x1,y2,x2) = door_bbox
        vertical_door = (y2-y1) > (x2-x1)
        orientation_analysis["door_type"] = "vertical" if vertical_door else "horizontal"
        orientation_analysis["analysis_method"] = "fallback"
        if vertical_door:
            orientation_analysis["estimated_swing"] = "opens_leftward" if (x1+x2)/2 < image_width/2 else "opens_rightward"
            orientation_analysis["hinge_side"] = "left_edge" if orientation_analysis["estimated_swing"].endswith("leftward") else "right_edge"
        else:
            orientation_analysis["estimated_swing"] = "opens_upward" if (y1+y2)/2 > image_height/2 else "opens_downward"
            orientation_analysis["hinge_side"] = "bottom_edge" if orientation_analysis["estimated_swing"].endswith("upward") else "top_edge"
        orientation_analysis["confidence"] = 0.4
        return orientation_analysis
    
def enhancedDoorAnalysis(door_objects, masks, door_indices, image_width, image_height):
	"""
	Enhance door objects with orientation analysis
	"""
	enhanced_doors = []
	
	for i, door in enumerate(door_objects):
		# Get the correct mask index for this door
		door_mask_index = door_indices[i] if i < len(door_indices) else None
		door_mask = masks[:, :, door_mask_index] if door_mask_index is not None and door_mask_index < masks.shape[2] else None
		door_bbox = door["bbox"]
		
		# Convert bbox format for analysis
		bbox_array = [door_bbox["y1"], door_bbox["x1"], door_bbox["y2"], door_bbox["x2"]]
		
		# Perform orientation analysis
		orientation = analyzeDoorOrientation(door_mask, bbox_array, image_width, image_height)
		
		# Add orientation data to door object
		enhanced_door = door.copy()
		enhanced_door["orientation"] = orientation
		
		# Add architectural insights
		enhanced_door["architectural_notes"] = generateArchitecturalNotes(orientation, door_bbox)
		
		enhanced_doors.append(enhanced_door)
	
	return enhanced_doors

def generateArchitecturalNotes(orientation, bbox):
	"""Generate architectural insights about door placement and orientation"""
	notes = []
	
	door_width = bbox["x2"] - bbox["x1"] 
	door_height = bbox["y2"] - bbox["y1"]
	
	# Standard door size analysis
	if door_width > door_height:
		if door_width >= 80:  # Assuming pixel measurements
			notes.append("Likely a standard interior door (32+ inches)")
		else:
			notes.append("Possible narrow door or entry")
	
	# Swing direction implications
	swing = orientation.get("estimated_swing", "")
	if "upward" in swing:
		notes.append("May open into upper room/hallway")
	elif "downward" in swing:
		notes.append("May open into lower room/hallway") 
	elif "leftward" in swing:
		notes.append("Opens toward left side of floor plan")
	elif "rightward" in swing:
		notes.append("Opens toward right side of floor plan")
	
	# Confidence-based recommendations
	confidence = orientation.get("confidence", 0)
	if confidence < 0.5:
		notes.append("Low confidence - manual verification recommended")
	elif confidence > 0.7:
		notes.append("High confidence orientation prediction")
	
	return notes



def categorize_door_size(width, height):
	"""Categorize door size based on dimensions"""
	door_area = width * height
	
	if door_area < 2000:
		return "small"
	elif door_area < 6000:
		return "standard"
	elif door_area < 10000:
		return "large"
	else:
		return "oversized"

def assess_door_accessibility(width):
	"""Assess door accessibility based on width"""
	# Rough pixel-to-inch conversion (this would need calibration)
	if width > 80:  # Assuming roughly 32+ inches
		return "ADA_compliant_likely"
	elif width > 60:  # Roughly 24+ inches
		return "standard_width"
	else:
		return "narrow_door"

def generate_door_layout_insights(doors, image_width, image_height):
	"""Generate insights about door layout and positioning"""
	insights = []
	
	if len(doors) == 0:
		return ["No doors detected for layout analysis"]
	
	# Analyze door distribution
	door_positions = [(d["location"]["center"]["x"], d["location"]["center"]["y"]) for d in doors]
	
	# Check for clustered doors
	clustered_count = 0
	for i, (x1, y1) in enumerate(door_positions):
		for j, (x2, y2) in enumerate(door_positions[i+1:], i+1):
			distance = ((x2-x1)**2 + (y2-y1)**2)**0.5
			if distance < min(image_width, image_height) * 0.2:  # 20% of image size
				clustered_count += 1
	
	if clustered_count > 0:
		insights.append(f"Found {clustered_count} pairs of doors in close proximity")
	
	# Analyze door alignment
	horizontal_doors = [d for d in doors if d["orientation"]["door_type"] == "horizontal"]
	vertical_doors = [d for d in doors if d["orientation"]["door_type"] == "vertical"]
	
	if len(horizontal_doors) > len(vertical_doors):
		insights.append("Predominantly horizontal door layout")
	elif len(vertical_doors) > len(horizontal_doors):
		insights.append("Predominantly vertical door layout")
	else:
		insights.append("Mixed door orientation layout")
	
	# Check for doors near image boundaries
	boundary_doors = 0
	for door in doors:
		x, y = door["location"]["center"]["x"], door["location"]["center"]["y"]
		margin = 0.1  # 10% margin
		if (x < image_width * margin or x > image_width * (1-margin) or 
			y < image_height * margin or y > image_height * (1-margin)):
			boundary_doors += 1
	
	if boundary_doors > 0:
		insights.append(f"{boundary_doors} doors positioned near floor plan boundaries")
	
	return insights


def convert_door_center_to_mm(door_data, scale_factor_mm_per_pixel):
    """Convert door center coordinates from pixels to millimeters"""
    if not door_data or "location" not in door_data or "center" not in door_data["location"]:
        return door_data
    
    door_mm = door_data.copy()
    center_pixels = door_data["location"]["center"]
    if "x" in center_pixels and "y" in center_pixels:
        x_mm = pixels_to_mm(center_pixels["x"], scale_factor_mm_per_pixel)
        y_mm = pixels_to_mm(center_pixels["y"], scale_factor_mm_per_pixel)
        door_mm["location"]["center_mm"] = {"x": x_mm, "y": y_mm}
    
    return door_mm