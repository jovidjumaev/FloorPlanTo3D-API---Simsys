# Visualization module for floor plan analysis results
import numpy
from PIL import Image, ImageDraw, ImageFont
from utils.conversions import mm_to_pixels, pixels_to_mm
from utils.geometry import split_line_around_windows
from analysis.door_analysis import analyzeDoorOrientation
from analysis.wall_analysis import calculate_centered_straight_centerline


def create_wall_visualization(original_image, model_results, wall_parameters, junction_analysis, image_width, image_height, scale_factor_mm_per_pixel=1.0, exterior_walls=None, space_names=None):
	"""Create enhanced visualization with wall centerlines, junctions, parameters, and OCR-detected space names"""
	
	# Convert to RGB if needed
	if original_image.mode != 'RGB':
		vis_image = original_image.convert('RGB')
	else:
		vis_image = original_image.copy()
	
	# Create drawing context
	draw = ImageDraw.Draw(vis_image)
	
	# Define colors
	colors = {
		1: (255, 0, 0),     # Wall - Red
		2: (0, 255, 0),     # Window - Green  
		3: (0, 0, 255),     # Door - Blue
	}

	centerline_color = (255, 255, 0)     # Yellow for centerlines (more visible)
	junction_color = (255, 0, 255)       # Magenta for junctions
	text_color = (0, 0, 0)               # Black for text
	
	class_names = {1: 'Wall', 2: 'Window', 3: 'Door'}
	
	# First draw regular detection boxes
	bboxes = model_results['rois']
	class_ids = model_results['class_ids']
	scores = model_results['scores']
	masks = model_results.get('masks', None)
	
	for i in range(len(bboxes)):
		bbox = bboxes[i]
		class_id = class_ids[i]
		confidence = scores[i]
		
		y1, x1, y2, x2 = bbox
		color = colors.get(class_id, (128, 128, 128))
		
		# Draw bounding box (thinner for walls to not interfere with centerlines)
		width = 1 if class_id == 1 else 3
		draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
		
		# Enhanced door visualization with orientation and centerpoint
		if class_id == 3 and masks is not None:  # door
			door_mask = masks[:, :, i] if i < masks.shape[2] else None
			orientation = analyzeDoorOrientation(door_mask, bbox, image_width, image_height)
			
			# Calculate door center
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
			
			# Draw door centerpoint (cyan circle)
			center_radius = 4
			draw.ellipse([center_x - center_radius, center_y - center_radius, 
						 center_x + center_radius, center_y + center_radius], 
						fill=(0, 255, 255), outline=(0, 0, 0), width=2)
			
			# Yellow arrow drawing disabled as per user request
			pass
		
		# Enhanced window visualization with centerpoint
		if class_id == 2:  # window
			# Calculate window center
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
		
			
			# Draw window centerpoint (orange circle)
			center_radius = 6
			draw.ellipse([center_x - center_radius, center_y - center_radius, 
						 center_x + center_radius, center_y + center_radius], 
						fill=(255, 165, 0), outline=(0, 0, 0), width=2)
		
		# Draw label with confidence (enhanced for doors)
		if class_id == 3:  # door
			swing_info = ""
			if masks is not None and i < masks.shape[2]:
				door_mask = masks[:, :, i]
				orientation = analyzeDoorOrientation(door_mask, bbox, image_width, image_height)
				swing_info = f" | {orientation.get('estimated_swing', 'unknown')}"
			label = f"{class_names.get(class_id, 'Unknown')} ({confidence:.2f}){swing_info}"
		else:
			label = f"{class_names.get(class_id, 'Unknown')} ({confidence:.2f})"
		
		# Try to load a font, fall back to default if not available
		try:
			font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
		except:
			font = ImageFont.load_default()
		
		# Calculate text size and position
		text_bbox = draw.textbbox((0, 0), label, font=font)
		text_width = text_bbox[2] - text_bbox[0]
		text_height = text_bbox[3] - text_bbox[1]
		
		# Position label above the bounding box
		text_x = x1
		text_y = max(0, y1 - text_height - 5)
		
		# Draw text background
		draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
					  fill=color, outline=color)
		
		# Draw text
		draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
	
	# Draw wall centerlines and track which walls already have centerlines
	# Pre-compute door bounding boxes once (x1,y1,x2,y2)
	door_boxes_xy = []
	for idx, cid in enumerate(class_ids):
		if cid == 3:
			dy1, dx1, dy2, dx2 = bboxes[idx]
			# Ensure all coordinates are scalars
			dy1, dx1, dy2, dx2 = float(dy1), float(dx1), float(dy2), float(dx2)
			door_boxes_xy.append((dx1, dy1, dx2, dy2))

	# Create a set of exterior wall IDs for quick lookup
	exterior_wall_ids = set()
	if exterior_walls:
		exterior_wall_ids = {wall["wall_id"] for wall in exterior_walls}

	# Track which wall bounding boxes already have centerlines drawn
	walls_with_centerlines = set()

	for wall in wall_parameters:
		# Skip if wall bbox overlaps any door heavily (>35% of wall bbox area)
		wb = wall.get("bbox", {})
		x1w = wb.get("x1", 0)
		y1w = wb.get("y1", 0)
		x2w = wb.get("x2", 0)
		y2w = wb.get("y2", 0)
		wall_area = max(1, (x2w - x1w) * (y2w - y1w))
		skip_wall = False
		for (dx1, dy1, dx2, dy2) in door_boxes_xy:
			inter_x1 = max(x1w, dx1)
			inter_y1 = max(y1w, dy1)
			inter_x2 = min(x2w, dx2)
			inter_y2 = min(y2w, dy2)
			# Ensure all values are scalars for comparison
			inter_x1 = float(inter_x1)
			inter_y1 = float(inter_y1)
			inter_x2 = float(inter_x2)
			inter_y2 = float(inter_y2)
			if inter_x1 < inter_x2 and inter_y1 < inter_y2:
				inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
				if inter_area / wall_area > 0.35:
					skip_wall = True
					break
		if skip_wall:
			continue

		centerline = wall["centerline"]
		# Only draw interior wall centerlines (yellow), not exterior (orange)
		is_exterior = wall["wall_id"] in exterior_wall_ids
		if is_exterior:
			continue  # REMOVE drawing of exterior wall centerlines
		# Filter: Only draw wall if centerline is long enough and not degenerate
		if len(centerline) > 1:
			total_length = 0
			unique_points = set()
			for i in range(1, len(centerline)):
				p1 = centerline[i-1]
				p2 = centerline[i]
				if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)):
					dx = p2[0] - p1[0]
					dy = p2[1] - p1[1]
					total_length += (dx**2 + dy**2) ** 0.5
				unique_points.add(tuple(p1))
				unique_points.add(tuple(p2))
			if total_length < 10 or len(unique_points) < 2:
				continue
			wall_color = centerline_color
			wall_width = 4
			for i in range(1, len(centerline)):
				p1 = centerline[i-1]
				p2 = centerline[i]
				if isinstance(p1, (list, numpy.ndarray)):
					p1 = tuple(p1) if isinstance(p1, list) else tuple(p1.tolist())
				if isinstance(p2, (list, numpy.ndarray)):
					p2 = tuple(p2) if isinstance(p2, list) else tuple(p2.tolist())
				
				# Skip drawing lines in legend area (top-left 40x140 pixel box)
				legend_area_x = 40
				legend_area_y = 140  # legend_y (10) + 8 items * 15 pixels
				if ((p1[0] < legend_area_x and p1[1] < legend_area_y) or 
					(p2[0] < legend_area_x and p2[1] < legend_area_y)):
					continue
				
				draw.line([p1, p2], fill=wall_color, width=wall_width)
			
			# Mark this wall region as having a centerline to avoid duplicates in fallback
			# Convert millimeter coordinates back to pixels for comparison
			x1w_px = mm_to_pixels(x1w, scale_factor_mm_per_pixel)
			y1w_px = mm_to_pixels(y1w, scale_factor_mm_per_pixel)
			x2w_px = mm_to_pixels(x2w, scale_factor_mm_per_pixel)
			y2w_px = mm_to_pixels(y2w, scale_factor_mm_per_pixel)
			walls_with_centerlines.add((int(x1w_px), int(y1w_px), int(x2w_px), int(y2w_px)))
		# Wall ID drawing is temporarily disabled as per user request
	
	# Draw junctions
	for junction in junction_analysis:
		# Convert millimeter position back to pixels for visualization
		pos_mm = junction["position"]
		pos = [mm_to_pixels(pos_mm[0], scale_factor_mm_per_pixel), mm_to_pixels(pos_mm[1], scale_factor_mm_per_pixel)]
		junction_id = junction["junction_id"]
		
		# Ensure pos is a list/tuple for safe indexing
		if isinstance(pos, numpy.ndarray):
			pos = pos.tolist()
		
		# Draw junction circle
		radius = 6
		draw.ellipse([pos[0]-radius, pos[1]-radius, pos[0]+radius, pos[1]+radius], 
					fill=junction_color, outline=(0, 0, 0), width=2)
		
		# Draw junction ID
		try:
			font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
		except:
			font = ImageFont.load_default()
		
		text_bbox = draw.textbbox((0, 0), junction_id, font=font)
		text_width = text_bbox[2] - text_bbox[0]
		text_height = text_bbox[3] - text_bbox[1]
		
		text_x = pos[0] - text_width // 2
		text_y = pos[1] + radius + 5
		
		# Draw text background
		draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
					  fill=(255, 255, 255), outline=junction_color)
		
		# Draw junction ID text
		draw.text((text_x, text_y), junction_id, fill=text_color, font=font)
	
	# Add legend
	legend_y = 10
	legend_items = [
		("Walls (Red boxes)", (255, 0, 0)),
		("Exterior Walls (Dark Orange)", (255, 140, 0)),
		("Interior Walls (Yellow)", centerline_color),
		("Junctions (Magenta)", junction_color),
		("Windows (Green)", (0, 255, 0)),
		("Window Centers (Orange circles)", (255, 165, 0)),
		("Doors (Blue)", (0, 0, 255)),
		("Door Centers (Cyan circles)", (0, 255, 255))
	]
	
	try:
		legend_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 10)
	except:
		legend_font = ImageFont.load_default()
	
	for i, (text, color) in enumerate(legend_items):
		y_pos = legend_y + i * 15
		# Draw color indicator (smaller)
		draw.rectangle([10, y_pos, 20, y_pos+8], fill=color, outline=(0, 0, 0))
		# Draw text
		draw.text((25, y_pos-1), text, fill=(0, 0, 0), font=legend_font)
	
	# Improved fallback centerlines only for walls that don't already have centerlines
	for i in range(len(bboxes)):
		if class_ids[i] != 1:
			continue
		
		y1, x1, y2, x2 = bboxes[i]
		
		# Check if this wall already has a centerline drawn
		wall_bbox = (int(x1), int(y1), int(x2), int(y2))
		wall_already_has_centerline = False
		
		# Check if this bbox overlaps significantly with any wall that already has a centerline
		for (wx1, wy1, wx2, wy2) in walls_with_centerlines:
			# Calculate overlap
			overlap_x1 = max(x1, wx1)
			overlap_y1 = max(y1, wy1)
			overlap_x2 = min(x2, wx2)
			overlap_y2 = min(y2, wy2)
			
			if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
				overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
				bbox_area = (x2 - x1) * (y2 - y1)
				if overlap_area / bbox_area > 0.5:  # 50% overlap threshold
					wall_already_has_centerline = True
					break
		
		if wall_already_has_centerline:
			continue  # Skip this wall since it already has a centerline
		
		width_box = x2 - x1
		height_box = y2 - y1
		if width_box < 3 or height_box < 3:
			continue  # skip extremely small boxes
		
		# Get the wall mask for this bounding box if available
		wall_mask = None
		if masks is not None and i < masks.shape[2]:
			wall_mask = masks[:, :, i]
		
		centerline_drawn = False
		
		# If we have a mask, try to compute a better centerline
		if wall_mask is not None:
			try:
				# Use the new centered straight centerline approach
				bbox = (x1, y1, x2, y2)
				centerline_coords = calculate_centered_straight_centerline(wall_mask, bbox)
				
				if len(centerline_coords) > 1:
					# Draw the centered straight centerline
					valid_segments = []
					for j in range(1, len(centerline_coords)):
						p1 = (centerline_coords[j-1][0], centerline_coords[j-1][1])  # (x, y)
						p2 = (centerline_coords[j][0], centerline_coords[j][1])    # (x, y)
						
						# Skip drawing lines in legend area
						legend_area_x = 40
						legend_area_y = 140
						if ((p1[0] < legend_area_x and p1[1] < legend_area_y) or 
							(p2[0] < legend_area_x and p2[1] < legend_area_y)):
							continue
						
						# Split centerline segment around windows instead of excluding entirely
						segments_to_draw = split_line_around_windows(p1[0], p1[1], p2[0], p2[1], bboxes, class_ids)
						valid_segments.extend(segments_to_draw)
					
					# Draw only the valid segments that don't cross windows
					for seg_data in valid_segments:
						if len(seg_data) == 2:  # It's a pair of points
							p1, p2 = seg_data
							draw.line([p1, p2], fill=centerline_color, width=4)
					
					if valid_segments:  # Only mark as drawn if we actually drew something
						centerline_drawn = True
						walls_with_centerlines.add(wall_bbox)  # Mark as having centerline
			except:
				# If centered approach fails, fall back to simple line
				pass
		
		# Simple fallback for when mask is not available or skeleton fails
		if not centerline_drawn:
			# Create a more centered line by using multiple intermediate points
			if width_box >= height_box:
				# Horizontal wall - create points along the centerline
				cy = (y1 + y2) // 2
				num_points = max(3, min(10, width_box // 20))  # Adaptive number of points
				x_points = numpy.linspace(x1, x2, num_points)
				centerline_points = [(x, cy) for x in x_points]
			else:
				# Vertical wall - create points along the centerline
				cx = (x1 + x2) // 2
				num_points = max(3, min(10, height_box // 20))  # Adaptive number of points
				y_points = numpy.linspace(y1, y2, num_points)
				centerline_points = [(cx, y) for y in y_points]
			
			# Draw the centerline with multiple segments
			valid_segments = []
			for j in range(1, len(centerline_points)):
				p1 = centerline_points[j-1]
				p2 = centerline_points[j]
				
				# Skip drawing lines in legend area
				legend_area_x = 40
				legend_area_y = 140
				if ((p1[0] < legend_area_x and p1[1] < legend_area_y) or 
					(p2[0] < legend_area_x and p2[1] < legend_area_y)):
					continue
				
				# Split centerline segment around windows instead of excluding entirely
				segments_to_draw = split_line_around_windows(p1[0], p1[1], p2[0], p2[1], bboxes, class_ids)
				valid_segments.extend(segments_to_draw)
			
			if valid_segments:  # If we have any valid segments to draw
				for seg_start, seg_end in valid_segments:
					draw.line([seg_start, seg_end], fill=centerline_color, width=4)
				walls_with_centerlines.add(wall_bbox)  # Mark as having centerline
	
	# ----------------------
	# Final fallback centerlines (only for walls without any centerlines)
	# This section is now disabled since the improved fallback above handles everything
	# ----------------------
	
	# This section is commented out to avoid duplicate centerlines
	# The improved fallback above now handles both skeleton-based and simple fallback centerlines

	# Draw OCR-detected space names
	if space_names:
		print(f"Drawing {len(space_names)} detected space names on visualization")
		space_name_color = (255, 128, 0)  # Orange color for space names
		space_center_color = (255, 0, 128)  # Pink color for space centerpoints
		
		try:
			space_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
		except:
			space_font = ImageFont.load_default()
		
		for space in space_names:
			bbox = space['bbox']
			x1, y1, x2, y2 = bbox
			text = space['name']
			confidence = space['confidence']
			
			# Calculate centerpoint of the text
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
			
			# Draw bounding box around detected text
			draw.rectangle([x1, y1, x2, y2], outline=space_name_color, width=2)
			
			# Draw centerpoint circle
			center_radius = 8
			draw.ellipse([center_x - center_radius, center_y - center_radius, 
						 center_x + center_radius, center_y + center_radius], 
						fill=space_center_color, outline=(0, 0, 0), width=2)
			
			# Draw small cross inside the circle for better visibility
			cross_size = 4
			draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y], 
					 fill=(255, 255, 255), width=2)
			draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size], 
					 fill=(255, 255, 255), width=2)
			
			# Draw text label with confidence
			label = f"{text} ({confidence:.2f})"
			text_bbox = draw.textbbox((0, 0), label, font=space_font)
			text_width = text_bbox[2] - text_bbox[0]
			text_height = text_bbox[3] - text_bbox[1]
			
			# Position label above the text bounding box
			text_x = x1
			text_y = max(0, y1 - text_height - 5)
			
			# Draw text background
			draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
						  fill=space_name_color, outline=space_name_color)
			
			# Draw text
			draw.text((text_x, text_y), label, fill=(255, 255, 255), font=space_font)
		
		# Add space names to legend
		legend_items.append(("Space Names (Orange boxes)", space_name_color))
		legend_items.append(("Space Centers (Pink circles)", space_center_color))

	return vis_image