def categorize_window_size(width, height):
	"""Categorize window size based on dimensions"""
	window_area = width * height
	
	if window_area < 1000:
		return "small"
	elif window_area < 4000:
		return "standard"
	elif window_area < 8000:
		return "large"
	else:
		return "oversized"

def assess_window_glazing(width, height):
	"""Assess window glazing type based on dimensions"""
	# Determine if it's likely single, double, or triple glazing based on size
	window_area = width * height
	
	if window_area < 2000:
		return "single_glazing_likely"
	elif window_area < 6000:
		return "double_glazing_likely"
	else:
		return "triple_glazing_likely"

def generate_window_notes(width, height, window_type):
	"""Generate architectural insights about window placement and type"""
	notes = []
	
	window_area = width * height
	

	if window_area < 1000:
		notes.append("Small window - possibly for ventilation or light")
	elif window_area > 8000:
		notes.append("Large window - likely for significant natural lighting")
	

	if window_type == "horizontal":
		notes.append("Horizontal window orientation")
		if width > height * 2:
			notes.append("Wide horizontal window - panoramic view")
	else:
		notes.append("Vertical window orientation")
		if height > width * 2:
			notes.append("Tall vertical window - floor-to-ceiling style")
	

	aspect_ratio = width / height if height > 0 else 0
	if aspect_ratio > 3:
		notes.append("Very wide window - modern architectural style")
	elif aspect_ratio < 0.5:
		notes.append("Very tall window - contemporary design")
	
	return notes
