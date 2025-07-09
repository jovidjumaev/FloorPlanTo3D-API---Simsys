import numpy
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects, skeletonize
from skimage.measure import label, regionprops
from utils.geometry import safe_logical_and, find_nearest_valid_point
from analysis.junction_analysis import find_junction_points_simple

def extract_wall_masks(model_results):
	"""Extract all wall masks from model results"""
	wall_masks = []
	wall_indices = []
	
	for i, class_id in enumerate(model_results['class_ids']):
		if class_id == 1:  # wall class
			wall_indices.append(i)
			wall_masks.append(model_results['masks'][:, :, i])
	
	return wall_masks, wall_indices

def improve_mask_for_skeletonization(wall_mask):
    """Improve wall mask quality for better skeletonization of angled walls."""
    
    # Apply morphological operations to smooth the wall mask
    # This helps with angled walls by filling small gaps and smoothing edges
    structuring_element = disk(2)  # Small disk for gentle smoothing
    
    # Fill small gaps
    improved_mask = binary_closing(wall_mask, structuring_element)
    
    # Remove small protrusions
    improved_mask = binary_opening(improved_mask, disk(1))
    
    # Ensure the mask is still roughly the same size
    if numpy.sum(improved_mask) < 0.8 * numpy.sum(wall_mask):
        # If we lost too much, use the original mask
        return wall_mask
    
    return improved_mask

def keep_largest_component(binary_img):
    """Keep only the largest connected component"""
    from skimage.measure import label
    labeled_img = label(binary_img)
    if labeled_img.max() == 0:
        return binary_img
    
    # Find the largest component
    component_sizes = numpy.bincount(labeled_img.flatten())
    component_sizes[0] = 0  # Ignore background
    largest_component = numpy.argmax(component_sizes)
    
    # Keep only the largest component
    return labeled_img == largest_component



def clean_skeleton(skeleton):
    """Clean up skeleton to remove small artifacts and improve centerline quality."""
    from skimage.morphology import remove_small_objects
    from skimage.measure import label
    
    # Remove very small skeleton components that are likely artifacts
    labeled_skeleton = label(skeleton)
    cleaned_skeleton = remove_small_objects(labeled_skeleton, min_size=3) > 0
    
    return cleaned_skeleton.astype(bool)

def segment_individual_walls(wall_mask):
    """Segment connected wall regions into individual wall segments with robust per-region processing."""
    from skimage.morphology import remove_small_objects, skeletonize, binary_closing, binary_opening, disk
    from skimage.measure import label, regionprops
    import numpy as np
    
    cleaned_mask = remove_small_objects(wall_mask.astype(bool), min_size=50)
    labeled_regions = label(cleaned_mask)
    segments = []
    all_junctions = []
    num_regions = labeled_regions.max()
    
    for region_idx in range(1, num_regions + 1):
        region_mask = (labeled_regions == region_idx)
        if numpy.sum(region_mask) < 20:
            continue
            
        # Improve skeleton quality for angled walls
        improved_mask = improve_mask_for_skeletonization(region_mask)
        skeleton = skeletonize(improved_mask)
        skeleton = safe_logical_and(skeleton.astype(bool), region_mask.astype(bool))  # FIXED: explicit bool casting
        
        # Clean up skeleton to remove small artifacts
        skeleton = clean_skeleton(skeleton)
        
        junctions = find_junction_points_simple(skeleton)
        all_junctions.extend(junctions)
        
        skeleton_segmented = skeleton.copy()
        for jx, jy in junctions:
            y_start = max(0, jy-1)
            y_end = min(skeleton.shape[0], jy+2)
            x_start = max(0, jx-1)
            x_end = min(skeleton.shape[1], jx+2)
            skeleton_segmented[y_start:y_end, x_start:x_end] = False
            
        labeled_skeleton = label(skeleton_segmented)
        for region in regionprops(labeled_skeleton):
            if region.area > 5:
                seg_coords = region.coords
                # Keep only points strictly inside region mask
                filtered_coords = validate_centerline_boundary(seg_coords, region_mask)
                if len(filtered_coords) >= 2:
                    segments.append(filtered_coords)
    
    return segments, all_junctions


def validate_centerline_boundary(segment_coords, wall_mask):
    """Validate that centerline points stay within wall boundaries"""
    validated_coords = []
    
    for coord in segment_coords:
        y, x = coord
        # Check if the point is within the wall mask
        if 0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1]:
            if wall_mask[y, x].item():  # FIXED: removed numpy.any()
                validated_coords.append(coord)
            else:
                # If point is outside wall, try to find nearest valid point
                nearest_valid = find_nearest_valid_point(x, y, wall_mask)
                if nearest_valid is not None:
                    validated_coords.append(nearest_valid)
        else:
            # Point is outside image bounds, skip it
            continue
    
    return numpy.array(validated_coords) if validated_coords else numpy.array([])