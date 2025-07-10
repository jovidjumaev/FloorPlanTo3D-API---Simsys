"""
JSON building and response formatting services
"""

import numpy
from datetime import datetime
from image_processing.image_loader import (
    calculateObjectArea, calculateObjectCenter, encodeMaskSummary, getClassName)
from analysis.door_analysis import enhancedDoorAnalysis


def buildEnhancedJson(model_results, image_width, image_height, original_image):
    """Build comprehensive JSON with all semantic information"""
    
    # Extract all data from model results
    bboxes = model_results['rois']
    class_ids = model_results['class_ids'] 
    scores = model_results['scores']
    masks = model_results['masks']
    
    # Build enhanced object list
    objects = []
    door_objects = []
    door_sizes = []
    
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        class_id = class_ids[i]
        confidence = float(scores[i])
        mask = masks[:, :, i] if i < masks.shape[2] else None
        
        # Calculate geometric properties
        area = calculateObjectArea(mask) if mask is not None else 0
        center = calculateObjectCenter(bbox)
        width = float(bbox[3] - bbox[1])
        height = float(bbox[2] - bbox[0])
        
        # Track door sizes for average calculation
        if class_id == 3:  # door
            door_size = max(width, height)
            door_sizes.append(door_size)
        
        # Build object data
        obj_data = {
            "id": i,
            "type": getClassName(class_id),
            "confidence": confidence,
            "bbox": {
                "x1": float(bbox[1]),
                "y1": float(bbox[0]), 
                "x2": float(bbox[3]),
                "y2": float(bbox[2])
            },
            "dimensions": {
                "width": width,
                "height": height,
                "area": float(area)
            },
            "center": center
        }
        
        # Add mask summary data if available
        if mask is not None:
            obj_data["mask_analysis"] = encodeMaskSummary(mask)
            obj_data["mask_analysis"]["shape"] = {"height": int(mask.shape[0]), "width": int(mask.shape[1])}
        
        # Collect door objects for enhanced analysis
        if class_id == 3:  # door
            door_objects.append(obj_data)
        
        objects.append(obj_data)
    
    # Enhance doors with orientation analysis
    if door_objects:
        # Get door indices and extract masks properly
        door_indices = [i for i, cid in enumerate(class_ids) if cid == 3]
        enhanced_doors = enhancedDoorAnalysis(door_objects, masks, door_indices, image_width, image_height)
        
        # Replace door objects in main list with enhanced versions
        door_index = 0
        for i, obj in enumerate(objects):
            if obj["type"] == "door":
                objects[i] = enhanced_doors[door_index]
                door_index += 1
    
    # Calculate statistics
    average_door_size = sum(door_sizes) / len(door_sizes) if door_sizes else 0
    
    # Count object types
    object_counts = {}
    for obj in objects:
        obj_type = obj["type"]
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
    
    # Build comprehensive JSON structure
    enhanced_json = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": {
                "width": image_width,
                "height": image_height
            },
            "total_objects_detected": len(objects),
            "object_counts": object_counts
        },
        "objects": objects,
        "statistics": {
            "average_door_size": float(average_door_size),
            "total_area_detected": sum(obj["dimensions"]["area"] for obj in objects),
            "confidence_scores": {
                "min": float(min(scores)) if len(scores) > 0 else 0,
                "max": float(max(scores)) if len(scores) > 0 else 0,
                "average": float(numpy.mean(scores)) if len(scores) > 0 else 0
            }
        },
        "legacy_format": {
            "Width": image_width,
            "Height": image_height,
            "averageDoor": float(average_door_size),
            "classes": [{"name": getClassName(cid)} for cid in class_ids],
            "points": [
                {
                    "x1": float(bbox[1]),
                    "y1": float(bbox[0]),
                    "x2": float(bbox[3]), 
                    "y2": float(bbox[2])
                }
                for bbox in bboxes
            ]
        }
    }
    
    return enhanced_json 