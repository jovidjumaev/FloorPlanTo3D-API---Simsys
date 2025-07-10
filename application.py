import os
import sys
import time
import json
import traceback
import logging
from datetime import datetime

# Third-party imports
import numpy
import cv2
import tensorflow as tf
from PIL import Image

# Flask and web framework imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Machine learning and image processing imports
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
from numpy import expand_dims

# Image processing libraries
import skimage.color

# Local imports
from config.constants import *
from config.settings import get_config

# Get application configuration
app_config = get_config()

# Utility imports
from utils.geometry import (
	line_intersects_rectangle, line_segments_intersect,
	split_line_around_windows, calculateOverlap, safe_logical_or, safe_logical_and)
from utils.conversions import (
	pixels_to_mm, mm_to_pixels, convert_centerline_to_mm,
	convert_thickness_to_mm, convert_bbox_to_mm, convert_junction_position_to_mm,
	save_wall_analysis)
from utils.file_utils import (
	getNextTestNumber, saveJsonToFile, saveAccuracyAnalysis)

from image_processing.image_loader import (
	myImageLoader, getClassNames, normalizePoints, turnSubArraysToJson,
	calculateObjectArea, calculateObjectCenter, encodeMaskSummary, getClassName)
from image_processing.mask_processing import (
	extract_wall_masks, improve_mask_for_skeletonization, keep_largest_component,
	clean_skeleton, segment_individual_walls, validate_centerline_boundary,
	find_nearest_valid_point)

from analysis.door_analysis import (
    analyzeDoorOrientation, enhancedDoorAnalysis, generateArchitecturalNotes,
    categorize_door_size, assess_door_accessibility, generate_door_layout_insights,
    convert_door_center_to_mm)

from analysis.wall_analysis import (
    calculate_wall_thickness, validate_centerline_in_walls, calculate_wall_length,
    calculate_wall_orientation, find_wall_connections, analyze_junction_types,
    extract_wall_parameters, extract_wall_parameters_with_regions, generate_wall_insights,
    identify_exterior_walls, calculate_perimeter_dimensions, calculate_centered_straight_centerline)

from analysis.junction_analysis import (
    find_junction_points, find_junction_points_simple, find_junctions_from_bboxes,
    extract_centerline_coords, extract_centerline_coords_with_validation,
    smooth_centerline_curve, order_centerline_points_connectivity, order_centerline_points)

from analysis.window_analysis import (
    categorize_window_size, assess_window_glazing, generate_window_notes)

from visualization.wall_visualization import create_wall_visualization

# Import OCR detector
from ocr_detector import detect_space_names

# Configure logging
logging.basicConfig(
	level=getattr(logging, app_config.LOG_LEVEL),
	format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Optional memory monitoring
try:
    import psutil
    import gc
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")

# Global variables
global _model
global cfg
ROOT_DIR = os.path.abspath("./")

# Add root directory to path
sys.path.append(ROOT_DIR)

# Initialize Flask application
application = Flask(__name__)
application.debug = app_config.DEBUG
cors = CORS(application, resources={r"/*": {"origins": app_config.CORS_ORIGINS}})


class PredictionConfig(Config):
	"""MaskRCNN model configuration"""
	# define the name of the configuration
	NAME = app_config.MODEL_NAME
	# number of classes (background + door + wall + window)
	NUM_CLASSES = app_config.NUM_CLASSES
	# simplify GPU config
	GPU_COUNT = app_config.GPU_COUNT
	IMAGES_PER_GPU = app_config.IMAGES_PER_GPU
	# Lower detection threshold to capture faint walls/doors
	DETECTION_MIN_CONFIDENCE = app_config.DETECTION_MIN_CONFIDENCE
	# Allow larger input images to preserve line detail
	IMAGE_MAX_DIM = app_config.IMAGE_MAX_DIM


def initialize_model():
	"""Initialize the MaskRCNN model once at startup"""
	global cfg, _model
	
	try:
		model_folder_path = os.path.abspath("./") + "/mrcnn"
		weights_path = os.path.join(app_config.WEIGHTS_FOLDER, app_config.WEIGHTS_FILE_NAME)
		
		# Check if weights file exists
		if not os.path.exists(weights_path):
			raise FileNotFoundError(f"Weights file not found: {weights_path}")
		
		cfg = PredictionConfig()
		logger.info(f"Model configuration: {cfg.IMAGE_RESIZE_MODE}")
		logger.info('==============Initializing model=========')
		
		_model = MaskRCNN(mode='inference', model_dir=model_folder_path, config=cfg)
		logger.info('=================Model created==============')
		
		_model.load_weights(weights_path, by_name=True)
		logger.info('=================Model loaded successfully==============')
		
		return _model, cfg
		
	except Exception as e:
		logger.error(f"Error initializing model: {str(e)}")
		raise e


# Initialize model at startup
logger.info("Starting model initialization...")
try:
    _model, cfg = initialize_model()
    logger.info("Model initialization completed successfully!")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    _model = None
    cfg = None


def check_memory_usage():
    """Monitor memory usage and warn if high"""
    if not MEMORY_MONITORING_AVAILABLE or not app_config.ENABLE_MEMORY_MONITORING:
        return 0
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > app_config.MAX_MEMORY_USAGE_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            gc.collect()  # Force garbage collection
        
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return 0


def validate_and_resize_image(image, max_size=None):
    """
    Validate image size and resize if necessary
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height), defaults to config
    
    Returns:
        PIL Image: Original or resized image
        dict: Information about the resize operation
    """
    if max_size is None:
        max_size = app_config.MAX_IMAGE_SIZE
    
    original_size = image.size
    max_dimension = max(original_size)
    min_dimension = min(original_size)
    
    resize_info = {
        "original_size": original_size,
        "resized": False,
        "resize_factor": 1.0,
        "reason": "no_resize_needed"
    }
    
    # Check minimum size
    if min_dimension < app_config.MIN_IMAGE_SIZE:
        resize_info["reason"] = "image_too_small"
        logger.warning(f"Image too small: {original_size} (min: {app_config.MIN_IMAGE_SIZE})")
        return image, resize_info
    
    # Check maximum size
    if max_dimension > max_size and app_config.ALLOW_IMAGE_RESIZE:
        # Calculate resize factor
        resize_factor = max_size / max_dimension
        
        # Calculate new size maintaining aspect ratio
        new_width = int(original_size[0] * resize_factor)
        new_height = int(original_size[1] * resize_factor)
        new_size = (new_width, new_height)
        
        # Ensure minimum size after resize
        if min(new_size) < app_config.MIN_IMAGE_SIZE:
            logger.warning(f"Resize would make image too small: {new_size}")
            resize_info["reason"] = "resize_would_make_too_small"
            return image, resize_info
        
        # Resize image using high-quality method
        resize_method = getattr(Image, app_config.RESIZE_QUALITY, Image.LANCZOS)
        image = image.resize(new_size, resize_method)
        
        resize_info.update({
            "resized": True,
            "resize_factor": resize_factor,
            "new_size": new_size,
            "reason": "resized_for_performance"
        })
        
        logger.info(f"Image resized from {original_size} to {new_size} (factor: {resize_factor:.2f})")
    
    elif max_dimension > max_size and not app_config.ALLOW_IMAGE_RESIZE:
        resize_info["reason"] = "image_too_large_resize_disabled"
        logger.warning(f"Image too large: {original_size} (max: {max_size}) but resize is disabled")
    
    return image, resize_info


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


@application.route('/health', methods=['GET'])
def health_check():
	"""Health check endpoint to verify model status"""
	try:
		if _model is None or cfg is None:
			return jsonify({
				"status": "error",
				"message": "Model not initialized",
				"model_loaded": False
			}), 503
		
		return jsonify({
			"status": "healthy",
			"message": "Model loaded and ready",
			"model_loaded": True,
			"model_config": {
				"name": cfg.NAME,
				"num_classes": cfg.NUM_CLASSES,
				"detection_min_confidence": cfg.DETECTION_MIN_CONFIDENCE,
				"image_max_dim": cfg.IMAGE_MAX_DIM
			}
		}), 200
	except Exception as e:
		return jsonify({
			"status": "error",
			"message": f"Health check failed: {str(e)}",
			"model_loaded": False
		}), 500


@application.route('/analyze_accuracy', methods=['POST'])
def analyze_accuracy():
	"""Analyze the accuracy and reliability of the model predictions"""
	
	# Check if model is initialized
	if _model is None or cfg is None:
		return jsonify({"error": "Model not initialized. Please check server logs."}), 503
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		
		# Validate and resize image
		imagefile, resize_info = validate_and_resize_image(imagefile)
		
		# Check if image validation failed
		if resize_info["reason"] in ["image_too_small", "resize_would_make_too_small", "image_too_large_resize_disabled"]:
			return jsonify({
				"error": f"Image validation failed: {resize_info['reason']}",
				"details": {
					"original_size": resize_info["original_size"],
					"min_size": app_config.MIN_IMAGE_SIZE,
					"max_size": app_config.MAX_IMAGE_SIZE,
					"resize_allowed": app_config.ALLOW_IMAGE_RESIZE
				}
			}), 400
		
		# Check memory usage before processing
		memory_before = check_memory_usage()
		logger.debug(f"Memory before processing: {memory_before:.1f}MB")
		
		image, w, h = myImageLoader(imagefile)
		logger.info(f"Analyzing accuracy for image: {h}x{w}")
		
		# Log resize information if image was resized
		if resize_info["resized"]:
			logger.info(f"Image was resized: {resize_info['original_size']} -> {resize_info['new_size']}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
	
		# Perform accuracy analysis
		accuracy_report = performAccuracyAnalysis(r, w, h)
		
		# Save accuracy analysis with simple naming
		test_num = getNextTestNumber()
		filename = saveAccuracyAnalysis(accuracy_report, test_num)
		
		# Check memory usage after processing
		memory_after = check_memory_usage()
		logger.debug(f"Memory after processing: {memory_after:.1f}MB")
		
		# Add filename and image processing info to response
		response = accuracy_report.copy()
		response["analysis_file"] = filename
		response["image_processing"] = {
			"original_size": resize_info["original_size"],
			"processed_size": resize_info.get("new_size", resize_info["original_size"]),
			"resized": resize_info["resized"],
			"resize_factor": resize_info["resize_factor"],
			"resize_reason": resize_info["reason"]
		}
		response["memory_usage"] = {
			"before_processing_mb": memory_before,
			"after_processing_mb": memory_after,
			"memory_increase_mb": memory_after - memory_before
		}
		
		return jsonify(response)
		
	except Exception as e:
		logger.error(f"Error in accuracy analysis: {str(e)}")
		return jsonify({"error": str(e)}), 500

def performAccuracyAnalysis(model_results, image_width, image_height):
	"""Perform comprehensive accuracy analysis of model predictions"""
	
	bboxes = model_results['rois']
	class_ids = model_results['class_ids']
	scores = model_results['scores']
	masks = model_results['masks']
	
	analysis = {
		"overall_metrics": {
			"total_detections": len(bboxes),
			"image_coverage": 0.0,
			"average_confidence": float(numpy.mean(scores)) if len(scores) > 0 else 0.0,
			"confidence_distribution": {}
		},
		"detection_quality": {
			"high_confidence": [],  # > 0.8
			"medium_confidence": [], # 0.5-0.8
			"low_confidence": []     # < 0.5
		},
		"object_analysis": {
			"walls": {"count": 0, "avg_confidence": 0.0, "avg_size": 0.0},
			"windows": {"count": 0, "avg_confidence": 0.0, "avg_size": 0.0},
			"doors": {"count": 0, "avg_confidence": 0.0, "avg_size": 0.0}
		},
		"spatial_analysis": {
			"bbox_overlaps": 0,
			"size_anomalies": [],
			"position_analysis": {}
		},
		"reliability_score": 0.0,
		"recommendations": []
	}
	
	# Analyze each detection
	total_area_covered = 0
	class_data = {1: [], 2: [], 3: []}  # walls, windows, doors
	
	for i in range(len(bboxes)):
		bbox = bboxes[i]
		class_id = class_ids[i]
		confidence = float(scores[i])
		
		# Calculate bbox area
		y1, x1, y2, x2 = bbox
		bbox_area = (x2 - x1) * (y2 - y1)
		total_area_covered += bbox_area
		
		# Store class-specific data
		class_data[class_id].append({
			"confidence": confidence,
			"area": bbox_area,
			"bbox": [float(x1), float(y1), float(x2), float(y2)]
		})
		
		# Categorize by confidence
		detection_info = {
			"id": i,
			"type": getClassName(class_id),
			"confidence": confidence,
			"area": float(bbox_area),
			"bbox": [float(x1), float(y1), float(x2), float(y2)]
		}
		
		if confidence > 0.8:
			analysis["detection_quality"]["high_confidence"].append(detection_info)
		elif confidence > 0.5:
			analysis["detection_quality"]["medium_confidence"].append(detection_info)
		else:
			analysis["detection_quality"]["low_confidence"].append(detection_info)
	
	# Calculate overall metrics
	image_area = image_width * image_height
	analysis["overall_metrics"]["image_coverage"] = float(total_area_covered / image_area * 100)
	
	# Analyze each object class
	class_names = {1: "walls", 2: "windows", 3: "doors"}
	for class_id, class_name in class_names.items():
		if class_data[class_id]:
			data = class_data[class_id]
			analysis["object_analysis"][class_name] = {
				"count": len(data),
				"avg_confidence": float(numpy.mean([d["confidence"] for d in data])),
				"avg_size": float(numpy.mean([d["area"] for d in data])),
				"confidence_range": {
					"min": float(min([d["confidence"] for d in data])),
					"max": float(max([d["confidence"] for d in data]))
				}
			}
	
	# Check for overlapping bboxes (potential duplicates)
	overlaps = 0
	for i in range(len(bboxes)):
		for j in range(i + 1, len(bboxes)):
			if calculateOverlap(bboxes[i], bboxes[j]) > 0.3:  # 30% overlap threshold
				overlaps += 1
	
	analysis["spatial_analysis"]["bbox_overlaps"] = overlaps
	
	# Detect size anomalies
	if class_data[1]:  # walls
		wall_areas = [d["area"] for d in class_data[1]]
		median_wall_area = numpy.median(wall_areas)
		for i, area in enumerate(wall_areas):
			if area > median_wall_area * 5 or area < median_wall_area * 0.2:
				analysis["spatial_analysis"]["size_anomalies"].append({
					"type": "wall",
					"detection_id": i,
					"area": float(area),
					"median_area": float(median_wall_area),
					"reason": "unusually_large" if area > median_wall_area * 5 else "unusually_small"
				})
	
	# Calculate reliability score (0-100)
	reliability_factors = []
	
	# Factor 1: Average confidence (0-40 points)
	avg_confidence = analysis["overall_metrics"]["average_confidence"]
	reliability_factors.append(min(40, avg_confidence * 40))
	
	# Factor 2: High confidence detections ratio (0-30 points)
	high_conf_ratio = len(analysis["detection_quality"]["high_confidence"]) / max(1, len(bboxes))
	reliability_factors.append(high_conf_ratio * 30)
	
	# Factor 3: Penalize overlaps (0-20 points)
	overlap_penalty = max(0, 20 - overlaps * 5)
	reliability_factors.append(overlap_penalty)
	
	# Factor 4: Reasonable detection count (0-10 points)
	detection_count_score = 10 if 1 <= len(bboxes) <= 50 else max(0, 10 - abs(len(bboxes) - 25))
	reliability_factors.append(detection_count_score)
	
	analysis["reliability_score"] = sum(reliability_factors)
	
	# Generate recommendations
	recommendations = []
	
	if avg_confidence < 0.6:
		recommendations.append("Low average confidence detected. Consider using higher quality images or retraining the model.")
	
	if len(analysis["detection_quality"]["low_confidence"]) > len(bboxes) * 0.3:
		recommendations.append("Many low-confidence detections found. Review these detections manually.")
	
	if overlaps > 0:
		recommendations.append(f"Found {overlaps} overlapping detections. Check for duplicate objects.")
	
	if len(analysis["spatial_analysis"]["size_anomalies"]) > 0:
		recommendations.append(f"Found {len(analysis['spatial_analysis']['size_anomalies'])} size anomalies. Review unusual object sizes.")
	
	if analysis["reliability_score"] > 80:
		recommendations.append("High reliability score! Results appear very accurate.")
	elif analysis["reliability_score"] > 60:
		recommendations.append("Good reliability score. Results are generally trustworthy.")
	else:
		recommendations.append("Low reliability score. Carefully review all detections.")
	
	analysis["recommendations"] = recommendations
	
	return analysis








@application.route('/visualize_walls', methods=['POST'])
def visualize_wall_analysis():
	"""Create enhanced visualization showing wall centerlines, junctions, and wall parameters"""
	
	# Check if model is initialized
	if _model is None or cfg is None:
		return jsonify({"error": "Model not initialized. Please check server logs."}), 503
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		
		# Get scale factor from request (default to 1.0 if not provided)
		scale_factor_mm_per_pixel = float(request.form.get('scale_factor_mm_per_pixel', 1.0))
		
		# Validate and resize image
		imagefile, resize_info = validate_and_resize_image(imagefile)
		
		# Check if image validation failed
		if resize_info["reason"] in ["image_too_small", "resize_would_make_too_small", "image_too_large_resize_disabled"]:
			return jsonify({
				"error": f"Image validation failed: {resize_info['reason']}",
				"details": {
					"original_size": resize_info["original_size"],
					"min_size": app_config.MIN_IMAGE_SIZE,
					"max_size": app_config.MAX_IMAGE_SIZE,
					"resize_allowed": app_config.ALLOW_IMAGE_RESIZE
				}
			}), 400
		
		# Check memory usage before processing
		memory_before = check_memory_usage()
		logger.debug(f"Memory before processing: {memory_before:.1f}MB")
		
		# Adjust scale factor if image was resized
		if resize_info["resized"]:
			original_scale = scale_factor_mm_per_pixel
			scale_factor_mm_per_pixel *= resize_info["resize_factor"]
			logger.info(f"Adjusted scale factor from {original_scale:.4f} to {scale_factor_mm_per_pixel:.4f} due to image resize")
		
		original_image = imagefile.copy()
		
		# Office plan detection
		img_rgb_tmp2 = original_image.convert('RGB')
		gray_tmp = cv2.cvtColor(numpy.array(img_rgb_tmp2), cv2.COLOR_RGB2GRAY)
		edges_tmp = cv2.Canny(gray_tmp, 50, 150)
		avg_col = numpy.mean(numpy.sum(edges_tmp > 0, axis=0))
		avg_row = numpy.mean(numpy.sum(edges_tmp > 0, axis=1))
		is_office_plan = max(avg_col, avg_row) < 7
		image, w, h = myImageLoader(imagefile, enhance_for_office=is_office_plan)
		logger.info(f"Creating wall analysis visualization for image: {h}x{w} {'(office plan)' if is_office_plan else ''}")
		
		# Log resize information if image was resized
		if resize_info["resized"]:
			logger.info(f"Image was resized: {resize_info['original_size']} -> {resize_info['new_size']}")
		
		# --- timing start ---
		t0 = time.time()
		# Preprocess for model
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)
		logger.debug(f"Time - preprocessing: {time.time()-t0:.2f}s")
		
		# Model detection
		t0 = time.time()
		r = _model.detect(sample, verbose=0)[0]
		
		# Extract wall masks and perform analysis
		t0 = time.time()
		wall_masks, wall_indices = extract_wall_masks(r)
		logger.info(f"Extracted {len(wall_masks)} wall masks from model output")
		combined_wall_mask = numpy.zeros((h, w), dtype=bool)
		for mask in wall_masks:
			combined_wall_mask = safe_logical_or(combined_wall_mask.astype(bool), mask.astype(bool))
		
		# Build combined door mask (dilated door masks + expanded bounding boxes)
		combined_door_mask = numpy.zeros((h, w), dtype=bool)
		for idx, cid in enumerate(r['class_ids']):
			if cid == 3:
				# Ensure bbox coordinates are ints for slicing
				bbox = r['rois'][idx]
				y1, x1, y2, x2 = [int(round(v)) for v in bbox]
				# Add mask if available
				if 'masks' in r and idx < r['masks'].shape[2]:
					dm = r['masks'][:, :, idx]
					dilated_dm = cv2.dilate(dm.astype(numpy.uint8), numpy.ones((15,15), numpy.uint8), iterations=1).astype(bool)
					# enlarge buffer further to ensure no wall pixels remain near doors
					dilated_dm = cv2.dilate(dilated_dm.astype(numpy.uint8), numpy.ones((35,35), numpy.uint8), iterations=1).astype(bool)
					combined_door_mask = safe_logical_or(combined_door_mask.astype(bool), dilated_dm.astype(bool))
				# Add expanded bounding box area (with margin)
				margin = 40
				x1e = max(0, x1 - margin)
				y1e = max(0, y1 - margin)
				x2e = min(w-1, x2 + margin)
				y2e = min(h-1, y2 + margin)
				temp_mask = numpy.zeros_like(combined_door_mask)
				temp_mask[y1e:y2e+1, x1e:x2e+1] = True
				combined_door_mask = safe_logical_or(combined_door_mask, temp_mask)

		# Build combined window mask (dilated window masks + expanded bounding boxes)
		combined_window_mask = numpy.zeros((h, w), dtype=bool)
		for idx, cid in enumerate(r['class_ids']):
			if cid == 2:  # window class
				# Ensure bbox coordinates are ints for slicing
				bbox = r['rois'][idx]
				y1, x1, y2, x2 = [int(round(v)) for v in bbox]
				# Add mask if available
				if 'masks' in r and idx < r['masks'].shape[2]:
					wm = r['masks'][:, :, idx]
					dilated_wm = cv2.dilate(wm.astype(numpy.uint8), numpy.ones((10,10), numpy.uint8), iterations=1).astype(bool)
					# enlarge buffer to ensure no wall pixels remain near windows
					dilated_wm = cv2.dilate(dilated_wm.astype(numpy.uint8), numpy.ones((20,20), numpy.uint8), iterations=1).astype(bool)
					combined_window_mask = safe_logical_or(combined_window_mask.astype(bool), dilated_wm.astype(bool))
				# Add expanded bounding box area (with margin)
				margin = 25
				x1e = max(0, x1 - margin)
				y1e = max(0, y1 - margin)
				x2e = min(w-1, x2 + margin)
				y2e = min(h-1, y2 + margin)
				temp_mask = numpy.zeros_like(combined_window_mask)
				temp_mask[y1e:y2e+1, x1e:x2e+1] = True
				combined_window_mask = safe_logical_or(combined_window_mask, temp_mask)

		# Remove door and window areas from combined wall mask
		combined_wall_mask = safe_logical_and(combined_wall_mask.astype(bool), numpy.logical_not(combined_door_mask.astype(bool)))
		combined_wall_mask = safe_logical_and(combined_wall_mask.astype(bool), numpy.logical_not(combined_window_mask.astype(bool)))
		logger.info("Combined wall mask ready; starting skeletonisation & segment extraction …")
		wall_segments, junctions = segment_individual_walls(combined_wall_mask)
		logger.info(f"Found {len(wall_segments)} wall segments and {len(junctions)} raw junctions")
		wall_parameters = extract_wall_parameters(wall_segments, combined_wall_mask, junctions, scale_factor_mm_per_pixel)
		logger.info(f"Computed parameters for {len(wall_parameters)} walls")
		wall_connections_viz = find_wall_connections(wall_segments, junctions)
		junction_analysis = analyze_junction_types(junctions, wall_connections_viz)
		
		# Convert junction positions to millimeters
		for junction in junction_analysis:
			junction.update(convert_junction_position_to_mm(junction, scale_factor_mm_per_pixel))
		logger.info(f"Final junction list contains {len(junction_analysis)} junctions")
		
		# Identify exterior walls and calculate perimeter dimensions
		exterior_walls, interior_walls = identify_exterior_walls(wall_parameters, w, h, scale_factor_mm_per_pixel)
		perimeter_dimensions = calculate_perimeter_dimensions(exterior_walls)
		logger.info(f"Identified {len(exterior_walls)} exterior walls and {len(interior_walls)} interior walls")
		logger.debug(f"Time - wall segmentation & analysis: {time.time()-t0:.2f}s")
		
		# Fallback junctions from wall bounding boxes if none detected (or very few)
		if len(junction_analysis) < 4:  # heuristic: expect at least the four outer corners
			wall_bboxes = [r['rois'][idx] for idx in wall_indices]
			fallback_juncs = find_junctions_from_bboxes(wall_bboxes)
			for jx, jy in fallback_juncs:
				junction_data = {
					"junction_id": f"J{len(junction_analysis)+1}",
					"position": [float(jx), float(jy)],
					"connected_walls": [],
					"junction_type": "corner",
					"wall_count": 2
				}
				# Convert position to millimeters
				junction_data.update(convert_junction_position_to_mm(junction_data, scale_factor_mm_per_pixel))
				junction_analysis.append(junction_data)
		
		# Extract and analyze doors
		t0 = time.time()
		door_indices = [i for i, class_id in enumerate(r['class_ids']) if class_id == 3]
		detailed_doors = []
		
		if door_indices:
			# Extract door-specific data
			door_bboxes = [r['rois'][i] for i in door_indices]
			door_scores = [r['scores'][i] for i in door_indices]
			door_masks = r['masks'] if len(door_indices) > 0 else None
			
			# Perform detailed door analysis
			for i, (bbox, confidence) in enumerate(zip(door_bboxes, door_scores)):
				# Get the correct mask index for this door
				door_mask_index = door_indices[i] if i < len(door_indices) else None
				door_mask = door_masks[:, :, door_mask_index] if door_masks is not None and door_mask_index is not None else None
				
				# Basic door properties
				y1, x1, y2, x2 = bbox
				# Orientation analysis
				orientation = analyzeDoorOrientation(door_mask, bbox, w, h)
				# Determine width based on orientation
				if orientation.get("door_type") == "vertical":
					width_px = abs(x2 - x1)
				else:
					width_px = abs(y2 - y1)
				width_mm = pixels_to_mm(width_px, scale_factor_mm_per_pixel)

				door_width = float(x2 - x1)
				door_height = float(y2 - y1)
				door_area = door_width * door_height

				# Architectural analysis
				door_bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
				architectural_notes = generateArchitecturalNotes(orientation, door_bbox_dict)
				
				# Build comprehensive door data
				door_data = {
					"door_id": i + 1,
					"confidence": float(confidence),
					"location": {
						"center": {
							"x": float(pixels_to_mm((x1 + x2) / 2, scale_factor_mm_per_pixel)),
							"y": float(pixels_to_mm((y1 + y2) / 2, scale_factor_mm_per_pixel))
						},
						"relative_position": {
							"from_left": f"{(x1/w)*100:.1f}%",
							"from_top": f"{(y1/h)*100:.1f}%"
						}
					},
					"dimensions": {
						"width": width_mm,
						"height": float(pixels_to_mm(door_height, scale_factor_mm_per_pixel)),
						"area": float(pixels_to_mm(door_area, scale_factor_mm_per_pixel)),
						"aspect_ratio": door_width / door_height if door_height > 0 else 0
					},
					"orientation": orientation,
					"architectural_analysis": {
						"door_type": "interior" if door_width < door_height else "entrance",
						"size_category": categorize_door_size(door_width, door_height),
						"accessibility": assess_door_accessibility(door_width),
						"notes": architectural_notes
					}
				}
				detailed_doors.append(door_data)
			
			logger.info(f"Analyzed {len(detailed_doors)} doors")
		logger.debug(f"Time - door analysis: {time.time()-t0:.2f}s")
		
		# Extract and analyze windows
		t0 = time.time()
		window_indices = [i for i, class_id in enumerate(r['class_ids']) if class_id == 2]
		detailed_windows = []
		
		if window_indices:
			# Extract window-specific data
			window_bboxes = [r['rois'][i] for i in window_indices]
			window_scores = [r['scores'][i] for i in window_indices]
			window_masks = r['masks'] if len(window_indices) > 0 else None
			
			# Perform detailed window analysis
			for i, (bbox, confidence) in enumerate(zip(window_bboxes, window_scores)):
				# Get the correct mask index for this window
				window_mask_index = window_indices[i] if i < len(window_indices) else None
				window_mask = window_masks[:, :, window_mask_index] if window_masks is not None and window_mask_index is not None else None
				
				# Basic window properties
				y1, x1, y2, x2 = bbox
				window_width = float(x2 - x1)
				window_height = float(y2 - y1)
				window_area = window_width * window_height
				
				# Determine window orientation and width
				if window_width > window_height:
					window_type = "horizontal"
					width_px = window_width
				else:
					window_type = "vertical"
					width_px = window_height
				width_mm = pixels_to_mm(width_px, scale_factor_mm_per_pixel)
				
				# Build comprehensive window data
				window_data = {
					"window_id": i + 1,
					"confidence": float(confidence),
					"location": {
						"center": {
							"x": float(pixels_to_mm((x1 + x2) / 2, scale_factor_mm_per_pixel)),
							"y": float(pixels_to_mm((y1 + y2) / 2, scale_factor_mm_per_pixel))
						},
						"relative_position": {
							"from_left": f"{(x1/w)*100:.1f}%",
							"from_top": f"{(y1/h)*100:.1f}%"
						}
					},
					"dimensions": {
						"width": width_mm,
						"height": float(pixels_to_mm(window_height, scale_factor_mm_per_pixel)),
						"area": float(pixels_to_mm(window_area, scale_factor_mm_per_pixel)),
						"aspect_ratio": window_width / window_height if window_height > 0 else 0
					},
					"window_type": window_type,
					"architectural_analysis": {
						"size_category": categorize_window_size(window_width, window_height),
						"glazing_type": assess_window_glazing(window_width, window_height),
						"notes": generate_window_notes(window_width, window_height, window_type)
					}
				}
				detailed_windows.append(window_data)
			
			logger.info(f"Analyzed {len(detailed_windows)} windows")
		logger.debug(f"Time - window analysis: {time.time()-t0:.2f}s")
		
		# Perform OCR detection for space names
		t0 = time.time()
		logger.info("Starting OCR detection for space names...")
		space_names = detect_space_names(numpy.array(original_image))
		
		# Convert space name coordinates to millimeters
		for space in space_names:
			# Add millimeter coordinates for centerpoint
			space['center_mm'] = {
				'x': float(pixels_to_mm(space['center']['x'], scale_factor_mm_per_pixel)),
				'y': float(pixels_to_mm(space['center']['y'], scale_factor_mm_per_pixel))
			}
			
			# Add millimeter coordinates for bounding box
			space['bbox_mm'] = {
				'x1': float(pixels_to_mm(space['bbox'][0], scale_factor_mm_per_pixel)),
				'y1': float(pixels_to_mm(space['bbox'][1], scale_factor_mm_per_pixel)),
				'x2': float(pixels_to_mm(space['bbox'][2], scale_factor_mm_per_pixel)),
				'y2': float(pixels_to_mm(space['bbox'][3], scale_factor_mm_per_pixel))
			}
		
		logger.info(f"OCR detected {len(space_names)} space names in {time.time()-t0:.2f}s")
		
		# Create enhanced visualization
		t0 = time.time()
		vis_image = create_wall_visualization(original_image, r, wall_parameters, junction_analysis, w, h, scale_factor_mm_per_pixel, exterior_walls, space_names)
		logger.debug(f"Time - visualization drawing: {time.time()-t0:.2f}s")
		logger.info("Visualization image drawn; saving files …")
		
		# Get next test number for naming
		test_num = getNextTestNumber()
		
		# Save visualization and analysis
		wall_vis_filename = f"vis{test_num}.png"
		wall_vis_filepath = os.path.join(IMAGES_OUTPUT_DIR, wall_vis_filename)
		vis_image.save(wall_vis_filepath)
		
		# Save wall analysis data
		wall_analysis = {
			"metadata": {
				"timestamp": datetime.now().isoformat(),
				"image_dimensions": {"width": w, "height": h},
				"scale_factor_mm_per_pixel": scale_factor_mm_per_pixel,
				"analysis_type": "comprehensive_floor_plan_analysis",
				"units": "millimeters"
			},
			"summary": {
				"walls": {
					"total_walls": len(wall_parameters),
					"total_junctions": len(junction_analysis),
					"total_length_mm": sum(w["length"] for w in wall_parameters),
					"average_thickness_mm": sum(w["thickness"]["average"] for w in wall_parameters) / len(wall_parameters) if wall_parameters else 0
				},
				"exterior_walls": {
					"total_exterior_walls": len(exterior_walls),
					"total_interior_walls": len(interior_walls),
					"perimeter_length_mm": perimeter_dimensions["total_perimeter_length"],
					"perimeter_area_mm2": perimeter_dimensions["perimeter_area"],
					"average_exterior_wall_length_mm": perimeter_dimensions["average_exterior_wall_length"],
					"boundary_coverage": perimeter_dimensions["boundary_coverage"]
				},
				"doors": {
					"total_doors": len(detailed_doors),
					"average_confidence": float(numpy.mean([d["confidence"] for d in detailed_doors])) if detailed_doors else 0,
					"door_orientations": {
						"horizontal": sum(1 for d in detailed_doors if d["orientation"]["door_type"] == "horizontal"),
						"vertical": sum(1 for d in detailed_doors if d["orientation"]["door_type"] == "vertical")
					},
					"swing_directions": {}
				},
				"windows": {
					"total_windows": len(detailed_windows),
					"average_confidence": float(numpy.mean([d["confidence"] for d in detailed_windows])) if detailed_windows else 0,
					"window_types": {
						"horizontal": sum(1 for d in detailed_windows if d["window_type"] == "horizontal"),
						"vertical": sum(1 for d in detailed_windows if d["window_type"] == "vertical")
					},
					"glazing_types": {}
				},
				"space_names": {
					"total_spaces_detected": len(space_names),
					"average_confidence": float(numpy.mean([s["confidence"] for s in space_names])) if space_names else 0,
					"languages_detected": list(set([s.get("language", "unknown") for s in space_names])) if space_names else []
				}
			},
			"walls": {
				"individual_walls": wall_parameters,
				"junctions": junction_analysis,
				"exterior_walls": exterior_walls,
				"interior_walls": interior_walls,
				"perimeter_analysis": perimeter_dimensions
			},
			"doors": {
				"detailed_doors": detailed_doors
			},
			"windows": {
				"detailed_windows": detailed_windows
			},
			"space_names": {
				"total_spaces_detected": len(space_names),
				"spaces": space_names,
				"detection_summary": {
					"average_confidence": float(numpy.mean([s["confidence"] for s in space_names])) if space_names else 0,
					"confidence_range": {
						"min": float(min([s["confidence"] for s in space_names])) if space_names else 0,
						"max": float(max([s["confidence"] for s in space_names])) if space_names else 0
					},
					"detection_methods": list(set([s.get("method", "unknown") for s in space_names])) if space_names else [],
					"centerpoints_mm": [s["center_mm"] for s in space_names],
					"centerpoints_px": [s["center"] for s in space_names]
				}
			}
		}
		
		# Count door swing directions
		for door in detailed_doors:
			swing = door["orientation"]["estimated_swing"]
			wall_analysis["summary"]["doors"]["swing_directions"][swing] = wall_analysis["summary"]["doors"]["swing_directions"].get(swing, 0) + 1
		
		# Count window glazing types
		for window in detailed_windows:
			glazing = window["architectural_analysis"]["glazing_type"]
			wall_analysis["summary"]["windows"]["glazing_types"][glazing] = wall_analysis["summary"]["windows"]["glazing_types"].get(glazing, 0) + 1
		
		wall_json_filename = f"final{test_num}.json"
		save_wall_analysis(wall_analysis, wall_json_filename)
		
		# Check memory usage after processing
		memory_after = check_memory_usage()
		logger.debug(f"Memory after processing: {memory_after:.1f}MB")
		
		return jsonify({
			"message": "Comprehensive floor plan analysis completed successfully",
			"visualization_file": wall_vis_filename,
			"analysis_file": wall_json_filename,
			"image_processing": {
				"original_size": resize_info["original_size"],
				"processed_size": resize_info.get("new_size", resize_info["original_size"]),
				"resized": resize_info["resized"],
				"resize_factor": resize_info["resize_factor"],
				"resize_reason": resize_info["reason"],
				"scale_factor_adjusted": resize_info["resized"],
				"original_scale_factor": scale_factor_mm_per_pixel / resize_info["resize_factor"] if resize_info["resized"] else scale_factor_mm_per_pixel,
				"final_scale_factor": scale_factor_mm_per_pixel
			},
			"memory_usage": {
				"before_processing_mb": memory_before,
				"after_processing_mb": memory_after,
				"memory_increase_mb": memory_after - memory_before
			},
			"total_walls": len(wall_parameters),
			"total_doors": len(detailed_doors),
			"total_windows": len(detailed_windows),
			"total_junctions": len(junction_analysis),
			"total_space_names": len(space_names),
			"comprehensive_summary": {
				"wall_count": len(wall_parameters),
				"exterior_wall_count": len(exterior_walls),
				"interior_wall_count": len(interior_walls),
				"door_count": len(detailed_doors),
				"window_count": len(detailed_windows),
				"junction_count": len(junction_analysis),
				"space_name_count": len(space_names),
				"total_wall_length_mm": sum(w["length"] for w in wall_parameters),
				"total_wall_thickness_mm": sum(w["thickness"]["average"] for w in wall_parameters),
				"perimeter_length_mm": perimeter_dimensions["total_perimeter_length"],
				"perimeter_area_mm2": perimeter_dimensions["perimeter_area"]
			}
		})
		
	except Exception as e:
		logger.error(f"Error in wall visualization: {str(e)}")
		logger.error(traceback.format_exc())
		return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
	api_config = app_config.get_api_config()
	logger.info('===========Starting FloorPlanTo3D API==========')
	logger.info(f"Running on {api_config['HOST']}:{api_config['PORT']}")
	logger.info(f"Debug mode: {api_config['DEBUG']}")
	application.run(
		host=api_config['HOST'], 
		port=api_config['PORT'],
		debug=api_config['DEBUG']
	)
	logger.info('===========API stopped==========')