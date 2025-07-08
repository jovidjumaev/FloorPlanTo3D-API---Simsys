# Floor Plan to 3D API
# 
# This API provides comprehensive floor plan analysis including:
# - Wall detection and parameter extraction
# - Door and window analysis
# - Junction detection and classification
# - Room detection and analysis
#
# NEW FEATURE: Millimeter Conversion
# To convert pixel measurements to millimeters, include a 'scale_factor_mm_per_pixel' 
# parameter in your POST request. This factor represents how many millimeters 
# each pixel represents in the real world.
#
# Example usage:
# - If 1 pixel = 0.5 mm, use scale_factor_mm_per_pixel=0.5
# - If 1 pixel = 2 mm, use scale_factor_mm_per_pixel=2.0
# - If no scale factor is provided, it defaults to 1.0 (1 pixel = 1 mm)
#
# The API will now return both pixel and millimeter measurements in the JSON output:
# - centerline_px: centerline coordinates in pixels
# - centerline_mm: centerline coordinates in millimeters
# - length_px: wall length in pixels
# - length_mm: wall length in millimeters
# - thickness_px: wall thickness in pixels
# - thickness_mm: wall thickness in millimeters
# - bbox_px: bounding box in pixels
# - bbox_mm: bounding box in millimeters

import os
import numpy
import skimage.color
import cv2
import traceback 

from numpy.lib.function_base import average
from numpy import zeros
from numpy import asarray

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image

from skimage.draw import polygon2mask
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects, label, binary_erosion, disk
from skimage.measure import find_contours, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform

from datetime import datetime
from io import BytesIO
from mrcnn.utils import extract_bboxes
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import json
import os
from datetime import datetime
from flask import Flask, flash, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage.draw import line as skimage_line
import time

# Global variables
global _model
global cfg
ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"

from flask_cors import CORS, cross_origin

sys.path.append(ROOT_DIR)

MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "floorPlan_cfg"
	# number of classes (background + door + wall + window)
	NUM_CLASSES = 1 + 3
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# Lower detection threshold to capture faint walls/doors
	DETECTION_MIN_CONFIDENCE = 0.15
	# Allow larger input images to preserve line detail
	IMAGE_MAX_DIM = 1600


def safe_logical_and(a, b):
    """Safe wrapper for numpy.logical_and with proper boolean conversion"""
    return numpy.logical_and(numpy.asarray(a, dtype=bool), numpy.asarray(b, dtype=bool))

def safe_logical_or(a, b):
    """Safe wrapper for numpy.logical_or with proper boolean conversion"""
    return numpy.logical_or(numpy.asarray(a, dtype=bool), numpy.asarray(b, dtype=bool))

def safe_logical_not(a):
    """Safe wrapper for numpy.logical_not with proper boolean conversion"""
    return numpy.logical_not(numpy.asarray(a, dtype=bool))
	

# Load model on startup (Flask 2.x compatible way)
@application.before_request
def load_model():
	global cfg, _model
	
	# Only load once
	if '_model' not in globals() or _model is None:
		model_folder_path = os.path.abspath("./") + "/mrcnn"
		weights_path = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
		cfg = PredictionConfig()
		print(cfg.IMAGE_RESIZE_MODE)
		print('==============before loading model=========')
		_model = MaskRCNN(mode='inference', model_dir=model_folder_path, config=cfg)
		print('=================after loading model==============')
		_model.load_weights(weights_path, by_name=True)
		print('=================model loaded successfully==============')


def myImageLoader(imageInput, enhance_for_office=False):
	# Convert PIL image to RGB first to ensure consistent format
	if hasattr(imageInput, 'convert'):
		imageInput = imageInput.convert('RGB')
	image = numpy.asarray(imageInput)
	
	# Office plan enhancement: binarize and thicken lines
	if enhance_for_office:
		# Convert to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		
		# Apply adaptive thresholding to better handle varying line intensities
		binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
		
		# Remove noise
		kernel = numpy.ones((2,2), numpy.uint8)
		binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
		
		# Dilate to thicken lines - using a larger kernel for better connectivity
		kernel_dilate = numpy.ones((8,8), numpy.uint8)
		dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
		
		# Connect nearby lines that might be part of the same wall
		kernel_close = numpy.ones((7,7), numpy.uint8)
		dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
		
		# Convert back to black lines on white background
		dilated = cv2.bitwise_not(dilated)
		
		# Enhance contrast
		dilated = cv2.normalize(dilated, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		
		# Convert back to 3-channel RGB
		image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
	
	# Ensure the image is in the right data type and range
	if image.dtype != numpy.uint8:
		if image.max() <= 1.0:
			image = (image * 255).astype(numpy.uint8)
		else:
			image = image.astype(numpy.uint8)
	
	h, w, c = image.shape
	print(f"Processed image shape: {h}x{w}x{c}{' (office enhancement)' if enhance_for_office else ''}")
	return image, w, h


def getClassNames(classIds):
	result = list()
	for classid in classIds:
		data = {}
		if classid == 1:
			data['name'] = 'wall'
		if classid == 2:
			data['name'] = 'window'
		if classid == 3:
			data['name'] = 'door'
		result.append(data)	
	return result				


def normalizePoints(bbx, classNames):
	normalizingX = 1
	normalizingY = 1
	result = list()
	doorCount = 0
	index = -1
	doorDifference = 0
	for bb in bbx:
		index = index + 1
		if(classNames[index] == 3):
			doorCount = doorCount + 1
			if(abs(bb[3]-bb[1]) > abs(bb[2]-bb[0])):
				doorDifference = doorDifference + abs(bb[3]-bb[1])
			else:
				doorDifference = doorDifference + abs(bb[2]-bb[0])

		result.append([bb[0]*normalizingY, bb[1]*normalizingX, bb[2]*normalizingY, bb[3]*normalizingX])
	
	# Avoid division by zero
	if doorCount > 0:
		return result, (doorDifference/doorCount)
	else:
		return result, 0
		

def turnSubArraysToJson(objectsArr):
	result = list()
	for obj in objectsArr:
		data = {}
		data['x1'] = obj[1]
		data['y1'] = obj[0]
		data['x2'] = obj[3]
		data['y2'] = obj[2]
		result.append(data)
	return result


def calculateObjectArea(mask):
	"""Calculate the area of an object from its segmentation mask"""
	return numpy.sum(mask)

def calculateObjectCenter(bbox):
	"""Calculate the center point of a bounding box"""
	y1, x1, y2, x2 = bbox
	center_x = (x1 + x2) / 2
	center_y = (y1 + y2) / 2
	return {"x": float(center_x), "y": float(center_y)}

def encodeMaskSummary(mask):
	"""Create a summary of the segmentation mask instead of full encoding"""
	# Count non-zero pixels and get bounding box of the mask
	non_zero_pixels = numpy.sum(mask > 0)
	total_pixels = mask.shape[0] * mask.shape[1]
	coverage_percentage = (non_zero_pixels / total_pixels) * 100
	
	# Find bounding box of non-zero region
	rows = numpy.any(mask, axis=1)
	cols = numpy.any(mask, axis=0)
	
	if numpy.any(rows) and numpy.any(cols):
		rmin, rmax = numpy.where(rows)[0][[0, -1]]
		cmin, cmax = numpy.where(cols)[0][[0, -1]]
		mask_bbox = {"x1": int(cmin), "y1": int(rmin), "x2": int(cmax), "y2": int(rmax)}
	else:
		mask_bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
	
	return {
		"coverage_percentage": float(coverage_percentage),
		"non_zero_pixels": int(non_zero_pixels),
		"total_pixels": int(total_pixels),
		"mask_bbox": mask_bbox
	}

def getClassName(classId):
	"""Get class name from class ID"""
	class_map = {1: 'wall', 2: 'window', 3: 'door'}
	return class_map.get(classId, 'unknown')

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

def getNextTestNumber():
	"""Get the next available test number"""
	existing_files = []
	for f in os.listdir(ROOT_DIR):
		if f.startswith(('plan', 'vis', 'acc')) and any(f.endswith(ext) for ext in ['.json', '.png']):
			existing_files.append(f)
	
	# Find highest test number
	max_num = 0
	for f in existing_files:
		try:
			# Extract number from names like plan1.json, vis1.png, acc1.json
			if 'plan' in f or 'vis' in f or 'acc' in f:
				import re
				match = re.search(r'(\d+)', f)
				if match:
					num = int(match.group(1))
					max_num = max(max_num, num)
		except:
			pass
	
	return max_num + 1

def saveJsonToFile(json_data, custom_name=None):
	"""Save JSON data to file in root directory with simple naming"""
	if custom_name:
		filename = f"{custom_name}.json"
	else:
		test_num = getNextTestNumber()
		filename = f"plan{test_num}.json"
	
	filepath = os.path.join(ROOT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(json_data, f, indent=2)
		print(f"JSON saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving JSON file: {str(e)}")
		return None


@application.route('/', methods=['POST'])
def prediction():
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		image, w, h = myImageLoader(imagefile)
		print(f"Image dimensions: {h}x{w}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# TensorFlow 2.x uses eager execution by default, no need for graph context
		r = _model.detect(sample, verbose=0)[0]
		
		# Enhanced JSON structure with comprehensive semantic data
		enhanced_data = buildEnhancedJson(r, w, h, image)
		
		# Save JSON to file in root directory with simple naming
		saveJsonToFile(enhanced_data)
		
		return jsonify(enhanced_data)
		
	except Exception as e:
		print(f"Error during prediction: {str(e)}")
		return jsonify({"error": str(e)}), 500


@application.route('/visualize', methods=['POST'])
def visualize_results():
	"""Create a visualization image showing detected objects overlaid on the original floor plan"""
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		original_image = imagefile.copy()
		
		# Check if this is an office plan by looking for thin lines
		gray = cv2.cvtColor(numpy.array(original_image), cv2.COLOR_RGB2GRAY)
		edges = cv2.Canny(gray, 50, 150)
		line_thickness = numpy.sum(edges > 0) / numpy.sum(edges > 0, axis=1).max()
		is_office_plan = line_thickness < 5  # Threshold for thin lines
		
		image, w, h = myImageLoader(imagefile, enhance_for_office=is_office_plan)
		print(f"Creating visualization for image: {h}x{w} {'(office plan)' if is_office_plan else ''}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
		
		# Create visualization
		vis_image = createVisualization(original_image, r, w, h)
		
		# Get next test number for simple naming
		test_num = getNextTestNumber()
		
		# Save visualization image with simple name
		vis_filename = f"vis{test_num}.png"
		vis_filepath = os.path.join(ROOT_DIR, vis_filename)
		vis_image.save(vis_filepath)
		
		# Create the JSON analysis with matching name
		enhanced_data = buildEnhancedJson(r, w, h, image)
		json_filename = f"plan{test_num}.json"
		saveJsonToFile(enhanced_data, f"plan{test_num}")
		
		return jsonify({
			"message": "Visualization created successfully",
			"visualization_file": vis_filename,
			"json_file": json_filename,
			"total_objects_detected": len(r['rois']),
			"object_summary": {
				"walls": sum(1 for cid in r['class_ids'] if cid == 1),
				"windows": sum(1 for cid in r['class_ids'] if cid == 2), 
				"doors": sum(1 for cid in r['class_ids'] if cid == 3)
			},
			"confidence_range": {
				"min": float(min(r['scores'])) if len(r['scores']) > 0 else 0,
				"max": float(max(r['scores'])) if len(r['scores']) > 0 else 0,
				"average": float(numpy.mean(r['scores'])) if len(r['scores']) > 0 else 0
			}
		})
		
	except Exception as e:
		print(f"Error creating visualization: {str(e)}")
		return jsonify({"error": str(e)}), 500

def createVisualization(original_image, model_results, image_width, image_height):
	"""Create a visualization image with detected objects overlaid"""
	
	# Convert to RGB if needed
	if original_image.mode != 'RGB':
		vis_image = original_image.convert('RGB')
	else:
		vis_image = original_image.copy()
	
	# Create drawing context
	draw = ImageDraw.Draw(vis_image)
	
	# Define colors for different object types
	colors = {
		1: (255, 0, 0),     # Wall - Red
		2: (0, 255, 0),     # Window - Green  
		3: (0, 0, 255)      # Door - Blue
	}
	
	class_names = {1:'Wall', 2:'Window', 3:'Door'}
	
	# Draw each detected object
	bboxes = model_results['rois']
	class_ids = model_results['class_ids']
	scores = model_results['scores']
	masks = model_results.get('masks', None)
	
	for i in range(len(bboxes)):
		bbox = bboxes[i]
		class_id = class_ids[i]
		confidence = scores[i]
		
		# Get bounding box coordinates (y1, x1, y2, x2 format from model)
		y1, x1, y2, x2 = bbox
		
		# Get color for this object type
		color = colors.get(class_id, (128, 128, 128))  # Gray for unknown
		
		# Draw bounding box
		draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
		
		# Enhanced door visualization with orientation and centerpoint
		if class_id == 3 and masks is not None:  # door
			door_mask = masks[:, :, i] if i < masks.shape[2] else None
			orientation = analyzeDoorOrientation(door_mask, bbox, image_width, image_height)
			
			# Calculate door center
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
			
			# Draw door centerpoint (cyan circle)
			center_radius = 8
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
			font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
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
	
	return vis_image


@application.route('/analyze_accuracy', methods=['POST'])
def analyze_accuracy():
	"""Analyze the accuracy and reliability of the model predictions"""
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		image, w, h = myImageLoader(imagefile)
		print(f"Analyzing accuracy for image: {h}x{w}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
	
		# Perform accuracy analysis
		accuracy_report = performAccuracyAnalysis(r, w, h)
		
		# Save accuracy analysis with simple naming
		test_num = getNextTestNumber()
		filename = saveAccuracyAnalysis(accuracy_report, test_num)
		
		# Add filename to response
		response = accuracy_report.copy()
		response["analysis_file"] = filename
		
		return jsonify(response)
		
	except Exception as e:
		print(f"Error in accuracy analysis: {str(e)}")
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

def saveAccuracyAnalysis(accuracy_data, test_num):
	"""Save accuracy analysis to file with simple naming"""
	filename = f"acc{test_num}.json"
	filepath = os.path.join(ROOT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(accuracy_data, f, indent=2)
		print(f"Accuracy analysis saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving accuracy analysis: {str(e)}")
		return None

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

@application.route('/analyze_doors', methods=['POST'])
def analyze_door_orientation():
	"""Dedicated endpoint for detailed door orientation and architectural analysis"""
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		# Get scale factor from request (default to 1.0 if not provided)
		scale_factor_mm_per_pixel = float(request.form.get('scale_factor_mm_per_pixel', 1.0))
		image, w, h = myImageLoader(imagefile)
		print(f"Analyzing door orientations for image: {h}x{w}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
		
		# Filter for doors only
		door_indices = [i for i, class_id in enumerate(r['class_ids']) if class_id == 3]
		
		if not door_indices:
			return jsonify({
				"message": "No doors detected in the floor plan",
				"total_doors": 0,
				"doors": []
			})
		
		# Extract door-specific data
		door_bboxes = [r['rois'][i] for i in door_indices]
		door_scores = [r['scores'][i] for i in door_indices]
		door_masks = r['masks'] if len(door_indices) > 0 else None
		
		# Perform detailed door analysis
		detailed_doors = []
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
		
		# Overall door analysis summary
		door_summary = {
			"total_doors_detected": len(detailed_doors),
			"average_confidence": float(numpy.mean(door_scores)),
			"door_orientations": {
				"horizontal": sum(1 for d in detailed_doors if d["orientation"]["door_type"] == "horizontal"),
				"vertical": sum(1 for d in detailed_doors if d["orientation"]["door_type"] == "vertical")
			},
			"swing_directions": {},
			"high_confidence_doors": sum(1 for d in detailed_doors if d["confidence"] > 0.8),
			"layout_insights": generate_door_layout_insights(detailed_doors, w, h)
		}
		
		# Count swing directions
		for door in detailed_doors:
			swing = door["orientation"]["estimated_swing"]
			door_summary["swing_directions"][swing] = door_summary["swing_directions"].get(swing, 0) + 1
		
		# Save detailed door analysis
		test_num = getNextTestNumber()
		door_filename = f"doors{test_num}.json"
		door_analysis = {
			"metadata": {
				"timestamp": datetime.now().isoformat(),
				"image_dimensions": {"width": w, "height": h},
				"scale_factor_mm_per_pixel": scale_factor_mm_per_pixel,
				"analysis_type": "door_orientation_analysis",
				"units": "millimeters"
			},
			"summary": door_summary,
			"detailed_doors": detailed_doors
		}
		
		save_door_analysis(door_analysis, door_filename)
		
		return jsonify({
			"message": "Door orientation analysis completed successfully",
			"analysis_file": door_filename,
			**door_analysis
		})
		
	except Exception as e:
		print(f"Error in door orientation analysis: {str(e)}")
		return jsonify({"error": str(e)}), 500

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

def save_door_analysis(door_data, filename):
	"""Save door analysis to file in root directory"""
	filepath = os.path.join(ROOT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(door_data, f, indent=2)
		print(f"Door analysis saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving door analysis: {str(e)}")
		return None

def extract_wall_masks(model_results):
	"""Extract all wall masks from model results"""
	wall_masks = []
	wall_indices = []
	
	for i, class_id in enumerate(model_results['class_ids']):
		if class_id == 1:  # wall class
			wall_indices.append(i)
			wall_masks.append(model_results['masks'][:, :, i])
	
	return wall_masks, wall_indices

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
                from skimage.measure import label
                labeled_neighbors = label(neighbors, connectivity=2)
                num_components = numpy.max(labeled_neighbors)
                
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
    from scipy.ndimage import binary_erosion, binary_dilation
    import numpy as np
    
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
                score = numpy.sum(region)
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

def keep_largest_component(binary_img):
    """Keep only the largest connected component in a binary image."""
    labeled = label(binary_img)
    if labeled.max() == 0:
        return binary_img  # nothing to keep
    largest_cc = numpy.equal(labeled, numpy.argmax(numpy.bincount(labeled.flat)[1:]) + 1).astype(bool)
    return largest_cc

def segment_individual_walls(wall_mask):
    """Segment connected wall regions into individual wall segments with robust per-region processing."""
    from skimage.morphology import remove_small_objects, skeletonize
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
            
        skeleton = skeletonize(region_mask)
        skeleton = safe_logical_and(skeleton.astype(bool), region_mask.astype(bool))  # FIXED: explicit bool casting
        
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

def extract_centerline_coords(segment_coords):
	"""Legacy extractor that now leverages connectivity ordering."""
	if len(segment_coords) < 2:
		return segment_coords.tolist()
	ordered = order_centerline_points_connectivity(segment_coords)
	return ordered if ordered else segment_coords.tolist()

def order_centerline_points_connectivity(segment_coords):
    """Order centerline points following skeleton connectivity (8-neighbour) to avoid diagonal shortcuts."""
    # Convert to set of (x, y) tuples for fast lookup
    coords_set = set((int(c[1]), int(c[0])) for c in segment_coords)
    if not coords_set:
        return []
    # 8-neighbour offsets
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    # Compute degree of each pixel
    degrees = {p: sum(((p[0]+dx, p[1]+dy) in coords_set) for dx, dy in nbrs) for p in coords_set}
    # Endpoints have degree 1. If none, pick arbitrary start (loop)
    endpoints = [p for p, d in degrees.items() if d == 1]
    start = endpoints[0] if endpoints else next(iter(coords_set))
    ordered = [start]
    visited = {start}
    current = start
    while True:
        next_pixel = None
        for dx, dy in nbrs:
            cand = (current[0]+dx, current[1]+dy)
            if cand in coords_set and cand not in visited:
                next_pixel = cand
                break
        if next_pixel is None:
            break
        ordered.append(next_pixel)
        visited.add(next_pixel)
        current = next_pixel
    # Return as [x, y] lists
    return [[p[0], p[1]] for p in ordered]

def extract_centerline_coords_with_validation(segment_coords, wall_mask, min_length=2):
    """Extract ordered centerline coordinates with connectivity ordering and robust in-mask validation, splitting into subpaths."""
    if len(segment_coords) < 2:
        return []
    ordered_centerline = order_centerline_points_connectivity(segment_coords)
    if len(ordered_centerline) < min_length:
        return []
    valid_paths = []
    current_path = []
    for i in range(1, len(ordered_centerline)):
        p1 = ordered_centerline[i-1]
        p2 = ordered_centerline[i]
        rr, cc = skimage_line(p1[1], p1[0], p2[1], p2[0])
        # FIXED: simplified boolean array comparison
        points_valid = [(0 <= y < wall_mask.shape[0] and 0 <= x < wall_mask.shape[1] and wall_mask[y, x]) for y, x in zip(rr, cc)]
        if numpy.all(points_valid):
            if not current_path:
                current_path.append(p1)
            current_path.append(p2)
        else:
            if len(current_path) >= min_length:
                valid_paths.append(current_path)
            current_path = []
    if len(current_path) >= min_length:
        valid_paths.append(current_path)
    if not valid_paths:
        return []
    # Return the longest valid path
    return max(valid_paths, key=len)

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

@application.route('/analyze_walls', methods=['POST'])
def analyze_wall_parameters():
	"""Comprehensive wall parameter analysis including centerlines, thickness, and junctions"""
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		# Get scale factor from request (default to 1.0 if not provided)
		scale_factor_mm_per_pixel = float(request.form.get('scale_factor_mm_per_pixel', 1.0))
		# Detect if this is an office plan (thin lines)
		img_rgb_tmp = imagefile.copy().convert('RGB')
		gray_tmp = cv2.cvtColor(numpy.array(img_rgb_tmp), cv2.COLOR_RGB2GRAY)
		edges_tmp = cv2.Canny(gray_tmp, 50, 150)
		avg_col = numpy.mean(numpy.sum(edges_tmp > 0, axis=0))
		avg_row = numpy.mean(numpy.sum(edges_tmp > 0, axis=1))
		is_office_plan = max(avg_col, avg_row) < 7  # heuristic for thin lines
		image, w, h = myImageLoader(imagefile, enhance_for_office=is_office_plan)
		print(f"Analyzing wall parameters for image: {h}x{w} {'(office plan)' if is_office_plan else ''}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
		
		# Extract wall masks and door masks separately
		wall_masks, wall_indices = extract_wall_masks(r)
		
		door_masks = []
		if 'masks' in r:
			for idx, cid in enumerate(r['class_ids']):
				if cid == 3 and idx < r['masks'].shape[2]:
					door_masks.append(r['masks'][:, :, idx])
		
		# Build a combined door mask
		combined_door_mask = numpy.zeros((h, w), dtype=bool)
		for dm in door_masks:
			# Dilate door mask to cover regions near door edges, preventing centerline crossing
			dilated_dm = cv2.dilate(dm.astype(numpy.uint8), numpy.ones((15,15), numpy.uint8), iterations=1).astype(bool)
			# enlarge buffer further to ensure no wall pixels remain near doors
			dilated_dm = cv2.dilate(dilated_dm.astype(numpy.uint8), numpy.ones((35,35), numpy.uint8), iterations=1).astype(bool)
			combined_door_mask = safe_logical_or(combined_door_mask.astype(bool), dilated_dm.astype(bool))
		
		if not wall_masks:
			return jsonify({
				"message": "No walls detected in the floor plan",
				"total_walls": 0,
				"individual_walls": [],
				"junctions": []
			})
		
		print(f"Found {len(wall_masks)} wall regions")
		
		# Process each wall mask separately to prevent cross-wall centerlines
		all_wall_segments = []
		all_junctions = []
		combined_wall_mask = numpy.zeros((h, w), dtype=bool)
		
		for i, wall_mask in enumerate(wall_masks):
			# Remove door areas from wall mask to avoid centerlines through openings
			wall_mask = safe_logical_and(wall_mask, safe_logical_not(combined_door_mask))
			combined_wall_mask = safe_logical_or(combined_wall_mask.astype(bool), wall_mask.astype(bool))
			
			# Segment individual walls from this mask
			wall_segments, junctions = segment_individual_walls(wall_mask)
			
			# Add wall ID prefix to distinguish between different wall regions
			for segment in wall_segments:
				all_wall_segments.append((segment, f"R{i+1}_"))
			
			# Adjust junction coordinates and add to global list
			for junction in junctions:
				all_junctions.append(junction)
		
		# Ensure combined_wall_mask does not include doors
		combined_wall_mask = safe_logical_and(combined_wall_mask.astype(bool), numpy.logical_not(combined_door_mask.astype(bool)))  # Fixed boolean operation

		print(f"Segmented into {len(all_wall_segments)} individual walls with {len(all_junctions)} junctions")
		
		# Extract detailed wall parameters with region prefixes
		wall_parameters = extract_wall_parameters_with_regions(all_wall_segments, combined_wall_mask, all_junctions, scale_factor_mm_per_pixel)
		
		# Analyze junctions
		wall_connections = find_wall_connections([seg for seg, _ in all_wall_segments], all_junctions)
		junction_analysis = analyze_junction_types(all_junctions, wall_connections)
		
		# Convert junction positions to millimeters
		for junction in junction_analysis:
			junction.update(convert_junction_position_to_mm(junction, scale_factor_mm_per_pixel))
		
		# Identify exterior walls and calculate perimeter dimensions
		exterior_walls, interior_walls = identify_exterior_walls(wall_parameters, w, h, scale_factor_mm_per_pixel)
		perimeter_dimensions = calculate_perimeter_dimensions(exterior_walls)
		
		# Fallback junctions from wall bounding boxes if none detected
		if len(junction_analysis) == 0:
			wall_bboxes = [r['rois'][idx] for idx in wall_indices]
			fallback_juncs = find_junctions_from_bboxes(wall_bboxes)
			for jx, jy in fallback_juncs:
				junction_analysis.append({
					"junction_id": f"J{len(junction_analysis)+1}",
					"position": [float(jx), float(jy)],
					"connected_walls": [],
					"junction_type": "corner",
					"wall_count": 2
				})
		
		# Calculate summary statistics
		total_wall_length_px = sum(wall["length_px"] for wall in wall_parameters)
		total_wall_length_mm = sum(wall["length_mm"] for wall in wall_parameters)
		avg_wall_thickness_px = numpy.mean([wall["thickness"]["average"] for wall in wall_parameters if wall["thickness"]["average"] > 0])
		avg_wall_thickness_mm = numpy.mean([wall["thickness"]["average"] for wall in wall_parameters if wall["thickness"]["average"] > 0])
		
		# Wall orientation distribution
		orientation_categories = {
			"horizontal": 0,     # 0-30 or 150-180 degrees
			"vertical": 0,       # 60-120 degrees  
			"diagonal": 0        # 30-60 or 120-150 degrees
		}
		
		for wall in wall_parameters:
			angle = wall["orientation_degrees"]
			if angle <= 30 or angle >= 150:
				orientation_categories["horizontal"] += 1
			elif 60 <= angle <= 120:
				orientation_categories["vertical"] += 1
			else:
				orientation_categories["diagonal"] += 1
		
		# Build comprehensive analysis
		wall_analysis = {
			"metadata": {
				"timestamp": datetime.now().isoformat(),
				"image_dimensions": {"width": w, "height": h},
				"scale_factor_mm_per_pixel": scale_factor_mm_per_pixel,
				"analysis_type": "wall_parameter_analysis",
				"units": "millimeters"
			},
			"summary": {
				"total_walls_detected": len(wall_parameters),
				"total_junctions": len(junctions),
				"total_wall_length_mm": float(total_wall_length_mm),
				"average_wall_thickness_mm": float(avg_wall_thickness_mm) if not numpy.isnan(avg_wall_thickness_mm) else 0.0,
				"wall_orientation_distribution": orientation_categories,
				"junction_types": {
					"corners": len([j for j in junction_analysis if j["junction_type"] == "corner"]),
					"T_junctions": len([j for j in junction_analysis if j["junction_type"] == "T_junction"]),
					"cross_junctions": len([j for j in junction_analysis if j["junction_type"] == "cross_junction"]),
					"complex_junctions": len([j for j in junction_analysis if j["junction_type"] == "complex_junction"])
				}
			},
			"individual_walls": wall_parameters,
			"junctions": junction_analysis,
			"architectural_insights": generate_wall_insights(wall_parameters, junction_analysis)
		}
		
		# Save wall analysis
		test_num = getNextTestNumber()
		wall_filename = f"walls{test_num}.json"
		save_wall_analysis(wall_analysis, wall_filename)
		
		return jsonify({
			"message": "Wall parameter analysis completed successfully",
			"analysis_file": wall_filename,
			**wall_analysis
		})
		
	except Exception as e:
		print(f"Error in wall parameter analysis: {str(e)}")
		return jsonify({"error": str(e)}), 500

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

def save_wall_analysis(wall_data, filename):
	"""Save wall analysis to file in root directory"""
	filepath = os.path.join(ROOT_DIR, filename)
	
	try:
		with open(filepath, 'w') as f:
			json.dump(wall_data, f, indent=2)
		print(f"Wall analysis saved to: {filepath}")
		return filename
	except Exception as e:
		print(f"Error saving wall analysis: {str(e)}")
		return None

@application.route('/visualize_walls', methods=['POST'])
def visualize_wall_analysis():
	"""Create enhanced visualization showing wall centerlines, junctions, and wall parameters"""
	global cfg, _model
	
	try:
		imagefile = Image.open(request.files['image'].stream)
		# Get scale factor from request (default to 1.0 if not provided)
		scale_factor_mm_per_pixel = float(request.form.get('scale_factor_mm_per_pixel', 1.0))
		original_image = imagefile.copy()
		# Office plan detection
		img_rgb_tmp2 = original_image.convert('RGB')
		gray_tmp = cv2.cvtColor(numpy.array(img_rgb_tmp2), cv2.COLOR_RGB2GRAY)
		edges_tmp = cv2.Canny(gray_tmp, 50, 150)
		avg_col = numpy.mean(numpy.sum(edges_tmp > 0, axis=0))
		avg_row = numpy.mean(numpy.sum(edges_tmp > 0, axis=1))
		is_office_plan = max(avg_col, avg_row) < 7
		image, w, h = myImageLoader(imagefile, enhance_for_office=is_office_plan)
		print(f"Creating wall analysis visualization for image: {h}x{w} {'(office plan)' if is_office_plan else ''}")
		
		# --- timing start ---
		t0 = time.time()
		# Preprocess for model
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)
		print(f"Time - preprocessing: {time.time()-t0:.2f}s")
		
		# Model detection
		t0 = time.time()
		r = _model.detect(sample, verbose=0)[0]
		
		# Extract wall masks and perform analysis
		t0 = time.time()
		wall_masks, wall_indices = extract_wall_masks(r)
		print(f"Extracted {len(wall_masks)} wall masks from model output")
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

		
		# Remove door areas from combined wall mask
		combined_wall_mask = safe_logical_and(combined_wall_mask.astype(bool), numpy.logical_not(combined_door_mask.astype(bool)))
		print("Combined wall mask ready; starting skeletonisation & segment extraction …")
		wall_segments, junctions = segment_individual_walls(combined_wall_mask)
		print(f"Found {len(wall_segments)} wall segments and {len(junctions)} raw junctions")
		wall_parameters = extract_wall_parameters(wall_segments, combined_wall_mask, junctions, scale_factor_mm_per_pixel)
		print(f"Computed parameters for {len(wall_parameters)} walls")
		wall_connections_viz = find_wall_connections(wall_segments, junctions)
		junction_analysis = analyze_junction_types(junctions, wall_connections_viz)
		
		# Convert junction positions to millimeters
		for junction in junction_analysis:
			junction.update(convert_junction_position_to_mm(junction, scale_factor_mm_per_pixel))
		print(f"Final junction list contains {len(junction_analysis)} junctions")
		
		# Identify exterior walls and calculate perimeter dimensions
		exterior_walls, interior_walls = identify_exterior_walls(wall_parameters, w, h, scale_factor_mm_per_pixel)
		perimeter_dimensions = calculate_perimeter_dimensions(exterior_walls)
		print(f"Identified {len(exterior_walls)} exterior walls and {len(interior_walls)} interior walls")
		print(f"Time - wall segmentation & analysis: {time.time()-t0:.2f}s")
		
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
			
			print(f"Analyzed {len(detailed_doors)} doors")
		print(f"Time - door analysis: {time.time()-t0:.2f}s")
		
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
			
			print(f"Analyzed {len(detailed_windows)} windows")
		print(f"Time - window analysis: {time.time()-t0:.2f}s")
		
		# Create enhanced visualization
		t0 = time.time()
		vis_image = create_wall_visualization(original_image, r, wall_parameters, junction_analysis, w, h, scale_factor_mm_per_pixel)
		print(f"Time - visualization drawing: {time.time()-t0:.2f}s")
		print("Visualization image drawn; saving files …")
		
		# Get next test number for naming
		test_num = getNextTestNumber()
		
		# Save visualization and analysis
		wall_vis_filename = f"wall_vis{test_num}.png"
		wall_vis_filepath = os.path.join(ROOT_DIR, wall_vis_filename)
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
		
		wall_json_filename = f"walls{test_num}.json"
		save_wall_analysis(wall_analysis, wall_json_filename)
		
		return jsonify({
			"message": "Comprehensive floor plan analysis completed successfully",
			"visualization_file": wall_vis_filename,
			"analysis_file": wall_json_filename,
			"total_walls": len(wall_parameters),
			"total_doors": len(detailed_doors),
			"total_windows": len(detailed_windows),
			"total_junctions": len(junction_analysis),
			"comprehensive_summary": {
				"wall_count": len(wall_parameters),
				"exterior_wall_count": len(exterior_walls),
				"interior_wall_count": len(interior_walls),
				"door_count": len(detailed_doors),
				"window_count": len(detailed_windows),
				"junction_count": len(junction_analysis),
				"total_wall_length_mm": sum(w["length"] for w in wall_parameters),
				"total_wall_thickness_mm": sum(w["thickness"]["average"] for w in wall_parameters),
				"perimeter_length_mm": perimeter_dimensions["total_perimeter_length"],
				"perimeter_area_mm2": perimeter_dimensions["perimeter_area"]
			}
		})
		
	except Exception as e:
		traceback.print_exc() 
		return jsonify({"error": str(e)}), 500

def create_wall_visualization(original_image, model_results, wall_parameters, junction_analysis, image_width, image_height, scale_factor_mm_per_pixel=1.0):
	"""Create enhanced visualization with wall centerlines, junctions, and parameters"""
	
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
			center_radius = 8
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
	
	# Draw wall centerlines
	# Pre-compute door bounding boxes once (x1,y1,x2,y2)
	door_boxes_xy = []
	for idx, cid in enumerate(class_ids):
		if cid == 3:
			dy1, dx1, dy2, dx2 = bboxes[idx]
			# Ensure all coordinates are scalars
			dy1, dx1, dy2, dx2 = float(dy1), float(dx1), float(dy2), float(dx2)
			door_boxes_xy.append((dx1, dy1, dx2, dy2))

	# Create a set of exterior wall IDs for quick lookup
	exterior_wall_ids = {wall["wall_id"] for wall in wall_parameters if "exterior_reasons" in wall}

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
		if len(centerline) > 1:
			# Determine if this is an exterior wall
			is_exterior = wall["wall_id"] in exterior_wall_ids
			
			# Choose color and width based on wall type
			if is_exterior:
				wall_color = (255, 140, 0)  # Dark orange for exterior walls
				wall_width = 6  # Thicker lines for exterior walls
			else:
				wall_color = centerline_color  # Yellow for interior walls
				wall_width = 4  # Normal width for interior walls
			
			# Draw centerline
			for i in range(1, len(centerline)):
				# Ensure coordinates are tuples (x, y)
				p1 = centerline[i-1]
				p2 = centerline[i]
				
				# Convert arrays or lists to tuples
				if isinstance(p1, (list, numpy.ndarray)):
					p1 = tuple(p1) if isinstance(p1, list) else tuple(p1.tolist())
				if isinstance(p2, (list, numpy.ndarray)):
					p2 = tuple(p2) if isinstance(p2, list) else tuple(p2.tolist())
				
				draw.line([p1, p2], fill=wall_color, width=wall_width)
			
			# Draw wall ID at midpoint
			if len(centerline) > 0:
				mid_idx = len(centerline) // 2
				mid_point = centerline[mid_idx]
				
				# Ensure mid_point is a list/tuple for safe indexing
				if isinstance(mid_point, numpy.ndarray):
					mid_point = mid_point.tolist()
				
				try:
					font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
				except:
					font = ImageFont.load_default()
				
				wall_id = wall["wall_id"]
				text_bbox = draw.textbbox((0, 0), wall_id, font=font)
				text_width = text_bbox[2] - text_bbox[0]
				text_height = text_bbox[3] - text_bbox[1]
				
				# Draw text background with different color for exterior walls
				text_x = mid_point[0] - text_width // 2
				text_y = mid_point[1] - text_height // 2
				bg_color = (255, 255, 255) if not is_exterior else (255, 200, 150)  # Light orange for exterior
				outline_color = wall_color
				
				draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
							  fill=bg_color, outline=outline_color)
				
				# Draw wall ID text
				draw.text((text_x, text_y), wall_id, fill=text_color, font=font)
	
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
		radius = 12
		draw.ellipse([pos[0]-radius, pos[1]-radius, pos[0]+radius, pos[1]+radius], 
					fill=junction_color, outline=(0, 0, 0), width=3)
		
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
		legend_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
	except:
		legend_font = ImageFont.load_default()
	
	for i, (text, color) in enumerate(legend_items):
		y_pos = legend_y + i * 20
		# Draw color indicator
		draw.rectangle([10, y_pos, 25, y_pos+10], fill=color, outline=(0, 0, 0))
		# Draw text
		draw.text((30, y_pos-2), text, fill=(0, 0, 0), font=legend_font)
	
	# After drawing centerlines (existing loop) add fallback for missing ones
	# Draw junctions
	for i in range(len(bboxes)):
		if class_ids[i] != 1:
			continue
		# Compute simple horizontal/vertical centerline within bbox
		y1, x1, y2, x2 = bboxes[i]
		width_box = x2 - x1
		height_box = y2 - y1
		if width_box < 3 or height_box < 3:
			continue  # skip extremely small boxes
		# Decide orientation: longer dimension defines line direction
		if width_box >= height_box:
			cy = (y1 + y2) // 2
			p_start = (x1, cy)
			p_end = (x2, cy)
		else:
			cx = (x1 + x2) // 2
			p_start = (cx, y1)
			p_end = (cx, y2)
		draw.line([p_start, p_end], fill=centerline_color, width=4)
	
	# ----------------------
	# Fallback centerlines (only if skeleton-based centerline missing)
	# Improved: split lines at door openings so they never cover doors.
	# ----------------------

	# Pre-compute door bounding boxes with small margin
	door_boxes = []
	for idx, cid in enumerate(class_ids):
		if cid == 3:  # door
			dy1, dx1, dy2, dx2 = bboxes[idx]
			margin = 25
			# Ensure all coordinates are scalars
			dx1, dy1, dx2, dy2 = float(dx1), float(dy1), float(dx2), float(dy2)
			door_boxes.append((dx1 - margin, dy1 - margin, dx2 + margin, dy2 + margin))

	for i in range(len(bboxes)):
		if class_ids[i] != 1:
			continue  # process only walls

		y1, x1, y2, x2 = bboxes[i]
		# Ensure all coordinates are scalars
		y1, x1, y2, x2 = float(y1), float(x1), float(y2), float(x2)
		width_box = x2 - x1
		height_box = y2 - y1
		if width_box < 3 or height_box < 3:
			continue  # ignore tiny walls

		if width_box >= height_box:  # horizontal wall
			cy = (y1 + y2) // 2
			segments = [(x1, x2)]
			# cut segments around door overlaps
			for (dx1, dy1, dx2, dy2) in door_boxes:
				if dy1 <= cy <= dy2:  # vertical overlap
					updated = []
					for sx, ex in segments:
						# no overlap
						if ex < dx1 or sx > dx2:
							updated.append((sx, ex))
						else:
							# segment before door
							if sx < dx1 - 1:
								updated.append((sx, dx1 - 1))
							# segment after door
							if ex > dx2 + 1:
								updated.append((dx2 + 1, ex))
					segments = updated
			# draw remaining segments
			for sx, ex in segments:
				if ex - sx > 2:
					draw.line([(sx, cy), (ex, cy)], fill=centerline_color, width=4)
		else:  # vertical wall
			cx = (x1 + x2) // 2
			segments = [(y1, y2)]
			for (dx1, dy1, dx2, dy2) in door_boxes:
				if dx1 <= cx <= dx2:  # horizontal overlap
					updated = []
					for sy, ey in segments:
						if ey < dy1 or sy > dy2:
							updated.append((sy, ey))
						else:
							if sy < dy1 - 1:
								updated.append((sy, dy1 - 1))
							if ey > dy2 + 1:
								updated.append((dy2 + 1, ey))
					segments = updated
			for sy, ey in segments:
				if ey - sy > 2:
					draw.line([(cx, sy), (cx, ey)], fill=centerline_color, width=4)

	return vis_image

# Fallback ordering using simple nearest-neighbor (kept for legacy calls)

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

# Add these functions after the existing utility functions, around line 90

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
	
	# Window size analysis
	if window_area < 1000:
		notes.append("Small window - possibly for ventilation or light")
	elif window_area > 8000:
		notes.append("Large window - likely for significant natural lighting")
	
	# Window orientation analysis
	if window_type == "horizontal":
		notes.append("Horizontal window orientation")
		if width > height * 2:
			notes.append("Wide horizontal window - panoramic view")
	else:
		notes.append("Vertical window orientation")
		if height > width * 2:
			notes.append("Tall vertical window - floor-to-ceiling style")
	
	# Aspect ratio insights
	aspect_ratio = width / height if height > 0 else 0
	if aspect_ratio > 3:
		notes.append("Very wide window - modern architectural style")
	elif aspect_ratio < 0.5:
		notes.append("Very tall window - contemporary design")
	
	return notes

def identify_exterior_walls(wall_parameters, image_width, image_height, scale_factor_mm_per_pixel):
	"""Identify walls that form the exterior boundary of the floor plan"""
	exterior_walls = []
	interior_walls = []
	
	# Convert image dimensions to mm for boundary analysis
	image_width_mm = pixels_to_mm(image_width, scale_factor_mm_per_pixel)
	image_height_mm = pixels_to_mm(image_height, scale_factor_mm_per_pixel)
	
	# Define boundary margins (walls within 5% of image edges are likely exterior)
	boundary_margin = 0.05  # 5% of image dimensions
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
		
		# Criterion 1: Near image boundaries
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
		
		# Criterion 2: Poorly connected walls (likely exterior)
		if connection_count <= 1 and not is_exterior:
			is_exterior = True
			exterior_reasons.append("poorly_connected")
		
		# Criterion 3: Long walls near boundaries (likely perimeter walls)
		wall_length = wall.get("length", 0)
		if wall_length > 100 and (is_near_left or is_near_right or is_near_top or is_near_bottom):
			is_exterior = True
			exterior_reasons.append("long_boundary_wall")
		
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

if __name__ == '__main__':
	application.debug = True
	print('===========before running==========')
	application.run(host='0.0.0.0', port=8080)
	print('===========after running==========')