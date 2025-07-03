import os
import numpy
import skimage.color

from numpy.lib.function_base import average
from numpy import zeros
from numpy import asarray

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image

from skimage.draw import polygon2mask
from skimage.io import imread
from skimage.morphology import skeletonize, remove_small_objects, label
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


def myImageLoader(imageInput):
	# Convert PIL image to RGB first to ensure consistent format
	if hasattr(imageInput, 'convert'):
		imageInput = imageInput.convert('RGB')
	
	image = numpy.asarray(imageInput)
	
	# Ensure we have a 3D array with 3 channels
	if image.ndim == 2:
		# Grayscale image
		image = skimage.color.gray2rgb(image)
	elif image.ndim == 3:
		if image.shape[-1] == 4:
			# RGBA -> RGB (remove alpha channel)
			image = image[..., :3]
		elif image.shape[-1] == 1:
			# Single channel -> RGB
			image = numpy.repeat(image, 3, axis=2)
		elif image.shape[-1] != 3:
			# Convert to RGB if not already
			image = skimage.color.gray2rgb(image)
	
	# Ensure the image is in the right data type and range
	if image.dtype != numpy.uint8:
		if image.max() <= 1.0:
			image = (image * 255).astype(numpy.uint8)
		else:
			image = image.astype(numpy.uint8)
	
	h, w, c = image.shape
	print(f"Processed image shape: {h}x{w}x{c}")
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
		image, w, h = myImageLoader(imagefile)
		print(f"Creating visualization for image: {h}x{w}")
		
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
	
	class_names = {1: 'Wall', 2: 'Window', 3: 'Door'}
	
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
		
		# Enhanced door visualization with orientation
		if class_id == 3 and masks is not None:  # door
			door_mask = masks[:, :, i] if i < masks.shape[2] else None
			orientation = analyzeDoorOrientation(door_mask, bbox, image_width, image_height)
			
			# Draw door orientation arrow
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
			arrow_length = min((x2 - x1), (y2 - y1)) * 0.4
			
			# Determine arrow direction based on estimated swing
			swing = orientation.get("estimated_swing", "")
			arrow_color = (255, 255, 0)  # Yellow arrow
			
			if "upward" in swing:
				arrow_end = (center_x, center_y - arrow_length)
			elif "downward" in swing:
				arrow_end = (center_x, center_y + arrow_length)
			elif "leftward" in swing:
				arrow_end = (center_x - arrow_length, center_y)
			elif "rightward" in swing:
				arrow_end = (center_x + arrow_length, center_y)
			else:
				arrow_end = (center_x, center_y)  # No arrow for unknown
			
			# Draw arrow line
			if arrow_end != (center_x, center_y):
				draw.line([(center_x, center_y), arrow_end], fill=arrow_color, width=3)
				
				# Draw arrow head (simple triangle)
				head_size = 8
				if "upward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] + head_size),
						(arrow_end[0] + head_size, arrow_end[1] + head_size)
					]
				elif "downward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] - head_size),
						(arrow_end[0] + head_size, arrow_end[1] - head_size)
					]
				elif "leftward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] + head_size, arrow_end[1] - head_size),
						(arrow_end[0] + head_size, arrow_end[1] + head_size)
					]
				elif "rightward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] - head_size),
						(arrow_end[0] - head_size, arrow_end[1] + head_size)
					]
				
				if 'arrow_points' in locals():
					draw.polygon(arrow_points, fill=arrow_color)
		
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
	"""
	Analyze door orientation and opening direction based on geometric features
	This is a post-processing approach using the detected door mask and position
	"""
	y1, x1, y2, x2 = door_bbox
	door_width = x2 - x1
	door_height = y2 - y1
	
	# Determine if door is horizontal or vertical based on aspect ratio
	is_horizontal = door_width > door_height
	
	# Extract door region from mask (ensure integer indices)
	door_region = door_mask[int(y1):int(y2), int(x1):int(x2)] if door_mask is not None else None
	
	orientation_analysis = {
		"door_type": "horizontal" if is_horizontal else "vertical",
		"estimated_swing": "unknown",
		"hinge_side": "unknown", 
		"confidence": 0.0,
		"analysis_method": "geometric_inference"
	}
	
	if door_region is not None:
		# Analyze the door mask shape for opening indicators
		door_center_x = (x1 + x2) / 2
		door_center_y = (y1 + y2) / 2
		
		# For horizontal doors (most common in floor plans)
		if is_horizontal:
			# Check door position relative to image bounds to infer swing
			distance_to_top = y1
			distance_to_bottom = image_height - y2
			distance_to_left = x1  
			distance_to_right = image_width - x2
			
			# Heuristic: doors usually open into the larger space
			if distance_to_top > distance_to_bottom:
				orientation_analysis.update({
					"estimated_swing": "opens_upward",
					"hinge_side": "bottom_edge",
					"confidence": 0.6
				})
			else:
				orientation_analysis.update({
					"estimated_swing": "opens_downward", 
					"hinge_side": "top_edge",
					"confidence": 0.6
				})
		
		# For vertical doors
		else:
			if door_center_x < image_width / 2:
				orientation_analysis.update({
					"estimated_swing": "opens_rightward",
					"hinge_side": "left_edge", 
					"confidence": 0.6
				})
			else:
				orientation_analysis.update({
					"estimated_swing": "opens_leftward",
					"hinge_side": "right_edge",
					"confidence": 0.6
				})
		
		# Analyze mask density for additional clues
		if hasattr(door_region, 'shape') and door_region.size > 0:
			mask_density = numpy.sum(door_region) / door_region.size
			if mask_density > 0.7:
				orientation_analysis["confidence"] = min(0.8, orientation_analysis["confidence"] + 0.2)
	
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
			door_width = float(x2 - x1)
			door_height = float(y2 - y1)
			door_area = door_width * door_height
			
			# Orientation analysis
			orientation = analyzeDoorOrientation(door_mask, bbox, w, h)
			
			# Architectural analysis
			door_bbox_dict = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}
			architectural_notes = generateArchitecturalNotes(orientation, door_bbox_dict)
			
			# Build comprehensive door data
			door_data = {
				"door_id": i + 1,
				"confidence": float(confidence),
				"location": {
					"bbox": door_bbox_dict,
					"center": {
						"x": float((x1 + x2) / 2),
						"y": float((y1 + y2) / 2)
					},
					"relative_position": {
						"from_left": f"{(x1/w)*100:.1f}%",
						"from_top": f"{(y1/h)*100:.1f}%"
					}
				},
				"dimensions": {
					"width_px": door_width,
					"height_px": door_height,
					"area_px": door_area,
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
				"analysis_type": "door_orientation_analysis"
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
			if skeleton[y, x]:
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
	"""Simple junction detection - find endpoints and intersections only"""
	junctions = []
	h, w = skeleton.shape
	
	for y in range(1, h-1):
		for x in range(1, w-1):
			if skeleton[y, x]:
				# Count skeleton neighbors in 8-connectivity
				neighbors = [
					skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
					skeleton[y, x-1],                     skeleton[y, x+1],
					skeleton[y+1, x-1], skeleton[y+1, x], skeleton[y+1, x+1]
				]
				neighbor_count = sum(neighbors)
				
				# Junction points: endpoints (1 neighbor) or intersections (3+ neighbors)
				if neighbor_count == 1 or neighbor_count >= 3:
					junctions.append((x, y))
	
	# Remove duplicates within 10 pixels
	filtered_junctions = []
	for junction in junctions:
		is_duplicate = False
		for existing in filtered_junctions:
			dist = ((junction[0] - existing[0])**2 + (junction[1] - existing[1])**2)**0.5
			if dist < 10:
				is_duplicate = True
				break
		if not is_duplicate:
			filtered_junctions.append(junction)
	
	return filtered_junctions

def segment_individual_walls(wall_mask):
	"""Segment connected wall regions into individual wall segments"""
	# Clean the mask
	cleaned_mask = remove_small_objects(wall_mask, min_size=50)
	
	# Skeletonize to find centerlines
	skeleton = skeletonize(cleaned_mask)
	
	# Find junction points using simpler method
	junctions = find_junction_points_simple(skeleton)
	
	# Remove junction points to separate wall segments
	skeleton_segmented = skeleton.copy()
	for jx, jy in junctions:
		# Remove small area around junction
		skeleton_segmented[max(0,jy-1):min(skeleton.shape[0],jy+2), 
						 max(0,jx-1):min(skeleton.shape[1],jx+2)] = False
	
	# Label connected components
	labeled_skeleton = label(skeleton_segmented)
	
	# Extract individual segments
	segments = []
	for region in regionprops(labeled_skeleton):
		if region.area > 10:  # Filter small segments
			segment_coords = region.coords
			segments.append(segment_coords)
	
	return segments, junctions

def extract_centerline_coords(segment_coords):
	"""Extract ordered centerline coordinates from segment"""
	if len(segment_coords) < 2:
		return segment_coords.tolist()
	
	# Convert to list of [x, y] coordinates
	centerline = [[int(coord[1]), int(coord[0])] for coord in segment_coords]
	
	# Sort coordinates to create ordered path
	if len(centerline) > 2:
		ordered_centerline = order_centerline_points(centerline)
		return ordered_centerline
	
	return centerline

def order_centerline_points(points):
	"""Order centerline points to create a continuous path"""
	if len(points) < 3:
		return points
	
	# Start with first point
	ordered = [points[0]]
	remaining = points[1:]
	
	while remaining:
		current = ordered[-1]
		# Find closest remaining point
		distances = [((p[0]-current[0])**2 + (p[1]-current[1])**2)**0.5 for p in remaining]
		closest_idx = numpy.argmin(distances)
		ordered.append(remaining[closest_idx])
		remaining.pop(closest_idx)
	
	return ordered

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

def extract_wall_parameters(segments, wall_mask, junctions):
	"""Extract comprehensive parameters for each wall segment"""
	wall_parameters = []
	
	# Find wall connections
	wall_connections = find_wall_connections(segments, junctions)
	
	for i, segment in enumerate(segments):
		wall_id = f"W{i+1}"
		
		# Extract centerline coordinates
		centerline = extract_centerline_coords(segment)
		
		# Calculate wall properties
		length = calculate_wall_length(centerline)
		thickness = calculate_wall_thickness(wall_mask, centerline)
		orientation = calculate_wall_orientation(centerline)
		
		# Get bounding box of wall segment
		if len(segment) > 0:
			min_y, min_x = numpy.min(segment, axis=0)
			max_y, max_x = numpy.max(segment, axis=0)
			bbox = {
				"x1": float(min_x), "y1": float(min_y),
				"x2": float(max_x), "y2": float(max_y)
			}
		else:
			bbox = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
		
		wall_params = {
			"wall_id": wall_id,
			"centerline": centerline,
			"length_px": length,
			"thickness": thickness,
			"orientation_degrees": orientation,
			"bbox": bbox,
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
		image, w, h = myImageLoader(imagefile)
		print(f"Analyzing wall parameters for image: {h}x{w}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
		
		# Extract wall masks
		wall_masks, wall_indices = extract_wall_masks(r)
		
		if not wall_masks:
			return jsonify({
				"message": "No walls detected in the floor plan",
				"total_walls": 0,
				"individual_walls": [],
				"junctions": []
			})
		
		print(f"Found {len(wall_masks)} wall regions")
		
		# Combine all wall masks for comprehensive analysis
		combined_wall_mask = numpy.zeros((h, w), dtype=bool)
		for mask in wall_masks:
			combined_wall_mask |= mask
		
		# Segment walls and find junctions
		wall_segments, junctions = segment_individual_walls(combined_wall_mask)
		print(f"Segmented into {len(wall_segments)} individual walls with {len(junctions)} junctions")
		
		# Extract detailed wall parameters
		wall_parameters = extract_wall_parameters(wall_segments, combined_wall_mask, junctions)
		
		# Analyze junctions
		wall_connections = find_wall_connections(wall_segments, junctions)
		junction_analysis = analyze_junction_types(junctions, wall_connections)
		
		# Calculate summary statistics
		total_wall_length = sum(wall["length_px"] for wall in wall_parameters)
		avg_wall_thickness = numpy.mean([wall["thickness"]["average"] for wall in wall_parameters if wall["thickness"]["average"] > 0])
		
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
				"analysis_type": "wall_parameter_analysis"
			},
			"summary": {
				"total_walls_detected": len(wall_parameters),
				"total_junctions": len(junctions),
				"total_wall_length_px": float(total_wall_length),
				"average_wall_thickness_px": float(avg_wall_thickness) if not numpy.isnan(avg_wall_thickness) else 0.0,
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
		original_image = imagefile.copy()
		image, w, h = myImageLoader(imagefile)
		print(f"Creating wall analysis visualization for image: {h}x{w}")
		
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)

		# Get model predictions
		r = _model.detect(sample, verbose=0)[0]
		
		# Extract wall masks and perform analysis
		wall_masks, wall_indices = extract_wall_masks(r)
		
		if not wall_masks:
			return jsonify({"error": "No walls detected for visualization"}), 400
		
		# Combine wall masks
		combined_wall_mask = numpy.zeros((h, w), dtype=bool)
		for mask in wall_masks:
			combined_wall_mask |= mask
		
		# Perform wall analysis
		wall_segments, junctions = segment_individual_walls(combined_wall_mask)
		wall_parameters = extract_wall_parameters(wall_segments, combined_wall_mask, junctions)
		junction_analysis = analyze_junction_types(junctions, find_wall_connections(wall_segments, junctions))
		
		# Create enhanced visualization
		vis_image = create_wall_visualization(original_image, r, wall_parameters, junction_analysis, w, h)
		
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
				"analysis_type": "wall_visualization_analysis"
			},
			"individual_walls": wall_parameters,
			"junctions": junction_analysis
		}
		
		wall_json_filename = f"walls{test_num}.json"
		save_wall_analysis(wall_analysis, wall_json_filename)
		
		return jsonify({
			"message": "Wall analysis visualization created successfully",
			"visualization_file": wall_vis_filename,
			"analysis_file": wall_json_filename,
			"total_walls": len(wall_parameters),
			"total_junctions": len(junction_analysis),
			"wall_summary": {
				"wall_count": len(wall_parameters),
				"junction_count": len(junction_analysis),
				"total_length": sum(w["length_px"] for w in wall_parameters)
			}
		})
		
	except Exception as e:
		print(f"Error creating wall visualization: {str(e)}")
		return jsonify({"error": str(e)}), 500

def create_wall_visualization(original_image, model_results, wall_parameters, junction_analysis, image_width, image_height):
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
		3: (0, 0, 255)      # Door - Blue
	}
	
	centerline_color = (0, 255, 255)     # Cyan for centerlines (more visible)
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
		
		# Enhanced door visualization with orientation
		if class_id == 3 and masks is not None:  # door
			door_mask = masks[:, :, i] if i < masks.shape[2] else None
			orientation = analyzeDoorOrientation(door_mask, bbox, image_width, image_height)
			
			# Draw door orientation arrow
			center_x = (x1 + x2) / 2
			center_y = (y1 + y2) / 2
			arrow_length = min((x2 - x1), (y2 - y1)) * 0.4
			
			# Determine arrow direction based on estimated swing
			swing = orientation.get("estimated_swing", "")
			arrow_color = (255, 255, 0)  # Yellow arrow
			
			if "upward" in swing:
				arrow_end = (center_x, center_y - arrow_length)
			elif "downward" in swing:
				arrow_end = (center_x, center_y + arrow_length)
			elif "leftward" in swing:
				arrow_end = (center_x - arrow_length, center_y)
			elif "rightward" in swing:
				arrow_end = (center_x + arrow_length, center_y)
			else:
				arrow_end = (center_x, center_y)  # No arrow for unknown
			
			# Draw arrow line
			if arrow_end != (center_x, center_y):
				draw.line([(center_x, center_y), arrow_end], fill=arrow_color, width=3)
				
				# Draw arrow head (simple triangle)
				head_size = 8
				if "upward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] + head_size),
						(arrow_end[0] + head_size, arrow_end[1] + head_size)
					]
				elif "downward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] - head_size),
						(arrow_end[0] + head_size, arrow_end[1] - head_size)
					]
				elif "leftward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] + head_size, arrow_end[1] - head_size),
						(arrow_end[0] + head_size, arrow_end[1] + head_size)
					]
				elif "rightward" in swing:
					arrow_points = [
						(arrow_end[0], arrow_end[1]),
						(arrow_end[0] - head_size, arrow_end[1] - head_size),
						(arrow_end[0] - head_size, arrow_end[1] + head_size)
					]
				
				if 'arrow_points' in locals():
					draw.polygon(arrow_points, fill=arrow_color)
					del arrow_points  # Clean up for next iteration
		
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
	for wall in wall_parameters:
		centerline = wall["centerline"]
		if len(centerline) > 1:
			# Draw centerline
			for i in range(1, len(centerline)):
				# Ensure coordinates are tuples (x, y)
				p1 = tuple(centerline[i-1]) if isinstance(centerline[i-1], list) else centerline[i-1]
				p2 = tuple(centerline[i]) if isinstance(centerline[i], list) else centerline[i]
				draw.line([p1, p2], fill=centerline_color, width=4)
			
			# Draw wall ID at midpoint
			if len(centerline) > 0:
				mid_idx = len(centerline) // 2
				mid_point = centerline[mid_idx]
				
				try:
					font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
				except:
					font = ImageFont.load_default()
				
				wall_id = wall["wall_id"]
				text_bbox = draw.textbbox((0, 0), wall_id, font=font)
				text_width = text_bbox[2] - text_bbox[0]
				text_height = text_bbox[3] - text_bbox[1]
				
				# Draw text background
				text_x = mid_point[0] - text_width // 2
				text_y = mid_point[1] - text_height // 2
				draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
							  fill=(255, 255, 255), outline=centerline_color)
				
				# Draw wall ID text
				draw.text((text_x, text_y), wall_id, fill=text_color, font=font)
	
	# Draw junctions
	for junction in junction_analysis:
		pos = junction["position"]
		junction_id = junction["junction_id"]
		
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
		("Centerlines (Cyan)", centerline_color),
		("Junctions (Magenta)", junction_color),
		("Windows (Green)", (0, 255, 0)),
		("Doors (Blue)", (0, 0, 255)),
		("Door Arrows (Yellow)", (255, 255, 0))
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
	
	return vis_image

if __name__ == '__main__':
	application.debug = True
	print('===========before running==========')
	application.run(host='0.0.0.0', port=8080)
	print('===========after running==========')
