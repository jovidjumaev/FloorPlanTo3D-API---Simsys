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
		door_masks = masks[:, :, [i for i, cid in enumerate(class_ids) if cid == 3]]
		enhanced_doors = enhancedDoorAnalysis(door_objects, door_masks, image_width, image_height)
		
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
	
	# Extract door region from mask
	door_region = door_mask[y1:y2, x1:x2] if door_mask is not None else None
	
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

def enhancedDoorAnalysis(door_objects, masks, image_width, image_height):
	"""
	Enhance door objects with orientation analysis
	"""
	enhanced_doors = []
	
	for i, door in enumerate(door_objects):
		door_mask = masks[:, :, i] if i < masks.shape[2] else None
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
		door_masks = r['masks'][:, :, door_indices] if len(door_indices) > 0 else None
		
		# Perform detailed door analysis
		detailed_doors = []
		for i, (bbox, confidence) in enumerate(zip(door_bboxes, door_scores)):
			door_mask = door_masks[:, :, i] if door_masks is not None else None
			
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

if __name__ == '__main__':
	application.debug = True
	print('===========before running==========')
	application.run(host='0.0.0.0', port=8080)
	print('===========after running==========')
