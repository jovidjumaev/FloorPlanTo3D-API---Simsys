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

# Import new modular components
from models.mask_rcnn_model import initialize_model, get_model, get_config, is_model_initialized
from services.image_validation import validate_and_resize_image, check_memory_usage
from services.json_service import buildEnhancedJson
from services.accuracy_service import performAccuracyAnalysis

# Import route blueprints
from routes.health_routes import bp as health_bp
from routes.accuracy_routes import bp as accuracy_bp
from routes.visualization_routes import bp as visualization_bp
   
# Configure logging
logging.basicConfig(
	level=getattr(logging, app_config.LOG_LEVEL),
	format=app_config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Global variables
ROOT_DIR = os.path.abspath("./")

# Add root directory to path
sys.path.append(ROOT_DIR)

# Initialize Flask application
application = Flask(__name__)
application.debug = app_config.DEBUG
cors = CORS(application, resources={r"/*": {"origins": app_config.CORS_ORIGINS}})

# Register blueprints
application.register_blueprint(health_bp)
application.register_blueprint(accuracy_bp)
application.register_blueprint(visualization_bp)

# Initialize model at startup
logger.info("Starting model initialization...")
try:
    initialize_model()
    logger.info("Model initialization completed successfully!")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")


if __name__ == '__main__':
    api_config = app_config.get_api_config()
    logger.info('===========Starting FloorPlanTo3D API==========')
    logger.info(f"Running on {api_config['HOST']}:{api_config['PORT']}")
    logger.info(f"Debug mode: {api_config['DEBUG']}")
    try:
        application.run(
            host=api_config['HOST'],
            port=api_config['PORT'],
            debug=api_config['DEBUG']
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
    logger.info('===========API stopped==========')