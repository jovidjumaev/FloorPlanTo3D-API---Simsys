"""
Visualization routes for wall analysis
"""

from flask import Blueprint, request, jsonify
import logging
import traceback
import time
import os
import cv2
import numpy
from datetime import datetime
from PIL import Image

from models.mask_rcnn_model import get_model, get_config, is_model_initialized
from services.image_validation import validate_and_resize_image, check_memory_usage
from image_processing.image_loader import myImageLoader
from mrcnn.model import mold_image
from numpy import expand_dims

from utils.geometry import safe_logical_or, safe_logical_and
from utils.conversions import (
    pixels_to_mm, convert_junction_position_to_mm, save_wall_analysis)
from utils.file_utils import getNextTestNumber

from image_processing.mask_processing import (
    extract_wall_masks, segment_individual_walls)

from analysis.door_analysis import (
    analyzeDoorOrientation, generateArchitecturalNotes,
    categorize_door_size, assess_door_accessibility)

from analysis.wall_analysis import (
    extract_wall_parameters, find_wall_connections, analyze_junction_types,
    identify_exterior_walls, calculate_perimeter_dimensions)

from analysis.junction_analysis import find_junctions_from_bboxes

from analysis.window_analysis import (
    categorize_window_size, assess_window_glazing, generate_window_notes)

from visualization.wall_visualization import create_wall_visualization

from ocr_detector import detect_space_names

from config.constants import IMAGES_OUTPUT_DIR

logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('visualization', __name__)


@bp.route('/visualize_walls', methods=['POST'])
def visualize_wall_analysis():
    """Create enhanced visualization showing wall centerlines, junctions, and wall parameters"""
    
    # Check if model is initialized
    if not is_model_initialized():
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
                    "min_size": 100,  # This should come from config
                    "max_size": 2048,  # This should come from config
                    "resize_allowed": True  # This should come from config
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
        cfg = get_config()
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        logger.debug(f"Time - preprocessing: {time.time()-t0:.2f}s")
        
        # Model detection
        t0 = time.time()
        model = get_model()
        r = model.detect(sample, verbose=0)[0]
        
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