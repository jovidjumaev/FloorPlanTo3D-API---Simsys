"""
OCR-based room detection module for floor plan analysis
Extracts room labels and maps them to wall regions
"""

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
import re
from typing import List, Dict, Tuple, Optional
import json

class RoomDetector:
    """OCR-based room detection and wall-room mapping"""
    
    def __init__(self, languages=['en']):
        """Initialize OCR reader"""
        try:
            self.reader = easyocr.Reader(languages, gpu=False)
            self.initialized = True
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            self.initialized = False
    
    def extract_text_from_image(self, image):
        """Extract text and positions from floor plan image"""
        if not self.initialized:
            return []
        
        try:
            # Convert PIL to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Run OCR with more aggressive settings for better detection
            results = self.reader.readtext(
                image_array, 
                detail=1, 
                paragraph=False,
                width_ths=0.5,  # Reduced for better text detection
                height_ths=0.5  # Reduced for better text detection
            )
            
            # Process results
            text_detections = []
            for (bbox, text, confidence) in results:
                if confidence > 0.2:  # Lower threshold to catch more text
                    # Calculate center point
                    center_x = int(np.mean([point[0] for point in bbox]))
                    center_y = int(np.mean([point[1] for point in bbox]))
                    
                    # Calculate dimensions
                    width = int(max([point[0] for point in bbox]) - min([point[0] for point in bbox]))
                    height = int(max([point[1] for point in bbox]) - min([point[1] for point in bbox]))
                    
                    text_detections.append({
                        'text': text.strip(),
                        'confidence': float(confidence),
                        'bbox': bbox,
                        'center': [center_x, center_y],
                        'width': width,
                        'height': height
                    })
            
            return text_detections
            
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return []
    
    def classify_room_text(self, text_detections):
        """Classify detected text as room names or other content"""
        
        # Enhanced room keywords with more variations
        room_keywords = {
            'bedroom': ['bedroom', 'bed room', 'br', 'master', 'guest room', 'bed', 'bedrm'],
            'bathroom': ['bathroom', 'bath', 'toilet', 'wc', 'powder room', 'bth', 'bathrm'],
            'kitchen': ['kitchen', 'kitchenette', 'pantry', 'kit'],
            'living_room': ['living room', 'living', 'family room', 'great room', 'lounge', 'greatroom'],
            'dining_room': ['dining room', 'dining', 'breakfast', 'nook', 'diningroom'],
            'office': ['office', 'study', 'den', 'library'],
            'closet': ['closet', 'walk-in', 'wardrobe', 'storage', 'clst'],
            'laundry': ['laundry', 'utility', 'mud room', 'utility room'],
            'garage': ['garage', 'carport'],
            'hallway': ['hallway', 'hall', 'corridor', 'foyer', 'entry', 'entrance'],
            'other': ['room', 'space', 'area']
        }
        
        room_labels = []
        
        for detection in text_detections:
            text_lower = detection['text'].lower().strip()
            original_text = detection['text'].strip()
            
            # Skip very short text, pure numbers, or single characters
            if len(text_lower) < 2 or text_lower.isdigit() or len(text_lower) == 1:
                continue
            
            # Skip common non-room text with improved filtering
            skip_words = [
                'door', 'window', 'wall', 'floor', 'ceiling', 
                'ft', 'in', 'sq', 'm²', 'cm', 'mm', 'inch',
                'plan', 'house', 'floor plan', 'scale',
                'north', 'south', 'east', 'west',
                'up', 'down', 'stairs', 'stair'
            ]
            
            # More flexible skip check
            should_skip = False
            for skip_word in skip_words:
                if skip_word in text_lower and not any(keyword in text_lower for keywords in room_keywords.values() for keyword in keywords):
                    should_skip = True
                    break
            
            if should_skip:
                continue
            
            # Classify room type with improved matching
            room_type = 'unknown'
            for room_cat, keywords in room_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    room_type = room_cat
                    break
            
            # Enhanced heuristics for room name detection
            if room_type != 'unknown' or self._looks_like_room_name_enhanced(text_lower, original_text):
                room_labels.append({
                    **detection,
                    'room_type': room_type,
                    'is_room_label': True,
                    'processed_text': text_lower
                })
        
        return room_labels
    
    def _looks_like_room_name_enhanced(self, text_lower, original_text):
        """Enhanced heuristic to identify text that looks like a room name"""
        
        # Contains "room" anywhere
        if 'room' in text_lower:
            return True
        
        # Short text that could be abbreviation (2-4 chars, mostly letters)
        if 2 <= len(text_lower) <= 4 and text_lower.replace(' ', '').isalpha():
            return True
        
        # Contains typical room words or patterns
        room_indicators = [
            'master', 'guest', 'main', 'primary', 'secondary',
            'ensuite', 'en-suite', 'powder', 'half', 'full',
            'walk', 'in', 'closet'
        ]
        if any(indicator in text_lower for indicator in room_indicators):
            return True
        
        # Text with numbers that might be room numbers (like "BEDROOM 4", "BATH 2")
        if any(char.isdigit() for char in text_lower) and any(char.isalpha() for char in text_lower):
            # Check if it contains room-like words
            room_like_words = ['bed', 'bath', 'room', 'kit', 'liv', 'din']
            if any(word in text_lower for word in room_like_words):
                return True
        
        # Capitalized text that looks like labels (mostly uppercase original text)
        if len(original_text) > 2 and original_text.isupper():
            return True
        
        # Text that has reasonable length for room names
        if 3 <= len(text_lower) <= 15 and text_lower.replace(' ', '').isalpha():
            return True
        
        return False
    
    def find_enclosed_regions(self, wall_parameters, image_width=2000, image_height=2000):
        """Find enclosed regions using a simplified flood-fill approach"""
        regions = []
        
        if not wall_parameters:
            return regions
        
        # Create a simplified approach: find areas bounded by walls
        # Step 1: Create a mask from wall boundaries
        wall_mask = self._create_wall_mask(wall_parameters, image_width, image_height)
        
        # Step 2: Find enclosed regions using flood fill
        room_regions = self._find_regions_by_flood_fill(wall_mask, wall_parameters)
        
        # Step 3: Convert to region objects
        for i, region in enumerate(room_regions):
            if region['area'] > 5000:  # Minimum area threshold
                regions.append({
                    'region_id': f"R{i+1}",
                    'center': region['center'],
                    'bbox': region['bbox'],
                    'width': region['width'],
                    'height': region['height'],
                    'area': region['area'],
                    'walls': region['walls'],
                    'contour': region['contour'],
                    'confidence': region['confidence']
                })
        
        return regions
    
    def _create_wall_mask(self, wall_parameters, image_width=2000, image_height=2000):
        """Create a binary mask from wall boundaries"""
        
        # Auto-detect mask size based on wall coordinates if walls exist
        if wall_parameters:
            max_x = max_y = 0
            min_x = min_y = float('inf')
            
            for wall in wall_parameters:
                centerline = wall.get('centerline', [])
                for point in centerline:
                    if len(point) >= 2:
                        x, y = point[0], point[1]
                        max_x = max(max_x, x)
                        max_y = max(max_y, y)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
            
            if max_x > 0 and max_y > 0:
                # Add padding and use actual coordinate bounds
                padding = 100
                mask_width = int(max_x + padding)
                mask_height = int(max_y + padding)
                
                # Ensure minimum size but use actual dimensions
                mask_width = max(mask_width, image_width)
                mask_height = max(mask_height, image_height)
        else:
            mask_width = image_width
            mask_height = image_height
        
        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        
        # Draw wall centerlines as thick lines
        for wall in wall_parameters:
            centerline = wall.get('centerline', [])
            if len(centerline) >= 2:
                # Draw thick line for wall
                for i in range(len(centerline) - 1):
                    pt1 = tuple(map(int, centerline[i]))
                    pt2 = tuple(map(int, centerline[i + 1]))
                    try:
                        cv2.line(mask, pt1, pt2, 255, thickness=12)  # Increased thickness
                    except Exception as e:
                        pass  # Skip invalid coordinates
        
        return mask
    
    def _find_regions_by_flood_fill(self, wall_mask, wall_parameters):
        """Find room regions using flood fill on the wall mask"""
        regions = []
        

        
        # Create inverted mask (areas NOT covered by walls)
        open_space = 255 - wall_mask
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)  # Slightly larger kernel
        open_space = cv2.morphologyEx(open_space, cv2.MORPH_CLOSE, kernel)
        open_space = cv2.morphologyEx(open_space, cv2.MORPH_OPEN, kernel)
        
        # Find connected components (room regions)
        num_labels, labels = cv2.connectedComponents(open_space)
        
        for label in range(1, num_labels):  # Skip background (label 0)
            # Create mask for this region
            region_mask = (labels == label).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate region properties
                area = cv2.contourArea(largest_contour)
                
                if area > 5000:  # Reduced minimum area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate center
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    else:
                        center_x = x + w // 2
                        center_y = y + h // 2
                    
                    # Simplify contour for better polygon representation
                    epsilon = 0.01 * cv2.arcLength(largest_contour, True)  # More detailed contour
                    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                    
                    # Convert contour to list of points
                    contour_points = []
                    for point in simplified_contour:
                        contour_points.append([int(point[0][0]), int(point[0][1])])
                    
                    # Find connected walls
                    connected_walls = self._find_walls_for_region(
                        [center_x, center_y], contour_points, wall_parameters
                    )
                    
                    regions.append({
                        'center': [center_x, center_y],
                        'bbox': [x, y, x + w, y + h],
                        'width': w,
                        'height': h,
                        'area': area,
                        'walls': connected_walls,
                        'contour': contour_points,
                        'confidence': self._calculate_region_confidence_simple(area, len(connected_walls))
                    })
        return regions
    
    def _find_walls_for_region(self, region_center, contour_points, wall_parameters):
        """Find walls that bound a region"""
        connected_walls = set()
        
        # Check distance from contour to wall centerlines
        for wall in wall_parameters:
            centerline = wall['centerline']
            wall_id = wall['wall_id']
            
            min_distance = float('inf')
            for contour_point in contour_points:
                for wall_point in centerline:
                    distance = self._calculate_distance(contour_point, wall_point)
                    min_distance = min(min_distance, distance)
            
            # If contour is close to wall, consider it connected
            if min_distance < 50:  # Threshold for connection
                connected_walls.add(wall_id)
        
        return list(connected_walls)
    
    def _calculate_region_confidence_simple(self, area, wall_count):
        """Calculate confidence for a region based on area and wall connections"""
        base_confidence = 0.5
        
        # Area bonus (larger areas are more likely to be rooms)
        area_bonus = min(0.3, area / 10000)
        
        # Wall count bonus (more walls = more enclosed)
        wall_bonus = min(0.2, wall_count * 0.05)
        
        return min(1.0, base_confidence + area_bonus + wall_bonus)
    
    def _prepare_wall_geometry(self, wall_parameters):
        """Prepare wall geometry data for polygon detection"""
        wall_segments = []
        all_endpoints = []
        
        for wall in wall_parameters:
            if len(wall['centerline']) >= 2:
                # Use centerline to create wall segments
                centerline = wall['centerline']
                
                # Create segments from consecutive points in centerline
                for i in range(len(centerline) - 1):
                    start_point = centerline[i]
                    end_point = centerline[i + 1]
                    
                    segment = {
                        'wall_id': wall['wall_id'],
                        'start': start_point,
                        'end': end_point,
                        'length': self._calculate_distance(start_point, end_point),
                        'angle': self._calculate_angle(start_point, end_point)
                    }
                    wall_segments.append(segment)
                    
                    # Collect endpoints
                    all_endpoints.extend([start_point, end_point])
        
        # Find unique endpoints (junction points)
        unique_endpoints = self._find_unique_points(all_endpoints)
        
        return wall_segments, unique_endpoints
    
    def _find_wall_enclosed_polygons(self, wall_segments, endpoints):
        """Find polygonal areas enclosed by walls using connectivity analysis"""
        polygons = []
        
        # Build connectivity graph
        connectivity_graph = self._build_wall_connectivity_graph(wall_segments, endpoints)
        
        # Find cycles in the graph (enclosed areas)
        cycles = self._find_cycles_in_connectivity_graph(connectivity_graph)
        
        # Convert cycles to polygons
        for cycle in cycles:
            if len(cycle) >= 3:
                polygon = self._cycle_to_polygon(cycle, connectivity_graph)
                if polygon and self._is_valid_room_polygon(polygon):
                    polygons.append(polygon)
        
        return polygons
    
    def _build_wall_connectivity_graph(self, wall_segments, endpoints, connection_threshold=25):
        """Build a graph of wall connectivity"""
        graph = {}
        
        # Initialize graph nodes for each endpoint
        for i, endpoint in enumerate(endpoints):
            graph[i] = {
                'point': endpoint,
                'connections': [],
                'segments': []
            }
        
        # Find connections between endpoints
        for i, point1 in enumerate(endpoints):
            for j, point2 in enumerate(endpoints):
                if i != j:
                    distance = self._calculate_distance(point1, point2)
                    if distance < connection_threshold:
                        # Find wall segment that connects these points
                        connecting_segment = self._find_connecting_segment(point1, point2, wall_segments)
                        if connecting_segment:
                            graph[i]['connections'].append({
                                'to_node': j,
                                'segment': connecting_segment,
                                'distance': distance
                            })
                            graph[i]['segments'].append(connecting_segment)
        
        return graph
    
    def _find_connecting_segment(self, point1, point2, wall_segments, tolerance=20):
        """Find wall segment that connects two points"""
        for segment in wall_segments:
            # Check if segment connects the two points (within tolerance)
            start_to_p1 = self._calculate_distance(segment['start'], point1)
            end_to_p2 = self._calculate_distance(segment['end'], point2)
            start_to_p2 = self._calculate_distance(segment['start'], point2)
            end_to_p1 = self._calculate_distance(segment['end'], point1)
            
            if ((start_to_p1 < tolerance and end_to_p2 < tolerance) or
                (start_to_p2 < tolerance and end_to_p1 < tolerance)):
                return segment
        return None
    
    def _find_cycles_in_connectivity_graph(self, graph):
        """Find cycles in the connectivity graph that represent enclosed areas"""
        cycles = []
        visited_global = set()
        
        # Try to find cycles starting from each unvisited node
        for start_node in graph:
            if start_node not in visited_global:
                found_cycles = self._dfs_find_cycles(graph, start_node, visited_global)
                cycles.extend(found_cycles)
        
        # Filter out duplicate and invalid cycles
        unique_cycles = self._filter_unique_cycles(cycles)
        
        return unique_cycles
    
    def _dfs_find_cycles(self, graph, start_node, visited_global, max_depth=8):
        """DFS to find cycles starting from a node"""
        cycles = []
        
        def dfs(current_node, path, visited_path):
            if len(path) > max_depth:
                return
            
            if len(path) > 2 and current_node == start_node:
                # Found a cycle
                cycle = path[:-1]  # Remove duplicate start node
                if len(cycle) >= 3:
                    cycles.append(cycle[:])
                return
            
            if current_node in visited_path:
                return
            
            visited_path.add(current_node)
            
            # Explore connections
            for connection in graph[current_node]['connections']:
                next_node = connection['to_node']
                if next_node not in visited_path or (len(path) > 2 and next_node == start_node):
                    dfs(next_node, path + [next_node], visited_path.copy())
        
        dfs(start_node, [start_node], set())
        
        # Mark nodes as globally visited
        visited_global.add(start_node)
        
        return cycles
    
    def _filter_unique_cycles(self, cycles):
        """Filter out duplicate cycles and keep only unique ones"""
        unique_cycles = []
        
        for cycle in cycles:
            # Normalize cycle (start with smallest node index)
            if cycle:
                min_idx = cycle.index(min(cycle))
                normalized_cycle = cycle[min_idx:] + cycle[:min_idx]
                
                # Check if this cycle is already in unique_cycles
                is_duplicate = False
                for existing_cycle in unique_cycles:
                    if len(existing_cycle) == len(normalized_cycle):
                        # Check forward and reverse direction
                        if (existing_cycle == normalized_cycle or 
                            existing_cycle == normalized_cycle[::-1]):
                            is_duplicate = True
                            break
                
                if not is_duplicate and len(normalized_cycle) >= 3:
                    unique_cycles.append(normalized_cycle)
        
        return unique_cycles
    
    def _cycle_to_polygon(self, cycle, graph):
        """Convert a cycle of nodes to a polygon of points"""
        polygon = []
        
        for node_idx in cycle:
            if node_idx in graph:
                point = graph[node_idx]['point']
                polygon.append(point)
        
        return polygon
    
    def _is_valid_room_polygon(self, polygon):
        """Check if polygon represents a valid room"""
        if len(polygon) < 3:
            return False
        
        # Calculate area using shoelace formula
        area = self._calculate_polygon_area(polygon)
        
        # Check minimum area (rooms should be reasonably sized)
        if area < 1000:  # Minimum area threshold
            return False
        
        # Check if polygon is not too elongated
        bbox = self._get_polygon_bbox(polygon)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 15:  # Too elongated
                return False
        
        return True
    
    def _calculate_polygon_area(self, polygon):
        """Calculate polygon area using shoelace formula"""
        if len(polygon) < 3:
            return 0
        
        area = 0
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        
        return abs(area) / 2
    
    def _polygon_to_region(self, polygon, wall_segments):
        """Convert polygon to region information"""
        if len(polygon) < 3:
            return None
        
        # Calculate center (centroid)
        center_x = sum(p[0] for p in polygon) / len(polygon)
        center_y = sum(p[1] for p in polygon) / len(polygon)
        center = [center_x, center_y]
        
        # Calculate bounding box
        bbox = self._get_polygon_bbox(polygon)
        
        # Calculate dimensions
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Calculate area
        area = self._calculate_polygon_area(polygon)
        
        # Find connected walls
        connected_walls = self._find_walls_for_polygon(polygon, wall_segments)
        
        # Calculate confidence based on polygon regularity and wall connectivity
        confidence = self._calculate_region_confidence(polygon, connected_walls)
        
        return {
            'center': center,
            'bbox': bbox,
            'width': width,
            'height': height,
            'area': area,
            'connected_walls': connected_walls,
            'confidence': confidence
        }
    
    def _get_polygon_bbox(self, polygon):
        """Get bounding box of polygon"""
        if not polygon:
            return [0, 0, 0, 0]
        
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        return [min_x, min_y, max_x, max_y]
    
    def _find_walls_for_polygon(self, polygon, wall_segments):
        """Find wall segments that form the polygon boundary"""
        connected_walls = set()
        
        # For each edge of the polygon, find corresponding wall segment
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Find wall segment closest to this edge
            closest_wall = self._find_closest_wall_segment(p1, p2, wall_segments)
            if closest_wall:
                connected_walls.add(closest_wall['wall_id'])
        
        return list(connected_walls)
    
    def _find_closest_wall_segment(self, p1, p2, wall_segments, max_distance=50):
        """Find wall segment closest to the line between p1 and p2"""
        closest_wall = None
        min_distance = float('inf')
        
        for segment in wall_segments:
            # Calculate distance between segment and line p1-p2
            dist = self._line_to_line_distance(p1, p2, segment['start'], segment['end'])
            
            if dist < min_distance and dist < max_distance:
                min_distance = dist
                closest_wall = segment
        
        return closest_wall
    
    def _line_to_line_distance(self, p1, p2, p3, p4):
        """Calculate minimum distance between two line segments"""
        # Simplified distance calculation - could be improved
        distances = [
            self._point_to_line_segment_distance(p1, p3, p4),
            self._point_to_line_segment_distance(p2, p3, p4),
            self._point_to_line_segment_distance(p3, p1, p2),
            self._point_to_line_segment_distance(p4, p1, p2)
        ]
        return min(distances)
    
    def _point_to_line_segment_distance(self, point, line_start, line_end):
        """Calculate distance from point to line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Handle case where line_start and line_end are the same
        if x1 == x2 and y1 == y2:
            return self._calculate_distance(point, line_start)
        
        # Calculate distance to line segment
        A = x0 - x1
        B = y0 - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return self._calculate_distance(point, line_start)
        
        param = dot / len_sq
        
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        
        dx = x0 - xx
        dy = y0 - yy
        return (dx * dx + dy * dy)**0.5
    
    def _calculate_region_confidence(self, polygon, connected_walls):
        """Calculate confidence score for a region"""
        base_confidence = 0.5
        
        # Bonus for having more walls
        wall_bonus = min(0.3, len(connected_walls) * 0.1)
        
        # Bonus for reasonable polygon shape
        area = self._calculate_polygon_area(polygon)
        area_bonus = min(0.2, area / 10000)  # Larger areas get higher confidence
        
        return min(1.0, base_confidence + wall_bonus + area_bonus)
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def _calculate_angle(self, p1, p2):
        """Calculate angle of line from p1 to p2"""
        import math
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    
    def _find_unique_points(self, points, tolerance=15):
        """Find unique points from a list, merging nearby points"""
        unique_points = []
        
        for point in points:
            is_unique = True
            for existing_point in unique_points:
                if self._calculate_distance(point, existing_point) < tolerance:
                    is_unique = False
                    break
            
            if is_unique:
                unique_points.append(point)
        
        return unique_points
    
    def match_rooms_to_walls(self, room_labels, wall_regions):
        """Match detected room labels to wall regions with improved spatial accuracy"""
        room_wall_mapping = []
        
        for room in room_labels:
            room_center = room['center']
            room_text = room['text'].upper()
            best_match = None
            min_distance = float('inf')
            
            # Find the best matching region for this room label
            for region in wall_regions:
                region_center = region['center']
                
                # Calculate distance between room label and region center
                distance = ((room_center[0] - region_center[0])**2 + 
                           (room_center[1] - region_center[1])**2)**0.5
                
                # Check if room label is within the region bounds (with margin)
                bbox = region['bbox']
                margin = 100  # Increased margin for better matching
                
                # Check if the room text is within or reasonably close to the region
                within_bounds = (
                    bbox[0] - margin <= room_center[0] <= bbox[2] + margin and
                    bbox[1] - margin <= room_center[1] <= bbox[3] + margin
                )
                
                # Additional spatial validation - ensure the match makes sense
                region_area = region.get('area', 0)
                region_confidence = region.get('confidence', 0.5)
                
                # Score the match based on multiple factors
                spatial_score = 1.0 / (1.0 + distance / 100)  # Closer is better
                area_score = min(1.0, region_area / 1000)  # Reasonable size regions preferred
                confidence_score = region_confidence
                
                # Combined score
                match_score = (spatial_score * 0.5 + area_score * 0.2 + confidence_score * 0.3)
                
                if within_bounds and distance < min_distance:
                    # Additional validation: check if this is a reasonable match
                    if self._validate_room_region_match(room, region):
                        min_distance = distance
                        best_match = region
                        best_match['match_score'] = match_score
            
            # Only create mapping if we found a good match
            if best_match and min_distance < 300:  # Maximum reasonable distance
                room_wall_mapping.append({
                    'room_name': room['text'],
                    'room_type': room['room_type'],
                    'confidence': room['confidence'],
                    'ocr_confidence': room['confidence'],
                    'spatial_confidence': best_match.get('match_score', 0.5),
                    'room_center': room_center,
                    'connected_walls': best_match['walls'],
                    'region_info': {
                        'center': best_match['center'],
                        'area': best_match['area'],
                        'dimensions': {
                            'width': best_match['width'],
                            'height': best_match['height']
                        },
                        'confidence': best_match.get('confidence', 0.5),
                        'contour': best_match.get('contour', [])
                    },
                    'match_distance': min_distance,
                    'match_quality': self._assess_match_quality(room, best_match, min_distance)
                })
        
        # Remove duplicate or overlapping matches
        room_wall_mapping = self._resolve_mapping_conflicts(room_wall_mapping)
        
        return room_wall_mapping
    
    def _validate_room_region_match(self, room, region):
        """Validate if a room label reasonably matches a region"""
        
        # Check if the region size is reasonable for the room type
        area = region.get('area', 0)
        room_type = room.get('room_type', 'unknown')
        
        # Define reasonable area ranges for different room types (in pixels squared)
        area_ranges = {
            'bathroom': (1000, 15000),    # Smaller rooms
            'closet': (500, 8000),        # Small spaces
            'hallway': (1000, 20000),     # Narrow but potentially long
            'kitchen': (3000, 25000),     # Medium sized
            'bedroom': (5000, 40000),     # Larger rooms
            'living_room': (8000, 60000), # Large rooms
            'dining_room': (4000, 30000), # Medium to large
            'office': (3000, 20000),      # Medium sized
            'laundry': (1500, 10000),     # Small to medium
            'garage': (10000, 80000),     # Large spaces
            'unknown': (500, 80000)       # Wide range for unknown
        }
        
        min_area, max_area = area_ranges.get(room_type, area_ranges['unknown'])
        
        # Check if area is reasonable
        if not (min_area <= area <= max_area):
            return False
        
        # Check if the region has reasonable aspect ratio
        width = region.get('width', 0)
        height = region.get('height', 0)
        
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            # Most rooms shouldn't be extremely narrow
            if aspect_ratio > 10:  # Very narrow spaces
                # Allow for hallways and some closets
                if room_type not in ['hallway', 'closet', 'corridor']:
                    return False
        
        return True
    
    def _assess_match_quality(self, room, region, distance):
        """Assess the quality of a room-region match"""
        
        # Factors that indicate good match quality
        distance_score = max(0, 1 - distance / 200)  # Closer is better
        area_score = min(1.0, region.get('area', 0) / 5000)  # Reasonable size
        confidence_score = region.get('confidence', 0.5)
        
        # Text-based validation
        room_text = room['text'].lower()
        room_type = room['room_type']
        
        # Bonus for high-confidence OCR
        ocr_confidence_bonus = room['confidence'] if room['confidence'] > 0.7 else 0
        
        # Overall quality score
        quality = (distance_score * 0.4 + 
                  area_score * 0.2 + 
                  confidence_score * 0.2 + 
                  ocr_confidence_bonus * 0.2)
        
        # Categorize quality
        if quality > 0.8:
            return "excellent"
        elif quality > 0.6:
            return "good"
        elif quality > 0.4:
            return "fair"
        else:
            return "poor"
    
    def _resolve_mapping_conflicts(self, mappings):
        """Resolve conflicts where multiple rooms map to the same region or overlapping regions"""
        
        if len(mappings) <= 1:
            return mappings
        
        # Group mappings by overlapping wall sets
        resolved_mappings = []
        used_wall_sets = set()
        
        # Sort by match quality and distance
        mappings.sort(key=lambda m: (
            -m.get('spatial_confidence', 0),
            m.get('match_distance', float('inf'))
        ))
        
        for mapping in mappings:
            wall_set = frozenset(mapping['connected_walls'])
            
            # Check if this wall set has significant overlap with existing ones
            has_conflict = False
            for used_walls in used_wall_sets:
                overlap = len(wall_set & used_walls)
                if overlap > len(wall_set) * 0.5:  # More than 50% overlap
                    has_conflict = True
                    break
            
            if not has_conflict:
                resolved_mappings.append(mapping)
                used_wall_sets.add(wall_set)
        
        return resolved_mappings
    
    def analyze_floor_plan_rooms(self, image, wall_parameters):
        """Main function to analyze rooms in floor plan"""
        result = {
            'text_detections': [],
            'room_labels': [],
            'wall_regions': [],
            'room_wall_mapping': [],
            'success': False,
            'error': None
        }
        
        try:
            # Get image dimensions
            if hasattr(image, 'size'):  # PIL Image
                image_width, image_height = image.size
            elif hasattr(image, 'shape'):  # numpy array
                image_height, image_width = image.shape[:2]
            else:
                image_width, image_height = 2000, 2000  # fallback
            
            # Step 1: Extract text from image
            text_detections = self.extract_text_from_image(image)
            result['text_detections'] = text_detections
            
            # Step 2: Classify room labels
            room_labels = self.classify_room_text(text_detections)
            result['room_labels'] = room_labels
            
            # Step 3: Find enclosed regions from walls
            wall_regions = self.find_enclosed_regions(wall_parameters, image_width, image_height)
            result['wall_regions'] = wall_regions
            
            # Step 4: Match rooms to walls
            room_wall_mapping = self.match_rooms_to_walls(room_labels, wall_regions)
            result['room_wall_mapping'] = room_wall_mapping
            
            result['success'] = True
            
            # Summary stats
            result['summary'] = {
                'total_text_detected': len(text_detections),
                'room_labels_found': len(room_labels),
                'regions_detected': len(wall_regions),
                'successful_matches': len(room_wall_mapping)
            }
            
        except Exception as e:
            result['error'] = str(e)
            print(f"Error in room analysis: {e}")
        
        return result
    
    def create_room_visualization(self, original_image, room_analysis, wall_parameters):
        """Create enhanced visualization showing detected rooms and their wall associations"""
        
        # Convert to RGB if needed
        if original_image.mode != 'RGB':
            vis_image = original_image.convert('RGB')
        else:
            vis_image = original_image.copy()
        
        draw = ImageDraw.Draw(vis_image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 9)
            large_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
        
        # Colors for different room types (more subtle/transparent effect)
        room_colors = {
            'bedroom': (255, 182, 193, 80),      # Light pink with transparency
            'bathroom': (173, 216, 230, 80),     # Light blue with transparency
            'kitchen': (255, 218, 185, 80),      # Peach with transparency
            'living_room': (144, 238, 144, 80),  # Light green with transparency
            'dining_room': (221, 160, 221, 80),  # Plum with transparency
            'office': (255, 228, 181, 80),       # Moccasin with transparency
            'closet': (211, 211, 211, 80),       # Light gray with transparency
            'laundry': (255, 160, 122, 80),      # Light salmon with transparency
            'garage': (169, 169, 169, 80),       # Dark gray with transparency
            'hallway': (245, 245, 220, 80),      # Beige with transparency
            'unknown': (255, 255, 255, 60)       # White with transparency
        }
        
        # Draw all detected text boxes first (red boxes)
        text_detections = room_analysis.get('text_detections', [])
        for detection in text_detections:
            bbox = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            
            # Draw bounding box for all detected text
            points = [(int(p[0]), int(p[1])) for p in bbox]
            draw.polygon(points, outline=(255, 0, 0), width=1)
            
            # Draw text with confidence
            center = detection['center']
            label = f"{text} ({confidence:.2f})"
            draw.text((center[0], center[1] - 15), label, fill=(255, 0, 0), font=small_font)
        
        # Draw room regions and mappings
        room_mappings = room_analysis.get('room_wall_mapping', [])
        for mapping in room_mappings:
            room_type = mapping['room_type']
            room_name = mapping['room_name']
            region_info = mapping['region_info']
            match_quality = mapping.get('match_quality', 'unknown')
            
            # Get color for room type
            color_rgba = room_colors.get(room_type, room_colors['unknown'])
            color_rgb = color_rgba[:3]  # Just RGB for outlines
            
            # Draw room region with actual contour boundaries
            center = region_info['center']
            
            # Check if we have contour data (from new flood-fill method)
            if 'contour' in region_info and region_info['contour']:
                # Draw actual room contour
                contour_points = region_info['contour']
                
                # Convert to tuple format for PIL
                contour_tuples = [(int(pt[0]), int(pt[1])) for pt in contour_points]
                
                # Draw filled polygon
                if len(contour_tuples) >= 3:
                    draw.polygon(contour_tuples, outline=color_rgb, width=3)
                    
                    # Draw inner outline for visual distinction
                    draw.polygon(contour_tuples, outline=color_rgb, width=1)
            else:
                # Fallback to rectangle for backward compatibility
                width = region_info['dimensions']['width']
                height = region_info['dimensions']['height']
                
                margin = 10
                room_bbox = [
                    center[0] - width/2 + margin,
                    center[1] - height/2 + margin,
                    center[0] + width/2 - margin,
                    center[1] + height/2 - margin
                ]
                
                draw.rectangle(room_bbox, outline=color_rgb, width=3)
            
            # Draw room label with background
            label_text = f"{room_name}"
            
            # Calculate text position
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            label_x = center[0] - text_width // 2
            label_y = center[1] - text_height // 2
            
            # Draw label background
            draw.rectangle([
                label_x - 3, label_y - 2,
                label_x + text_width + 3, label_y + text_height + 2
            ], fill=(255, 255, 255), outline=color_rgb, width=1)
            
            # Draw room name
            draw.text((label_x, label_y), label_text, fill=(0, 0, 0), font=font)
            
            # Draw connected walls info (smaller text below)
            connected_walls = mapping['connected_walls']
            if connected_walls:
                wall_text = f"Walls: {', '.join(connected_walls[:4])}"  # Show first 4 walls
                if len(connected_walls) > 4:
                    wall_text += f" +{len(connected_walls)-4}"
                
                wall_label_y = label_y + text_height + 5
                draw.text((label_x, wall_label_y), wall_text, fill=color_rgb, font=small_font)
            
            # Draw match quality indicator
            quality_text = f"Quality: {match_quality}"
            quality_color = {
                'excellent': (0, 128, 0),
                'good': (0, 100, 0),
                'fair': (255, 165, 0),
                'poor': (255, 0, 0),
                'unknown': (128, 128, 128)
            }.get(match_quality, (128, 128, 128))
            
            draw.text((label_x, wall_label_y + 12), quality_text, fill=quality_color, font=small_font)
        
        # Draw room labels that weren't successfully matched (in yellow)
        room_labels = room_analysis.get('room_labels', [])
        matched_room_names = set(mapping['room_name'] for mapping in room_mappings)
        
        for room_label in room_labels:
            if room_label['text'] not in matched_room_names:
                center = room_label['center']
                text = room_label['text']
                
                # Draw unmatched room label
                draw.ellipse([
                    center[0] - 15, center[1] - 15,
                    center[0] + 15, center[1] + 15
                ], outline=(255, 255, 0), width=2, fill=(255, 255, 0, 100))
                
                draw.text((center[0] + 20, center[1] - 10), f"Unmatched: {text}", 
                         fill=(255, 165, 0), font=small_font)
        
        # Enhanced legend
        legend_y = 10
        draw.text((10, legend_y), "Room Detection Results:", fill=(0, 0, 0), font=large_font)
        legend_y += 25
        
        # Summary info with more detail
        summary = room_analysis.get('summary', {})
        summary_lines = [
            f"Text detected: {summary.get('total_text_detected', 0)}",
            f"Room labels identified: {summary.get('room_labels_found', 0)}",
            f"Spatial regions found: {summary.get('regions_detected', 0)}",
            f"Successful room-wall matches: {summary.get('successful_matches', 0)}"
        ]
        
        for line in summary_lines:
            draw.text((10, legend_y), line, fill=(0, 0, 0), font=small_font)
            legend_y += 15
        
        # Legend for colors
        legend_y += 10
        draw.text((10, legend_y), "Room Types:", fill=(0, 0, 0), font=font)
        legend_y += 18
        
        # Show legend for detected room types only
        detected_room_types = set(mapping['room_type'] for mapping in room_mappings)
        for room_type in detected_room_types:
            if room_type in room_colors:
                color = room_colors[room_type][:3]  # RGB only
                draw.rectangle([10, legend_y, 25, legend_y+10], fill=color, outline=(0, 0, 0))
                draw.text((30, legend_y-2), room_type.replace('_', ' ').title(), 
                         fill=(0, 0, 0), font=small_font)
                legend_y += 15
        
        # Legend for indicators
        legend_y += 5
        draw.text((10, legend_y), "Indicators:", fill=(0, 0, 0), font=small_font)
        legend_y += 15
        
        indicators = [
            ("Red boxes: All detected text", (255, 0, 0)),
            ("Colored regions: Matched rooms", (0, 100, 0)),
            ("Yellow circles: Unmatched text", (255, 255, 0))
        ]
        
        for text, color in indicators:
            draw.text((10, legend_y), text, fill=color, font=small_font)
            legend_y += 12
        
        return vis_image


def save_room_analysis(room_data, filename):
    """Save room analysis to JSON file"""
    import os
    
    ROOT_DIR = os.path.abspath("./")
    filepath = os.path.join(ROOT_DIR, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(room_data, f, indent=2)
        print(f"Room analysis saved to: {filepath}")
        return filename
    except Exception as e:
        print(f"Error saving room analysis: {str(e)}")
        return None 