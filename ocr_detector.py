# OCR Detector for Floor Plan Text Recognition
# Detects space names and labels in multiple languages

import cv2
import numpy as np
from PIL import Image
import easyocr
import re
from typing import List, Dict, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple watermark keywords to filter out (English and Korean)
WATERMARK_KEYWORDS = [
    # English watermarks
    'alamy', 'shutterstock', 'getty', 'watermark', 'stock', 'copyright', '©', 'www', 'http', '.com',
    # Korean watermarks
    '워터마크', '저작권', '샘플', '견본', '미리보기'
]

class FloorPlanOCRDetector:
    """
    OCR detector specifically designed for floor plan text recognition.
    Detects space names and labels in multiple languages.
    """
    
    def __init__(self, languages=['en', 'ko']):
        """
        Initialize the OCR detector with support for multiple languages.
        
        Args:
            languages (list): List of language codes to support
                            Default: ['en', 'ko'] for English and Korean
        """
        self.languages = languages
        self.reader = None
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize the EasyOCR reader with specified languages."""
        try:
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.reader = None
    
    def preprocess_image_for_ocr(self, image: np.ndarray, save_debug=False) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy for floor plan text.
        
        Args:
            image: Input image as numpy array
            save_debug: Boolean to save the preprocessed image for debugging
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
            
            if save_debug:
                cv2.imwrite("ocr_preprocessed_debug.png", blurred)
            
            return blurred
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all text regions in the image using multiple approaches.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected text regions
        """
        if self.reader is None:
            logger.error("OCR reader not initialized")
            return []
        
        try:
            all_detections = []
            
            # Method 1: OCR on original image
            logger.info("Running OCR on original image...")
            try:
                results = self.reader.readtext(image)
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.1:  # Very low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'original'
                        })
            except Exception as e:
                logger.error(f"Error in original OCR: {str(e)}")
            
            # Method 2: OCR on preprocessed image
            logger.info("Running OCR on preprocessed image...")
            try:
                preprocessed = self.preprocess_image_for_ocr(image, save_debug=True)
                results = self.reader.readtext(preprocessed)
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.1:  # Very low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'preprocessed'
                        })
            except Exception as e:
                logger.error(f"Error in preprocessed OCR: {str(e)}")
            
            # Method 3: OCR with different parameters
            logger.info("Running OCR with enhanced parameters...")
            try:
                results = self.reader.readtext(image, 
                                            width_ths=0.2,  # Even lower width threshold
                                            height_ths=0.2,  # Even lower height threshold
                                            paragraph=False,  # Don't group into paragraphs
                                            detail=1)  # Return detailed results
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.03:  # Very low threshold for small text
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'enhanced'
                        })
            except Exception as e:
                logger.error(f"Error in enhanced OCR: {str(e)}")
            
            # Method 4: OCR with very aggressive settings for small text
            logger.info("Running OCR with aggressive settings for small text...")
            try:
                results = self.reader.readtext(image, 
                                            width_ths=0.1,  # Very low width threshold
                                            height_ths=0.1,  # Very low height threshold
                                            paragraph=False,
                                            detail=1,
                                            min_size=5)  # Allow very small text
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.02:  # Extremely low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'aggressive'
                        })
            except Exception as e:
                logger.error(f"Error in aggressive OCR: {str(e)}")
            
            # Remove duplicates with improved logic
            unique_detections = []
            
            # Sort by confidence (highest first) to keep the best detections
            all_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            for det in all_detections:
                is_duplicate = False
                current_text = det['text'].lower().strip()
                current_bbox = det['bbox']
                current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
                
                for existing in unique_detections:
                    existing_text = existing['text'].lower().strip()
                    existing_bbox = existing['bbox']
                    existing_center = ((existing_bbox[0] + existing_bbox[2]) / 2, (existing_bbox[1] + existing_bbox[3]) / 2)
                    
                    # Calculate distance between centers
                    distance = ((current_center[0] - existing_center[0])**2 + (current_center[1] - existing_center[1])**2)**0.5
                    
                    # Check for various types of duplicates
                    if distance < 50:  # Within 50 pixels
                        # Exact text match
                        if current_text == existing_text:
                            is_duplicate = True
                            break
                        
                        # One text is contained in the other (e.g., "BEDROOM" vs "BEDROOM 1")
                        if current_text in existing_text or existing_text in current_text:
                            is_duplicate = True
                            break
                        
                        # Similar text with small differences (OCR variations)
                        if len(current_text) > 3 and len(existing_text) > 3:
                            # Calculate text similarity
                            common_chars = sum(1 for a, b in zip(current_text, existing_text) if a == b)
                            similarity = common_chars / max(len(current_text), len(existing_text))
                            if similarity > 0.8:  # 80% similar
                                is_duplicate = True
                                break
                    
                    # Check for bbox overlap (even if centers are far apart)
                    overlap_x = max(0, min(current_bbox[2], existing_bbox[2]) - max(current_bbox[0], existing_bbox[0]))
                    overlap_y = max(0, min(current_bbox[3], existing_bbox[3]) - max(current_bbox[1], existing_bbox[1]))
                    overlap_area = overlap_x * overlap_y
                    
                    current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                    existing_area = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                    
                    # If bboxes overlap significantly and text is similar
                    if overlap_area > 0:
                        overlap_ratio = overlap_area / min(current_area, existing_area)
                        if overlap_ratio > 0.5 and (current_text in existing_text or existing_text in current_text):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_detections.append(det)
            
            logger.info(f"Detected {len(unique_detections)} unique text regions from {len(all_detections)} total detections")
            return unique_detections
            
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return []
    
    def filter_space_names(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Apply minimal filtering to keep only likely space names.
        
        Args:
            detections: List of text detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Filtered list of space names
        """
        img_h, img_w = image_shape
        filtered = []
        
        # Patterns that indicate descriptive text (not room names)
        descriptive_patterns = [
            r'\d+\s*(bedrooms|bathrooms)',  # "4 Bedrooms", "3 Bathrooms" (plural only)
            r'(bedrooms|bathrooms)\s*/\s*\d+',  # "Bedrooms/3" (plural only)
            r'\d+\s*bed\s*/\s*\d+\s*bath',  # "4 bed/3 bath"
            r'(sq\s*ft|square\s*feet|sqft)',  # Square footage
            r'(floor\s*plan|house\s*plan|plan\s*view)',  # Plan descriptions
            r'(scale|drawing|blueprint|architect)',  # Technical terms
            r'(total|approx|approximately)',  # Measurement terms
            r'(main\s*floor|upper\s*floor|lower\s*floor)',  # Floor descriptions
            r'(first\s*floor|second\s*floor|ground\s*floor)',  # Floor levels
            r'(lot\s*size|home\s*size|building\s*size)',  # Size descriptions
        ]
        
        # Patterns for valid room names (including fractions and abbreviations)
        valid_room_patterns = [
            # English patterns
            r'^\d+/\d+\s*(bath|bathroom)$',  # "1/2 Bath", "3/4 Bathroom"
            r'^(half|quarter)\s*(bath|bathroom)$',  # "Half Bath", "Quarter Bathroom"
            r'^(powder|guest|master|en)\s*(room|bath|bathroom)$',  # "Powder Room", "Guest Bath"
            r'^(walk|walk-in)\s*(closet|pantry)$',  # "Walk-in Closet"
            
            # Korean patterns (common room names)
            r'.*?(주방|부엌)',  # Kitchen
            r'.*?(거실|응접실)',  # Living room
            r'.*?(침실|안방|방)',  # Bedroom
            r'.*?(화장실|욕실|변소)',  # Bathroom
            r'.*?(식당|다이닝)',  # Dining room
            r'.*?(서재|공부방|사무실)',  # Study/Office
            r'.*?(현관|입구)',  # Entrance
            r'.*?(베란다|발코니)',  # Balcony
            r'.*?(다용도실|세탁실)',  # Utility room
            r'.*?(창고|저장실)',  # Storage
            r'.*?(드레스룸|옷장)',  # Dressing room/Closet
            r'.*?(팬트리)',  # Pantry
            r'.*?(계단|층계)',  # Stairs
        ]
        
        # Korean room keywords for additional validation
        korean_room_keywords = [
            '주방', '부엌', '거실', '응접실', '침실', '안방', '방',
            '화장실', '욕실', '변소', '식당', '다이닝', '서재',
            '공부방', '사무실', '현관', '입구', '베란다', '발코니',
            '다용도실', '세탁실', '창고', '저장실', '드레스룸',
            '옷장', '팬트리', '계단', '층계'
        ]
        
        for det in detections:
            text = det['text'].strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Skip obvious watermarks
            text_lower = text.lower()
            if any(watermark in text_lower for watermark in WATERMARK_KEYWORDS):
                logger.debug(f"Filtered watermark: {text}")
                continue
            
            # Skip standalone numbers (likely room numbers detected separately)
            if text.strip().isdigit() and len(text.strip()) <= 2:
                logger.debug(f"Filtered standalone number: {text}")
                continue
            
            # Skip single characters or very short text (likely OCR noise)
            if len(text.strip()) <= 1:
                logger.debug(f"Filtered single character: {text}")
                continue
            
            # Check if it's a valid room name first (before filtering)
            is_valid_room = False
            
            # Check English patterns (case-insensitive)
            for pattern in valid_room_patterns:
                if re.search(pattern, text_lower):
                    logger.debug(f"Kept valid room pattern '{pattern}': {text}")
                    is_valid_room = True
                    break
            
            # Check Korean keywords (case-sensitive for Korean)
            if not is_valid_room:
                for keyword in korean_room_keywords:
                    if keyword in text:  # Korean text is case-sensitive
                        logger.debug(f"Kept Korean room keyword '{keyword}': {text}")
                        is_valid_room = True
                        break
            
            # If it's a valid room, skip all other filters
            if not is_valid_room:
                # Skip descriptive patterns
                is_descriptive = False
                for pattern in descriptive_patterns:
                    if re.search(pattern, text_lower):
                        logger.debug(f"Filtered descriptive pattern '{pattern}': {text}")
                        is_descriptive = True
                        break
                
                if is_descriptive:
                    continue
                
                # Skip text with numbers and room counts (like "4 Bedrooms/3 Bathrooms")
                if re.search(r'\d+.*\d+', text) and any(word in text_lower for word in ['bedrooms', 'bathrooms']):
                    logger.debug(f"Filtered room count: {text}")
                    continue
            
            # Skip very long text (likely descriptions or titles)
            # Korean text can be more compact, so adjust length limits
            max_length = 20 if any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in text) else 30
            if len(text) > max_length:
                logger.debug(f"Filtered long text: {text}")
                continue
            
            # Skip text with multiple words that might be descriptive
            # Korean text doesn't use spaces the same way, so be more lenient
            words = text.split()
            has_korean = any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in text)
            max_words = 5 if has_korean else 3  # Allow more "words" for Korean mixed text
            if len(words) > max_words:
                logger.debug(f"Filtered multi-word text: {text}")
                continue
            
            # Skip text that's too small or too large
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area < 20:  # Too small
                logger.debug(f"Filtered too small: {text}")
                continue
                
            if area > 8000:  # Reduced from 10000 to catch more descriptive text
                logger.debug(f"Filtered too large: {text}")
                continue
            
            # Skip text in the top 20% of the image (likely titles/descriptions)
            center_y = (y1 + y2) / 2
            if center_y < img_h * 0.20:  # Increased from 0.15 to catch more titles
                logger.debug(f"Filtered title area: {text}")
                continue
            
            # Skip text in the bottom 15% of the image (likely legends)
            if center_y > img_h * 0.85:  # Increased from 0.90 to catch more legends
                logger.debug(f"Filtered legend area: {text}")
                continue
            
            # Skip text in the left and right margins (5% each side)
            center_x = (x1 + x2) / 2
            if center_x < img_w * 0.05 or center_x > img_w * 0.95:
                logger.debug(f"Filtered margin text: {text}")
                continue
            
            # Keep everything else
            filtered.append(det)
            logger.debug(f"Kept space name: {text}")
        
        logger.info(f"Filtered to {len(filtered)} space names from {len(detections)} detections")
        return filtered
    
    def _clean_room_name(self, text: str) -> str:
        """
        Clean up room name text by removing common OCR artifacts and extra characters.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned room name
        """
        # Remove extra whitespace
        cleaned = text.strip()
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s/\-가-힣]', '', cleaned)  # Keep alphanumeric, spaces, slashes, hyphens, Korean
        
        # Remove trailing numbers that might be room numbers detected as part of the name
        # But keep fractions like "1/2"
        if not re.search(r'\d+/\d+', cleaned):  # If it's not a fraction
            cleaned = re.sub(r'\s+\d+$', '', cleaned)  # Remove trailing numbers
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Capitalize properly for English text
        if not any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in cleaned):  # If not Korean
            cleaned = cleaned.title()
        
        return cleaned.strip()

    def get_space_names(self, image: np.ndarray, wall_bboxes: Optional[List[List[int]]] = None) -> List[Dict]:
        """
        Main method to detect space names in a floor plan image.
        
        Args:
            image: Input floor plan image as numpy array
            wall_bboxes: List of wall bounding boxes [x1, y1, x2, y2] (optional)
            
        Returns:
            List of dictionaries containing space name information
        """
        try:
            # Detect all text regions
            text_regions = self.detect_text_regions(image)
            
            # Apply minimal filtering
            filtered_regions = self.filter_space_names(text_regions, image.shape[:2])
            
            # Format as space names with text cleanup
            space_names = []
            for i, region in enumerate(filtered_regions):
                # Clean up the text
                cleaned_text = self._clean_room_name(region['text'])
                
                space_name = {
                    'id': f'SPACE_{i+1}',
                    'name': cleaned_text,
                    'bbox': region['bbox'],
                    'confidence': region['confidence'],
                    'method': region.get('method', 'unknown'),
                    'center': {
                        'x': (region['bbox'][0] + region['bbox'][2]) / 2,
                        'y': (region['bbox'][1] + region['bbox'][3]) / 2
                    }
                }
                space_names.append(space_name)
            
            logger.info(f"Final result: {len(space_names)} space names detected")
            return space_names
            
        except Exception as e:
            logger.error(f"Error in space name detection: {str(e)}")
            return []
    
    def detect_text_in_region(self, image: np.ndarray, region_bbox: List[int]) -> List[Dict]:
        """
        Detect text within a specific region of the image.
        
        Args:
            image: Input image
            region_bbox: Bounding box [x1, y1, x2, y2] of the region to analyze
            
        Returns:
            List of text detections within the region
        """
        try:
            x1, y1, x2, y2 = region_bbox
            
            # Ensure coordinates are within image bounds
            img_h, img_w = image.shape[:2]
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1, min(x2, img_w))
            y2 = max(y1, min(y2, img_h))
            
            # Extract region
            region_image = image[y1:y2, x1:x2]
            
            # Detect text in region
            text_regions = self.detect_text_regions(region_image)
            
            # Adjust coordinates back to original image
            for region in text_regions:
                region['bbox'][0] += x1
                region['bbox'][1] += y1
                region['bbox'][2] += x1
                region['bbox'][3] += y1
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error in region text detection: {str(e)}")
            return []

# Global OCR detector instance
_ocr_detector = None

def get_ocr_detector() -> FloorPlanOCRDetector:
    """
    Get or create the global OCR detector instance.
    
    Returns:
        FloorPlanOCRDetector instance
    """
    global _ocr_detector
    if _ocr_detector is None:
        _ocr_detector = FloorPlanOCRDetector()
    return _ocr_detector

def detect_space_names(image: np.ndarray, wall_bboxes: Optional[List[List[int]]] = None) -> List[Dict]:
    """
    Convenience function to detect space names in an image.
    
    Args:
        image: Input image as numpy array
        wall_bboxes: List of wall bounding boxes [x1, y1, x2, y2] (optional)
        
    Returns:
        List of space name dictionaries
    """
    detector = get_ocr_detector()
    return detector.get_space_names(image, wall_bboxes=wall_bboxes)

def detect_text_in_region(image: np.ndarray, region_bbox: List[int]) -> List[Dict]:
    """
    Convenience function to detect text in a specific region.
    
    Args:
        image: Input image as numpy array
        region_bbox: Bounding box [x1, y1, x2, y2] of the region
        
    Returns:
        List of text detections within the region
    """
    detector = get_ocr_detector()
    return detector.detect_text_in_region(image, region_bbox) 