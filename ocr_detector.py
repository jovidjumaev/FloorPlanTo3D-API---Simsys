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

# Simple watermark keywords to filter out
WATERMARK_KEYWORDS = ['alamy', 'shutterstock', 'getty', 'watermark', 'stock', 'copyright', '©', 'www', 'http', '.com']

class FloorPlanOCRDetector:
    """
    OCR detector specifically designed for floor plan text recognition.
    Detects space names and labels in multiple languages.
    """
    
    def __init__(self, languages=['en']):
        """
        Initialize the OCR detector with support for multiple languages.
        
        Args:
            languages (list): List of language codes to support
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
                                            width_ths=0.3,  # Lower width threshold
                                            height_ths=0.3,  # Lower height threshold
                                            paragraph=False,  # Don't group into paragraphs
                                            detail=1)  # Return detailed results
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.05:  # Even lower threshold
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
            
            # Remove duplicates based on text and position
            unique_detections = []
            seen = set()
            
            for det in all_detections:
                # Create a key based on text and approximate position
                text_key = det['text'].lower().strip()
                bbox_key = (det['bbox'][0] // 10, det['bbox'][1] // 10)  # Approximate position
                key = (text_key, bbox_key)
                
                if key not in seen:
                    seen.add(key)
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
            r'\d+\s*(bedroom|bathroom|bed|bath|br|ba)',  # "4 Bedrooms", "3 Bathrooms"
            r'(bedroom|bathroom|bed|bath|br|ba)\s*/\s*\d+',  # "Bedrooms/3"
            r'\d+\s*bed\s*/\s*\d+\s*bath',  # "4 bed/3 bath"
            r'(sq\s*ft|square\s*feet|sqft)',  # Square footage
            r'(floor\s*plan|house\s*plan|plan\s*view)',  # Plan descriptions
            r'(scale|drawing|blueprint|architect)',  # Technical terms
            r'(total|approx|approximately)',  # Measurement terms
            r'(main\s*floor|upper\s*floor|lower\s*floor)',  # Floor descriptions
            r'(first\s*floor|second\s*floor|ground\s*floor)',  # Floor levels
            r'(lot\s*size|home\s*size|building\s*size)',  # Size descriptions
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
            if re.search(r'\d+.*\d+', text) and any(word in text_lower for word in ['bedroom', 'bathroom', 'bed', 'bath']):
                logger.debug(f"Filtered room count: {text}")
                continue
            
            # Skip very long text (likely descriptions or titles)
            if len(text) > 30:  # Reduced from 50 to catch more descriptive text
                logger.debug(f"Filtered long text: {text}")
                continue
            
            # Skip text with multiple words that might be descriptive
            words = text.split()
            if len(words) > 3:  # More than 3 words is likely descriptive
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
            
            # Format as space names
            space_names = []
            for i, region in enumerate(filtered_regions):
                space_name = {
                    'id': f'SPACE_{i+1}',
                    'name': region['text'],
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