# OCR Text Detection for Floor Plans

## Overview

The Floor Plan to 3D API now includes Optical Character Recognition (OCR) capabilities to detect and identify space names and labels within floor plan images. This feature automatically recognizes text in multiple languages and identifies room names, area labels, and other spatial identifiers.

## Features

### Multi-Language Support
The OCR system supports text detection in multiple languages:
- **English**: Room, Kitchen, Bathroom, etc.
- **Spanish**: Habitación, Cocina, Baño, etc.
- **French**: Chambre, Cuisine, Salle de bain, etc.
- **German**: Zimmer, Küche, Bad, etc.
- **Italian**: Stanza, Cucina, Bagno, etc.
- **Portuguese**: Quarto, Cozinha, Banheiro, etc.
- **Russian**: Комната, Кухня, Ванная, etc.
- **Japanese**: 部屋, キッチン, 浴室, etc.
- **Korean**: 방, 부엌, 욕실, etc.
- **Chinese**: 房间, 厨房, 浴室, etc.

### Smart Text Recognition
- **Space Name Detection**: Automatically identifies room names and area labels
- **Confidence Scoring**: Each detected text comes with a confidence score
- **Text Cleaning**: Removes OCR artifacts and normalizes text
- **Pattern Matching**: Recognizes common space name patterns and abbreviations

### Integration with Visualization
- **Visual Overlay**: Detected space names are highlighted on the floor plan visualization
- **Bounding Boxes**: Orange boxes around detected text regions
- **Confidence Display**: Shows confidence scores for each detection
- **Legend Integration**: Space names are included in the visualization legend

## Usage

### API Endpoint
The OCR functionality is integrated into the `/visualize_walls` endpoint:

```bash
curl -X POST -F "image=@floor_plan.png" -F "scale_factor_mm_per_pixel=1.0" http://localhost:8080/visualize_walls
```

### Response Format
The API response now includes space name information:

```json
{
  "message": "Comprehensive floor plan analysis completed successfully",
  "visualization_file": "wall_vis7.png",
  "analysis_file": "walls7.json",
  "total_walls": 15,
  "total_doors": 8,
  "total_windows": 6,
  "total_junctions": 12,
  "total_space_names": 5,
  "comprehensive_summary": {
    "wall_count": 15,
    "door_count": 8,
    "window_count": 6,
    "junction_count": 12,
    "space_name_count": 5
  }
}
```

### Detailed Space Names Data
The JSON analysis file includes detailed space name information:

```json
{
  "space_names": {
    "total_spaces_detected": 5,
    "spaces": [
      {
        "id": "SPACE_1",
        "name": "Kitchen",
        "bbox": [120, 150, 180, 170],
        "confidence": 0.85,
        "center": {
          "x": 150,
          "y": 160
        }
      },
      {
        "id": "SPACE_2",
        "name": "Living Room",
        "bbox": [200, 100, 280, 140],
        "confidence": 0.92,
        "center": {
          "x": 240,
          "y": 120
        }
      }
    ]
  }
}
```

## Technical Implementation

### OCR Engine
- **EasyOCR**: Uses the EasyOCR library for robust text detection
- **Multi-language Models**: Pre-trained models for each supported language
- **GPU Support**: Can utilize GPU acceleration if available

### Image Preprocessing
1. **Grayscale Conversion**: Converts color images to grayscale
2. **Contrast Enhancement**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
3. **Noise Reduction**: Applies Gaussian blur to reduce noise
4. **Thresholding**: Creates binary image for better text detection
5. **Inversion**: Inverts image if text is dark on light background

### Text Validation
The system validates detected text using multiple criteria:

1. **Confidence Threshold**: Minimum confidence score of 0.3
2. **Pattern Matching**: Recognizes common space name patterns
3. **Length Validation**: Filters out very short or very long text
4. **Abbreviation Support**: Recognizes common abbreviations (RM, BR, BA, etc.)
5. **Room Number Detection**: Identifies room numbers and codes

### Supported Text Patterns
- **Full Names**: "Kitchen", "Bedroom", "Bathroom", etc.
- **Abbreviations**: "RM", "BR", "BA", "KT", "LV", etc.
- **Room Numbers**: "101", "A1", "Room 2", etc.
- **Multi-language**: Supports space names in 10 different languages

## Testing

### Test Script
Use the provided test script to verify OCR functionality:

```bash
python test_ocr.py
```

This script will:
1. Test OCR detection on a sample image
2. Test the API endpoint integration
3. Display detected space names and confidence scores

### Test Image Requirements
- Clear, high-resolution floor plan image
- Text should be clearly visible
- Good contrast between text and background
- Supported image formats: PNG, JPG, JPEG

## Configuration

### Language Selection
You can modify the supported languages in `ocr_detector.py`:

```python
languages=['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']
```

### Confidence Thresholds
Adjust confidence thresholds in the `_is_valid_space_name` method:

```python
if confidence < 0.3:  # Lower threshold for floor plan text
    return False
```

### Text Patterns
Add custom text patterns in the `space_patterns` list:

```python
space_patterns = [
    r'room', r'bedroom', r'bathroom',
    # Add your custom patterns here
]
```

## Performance Considerations

### Processing Time
- OCR detection typically takes 2-5 seconds depending on image size
- Larger images may take longer to process
- GPU acceleration can significantly improve performance

### Memory Usage
- EasyOCR models are loaded into memory on first use
- Multiple language models increase memory usage
- Consider using fewer languages if memory is limited

### Accuracy Tips
1. **Image Quality**: Use high-resolution images with clear text
2. **Contrast**: Ensure good contrast between text and background
3. **Orientation**: Text should be roughly horizontal or vertical
4. **Size**: Text should be large enough to be readable (minimum 10px height)

## Troubleshooting

### Common Issues

1. **No Text Detected**
   - Check image quality and contrast
   - Verify text is large enough
   - Ensure text is not rotated significantly

2. **Low Confidence Scores**
   - Improve image preprocessing
   - Check text clarity and contrast
   - Consider adjusting confidence thresholds

3. **Incorrect Text Recognition**
   - Verify language support
   - Check for OCR artifacts
   - Review text validation patterns

4. **Performance Issues**
   - Reduce number of supported languages
   - Use smaller images for testing
   - Enable GPU acceleration if available

### Error Messages
- **"OCR reader not initialized"**: Check EasyOCR installation
- **"Failed to initialize EasyOCR"**: Verify dependencies and language models
- **"No valid text regions"**: Check image quality and text visibility

## Dependencies

The OCR functionality requires:
- `easyocr==1.7.0`
- `opencv-python`
- `numpy`
- `PIL` (Pillow)

All dependencies are included in the main `requirements.txt` file.

## Future Enhancements

Potential improvements for the OCR system:
- **Custom Training**: Train models on specific floor plan text styles
- **Text Orientation**: Support for rotated text detection
- **Handwriting Recognition**: Support for handwritten labels
- **Symbol Recognition**: Detect and interpret architectural symbols
- **Context Awareness**: Use spatial relationships to improve text interpretation 