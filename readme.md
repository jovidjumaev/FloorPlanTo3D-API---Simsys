# Enhanced FloorPlanTo3D API with Advanced Analysis

## Credits

This project is based on the original [FloorPlanTo3D-API](https://github.com/fadyazizz/FloorPlanTo3D-API) by Fady Aziz Ibrahim. We extended the original floor plan detection system with comprehensive architectural analysis capabilities and a modular, production-ready codebase.

## Features

This Flask API provides comprehensive floor plan analysis including:

- **Object Detection**: Walls, doors, and windows using Mask R-CNN
- **Wall Parameter Extraction**: Centerlines, thickness profiling, junction analysis
- **Door & Window Analysis**: Orientation, swing direction, and architectural insights
- **OCR Space Name Detection**: Multi-language text recognition with intelligent deduplication
- **Millimeter Conversion**: Real-world measurement conversion from pixel coordinates
- **Enhanced Visualization**: Floor plan overlays with analysis results
- **Organized Output**: Structured file system with separate directories for images and JSON
- **Accuracy Assessment**: Reliability scoring and detection quality analysis

## Architecture

The codebase has been completely refactored into a modular structure for better maintainability:

```
FloorPlanTo3D-API/
├── application.py              # Main Flask application
├── requirements.txt            # Dependencies (TensorFlow 2.13+)
├── config/
│   └── constants.py           # Configuration constants
├── utils/
│   ├── geometry.py            # Geometric calculations
│   ├── conversions.py         # Unit conversions
│   └── file_utils.py          # File management
├── image_processing/
│   ├── image_loader.py        # Image preprocessing
│   └── mask_processing.py     # Mask operations
├── analysis/
│   ├── door_analysis.py       # Door detection & analysis
│   ├── wall_analysis.py       # Wall parameter extraction
│   ├── junction_analysis.py   # Junction point detection
│   └── window_analysis.py     # Window analysis
├── visualization/
│   └── wall_visualization.py  # Enhanced visualizations
├── ocr_detector.py            # OCR space name detection
├── outputs/
│   ├── images/                # Generated PNG visualizations
│   └── json/                  # Analysis JSON files
└── weights/                   # Model weights
```

## Installation

### Prerequisites
- Python 3.8-3.11
- Conda or pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FloorPlanTo3D-API
   ```

2. **Create and activate environment**
   ```bash
   # Using conda (recommended)
   conda create -n floorplan python=3.11
   conda activate floorplan
   
   # Or using venv
   python -m venv floorplan
   source floorplan/bin/activate  # Linux/Mac
   # or
   floorplan\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**
   - Download `maskrcnn_15_epochs.h5` from the original repository
   - Place it in the `weights/` folder

5. **Run the application**
   ```bash
   python application.py
   ```

### Environment Notes
- **Avoid nested conda environments**: If you see `(floorplan)(base)`, deactivate base first:
  ```bash
  conda deactivate  # Exit base environment
  conda activate floorplan
  ```
- **M1 Mac users**: The requirements.txt includes optimized TensorFlow for Apple Silicon

## API Endpoints

### Comprehensive Floor Plan Analysis
```bash
# Complete analysis with walls, doors, windows, junctions, OCR, and visualization
curl -X POST \
  -F "image=@images/test3.png" \
  -F "scale_factor_mm_per_pixel=0.2645833333" \
  http://localhost:8080/visualize_walls
```

### Accuracy Assessment
```bash
# Reliability scoring and detection quality analysis
curl -X POST \
  -F "image=@images/test3.png" \
  http://localhost:8080/analyze_accuracy
```

## Millimeter Conversion

To get real-world measurements, include the `scale_factor_mm_per_pixel` parameter:

### Determining Scale Factor
- Measure a known distance on your floor plan (e.g., a wall that's 3000mm long)
- Count how many pixels that distance is in the image
- Scale factor = real_distance_mm / pixels_in_image

**Example**: If a 3000mm wall is 600 pixels long:
- Scale factor = 3000mm / 600px = 5.0 mm/pixel

### Common Scale Factors
- **High-resolution CAD drawings**: 0.1 - 0.5 mm/pixel
- **Standard architectural drawings**: 0.5 - 2.0 mm/pixel
- **Scanned blueprints**: 1.0 - 5.0 mm/pixel
- **Mobile phone photos**: 2.0 - 10.0 mm/pixel

## OCR Space Name Detection

### Enhanced Features
- **Intelligent Deduplication**: Removes duplicate detections from multiple OCR methods
- **Multi-language Support**: English and Korean text recognition
- **Compound Space Names**: Automatically merges "Great Room", "Master Bedroom", etc.
- **Quality Filtering**: Removes OCR noise and corrupted text

### Supported Text Patterns
- **Full Names**: "Kitchen", "Bedroom", "Bathroom", "Great Room", etc.
- **Abbreviations**: "RM", "BR", "BA", "KT", "LV", etc.
- **Room Numbers**: "101", "A1", "Room 2", "Bath1", "Bath2", etc.
- **Compound Names**: "Eat-in Kitchen", "Formal Living", "1/2 Bath", etc.
- **Korean Names**: "침실", "거실", "화장실", "주방", etc.

## Output Files

The API generates organized output files in structured directories:

### File Structure
```
outputs/
├── images/
│   └── vis{N}.png     # Enhanced wall visualizations
└── json/
    ├── final{N}.json  # Comprehensive analysis results
    └── acc{N}.json    # Accuracy assessment (when requested)
```

### File Naming Convention
- `vis1.png`, `vis2.png`, etc. - Visualization images
- `final1.json`, `final2.json`, etc. - Analysis results
- `acc1.json`, `acc2.json`, etc. - Accuracy reports

## Response Format

### Wall Analysis Response
```json
{
  "metadata": {
    "timestamp": "2025-01-09T14:07:13.123456",
    "scale_factor_mm_per_pixel": 0.2645833333,
    "units": "millimeters",
    "analysis_type": "comprehensive_floor_plan_analysis"
  },
  "summary": {
    "walls": {
      "total_walls": 15,
      "total_junctions": 12,
      "total_length_mm": 45000.0,
      "average_thickness_mm": 150.0
    },
    "exterior_walls": {
      "total_exterior_walls": 8,
      "total_interior_walls": 7,
      "perimeter_length_mm": 30000.0,
      "boundary_coverage": 0.85
    },
    "doors": {
      "total_doors": 8,
      "door_orientations": {
        "horizontal": 3,
        "vertical": 5
      }
    },
    "windows": {
      "total_windows": 6,
      "window_types": {
        "horizontal": 4,
        "vertical": 2
      }
    },
    "space_names": {
      "total_spaces_detected": 9,
      "average_confidence": 0.92,
      "languages_detected": ["en", "ko"]
    }
  },
  "walls": {
    "individual_walls": [
      {
        "wall_id": "W1",
        "centerline": [[50.0, 100.0], [75.0, 100.0]],
        "length": 25.0,
        "thickness": {"average": 7.5, "min": 6.0, "max": 9.0},
        "orientation_degrees": 0.0,
        "bbox": {"x1": 47.5, "y1": 97.5, "x2": 77.5, "y2": 102.5}
      }
    ],
    "junctions": [
      {
        "junction_id": "J1",
        "position": [50.0, 100.0],
        "junction_type": "corner",
        "connected_walls": ["W1", "W2"]
      }
    ],
    "exterior_walls": [...],
    "interior_walls": [...],
    "perimeter_analysis": {
      "total_perimeter_length": 30000.0,
      "perimeter_area": 75000000.0,
      "boundary_coverage": 0.85
    }
  },
  "doors": {
    "detailed_doors": [
      {
        "door_id": 1,
        "confidence": 0.95,
        "location": {
          "center": {"x": 150.0, "y": 160.0},
          "relative_position": {"from_left": "25.0%", "from_top": "30.0%"}
        },
        "dimensions": {"width": 800.0, "height": 2000.0},
        "orientation": {
          "door_type": "vertical",
          "estimated_swing": "opens_rightward",
          "confidence": 0.9
        },
        "architectural_analysis": {
          "door_type": "interior",
          "size_category": "standard",
          "accessibility": "compliant"
        }
      }
    ]
  },
  "windows": {
    "detailed_windows": [
      {
        "window_id": 1,
        "confidence": 0.88,
        "location": {
          "center": {"x": 200.0, "y": 100.0}
        },
        "dimensions": {"width": 1200.0, "height": 800.0},
        "window_type": "horizontal",
        "architectural_analysis": {
          "size_category": "standard",
          "glazing_type": "double_glazed"
        }
      }
    ]
  },
  "space_names": {
    "total_spaces_detected": 9,
    "spaces": [
      {
        "id": "SPACE_1",
        "name": "Kitchen",
        "confidence": 0.95,
        "method": "preprocessed",
        "center": {"x": 150.0, "y": 160.0},
        "center_mm": {"x": 39.7, "y": 42.3},
        "bbox": [100, 140, 200, 180],
        "bbox_mm": {"x1": 26.5, "y1": 37.0, "x2": 52.9, "y2": 47.6}
      }
    ],
    "detection_summary": {
      "average_confidence": 0.92,
      "detection_methods": ["original", "preprocessed", "aggressive", "preprocessed_upscaled"],
      "centerpoints_mm": [...]
    }
  }
}
```

## Technical Implementation

- **Mask R-CNN**: Object detection for walls, doors, and windows
- **Skeletonization**: Wall centerline extraction using morphological operations
- **Distance Transform**: Wall thickness calculation
- **Junction Detection**: Connectivity analysis for wall intersections
- **EasyOCR**: Multi-language text recognition with intelligent deduplication
- **Geometric Analysis**: Door orientation inference and window categorization
- **Modular Architecture**: Clean separation of concerns for maintainability

## Development Features

### Code Quality
- **Modular Design**: Organized into logical modules for easy maintenance
- **Import Resolution**: All circular imports and missing dependencies resolved
- **Error Handling**: Comprehensive error handling with logging
- **Type Safety**: Proper type hints and validation

### Output Management
- **Organized File Structure**: Separate directories for images and JSON files
- **Automatic Numbering**: Intelligent file numbering system
- **Backward Compatibility**: Checks both new and old file locations

### Performance Optimizations
- **Parallel Processing**: Multiple analysis steps run concurrently
- **Memory Efficiency**: Optimized mask processing and image handling
- **Caching**: Model loading optimization

## Important Notes

- **Scale Factor**: Include `scale_factor_mm_per_pixel` parameter for accurate real-world measurements (defaults to 1.0 if not provided)
- **Image Quality**: Higher resolution images provide more accurate measurements
- **Text Detection**: OCR works best with clear, high-contrast text
- **Processing Time**: Analysis typically takes 15-30 seconds depending on image size and complexity
- **File Management**: Output files are automatically organized in `outputs/` directory
- **Environment**: Use clean conda environment to avoid conflicts

## Troubleshooting

### Common Issues
- **TensorFlow errors**: Ensure you're using the correct Python version (3.8-3.11)
- **OCR not working**: Install EasyOCR dependencies: `pip install easyocr`
- **Missing weights**: Download `maskrcnn_15_epochs.h5` and place in `weights/` folder
- **Nested environments**: Deactivate base conda environment before activating floorplan

### Performance Tips
- **Use appropriate scale factors** for your image type
- **Provide high-resolution images** for better accuracy
- **Ensure good contrast** for text detection
- **Clean image backgrounds** improve detection quality

This enhanced system provides professional-grade architectural analysis capabilities for floor plan interpretation and building parameter extraction with a production-ready, modular codebase.
