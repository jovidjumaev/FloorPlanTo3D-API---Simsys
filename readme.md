# Enhanced FloorPlanTo3D API with Advanced Analysis

## Credits

This project is based on the original [FloorPlanTo3D-API](https://github.com/fadyazizz/FloorPlanTo3D-API) by Fady Aziz Ibrahim. We extended the original floor plan detection system with comprehensive architectural analysis capabilities and a modular, production-ready codebase.

## Features

- **Object Detection**: Walls, doors, and windows using Mask R-CNN
- **Wall Analysis**: Centerlines, thickness profiling, junction detection
- **Door & Window Analysis**: Orientation, swing direction, architectural insights
- **OCR Space Detection**: Multi-language text recognition with deduplication
- **Real-world Measurements**: Millimeter conversion from pixel coordinates
- **Enhanced Visualization**: Floor plan overlays with analysis results
- **Modular Architecture**: Clean, maintainable codebase with separated concerns

## Project Structure

```
├── models/           # Model configuration and initialization
├── services/         # Business logic (validation, analysis, JSON building)
├── routes/           # API endpoints organized as Flask blueprints
├── analysis/         # Floor plan analysis algorithms
├── image_processing/ # Image loading and processing
├── visualization/    # Visualization generation
├── config/          # Configuration management
└── utils/           # Utility functions
```

## Installation

1. **Setup Environment**
   ```bash
   conda create -n floorplan python=3.11
   conda activate floorplan
   pip install -r requirements.txt
   ```

2. **Download Model Weights**
   - Download `maskrcnn_15_epochs.h5` from original repository
   - Place in `weights/` folder

3. **Run Application**
   ```bash
   python application.py
   ```

## API Usage

### Complete Floor Plan Analysis
```bash
curl -X POST \
  -F "image=@your_image.png" \
  -F "scale_factor_mm_per_pixel=0.2645833333" \
  http://localhost:8080/visualize_walls
```

### Accuracy Assessment
```bash
curl -X POST \
  -F "image=@your_image.png" \
  http://localhost:8080/analyze_accuracy
```

### Health Check
```bash
curl http://localhost:8080/health
```

## Scale Factor Calculation

To get real-world measurements:
- Measure a known distance on your floor plan (e.g., 3000mm wall)
- Count pixels for that distance (e.g., 600 pixels)
- Scale factor = real_distance_mm / pixels = 3000/600 = 5.0 mm/pixel

**Common Values:**
- CAD drawings: 0.1-0.5 mm/pixel
- Architectural drawings: 0.5-2.0 mm/pixel
- Scanned blueprints: 1.0-5.0 mm/pixel

## Output Files

```
outputs/
├── images/vis{N}.png      # Visualization images
└── json/final{N}.json     # Analysis results
```

## Key Response Data

```json
{
  "summary": {
    "walls": {"total_walls": 15, "total_length_mm": 45000.0},
    "doors": {"total_doors": 8},
    "windows": {"total_windows": 6},
    "space_names": {"total_spaces_detected": 9}
  },
  "walls": {"individual_walls": [...], "junctions": [...]},
  "doors": {"detailed_doors": [...]},
  "windows": {"detailed_windows": [...]},
  "space_names": {"spaces": [...]}
}
```

## Notes

- Include `scale_factor_mm_per_pixel` for accurate measurements
- Processing time: 15-30 seconds depending on image complexity
- OCR supports English and Korean text
- Higher resolution images provide better accuracy
- Modular architecture enables easy maintenance and feature additions
