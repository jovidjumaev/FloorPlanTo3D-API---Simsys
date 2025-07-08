# Millimeter Conversion Feature

## Overview

The Floor Plan to 3D API now supports automatic conversion of pixel measurements to millimeters. This allows you to get real-world measurements from your floor plan analysis.

## How to Use

### 1. Determine Your Scale Factor

The scale factor represents how many millimeters each pixel represents in the real world. You need to determine this based on your floor plan:

- **If you know the real-world dimensions**: 
  - Measure a known distance on your floor plan (e.g., a wall that's 3000mm long)
  - Count how many pixels that distance is in the image
  - Scale factor = real_distance_mm / pixels_in_image

- **Example**: If a 3000mm wall is 600 pixels long in your image:
  - Scale factor = 3000mm / 600px = 5.0 mm/pixel
  - This means 1 pixel = 5.0 mm in the real world

### 2. API Usage

Include the `scale_factor_mm_per_pixel` parameter in your POST request:

```python
import requests

# Example with scale factor of 0.5 mm/pixel
with open('floorplan.jpg', 'rb') as f:
    files = {'image': f}
    data = {'scale_factor_mm_per_pixel': '0.5'}
    
    response = requests.post('http://localhost:5000/analyze_walls', 
                           files=files, data=data)
```

### 3. Response Format

The API now returns both pixel and millimeter measurements:

```json
{
  "metadata": {
    "scale_factor_mm_per_pixel": 0.5,
    "image_dimensions": {"width": 1200, "height": 800}
  },
  "summary": {
    "total_wall_length_px": 5000.0,
    "total_wall_length_mm": 2500.0,
    "average_wall_thickness_px": 15.0,
    "average_wall_thickness_mm": 7.5
  },
  "individual_walls": [
    {
      "wall_id": "W1",
      "centerline_px": [[100, 200], [150, 200]],
      "centerline_mm": [[50.0, 100.0], [75.0, 100.0]],
      "length_px": 50.0,
      "length_mm": 25.0,
      "thickness_px": {
        "average": 15.0,
        "min": 12.0,
        "max": 18.0,
        "profile": [15.0, 16.0, 14.0]
      },
      "thickness_mm": {
        "average": 7.5,
        "min": 6.0,
        "max": 9.0,
        "profile": [7.5, 8.0, 7.0]
      },
      "bbox_px": {"x1": 95, "y1": 195, "x2": 155, "y2": 205},
      "bbox_mm": {"x1": 47.5, "y1": 97.5, "x2": 77.5, "y2": 102.5}
    }
  ],
  "junctions": [
    {
      "junction_id": "J1",
      "position": [50.0, 100.0],  // Now in millimeters
      "junction_type": "corner"
    }
  ]
}
```

## Supported Endpoints

The millimeter conversion feature is available on these endpoints:

1. **`/analyze_walls`** - Comprehensive wall parameter analysis
2. **`/visualize_walls`** - Wall analysis with visualization

## Testing

Use the provided test script to verify the feature:

```bash
python test_mm_conversion.py your_floorplan.jpg 0.5
```

This will:
- Send your image to the API with the specified scale factor
- Display a summary of the results
- Save detailed results to a JSON file

## Common Scale Factors

Here are some typical scale factors for different floor plan types:

- **High-resolution CAD drawings**: 0.1 - 0.5 mm/pixel
- **Standard architectural drawings**: 0.5 - 2.0 mm/pixel
- **Scanned blueprints**: 1.0 - 5.0 mm/pixel
- **Mobile phone photos**: 2.0 - 10.0 mm/pixel

## Tips for Accurate Measurements

1. **Use known reference objects**: Include a scale bar or measure a known distance in your floor plan
2. **High-resolution images**: Higher resolution images provide more accurate measurements
3. **Consistent lighting**: Ensure good contrast between walls and background
4. **Orthogonal photography**: Take photos from directly above to minimize perspective distortion

## Troubleshooting

- **Default scale factor**: If no scale factor is provided, it defaults to 1.0 (1 pixel = 1 mm)
- **Negative values**: Scale factors must be positive
- **Very small values**: Use appropriate precision (e.g., 0.001 for very detailed drawings)
- **Large values**: For very coarse measurements, use larger scale factors

## Example Use Cases

1. **Construction planning**: Get exact wall lengths for material estimation
2. **Furniture placement**: Measure room dimensions for furniture planning
3. **Renovation projects**: Calculate areas and perimeters for cost estimation
4. **Architectural documentation**: Create precise as-built drawings 