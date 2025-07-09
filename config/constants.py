import os

# Project root directory
ROOT_DIR = os.path.abspath("./")

# Output directories
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGES_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "images")
JSON_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "json")

# Create output directories if they don't exist
os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# Image processing constants
DOOR_MASK_DILATION_KERNEL_SIZE = (15, 15)
DOOR_MASK_LARGE_DILATION_KERNEL_SIZE = (35, 35)
WINDOW_MASK_DILATION_KERNEL_SIZE = (10, 10)
WINDOW_MASK_LARGE_DILATION_KERNEL_SIZE = (20, 20)

# Margin constants
DOOR_BBOX_MARGIN = 40
WINDOW_BBOX_MARGIN = 25
WINDOW_EXCLUSION_MARGIN = 6

# Detection thresholds
OFFICE_PLAN_THRESHOLD = 7
BOUNDARY_MARGIN_PERCENT = 0.03
OVERLAP_THRESHOLD = 0.3

# Font settings
DEFAULT_FONT_SIZE = 12
SPACE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10

# Colors (RGB tuples)
WALL_COLOR = (255, 0, 0)
WINDOW_COLOR = (0, 255, 0)
DOOR_COLOR = (0, 0, 255)
CENTERLINE_COLOR = (255, 255, 0)
JUNCTION_COLOR = (255, 0, 255)