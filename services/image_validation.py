"""
Image validation and memory monitoring services
"""

import logging
from PIL import Image
from config.settings import get_config

# Get application configuration
app_config = get_config()
logger = logging.getLogger(__name__)

# Optional memory monitoring
try:
    import psutil
    import gc
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


def check_memory_usage():
    """Monitor memory usage and warn if high"""
    if not MEMORY_MONITORING_AVAILABLE or not app_config.ENABLE_MEMORY_MONITORING:
        return 0
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > app_config.MAX_MEMORY_USAGE_MB:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            gc.collect()  # Force garbage collection
        
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return 0


def validate_and_resize_image(image, max_size=None):
    """
    Validate image size and resize if necessary
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height), defaults to config
    
    Returns:
        PIL Image: Original or resized image
        dict: Information about the resize operation
    """
    if max_size is None:
        max_size = app_config.MAX_IMAGE_SIZE
    
    original_size = image.size
    max_dimension = max(original_size)
    min_dimension = min(original_size)
    
    resize_info = {
        "original_size": original_size,
        "resized": False,
        "resize_factor": 1.0,
        "reason": "no_resize_needed"
    }
    
    # Check minimum size
    if min_dimension < app_config.MIN_IMAGE_SIZE:
        resize_info["reason"] = "image_too_small"
        logger.warning(f"Image too small: {original_size} (min: {app_config.MIN_IMAGE_SIZE})")
        return image, resize_info
    
    # Check maximum size
    if max_dimension > max_size and app_config.ALLOW_IMAGE_RESIZE:
        # Calculate resize factor
        resize_factor = max_size / max_dimension
        
        # Calculate new size maintaining aspect ratio
        new_width = int(original_size[0] * resize_factor)
        new_height = int(original_size[1] * resize_factor)
        new_size = (new_width, new_height)
        
        # Ensure minimum size after resize
        if min(new_size) < app_config.MIN_IMAGE_SIZE:
            logger.warning(f"Resize would make image too small: {new_size}")
            resize_info["reason"] = "resize_would_make_too_small"
            return image, resize_info
        
        # Resize image using high-quality method
        resize_method = getattr(Image, app_config.RESIZE_QUALITY, Image.LANCZOS)
        image = image.resize(new_size, resize_method)
        
        resize_info.update({
            "resized": True,
            "resize_factor": resize_factor,
            "new_size": new_size,
            "reason": "resized_for_performance"
        })
        
        logger.info(f"Image resized from {original_size} to {new_size} (factor: {resize_factor:.2f})")
    
    elif max_dimension > max_size and not app_config.ALLOW_IMAGE_RESIZE:
        resize_info["reason"] = "image_too_large_resize_disabled"
        logger.warning(f"Image too large: {original_size} (max: {max_size}) but resize is disabled")
    
    return image, resize_info 