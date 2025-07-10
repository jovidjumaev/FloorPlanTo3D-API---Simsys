"""
Health check routes
"""

from flask import Blueprint, jsonify
import logging
from models.mask_rcnn_model import get_config, is_model_initialized

logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('health', __name__)


@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify model status"""
    try:
        if not is_model_initialized():
            return jsonify({
                "status": "error",
                "message": "Model not initialized",
                "model_loaded": False
            }), 503
        
        cfg = get_config()
        return jsonify({
            "status": "healthy",
            "message": "Model loaded and ready",
            "model_loaded": True,
            "model_config": {
                "name": cfg.NAME,
                "num_classes": cfg.NUM_CLASSES,
                "detection_min_confidence": cfg.DETECTION_MIN_CONFIDENCE,
                "image_max_dim": cfg.IMAGE_MAX_DIM
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "model_loaded": False
        }), 500 