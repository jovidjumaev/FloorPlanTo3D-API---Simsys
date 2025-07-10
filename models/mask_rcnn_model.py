"""
MaskRCNN model configuration and initialization
"""

import os
import logging
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from config.settings import get_config

# Get application configuration
app_config = get_config()
logger = logging.getLogger(__name__)

# Global variables for model
_model = None
_cfg = None


class PredictionConfig(Config):
    """MaskRCNN model configuration"""
    # define the name of the configuration
    NAME = app_config.MODEL_NAME
    # number of classes (background + door + wall + window)
    NUM_CLASSES = app_config.NUM_CLASSES
    # simplify GPU config
    GPU_COUNT = app_config.GPU_COUNT
    IMAGES_PER_GPU = app_config.IMAGES_PER_GPU
    # Lower detection threshold to capture faint walls/doors
    DETECTION_MIN_CONFIDENCE = app_config.DETECTION_MIN_CONFIDENCE
    # Allow larger input images to preserve line detail
    IMAGE_MAX_DIM = app_config.IMAGE_MAX_DIM


def initialize_model():
    """Initialize the MaskRCNN model once at startup"""
    global _cfg, _model
    
    try:
        model_folder_path = os.path.abspath("./") + "/mrcnn"
        weights_path = os.path.join(app_config.WEIGHTS_FOLDER, app_config.WEIGHTS_FILE_NAME)
        
        # Check if weights file exists
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        _cfg = PredictionConfig()
        logger.info(f"Model configuration: {_cfg.IMAGE_RESIZE_MODE}")
        logger.info('==============Initializing model=========')
        
        _model = MaskRCNN(mode='inference', model_dir=model_folder_path, config=_cfg)
        logger.info('=================Model created==============')
        
        _model.load_weights(weights_path, by_name=True)
        logger.info('=================Model loaded successfully==============')
        
        return _model, _cfg
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise e


def get_model():
    """Get the initialized model"""
    global _model
    return _model


def get_config():
    """Get the model configuration"""
    global _cfg
    return _cfg


def is_model_initialized():
    """Check if model is initialized"""
    global _model, _cfg
    return _model is not None and _cfg is not None 