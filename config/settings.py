"""
Application configuration settings
Centralized configuration management for the FloorPlanTo3D API
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration class"""
    
    # Model settings
    MODEL_NAME = "mask_rcnn_hq"
    WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'
    WEIGHTS_FOLDER = "./weights"
    
    # Model configuration
    NUM_CLASSES = 4  # background + door + wall + window
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.15
    IMAGE_MAX_DIM = 1600
    
    # Performance settings
    MAX_IMAGE_SIZE = 2048  # Maximum image dimension to prevent memory issues
    MIN_IMAGE_SIZE = 100   # Minimum image dimension
    ALLOW_IMAGE_RESIZE = True
    RESIZE_QUALITY = 'LANCZOS'  # or 'BICUBIC', 'BILINEAR'
    ENABLE_CACHING = True
    CACHE_TIMEOUT = 300  # 5 minutes
    
    # Memory management
    MAX_MEMORY_USAGE_MB = 1024  # 1GB limit
    ENABLE_MEMORY_MONITORING = True
    
    # API settings
    HOST = '0.0.0.0'
    PORT = 8080
    DEBUG = True
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # CORS settings
    CORS_ORIGINS = "*"
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration dictionary"""
        return {
            'NAME': cls.MODEL_NAME,
            'NUM_CLASSES': cls.NUM_CLASSES,
            'GPU_COUNT': cls.GPU_COUNT,
            'IMAGES_PER_GPU': cls.IMAGES_PER_GPU,
            'DETECTION_MIN_CONFIDENCE': cls.DETECTION_MIN_CONFIDENCE,
            'IMAGE_MAX_DIM': cls.IMAGE_MAX_DIM
        }
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration dictionary"""
        return {
            'HOST': cls.HOST,
            'PORT': cls.PORT,
            'DEBUG': cls.DEBUG
        }

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    ENABLE_CACHING = True
    CACHE_TIMEOUT = 600  # 10 minutes

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    ENABLE_CACHING = False

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config(environment: str = None) -> Config:
    """Get configuration for the specified environment"""
    if environment is None:
        environment = os.getenv('FLASK_ENV', 'development')
    
    return config_map.get(environment, DevelopmentConfig) 