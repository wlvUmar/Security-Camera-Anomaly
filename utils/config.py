import os
from pathlib import Path


class Settings:
    
    # Model settings
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "./utils/yolov8s.pt")
    ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.7"))
    MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "50"))
    SPEED_THRESHOLD = int(os.getenv("SPEED_THRESHOLD", "100"))

    # Storage settings
    MAX_STORAGE_ITEMS = int(os.getenv("MAX_STORAGE_ITEMS", "5000"))
    RETENTION_HOURS = int(os.getenv("RETENTION_HOURS", "48"))
    STREAM_BUFFER_SIZE = int(os.getenv("STREAM_BUFFER_SIZE", "200"))

    # Data paths
    DATA_PATH = os.getenv("DATA_PATH", "./utils/data")
    EXPORT_PATH = os.getenv("EXPORT_PATH", "./utils/exports")

    # Server settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "5"))
    MAX_OBJECTS_THRESHOLD = int(os.getenv("MAX_OBJECTS_THRESHOLD", "10"))
    MOTION_INTENSITY_THRESHOLD = float(
        os.getenv("MOTION_INTENSITY_THRESHOLD", "0.1"))

    WS_TIMEOUT = int(os.getenv("WS_TIMEOUT", "300"))

    @classmethod
    def create_directories(cls):
        Path(cls.DATA_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.EXPORT_PATH).mkdir(parents=True, exist_ok=True)
        Path(cls.DATA_PATH + "/extracted_frames").mkdir(parents=True, exist_ok=True)


settings = Settings()

settings.create_directories()
