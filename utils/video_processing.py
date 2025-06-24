import numpy as np 
from fastapi import Request, Depends
from typing import Union, Callable
import logging


logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, app_state):
        self.anomaly_detector = app_state.anomaly_detector
        self.anomaly_storage = app_state.anomaly_storage
        self.stream_dataset = app_state.stream_dataset
        self.processed_frames = 0
        self.total_anomalies = 0

    async def process_frame(self, frame: np.ndarray) -> dict:
        try:
            self.processed_frames += 1
            
            result = self.anomaly_detector.detect_anomaly(frame)
            
            if result['is_anomaly']:
                self.total_anomalies += 1
                
                anomaly_label = f"Motion Anomaly: {', '.join(result['anomaly_reasons'])}"
                
                anomaly_id = self.anomaly_storage.store_anomaly(
                    anomaly_label=anomaly_label,
                    confidence=result['confidence'],
                    metadata={
                        'features': result['features'],
                        'motion_data': result['motion_data'],
                        'frame_number': self.processed_frames
                    }
                )
                
                result['anomaly_id'] = anomaly_id
            
            label = "anomaly" if result['is_anomaly'] else "normal"
            self.stream_dataset.add_frame(frame, label, result['timestamp'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"error": str(e), "is_anomaly": False}




def get_from_state(*attrs: str) -> Callable[[Request], Union[object, tuple]]:
    def getter(request: Request):
        if len(attrs) == 1:
            return getattr(request.app.state, attrs[0])
        return tuple(getattr(request.app.state, attr) for attr in attrs)
    return getter