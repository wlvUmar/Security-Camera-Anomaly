import time
import torch
import cv2
import logging
import numpy as np
from fastapi import Request, Depends
from typing import Union, Callable
from models.model import YOLOFeatureExtractor, LongTermMemoryAnomalyDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, app_state):
        self.app_state = app_state
        self.anomaly_storage = app_state.anomaly_storage
        self.stream_dataset = app_state.stream_dataset
        self.processed_frames = 0
        self.total_anomalies = 0
        self.cap = cv2.VideoCapture("rtsp://192.168.100.184:8554/stream")
        self.frame_buffer = []
        self.sequence_length = 150
        self.stride = 30
        self.frame_count = 0
        self.feature_extractor = YOLOFeatureExtractor()
        self.prev_detections = {}

    async def process_frame(self, frame=None) -> dict:
        try:
            self.processed_frames += 1
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                frame = frame if frame else None
                logger.debug(
                    "couldn't connect to RTSP stream. processing a passed frame")

            if not ret or frame is None:
                raise ValueError("Failed to read frame from video source.")

            result = await self.process_streaming_anomaly(frame)

            if result['is_anomaly']:
                self.total_anomalies += 1

                anomaly_id = self.anomaly_storage.store_anomaly(
                    anomaly_label="nul",
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
        except RuntimeError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return {"error": str(e), "is_anomaly": False}

    async def process_streaming_anomaly(self, frame):
        try:
            frame = cv2.resize(frame, (640, 480))

            if self.frame_count % self.stride == 0:
                self.frame_buffer.append(frame)

            self.frame_count += 1

            if len(self.frame_buffer) < self.sequence_length:
                return {
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'timestamp': time.time(),
                    'features': {},
                    'motion_data': {}
                }

            if len(self.frame_buffer) > self.sequence_length:
                self.frame_buffer = self.frame_buffer[-self.sequence_length:]

            sequence_features = []
            temp_prev_detections = self.prev_detections.copy()

            for frame_idx, buff_frame in enumerate(self.frame_buffer):
                frame_features = self.feature_extractor.extract_frame_features(
                    buff_frame)

                enhanced_features = []
                for obj_idx, obj_features in enumerate(frame_features):
                    if np.sum(obj_features) == 0:
                        enhanced_features.append(np.zeros(20))
                        continue

                    obj_key = f"{obj_idx}"

                    if obj_key in temp_prev_detections:
                        prev_pos = temp_prev_detections[obj_key][:2]
                        curr_pos = obj_features[:2]

                        velocity_x = curr_pos[0] - prev_pos[0]
                        velocity_y = curr_pos[1] - prev_pos[1]
                        speed = np.sqrt(velocity_x**2 + velocity_y**2)

                        temporal_features = [
                            velocity_x, velocity_y, speed,
                            frame_idx / len(self.frame_buffer)
                        ]
                    else:
                        temporal_features = [
                            0, 0, 0, frame_idx / len(self.frame_buffer)]

                    temp_prev_detections[obj_key] = obj_features[:2]

                    combined_features = np.concatenate(
                        [obj_features, temporal_features])
                    enhanced_features.append(combined_features)

                sequence_features.append(enhanced_features)

            self.prev_detections = temp_prev_detections

            sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)

            with torch.no_grad():
                anomaly_score, _, attention_weights = self.app_state.model(sequence_tensor, return_attention=True)

                is_anomaly = anomaly_score.item() > 0.5
                confidence = anomaly_score.item()

            return {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'timestamp': time.time(),
                'features': {'attention_weights': attention_weights.cpu().numpy().tolist()},
                'motion_data': {'sequence_length': len(self.frame_buffer)}
            }

        except Exception as e:
            logger.error(f"Error in streaming anomaly detection: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'timestamp': time.time(),
                'features': {},
                'motion_data': {}
            }

    def load_model(self):
        from utils import settings
        if not hasattr(self.app_state, "model"):

            model = LongTermMemoryAnomalyDetector()

            try:
                model.load_state_dict(torch.load(
                    settings.MODEL_PATH, map_location='cpu'))
                model.eval()
                logger.info("Anomaly detection model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
            self.app_state.model = model
        


def get_from_state(*attrs: str) -> Callable[[Request], Union[object, tuple]]:
    def getter(request: Request):
        if len(attrs) == 1:
            return getattr(request.app.state, attrs[0])
        return tuple(getattr(request.app.state, attr) for attr in attrs)
    return getter

async def run_video_processing(vp):
    while True:
        await vp.process_frame()
 