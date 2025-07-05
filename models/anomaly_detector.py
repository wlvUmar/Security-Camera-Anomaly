import cv2
import time
import logging
import numpy as np
from typing import List
from ultralytics import YOLO
from collections import deque, defaultdict
from model import predict_video_anomaly

logger = logging.getLogger(__name__)
logger.info(__name__)


class AnomalyDetector:
    def __init__(self, model_path: str = "utils/yolov8s.pt"):
        self.yolo_model = YOLO(model_path)
        self.speed_ratio = 2.0
        self.stride = 30
        self.sequence_length = 150
        self.distance_ratio = 0.5
        self.trajectory_memory = 30
        self.object_tracks = {}
        self.frame_buffer = deque(maxlen=self.trajectory_memory)
        self.anomaly_memory = defaultdict(lambda: deque(maxlen=5))
        self.allowed_objects = {"person", "chair", "tv"}

    def process_stream(self, rtsp_url, threshold=0.5):
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"Warning: Could not open {rtsp_url}")
            return

        frames = []
        frame_count = 0
        sequence_length = self.sequence_length
        stride = self.stride
        overlap = sequence_length // 2

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or failed. Reopening...")
                cap.release()
                time.sleep(2)  
                cap = cv2.VideoCapture(rtsp_url)
                continue

            if frame_count % stride == 0:
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame)

            frame_count += 1

            while len(frames) >= sequence_length:
                sequence_frames = frames[:sequence_length]

                results = predict_video_anomaly(sequence_frames, threshold=threshold)

                yield {
                    'start_frame': frame_count - len(frames),
                    'end_frame': frame_count - len(frames) + sequence_length,
                    'anomaly_results': results
                }

                frames = frames[overlap:]


    def get_unexpected_objects(self) -> List[str]:
        unexpected = set()

        if not self.frame_buffer:
            return []

        latest_frame = self.frame_buffer[-1]
        for obj in latest_frame['objects']:
            if obj['class_name'] not in self.allowed_objects and obj["confidence"] > 0.6:
                logger.debug(f"Obj: {obj['class_name']}, Conf: {obj['confidence']}")
                unexpected.add(obj['class_name'])

        return list(unexpected)

   
    def detect_anomaly(self, frame: np.ndarray):
        timestamp = time.time()
        