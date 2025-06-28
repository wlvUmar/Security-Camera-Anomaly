import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import cv2
from collections import deque, defaultdict, Counter
import time
import logging

logger = logging.getLogger(__name__)
logger.info(__name__)


class AnomalyDetector:
    def __init__(self, model_path: str = "utils/yolov8s.pt"):
        self.yolo_model = YOLO(model_path)
        self.speed_ratio = 2.0
        self.distance_ratio = 0.5
        self.trajectory_memory = 30
        self.object_tracks = {}
        self.frame_buffer = deque(maxlen=self.trajectory_memory)
        self.anomaly_memory = defaultdict(lambda: deque(maxlen=5))
        self.allowed_objects = {"person", "chair", "tv"}

    def extract_features(self, frame: np.ndarray) -> Dict:
        results = self.yolo_model(frame, verbose=False)

        features = {
            'objects': [],
            'total_objects': 0,
            'frame_timestamp': time.time()
        }

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for i, (box, clas, conf) in enumerate(zip(boxes, classes, confidences)):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    obj_data = {
                        'id': i,
                        'class': int(clas),
                        'class_name': self.yolo_model.names[int(clas)],
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center_x), float(center_y)],
                        'area': float((x2 - x1) * (y2 - y1))
                    }
                    features['objects'].append(obj_data)

                features['total_objects'] = len(features['objects'])
        logger.debug(f"Detected {features['total_objects']} objects.")

        return features

    def calculate_motion(self, current_features: Dict, frame_shape: Tuple) -> Dict:
        motion_data = {
            'total_motion': 0,
            'anomalous_objects': [],
            'motion_intensity': 0
        }

        if len(self.frame_buffer) < 2:
            self.frame_buffer.append(current_features)
            return motion_data

        prev_features = self.frame_buffer[-1]
        time_diff = current_features['frame_timestamp'] - \
            prev_features['frame_timestamp']

        if time_diff <= 0:
            return motion_data

        total_motion_pixels = 0

        for curr_obj in current_features['objects']:
            curr_center = curr_obj['center']
            curr_box = curr_obj['bbox']  # [x1, y1, x2, y2]
            box_w = curr_box[2] - curr_box[0]
            box_h = curr_box[3] - curr_box[1]
            obj_diag = (box_w ** 2 + box_h ** 2) ** 0.5

            distance_threshold = self.distance_ratio * obj_diag
            speed_threshold = self.speed_ratio * obj_diag / time_diff

            min_distance = float('inf')
            matched_obj = None

            for prev_obj in prev_features['objects']:
                if prev_obj['class'] == curr_obj['class']:
                    prev_center = prev_obj['center']
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 +
                                       (curr_center[1] - prev_center[1])**2)
                    if distance < min_distance:
                        min_distance = distance
                        matched_obj = prev_obj

            if matched_obj and min_distance < distance_threshold:
                speed = min_distance / time_diff
                total_motion_pixels += min_distance

                if speed > speed_threshold:
                    motion_data['anomalous_objects'].append({
                        'object': curr_obj,
                        'speed': speed,
                        'distance_moved': min_distance,
                        'anomaly_type': 'high_speed'
                    })

        motion_data['total_motion'] = total_motion_pixels
        motion_data['motion_intensity'] = total_motion_pixels / \
            (frame_shape[0] * frame_shape[1])

        self.frame_buffer.append(current_features)
        return motion_data

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

    def detect_covered(self, frame: np.ndarray, base_threshold: float = 50.0) -> np.bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        std_dev = np.std(gray)
        edge_density = np.sum(edge > 0) / gray.size
        is_covered = std_dev < 10 and edge_density < 0.01
        logger.debug(f"std: {std_dev}, edge_density: {edge_density}")
        return is_covered

    def detect_anomaly(self, frame: np.ndarray) -> Dict:
        timestamp = time.time()
        is_covered = self.detect_covered(frame)
        self.anomaly_memory["covered"].append(is_covered)
        if sum(self.anomaly_memory["covered"]) >= 3:
            return {
                'is_anomaly': True,
                'confidence': 1.0,
                'anomaly_reasons': [f"Camera is covered"],
                'features': None,
                'motion_data': None,
                'timestamp': timestamp
            }

        features = self.extract_features(frame)
        motion_data = self.calculate_motion(features, frame.shape[:2])
        unallowed = set(self.get_unexpected_objects())
        self.anomaly_memory["unallowed_objects"].append(unallowed)
        flat = [cls for frame_objs in self.anomaly_memory["unallowed_objects"]
                for cls in frame_objs]
        counts = Counter(flat)

        persistent_unallowed = [cls for cls,
                                count in counts.items() if count >= 3]

        reasons = []

        if persistent_unallowed:
            logger.warning(f"Unexpected objects detected: {persistent_unallowed}")
            reasons.append(f"Unallowed objects detected: {', '.join(persistent_unallowed)}")

        self.anomaly_memory["unusual_movement"].append(
            bool(motion_data["anomalous_objects"]))
        self.anomaly_memory["intense_motion"].append(
            motion_data["motion_intensity"] > 0.1)

        if sum(self.anomaly_memory["unusual_movement"]) >= 3:
            logger.warning("Unusual object movement detected")
            reasons.append("Unusual object movement detected")

        if sum(self.anomaly_memory["intense_motion"]) >= 3:
            logger.warning("High motion intensity")
            reasons.append("High motion intensity")

        is_anomaly = bool(reasons)
        if features['objects']:
            avg_conf = np.mean([obj['confidence']
                                for obj in features['objects']])
        else:
            avg_conf = 0.0
        motion_score = motion_data['motion_intensity']
        confidence = min(1.0, 0.5 * avg_conf + 0.5 * motion_score * 2)
        if motion_data['motion_intensity'] > 0:
            confidence = min(1.0, motion_data['motion_intensity'] * 2)

        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'anomaly_reasons': reasons,
            'features': features,
            'motion_data': motion_data,
            'timestamp': timestamp
        }
