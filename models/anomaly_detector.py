import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
import cv2
from collections import deque
import time

class AnomalyDetector:
    def __init__(self, model_path: str = "utils/yolov8s.pt"):
        self.yolo_model = YOLO(model_path)
        self.motion_threshold = 50  # Minimum motion pixels for detection
        self.speed_threshold = 100  # Pixels per second for anomaly
        self.trajectory_memory = 10  # Frames to remember for motion analysis
        
        self.object_tracks = {}
        self.frame_buffer = deque(maxlen=self.trajectory_memory)
        
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
        current_time = current_features['frame_timestamp']
        prev_time = prev_features['frame_timestamp']
        time_diff = current_time - prev_time
        
        if time_diff <= 0:
            return motion_data
        
        total_motion_pixels = 0
        
        for curr_obj in current_features['objects']:
            curr_center = curr_obj['center']
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
            
            if matched_obj and min_distance < 200:  
                speed = min_distance / time_diff
                total_motion_pixels += min_distance
                
                if speed > self.speed_threshold:
                    motion_data['anomalous_objects'].append({
                        'object': curr_obj,
                        'speed': speed,
                        'distance_moved': min_distance,
                        'anomaly_type': 'high_speed'
                    })
        
        motion_data['total_motion'] = total_motion_pixels
        motion_data['motion_intensity'] = total_motion_pixels / (frame_shape[0] * frame_shape[1])
        
        self.frame_buffer.append(current_features)
        return motion_data
    
    def detect_anomaly(self, frame: np.ndarray) -> Dict:

        features = self.extract_features(frame)
        motion_data = self.calculate_motion(features, frame.shape[:2])
        is_anomaly = False
        anomaly_reasons = []
        
        if motion_data['anomalous_objects']:
            is_anomaly = True
            anomaly_reasons.append("Unusual object movement detected")
        
        if motion_data['motion_intensity'] > 0.1:  # 10% of frame area
            is_anomaly = True
            anomaly_reasons.append("High motion intensity")
        
        if features['total_objects'] > 10:  # Threshold for crowding
            is_anomaly = True
            anomaly_reasons.append("High object density")
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': min(1.0, motion_data['motion_intensity'] * 2),
            'anomaly_reasons': anomaly_reasons,
            'features': features,
            'motion_data': motion_data,
            'timestamp': time.time()
        }