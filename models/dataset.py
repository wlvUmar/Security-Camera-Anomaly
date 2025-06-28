import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from typing import List, Dict, Optional, Union
import os
import json
from pathlib import Path

class SecurityCameraDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 transform=None,
                 video_extensions: List[str] = ['.mp4', '.avi', '.mov'],
                 image_extensions: List[str] = ['.jpg', '.jpeg', '.png']):
        """
        Custom dataset for security camera data
        Supports both video files and image sequences
        Uses YOLOv8 for automatic labeling when labels are not provided
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.video_extensions = video_extensions
        self.image_extensions = image_extensions
        
        self.data_items = []
        self._load_data()
    
    def _load_data(self):
        if not self.data_path.exists():
            print(f"Warning: Data path {self.data_path} does not exist")
            return
        
        for ext in self.video_extensions:
            video_files = list(self.data_path.glob(f"*{ext}"))
            for video_file in video_files:
                self._extract_frames_from_video(video_file)

        for ext in self.image_extensions:
            image_files = list(self.data_path.glob(f"*{ext}"))
            for image_file in image_files:
                self.data_items.append({
                    'type': 'image',
                    'path': str(image_file),
                    'label': self._get_label(image_file)
                })
    
    def _extract_frames_from_video(self, video_path: Path, frame_skip: int = 5):

        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                frame_filename = f"{video_path.stem}_frame_{frame_count:06d}.jpg"
                frame_path = self.data_path / "extracted_frames" / frame_filename
                frame_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(frame_path), frame)
                
                self.data_items.append({
                    'type': 'video_frame',
                    'path': str(frame_path),
                    'video_source': str(video_path),
                    'frame_number': frame_count,
                    'label': self._get_label(frame_path)
                })
            
            frame_count += 1
        
        cap.release()
    
    def _get_label(self, file_path: Path) -> Optional[str]:
        label_file = file_path.with_suffix('.json')
        if label_file.exists():
            try:
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    return label_data.get('anomaly_label', 'normal')
            except:
                pass
        return None 
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        image = cv2.imread(item['path'])
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': item.get('label', 'unknown'),
            'path': item['path'],
            'metadata': {
                'type': item['type'],
                'frame_number': item.get('frame_number', 0),
                'video_source': item.get('video_source', '')
            }
        }
    
    def get_unlabeled_items(self) -> List[int]:
        unlabeled_indices = []
        for i, item in enumerate(self.data_items):
            if item.get('label') is None:
                unlabeled_indices.append(i)
        return unlabeled_indices
    
    def update_label(self, idx: int, label: str):
        if 0 <= idx < len(self.data_items):
            self.data_items[idx]['label'] = label
            
            # Save label to file
            item_path = Path(self.data_items[idx]['path'])
            label_file = item_path.with_suffix('.json')
            
            label_data = {
                'anomaly_label': label,
                'timestamp': str(np.datetime64('now')),
                'auto_generated': True
            }
            
            with open(label_file, 'w') as f:
                json.dump(label_data, f, indent=2)

class StreamDataset:
    def __init__(self, buffer_size: int = 100):
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.labels_buffer = []
        self.timestamps = []
    
    def add_frame(self, frame: np.ndarray, label: str = None, timestamp: float = None):
        if timestamp is None:
            timestamp = time.time()
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.frame_buffer.append(frame)
        self.labels_buffer.append(label)
        self.timestamps.append(timestamp)
        
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
            self.labels_buffer.pop(0)
            self.timestamps.pop(0)
    
    def get_recent_frames(self, count: int = 10) -> List[Dict]:
        recent_count = min(count, len(self.frame_buffer))
        recent_frames = []
        
        for i in range(-recent_count, 0):
            recent_frames.append({
                'frame': self.frame_buffer[i],
                'label': self.labels_buffer[i],
                'timestamp': self.timestamps[i]
            })
        
        return recent_frames
    
    def __len__(self):
        return len(self.frame_buffer)