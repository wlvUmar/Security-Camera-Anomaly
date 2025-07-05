import cv2
import json
import torch
import logging 
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from collections import Counter
from torch.utils.data import Dataset

from utils import settings, setup_logging

warnings.filterwarnings('ignore')
setup_logging("model.log")
logger = logging.getLogger(__name__)
# feature shape (20 , 16)
class YOLOFeatureExtractor:
    def __init__(self):
        self.yolo = YOLO(settings.YOLO_MODEL_PATH)
        self.yolo.fuse()

    def extract_frame_features(self, frame, conf_threshold=0.3):
        results = self.yolo(frame, conf=conf_threshold, verbose=False)

        frame_features = []
        h, w = frame.shape[:2]
        object_counter = Counter()
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    class_name = int(boxes.cls[i].cpu().numpy())
                    object_counter[class_name] += 1
                    logger.debug(f"Detected: {class_name} (conf: {conf:.2f})")
                    cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
                    width, height = (x2 - x1) / w, (y2 - y1) / h
                    area = width * height
                    aspect_ratio = width / (height + 1e-6)

                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size > 0:
                        mean_intensity = np.mean(roi)
                        std_intensity = np.std(roi)

                        features = [
                            cx, cy,                    # Normalized center position
                            width, height,             # Normalized size
                            area,                      # Normalized area
                            aspect_ratio,              # Shape
                            conf,                      # Detection confidence
                            # Normalized class (COCO has 80 classes)
                            class_name / 80.0,
                            mean_intensity / 255.0,   # Appearance
                            std_intensity / 255.0,    # Texture variation
                            abs(cx - 0.5),            # Distance from center X
                            abs(cy - 0.5),            # Distance from center Y
                            min(cx, 1-cx),            # Distance to edge X
                            min(cy, 1-cy),            # Distance to edge Y
                            area * 100,               # Relative size
                            1.0 if area > 0.1 else 0.0,  # Large object flag
                        ]

                        frame_features.append(features)
        if object_counter:
            logger.debug(f"Frame object counts: {dict(object_counter)}")
        max_objects = 20
        feature_dim = 16

        if len(frame_features) == 0:
            return np.zeros((max_objects, feature_dim))

        frame_features = np.array(frame_features)

        if len(frame_features) > max_objects:
            confidences = frame_features[:, 6]  # Confidence is at index 6
            top_indices = np.argsort(confidences)[-max_objects:]
            frame_features = frame_features[top_indices]

        padded_features = np.zeros((max_objects, feature_dim))
        padded_features[:len(frame_features)] = frame_features

        return padded_features


class VideoAnomalyDataset(Dataset):
    def __init__(self, video_paths, labels, sequence_length=150, stride=30, transform=None):
        """
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=normal, 1=anomaly) for each video
            sequence_length: Number of frames per sequence (5-10 sec at 15-30fps)
            stride: Frame stride for sampling
        """
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform

        self.feature_extractor = YOLOFeatureExtractor()
        self.sequences = self._extract_sequences()

    def _extract_sequences(self):
        sequences = []

        for video_path, label in zip(self.video_paths, self.labels):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Warning: Could not open {video_path}")
                continue

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.stride == 0:
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)

                frame_count += 1

            cap.release()
            total_frames = len(frames)

            if total_frames < self.sequence_length:
                pad_count = self.sequence_length - total_frames 
                last_frame = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
                frames.extend([last_frame] * pad_count)
                sequences.append((frames, label))
            else:     
                for start_idx in range(0    , total_frames - self.sequence_length + 1, self.sequence_length // 2):
                    sequence_frames = frames[start_idx:start_idx + self.sequence_length]
                    if label == 1 and start_idx >= 0.7 * (total_frames - self.sequence_length):
                        seq_label = 1
                    else:
                        seq_label = 0
                    sequences.append((sequence_frames, seq_label))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames, label = self.sequences[idx]

        sequence_features = []
        prev_detections = {}  

        for frame_idx, frame in enumerate(frames):
            frame_features = self.feature_extractor.extract_frame_features(
                frame)

            enhanced_features = []

            for obj_idx, obj_features in enumerate(frame_features):
                if np.sum(obj_features) == 0:
                    enhanced_features.append(np.zeros(20))
                    continue

                basic_features = obj_features.copy()

                obj_key = f"{obj_idx}"

                if obj_key in prev_detections:
                    prev_pos = prev_detections[obj_key][:2]
                    curr_pos = obj_features[:2]

                    velocity_x = curr_pos[0] - prev_pos[0]
                    velocity_y = curr_pos[1] - prev_pos[1]
                    speed = np.sqrt(velocity_x**2 + velocity_y**2)

                    temporal_features = [
                        velocity_x, velocity_y, speed,
                        frame_idx / len(frames)  # Temporal position
                    ]
                else:
                    temporal_features = [0, 0, 0, frame_idx / len(frames)]

                prev_detections[obj_key] = obj_features[:2]

                combined_features = np.concatenate(
                    [basic_features, temporal_features])
                enhanced_features.append(combined_features)

            sequence_features.append(enhanced_features)

        sequence_tensor = torch.FloatTensor(sequence_features)
        label_tensor = torch.FloatTensor([label])

        return sequence_tensor, label_tensor


class LongTermMemoryAnomalyDetector(nn.Module):
    def __init__(self,
                 feature_dim=16,
                 max_objects=20,
                 hidden_size=256,
                 memory_size=512,
                 num_layers=3):
        super().__init__()

        self.feature_dim = feature_dim
        self.max_objects = max_objects
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.object_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size)
        )

        self.frame_aggregator = nn.Sequential(
            nn.Linear(hidden_size * max_objects, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.memory_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=memory_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=memory_size * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        self.anomaly_classifier = nn.Sequential(
            nn.Linear(memory_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(memory_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * max_objects),
            nn.ReLU(),
            nn.Linear(hidden_size * max_objects, feature_dim * max_objects)
        )

    def forward(self, sequence, return_attention=False):
        """
        Args:
            sequence: [batch_size, seq_len, max_objects, feature_dim]
        """
        batch_size, seq_len, max_objects, feature_dim = sequence.shape

        # Process each frame
        frame_representations = []

        for t in range(seq_len):
            # [batch, max_objects, feature_dim]
            sequence_frames = sequence[:, t, :, :]

            object_features = self.object_encoder(sequence_frames.view(-1, feature_dim)) # (20n, 256)
            
            object_features = object_features.view(batch_size, max_objects, -1) # (n, 20, 256)

            frame_feature = object_features.view(batch_size, -1) # (n, 5120)
            frame_representation = self.frame_aggregator(frame_feature) # (n, 256)

            frame_representations.append(frame_representation) # (, n, 256)

        sequence_features = torch.stack(frame_representations, dim=1) # (n, 150, 256)

        memory_output, _ = self.memory_lstm(sequence_features) # (n , 150 , 512)

        attended_features, attention_weights = self.attention(
            memory_output, memory_output, memory_output
        ) # (n, 150 ,1024) , weights 

        sequence_representation = torch.mean(attended_features, dim=1) # (n, 1024) 

        anomaly_score = self.anomaly_classifier(sequence_representation) # (n , 1)

        reconstruction = self.decoder(sequence_representation) # (n, 320)
        reconstruction = reconstruction.view(batch_size, max_objects, feature_dim) # (n, 20, 16)

        if return_attention:
            return anomaly_score, reconstruction, attention_weights

        return anomaly_score, reconstruction # (n,1) , (n,20,16)


class AnomalyTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # Combined loss: classification + reconstruction
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def train_epoch(self, dataloader, alpha=0.7):
        self.model.train()
        total_loss = 0

        for _, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device).squeeze()

            self.optimizer.zero_grad()

            anomaly_scores, reconstructions = self.model(sequences) # (n,1) , (n,20,16)
            anomaly_scores = anomaly_scores.squeeze() 

            clf_loss = self.bce_loss(anomaly_scores, labels)

            target_frame = sequences[:, 0, :, :]  # First frame
            recon_loss = self.mse_loss(reconstructions, target_frame)
            total_loss_batch = alpha * clf_loss + (1 - alpha) * recon_loss

            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total_loss_batch.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs=50):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), settings.MODEL_PATH)

            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device).squeeze()

                anomaly_scores, _ = self.model(sequences)
                anomaly_scores = anomaly_scores.squeeze()

                loss = self.bce_loss(anomaly_scores, labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)


def predict_video_anomaly(frames, threshold=0.5):
    model = LongTermMemoryAnomalyDetector()
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location='cpu'))
    model.eval()

    feature_extractor = YOLOFeatureExtractor()
    
    sequence_length = 150
    anomaly_scores = []

    for start_idx in range(0, len(frames) - sequence_length + 1, sequence_length // 2):
        sequence_frames = frames[start_idx:start_idx + sequence_length]

        sequence_features = []

        for frame_idx, frame in enumerate(sequence_frames):
            frame_features = feature_extractor.extract_frame_features(frame)

            enhanced_features = []
            for _, obj_features in enumerate(frame_features):
                if np.sum(obj_features) == 0:
                    enhanced_features.append(np.zeros(20))
                    continue

                temporal_features = [0, 0, 0, frame_idx / len(sequence_frames)]
                combined_features = np.concatenate(
                    [obj_features, temporal_features])
                enhanced_features.append(combined_features)

            sequence_features.append(enhanced_features)

        sequence_tensor = torch.FloatTensor(sequence_features).unsqueeze(0)

        with torch.no_grad():
            anomaly_score, _, attention_weights = model(
                sequence_tensor, return_attention=True)
            anomaly_scores.append({
                'start_frame': start_idx,
                'end_frame': start_idx + sequence_length,
                'anomaly_score': anomaly_score.item(),
                'is_anomaly': anomaly_score.item() > threshold,
                'attention_weights': attention_weights.cpu().numpy()
            })

    return anomaly_scores


