# Security Camera Anomaly Detection System

A scalable system for detecting anomalies in security camera feeds using YOLOv8 for feature extraction and motion tracking.

## Features

- **Real-time Processing**: WebSocket support for live video streams
- **Motion Tracking**: Advanced motion detection and speed analysis
- **Automatic Labeling**: Uses YOLOv8 for feature extraction and labeling
- **Temporal Storage**: Thread-safe storage with configurable retention
- **RESTful API**: Complete API for frame upload and anomaly retrieval
- **Scalable Architecture**: Clean separation of concerns

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/wlvUmar/Security-Camera-Anomaly.git
cd Security-Camera-Anomaly

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 
```

### 3. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

#### Upload Frame
```bash
curl -X POST "http://localhost:8000/upload/frame" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Get Anomalies
```bash
# Get recent anomalies
curl "http://localhost:8000/anomalies?limit=10"

# Filter by timeframe
curl "http://localhost:8000/anomalies?timeframe=24h"

# Get statistics
curl http://localhost:8000/anomalies/stats/summary
```

## Architecture

### Core Components

1. **AnomalyDetector** (`models/anomaly_detector.py`)
   - YOLOv8-based feature extraction
   - Motion tracking and speed analysis
   - Configurable thresholds

2. **Dataset Classes** (`models/dataset.py`)
   - `SecurityCameraDataset`: For batch processing
   - `StreamDataset`: For real-time streaming

3. **Storage System** (`utils/storage.py`)
   - Thread-safe anomaly storage
   - Configurable retention policies
   - Export capabilities

### Data Flow

```
Video Frame → YOLOv8 Feature Extraction → Motion Analysis → Anomaly Detection → Storage
```

## Configuration

Environment variables (optional):

```bash
# Model settings
export YOLO_MODEL_PATH="yolov8s.pt"
export ANOMALY_THRESHOLD="0.7"
export MOTION_THRESHOLD="50"
export SPEED_THRESHOLD="100"

# Storage settings
export MAX_STORAGE_ITEMS="5000"
export RETENTION_HOURS="48"

# Server settings
export HOST="0.0.0.0"
export PORT="8000"
```

## WebSocket Usage

Connect to `ws://localhost:8000/ws/video` and send frame data as bytes:

```python
import websocket
import cv2
import numpy as np

def on_message(ws, message):
    result = json.loads(message)
    if result['is_anomaly']:
        print(f"Anomaly detected! Confidence: {result['confidence']}")

ws = websocket.WebSocketApp("ws://localhost:8000/ws/video",
                          on_message=on_message)

# Send frame
frame = cv2.imread("test_image.jpg")
_, buffer = cv2.imencode('.jpg', frame)
ws.send(buffer.tobytes(), websocket.ABNF.OPCODE_BINARY)
```

## Dataset Creation

```python
from models.dataset import SecurityCameraDataset

# Create dataset from video directory
dataset = SecurityCameraDataset("./path/to/videos")

# Access data
for i in range(len(dataset)):
    item = dataset[i]
    print(f"Image shape: {item['image'].shape}")
    print(f"Label: {item['label']}")
```

## Anomaly Types Detected

1. **High Speed Movement**: Objects moving faster than threshold
2. **High Motion Intensity**: Significant portion of frame in motion
3. **Object Density**: Too many objects in frame (crowding)

## API Documentation

Once running, visit `/docs` for interactive API documentation.

## Scaling Considerations

- **Horizontal Scaling**: Multiple instances can run simultaneously
- **Database Integration**: Replace in-memory storage with Redis/PostgreSQL
- **Load Balancing**: Use nginx or similar for production deployments
- **Container Deployment**: Dockerfile ready for containerization

## Performance Notes

- **YOLOv8s Model**: Good balance of speed and accuracy
- **Frame Skipping**: Configurable for video processing
- **Memory Management**: Automatic cleanup of old data
- **Threading**: Background cleanup doesn't block main processing



## Development

### Project Structure
```
security_camera_anomaly/
├── main.py                 # FastAPI application
├── models/
│   ├── anomaly_detector.py # Core detection logic
│   └── dataset.py          # Dataset classes
│
├── routes/
│   ├── anomaly.py 
│   └── stream.py    
│
├── utils/
│   ├── video_processing.py  
│   ├── storage.py          # Storage utilities
├── └── config.py
│  
└── requirements.txt
```


## License and Attribution
This project includes code from the YOLO project by Joseph Redmon et al., licensed under the GPL v3.

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
You can find the full license in the LICENSE.txt file.

Because this project uses AGPL, if you modify or use it over a network, you must make your source code available under the same license.

