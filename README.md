# Security Camera Anomaly Detection System

A scalable system for detecting anomalies in security camera feeds using YOLOv8 for feature extraction and motion tracking.

## Features

- **Real-time Processing**: via CCTV / RTSP streams
- **Deep Learnng**: Advanced per object attention model  
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
curl http://localhost:8000/health
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

1. **Video Processor** (`utils/video_processor.py`)
   - processes streamed data
   - Configurable thresholds

2. **Model** (`models/model.py`)
   - `LongTermMemoryAnomalyDetector`: For anomaly detection
   - `AnomalyTrainer`: For easy Training/Tuning
   - `YOLOFeatureExtractor`: Extracts features from frames

3. **Storage System** (`utils/storage.py`)
   - Thread-safe anomaly storage
   - Configurable retention policies
   - Export capabilities

### Data Flow

```
Video stream → Video Processor. → Feature Extraction → model prediction → Storage -> API
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

## API Documentation

Once running, visit `/docs` for interactive API documentation.

## Scaling Considerations

- **Horizontal Scaling**: Multiple instances can run simultaneously
- **Database Integration**: Replace in-memory storage with Redis/PostgreSQL
- **Load Balancing**: Use nginx or similar for production deployments
- **Container Deployment**: Dockerfile ready for containerization

## Performance Notes
- **MHA + LSTM** : Can be Heavy
- **YOLOv8s Model**: Good balance of speed and accuracy
- **Frame Skipping**: Configurable for video processing
- **Memory Management**: Automatic cleanup of old data
- **Threading**: Background cleanup doesn't block main processing



## Development

### Project Structure
```
security_camera_anomaly/
├── main.py                 
├── models/
│   ├── anomaly_detector.py # Core detection logic
│   ├── dataset.py  # Dataset classes
│   └── model.py   # Model 
│
├── routes/
│   └── anomaly.py 
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

