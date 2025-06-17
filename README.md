👁️ 3. Security Camera Anomaly Detector

A FastAPI-based security camera anomaly detection system that processes video streams and detects unusual movements or behaviors.

Features:
- Real-time video processing via WebSocket
- Video file upload support
- Anomaly detection using ML models
- RESTful API endpoints

Setup:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server:
   ```bash
   python main.py
   ```

API Endpoints:
- GET `/`: Health check
- WebSocket `/ws/video`: Real-time video processing
- POST `/upload/video`: Upload video file for processing
- GET `/anomalies`: Get list of detected anomalies

Input: Live video stream or video frames
Output: JSON alerts for unusual movements/behavior

ML: Autoencoder, YOLO + rules (to be implemented)
✅ Great for smart surveillance