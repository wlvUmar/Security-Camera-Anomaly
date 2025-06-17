from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
from typing import List
import json

app = FastAPI(title="Security Camera Anomaly Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoProcessor:
    def __init__(self):
        self.anomaly_threshold = 0.8
        self.detected_anomalies = []

    async def process_frame(self, frame: np.ndarray) -> dict:
        # TODO: Implement actual anomaly detection logic
        # This is a placeholder that simulates anomaly detection
        is_anomaly = np.random.random() > self.anomaly_threshold
        if is_anomaly:
            self.detected_anomalies.append({
                "timestamp": str(asyncio.get_event_loop().time()),
                "confidence": float(np.random.random())
            })
        return {"is_anomaly": is_anomaly}

video_processor = VideoProcessor()

@app.get("/")
async def root():
    return {"status": "active", "service": "Security Camera Anomaly Detector"}

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            result = await video_processor.process_frame(frame)
            await websocket.send_json(result)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = await video_processor.process_frame(frame)
    return result

@app.get("/anomalies")
async def get_anomalies():
    return {"anomalies": video_processor.detected_anomalies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 