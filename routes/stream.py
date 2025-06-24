import cv2
import time
import logging
import numpy as np

from fastapi import APIRouter, WebSocket, UploadFile, File, HTTPException, Query, Depends
from starlette.websockets import WebSocketState
from models.dataset import SecurityCameraDataset
from utils import get_from_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stream_router = APIRouter(prefix="/stream", tags=["Stream"])

@stream_router.websocket("/video")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    vp = websocket.app.state.video_processor
    
    try:
        while True:
            data = await websocket.receive_bytes()
            
            nparr = np.frombuffer(data, np.uint8)
            try:
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Frame decode failed")
            except Exception as e:
                logger.error(f"Decode error: {e}")
            
            result = await vp.process_frame(frame)
            
            response = {
                "is_anomaly": result.get('is_anomaly', False),
                "confidence": result.get('confidence', 0.0),
                "timestamp": result.get('timestamp', time.time()),
                "anomaly_reasons": result.get('anomaly_reasons', []),
                "frame_number": vp.processed_frames
            }
            
            if 'anomaly_id' in res ult:
                response['anomaly_id'] = result['anomaly_id']
            
            await websocket.send_json(response)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"Connection state before close: {websocket.application_state}")
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
        logger.info("WebSocket connection closed")


@stream_router.get("/recent")
async def get_recent_frames(count: int = Query(10, ge=1, le=50), dataset = Depends(get_from_state("stream_dataset"))):
    """Get recent frames from the stream buffer"""
    try:
        recent_frames = dataset.get_recent_frames(count)
        
        frame_data = []
        for frame_info in recent_frames:
            frame_data.append({
                "timestamp": frame_info['timestamp'],
                "label": frame_info['label'],
                "shape": frame_info['frame'].shape,
                "has_frame": True  # We don't send actual frame data via REST
            })
        
        return {
            "recent_frames": frame_data,
            "count": len(frame_data)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving recent frames: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/upload-frame")
async def upload_frame(file: UploadFile = File(...), vp=Depends(get_from_state("video_processor"))):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        result = await vp.process_frame(frame)
        
        return {
            "filename": file.filename,
            "is_anomaly": result.get('is_anomaly', False),
            "confidence": result.get('confidence', 0.0),
            "anomaly_reasons": result.get('anomaly_reasons', []),
            "timestamp": result.get('timestamp', time.time()),
            "anomaly_id": result.get('anomaly_id')
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@stream_router.post("/dataset-create")
async def create_dataset(data_path: str):
    """Create a dataset from a directory of videos/images"""
    try:
        if not os.path.exists(data_path):
            raise HTTPException(status_code=400, detail="Data path does not exist")
        
        dataset = SecurityCameraDataset(data_path)
        unlabeled_indices = dataset.get_unlabeled_items()
        
        return {
            "message": "Dataset created successfully",
            "total_items": len(dataset),
            "unlabeled_items": len(unlabeled_indices),
            "data_path": data_path
        }
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))