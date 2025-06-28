import cv2
import time
import json
import logging
import numpy as np
from io import BytesIO, StringIO
from typing import Optional

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Depends, WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
from utils import TimeFrameManager, get_from_state


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

anomaly_router = APIRouter(prefix="/anomalies", tags=["Anomalies"])

@anomaly_router.get("/")
async def get_anomalies(
    limit: Optional[int] = Query(50, ge=1, le=1000),
    timeframe: Optional[str] = Query(None, regex="^(1h|6h|12h|24h|7d|30d)$"),
    label_filter: Optional[str] = Query(None),
    storage = Depends(get_from_state("anomaly_storage"))
):
    try:
        since_timestamp = None
        if timeframe:
            since_timestamp = TimeFrameManager.get_timeframe_start(timeframe)
        
        anomalies = storage.get_anomalies(
            limit=limit,
            since_timestamp=since_timestamp,
            label_filter=label_filter
        )
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "timeframe": timeframe,
            "filter_applied": label_filter is not None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.get("/{anomaly_id}")
async def get_anomaly_details(anomaly_id: str, storage = Depends(get_from_state("anomaly_storage"))):
    """Get detailed information about a specific anomaly"""
    try:
        anomaly = storage.get_anomaly_by_id(anomaly_id)
        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        return anomaly
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving anomaly {anomaly_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.get("/stats/summary")
async def get_anomaly_stats(as_and_vp = Depends(get_from_state("anomaly_storage", "video_processor"))):
    """Get anomaly statistics and summary"""
    try:
        storage, vp = as_and_vp
        stats = storage.get_stats()
        
        recent_anomalies = storage.get_anomalies(
            since_timestamp=TimeFrameManager.get_timeframe_start('24h')
        )
        
        hourly_data = TimeFrameManager.group_by_hour(recent_anomalies)
        
        return {
            "summary": stats,
            "trends": {
                "hourly_counts": hourly_data,
                "peak_hour": max(hourly_data.items(), key=lambda x: x[1])[0] if hourly_data else None
            },
            "processing_stats": {
                "total_frames_processed": vp.processed_frames,
                "anomaly_detection_rate": (
                    vp.total_anomalies / vp.processed_frames * 100
                    if vp.processed_frames > 0 else 0
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating anomaly stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.delete("/")
async def clear_anomalies(storage = Depends(get_from_state("anomaly_storage"))):
    try:
        storage.clear_all()
        return {"message": "All anomalies cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@anomaly_router.post("/export")
async def export_anomalies(storage = Depends(get_from_state("anomaly_storage"))):
    try:
        data = {
            'export_timestamp': time.time(),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'anomalies': storage.get_anomalies()
        }
        
        str_buffer = StringIO()
        json.dump(data, str_buffer, indent=2)
        str_buffer.seek(0)

        # Convert to bytes for StreamingResponse
        byte_buffer = BytesIO(str_buffer.read().encode("utf-8"))
        byte_buffer.seek(0)

        return StreamingResponse(
            byte_buffer,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=anomalies_export.json"}
        )

    except Exception as e:
        logger.error(f"Error exporting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
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
            logger.debug(response)
            # if 'anomaly_id' in result:
            #     response['anomaly_id'] = result['anomaly_id']
            # await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"Connection state before close: {websocket.application_state}")
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
                logger.info("WebSocket connection closed")
            except RuntimeError as e:
                logger.debug(e)
        
 